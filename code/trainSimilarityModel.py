import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.utils import ToTensor
from utils.ExecutionTimeTracker import ExecutionTimeTracker
import numpy as np
import pickle

from config import cfg
from models.orthosimilarity import BranchNet, DecisionNet
from datasourcing.BitmapDataset import BitmapDataset
from torchvision import transforms
import matplotlib.pyplot as plt

from datasourcing.transformations import albTransform

def getConcatenatedBitmapDataset(bitmapDatasetDescriptions, albTransform=None, verbose=False):
    dataset_parts = []
    
    for area in bitmapDatasetDescriptions:
            if (verbose):
                print("Loading dataset {}".format(area))

            dataset = BitmapDataset(bitmapDatasetDescriptions[area],
                cfg.sampledimensions.dimension_px,
                cfg.training.marginBetweenSamples_px,
                cfg.training.homographyCornerErrorStd_px,
                cfg.training.rotationErrorStd_deg,
                cfg.training.translationErrorStd_px,
                transform=transforms.Compose([ToTensor()]),
                apShiftingStd=cfg.training.apShiftingStd,
                scaleStd=cfg.training.scaleStd,
                albTransform=albTransform)
            
            dataset_parts.append(dataset)
    
    concatenatedDataset = torch.utils.data.ConcatDataset(dataset_parts)
    return concatenatedDataset

def getTrainingAndTestingDatasets(printStatistics=False, albTransform_bitmap=None):
    if (printStatistics):
        print("Loading training data")

    bitmapDatasetDescriptions_training = cfg.data.path.trainingdata_bitmaps
    trainingdataset_bitmap = getConcatenatedBitmapDataset(bitmapDatasetDescriptions_training,albTransform_bitmap,verbose=printStatistics)
     
    if (printStatistics):
        print("Loading testing datasets")
    
    bitmapDatasetDescriptions_testing = cfg.data.path.testingdata_bitmaps
    testingdataset_bitmap = getConcatenatedBitmapDataset(bitmapDatasetDescriptions_testing,verbose=printStatistics)
    
    if (printStatistics):
        print("Loading datasets completed.")
        print("Number of training samples: bitmap: {}".format(len(trainingdataset_bitmap)))
        print("Number of testing samples: bitmap: {}".format(len(testingdataset_bitmap)))

    return (trainingdataset_bitmap, testingdataset_bitmap)

if __name__ == '__main__':

    device = torch.device(cfg.device)
    print("device={}".format(device))

    (trainingdataset_bitmap, testingdataset_bitmap) = getTrainingAndTestingDatasets(printStatistics=True,albTransform_bitmap=albTransform)

    # Set up testing data loaders.
    testingloader_bitmap = torch.utils.data.DataLoader(testingdataset_bitmap, batch_size=cfg.training.batchsize)
    testingloaders = [testingloader_bitmap]
    testingloader_descriptions = ["bitmap"]

    # Initialize model
    branchmodel = BranchNet().to(device)
    decisionmodel = DecisionNet().to(device)

    # Initialize optimizer
    optimizer = optim.Adam(list(branchmodel.parameters())+list(decisionmodel.parameters()), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)

    training_losses_per_epoch = []
    testing_losses_per_epoch = []
    first_epoch = 0

    mean_testing_losses = []
    for testingloader in testingloaders:
        mean_testing_losses.append([])

    # If a checkpoint for training was specified, use it for initializing the model and optimizer
    print("initial model checkpoint: {}".format(cfg.training.initial_model_checkpoint))
    if (os.path.isfile(cfg.training.initial_model_checkpoint)):
        # Load checkpoint file
        checkpoint = torch.load(cfg.training.initial_model_checkpoint)
        
        # Initialize model
        branchmodel.load_state_dict(checkpoint['branchmodel_state_dict'])
        decisionmodel.load_state_dict(checkpoint['decisionmodel_state_dict'])

        # If we want to continue from an interrupted training session, load also optimizer, epoch and loss history
        if (not cfg.training.use_only_network_params):
            print("Loading optimizer state dict and epoch history")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            first_epoch = checkpoint['epoch']

            training_losses_per_epoch = checkpoint['training_losses_per_epoch']
            mean_testing_losses = checkpoint['testing_losses_per_epoch']

            for lossdataset_idx, lossdataset_descriptor in enumerate(testingloader_descriptions):
                print('testing losses loaded for {}:'.format(lossdataset_descriptor))
                print(mean_testing_losses[lossdataset_idx])

        print("file loaded")
    else:
        print("file not found, starting with default initialization")

    # Initialize cirterion for loss
    criterion = torch.nn.BCELoss(reduction='none')

    # Set up a figure for plotting loss
    loss_fig = plt.figure()

    # Start tracking execution time.
    ett = ExecutionTimeTracker()
    ett.markStartOfLoop(first_epoch)

    trainingloader_bitmap = torch.utils.data.DataLoader(trainingdataset_bitmap, batch_size=cfg.training.batchsize)
   
    trainingloaders = [trainingloader_bitmap]
    trainingloader_descriptions = ["Bitmap"]

    for epoch in range(first_epoch,cfg.training.numEpochs):
        
        running_loss = 0.0
        training_loss_over_epoch = 0.0

        # Training loop
        print("Starting epoch {} of {}".format(epoch+1,cfg.training.numEpochs))
        branchmodel.train()
        decisionmodel.train()

        cumulative_loss_alltrls = 0
        cumulative_samples_alltrls = 0

        for tl_idx, (trainingloader, trainingloader_description) in enumerate(zip(trainingloaders, trainingloader_descriptions)):

            numBatches = len(trainingloader)
            print("Training with trainingloader \"{}\". This trainingloader has {} batches.".format(trainingloader_description, numBatches))

            for i, data in enumerate(trainingloader):

                (anchorImage, positiveImage, negativeImage) = data
                anchorImage = anchorImage.to(device)
                positiveImage = positiveImage.to(device)
                negativeImage = negativeImage.to(device)
                
                optimizer.zero_grad()

                anc = branchmodel(anchorImage)
                pos = branchmodel(positiveImage)
                neg = branchmodel(negativeImage)

                ancToPos = decisionmodel(anc,pos)
                ancToNeg = decisionmodel(anc,neg)

                samples = torch.cat((ancToPos,ancToNeg))
                targets = torch.cat((torch.ones_like(ancToPos),torch.zeros_like(ancToNeg)))

                loss = criterion(samples,targets)

                numSamplesInBatch = anchorImage.shape[0]
                cumulative_loss_alltrls += loss.mean().item()*numSamplesInBatch
                cumulative_samples_alltrls += numSamplesInBatch
                
                loss.mean().backward()
                optimizer.step()
        
                if i % cfg.training.printAfterEveryNthBatch == 0 and i != 0:
                    print("at batch {} of {}, mean training loss is {:.4f}".format(i, numBatches, cumulative_loss_alltrls/cumulative_samples_alltrls))

        mean_training_loss = cumulative_loss_alltrls/cumulative_samples_alltrls
        print("Training completed for this epoch, mean training loss: {}".format(mean_training_loss))
        training_losses_per_epoch.append(mean_training_loss)

        # Testing loop with all testing loaders
        branchmodel.eval()
        decisionmodel.eval()

        for testingloader_idx, (testingloader, testingloader_description) in enumerate(zip(testingloaders, testingloader_descriptions)):

            cumulative_testing_loss = 0
            cumulative_testing_sample_count = 0

            with torch.no_grad():
                cumulative_loss = 0

                for i, data in enumerate(testingloader):
                    (anchorImage, positiveImage, negativeImage) = data
                    anchorImage = anchorImage.to(device)
                    positiveImage = positiveImage.to(device)
                    negativeImage = negativeImage.to(device)

                    anc = branchmodel(anchorImage)
                    pos = branchmodel(positiveImage)
                    neg = branchmodel(negativeImage)

                    ancToPos = decisionmodel(anc,pos)
                    ancToNeg = decisionmodel(anc,neg)

                    samples = torch.cat((ancToPos,ancToNeg))
                    targets = torch.cat((torch.ones_like(ancToPos),torch.zeros_like(ancToNeg)))

                    loss = criterion(samples,targets)

                    numSamplesInBatch = anchorImage.shape[0]
                    cumulative_testing_loss += loss.mean().item()*numSamplesInBatch
                    cumulative_testing_sample_count += numSamplesInBatch
                
            mean_testing_loss_subset = cumulative_testing_loss/cumulative_testing_sample_count
            print("Mean testing loss ({}): {:.4f}".format(testingloader_description, mean_testing_loss_subset))

            mean_testing_losses[testingloader_idx].append(mean_testing_loss_subset)

        # Only save every Nth epoch state on disk (as determined in configuration) or the output of the last training epoch
        if ((epoch+1) % cfg.training.save_every == 0 or epoch == cfg.training.numEpochs-1):
            ckptFilename = "{}_epoch_{:05d}_of_{}.pt".format(cfg.training.experiment_name,epoch+1,cfg.training.numEpochs)
            ckptFilenameWithPath = os.path.join(cfg.training.checkpoint_saving_path,ckptFilename)

            torch.save({
                    'epoch': epoch,
                    'branchmodel_state_dict': branchmodel.state_dict(),
                    'decisionmodel_state_dict': decisionmodel.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'training_losses_per_epoch': training_losses_per_epoch,
                    'testing_losses_per_epoch': mean_testing_losses,
                    'testing_loss_descriptions': testingloader_descriptions
                    }, ckptFilenameWithPath)

            print("Saved state to {}".format(ckptFilename))

        if (cfg.training.plotLossFigure):
            # Plot a graph showing progression of loss
            loss_fig.clf()
            plt.ion()
            plt.show()
            ax = loss_fig.add_subplot(1,1,1)
            ax.plot(training_losses_per_epoch,label='training')
            for lossdataset_idx, lossdataset_descriptor in enumerate(testingloader_descriptions):
                ax.plot(mean_testing_losses[lossdataset_idx],label='testing, {}'.format(lossdataset_descriptor))
            ax.legend()
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            loss_fig.set_figwidth(10)
            loss_fig.set_figheight(5)
            ax.grid(True)
            plt.draw()
            lossFigureFilepath = os.path.join(cfg.training.lossFigureSavePath,"{}_loss.svg".format(cfg.training.experiment_name))
            plt.savefig(lossFigureFilepath, bbox_inches = 'tight')
            plt.pause(0.001)
        
        ett.markCompletedIteration(epoch+1,cfg.training.numEpochs)
        print("Expected time of completion: {}".format(ett.getExpectedTimeOfFinishingStr()))