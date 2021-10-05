import torch
import matplotlib.pyplot as plt
from config import cfg
import numpy as np
from scipy import stats
from models.orthosimilarity import BranchNet, DecisionNet
from trainSimilarityModel import getConcatenatedBitmapDataset

def plotHistogramsAndLikelihood(ys,likelihood,pos_scores,neg_scores,titletext=""):

    fig_hist, ax_hist = plt.subplots()
    fig_hist.canvas.manager.set_window_title(titletext)
    fig_lik, ax_lik = plt.subplots()
    fig_lik.canvas.manager.set_window_title(titletext)

    ymin = min(ys)
    ymax = max(ys)

    bins = np.linspace(ymin,ymax,num=50,endpoint=True)

    ax_hist.hist(pos_scores,bins=bins,label='S=s',alpha=0.5,density=False)
    ax_hist.hist(neg_scores,bins=bins,label='S=u',alpha=0.5,density=False)
    ax_hist.set_xlabel("c")
    ax_hist.set_ylabel('frequency')
    ax_hist.set_xlim([ymin,ymax])

    ax_hist.legend()

    ax_lik.plot(ys,likelihood)
    ax_lik.set_ylim([0,1])
    ax_lik.set_xlim([ymin,ymax])
    ax_lik.set_xlabel("c")
    ax_lik.set_ylabel("likelihood")

    return (fig_hist, ax_hist, fig_lik, ax_lik)

def getLookup(pos_scores,neg_scores,scoreInterpRangeMin=None, scoreInterpRangeMax=None, titletext=""):

    all_scores = np.concatenate((np.ravel(pos_scores),np.ravel(neg_scores)))
    minScore = np.min(all_scores)
    maxScore = np.max(all_scores)

    # If score interpolation range is not given, interpolate probability values over a wider area than the observed values.
    if (scoreInterpRangeMin is None):
        scoreInterpRangeMin = minScore-(maxScore-minScore)*0.25
    
    if (scoreInterpRangeMax is None):
        scoreInterpRangeMax = maxScore+(maxScore-minScore)*0.25
    
    gaussiankde_positive = stats.gaussian_kde(pos_scores)
    gaussiankde_negative = stats.gaussian_kde(neg_scores)
    
    posDistributionPdfFcn = gaussiankde_positive.evaluate
    negDistributionPdfFcn = gaussiankde_negative.evaluate

    rand_distribution_frozen_rv = stats.uniform(loc=scoreInterpRangeMin,scale=(scoreInterpRangeMax-scoreInterpRangeMin))
    randDistributionPdfFcn = rand_distribution_frozen_rv.pdf

    num_r_points = 1000

    ys = np.linspace(scoreInterpRangeMin,scoreInterpRangeMax,num_r_points)

    probability_of_pos = np.zeros(num_r_points)

    w_random = 0.1
    w_correct = 1
    w_generated = 1
    for idx, y in enumerate(ys):
        sum_of_all_densities = (w_correct * posDistributionPdfFcn(y) + w_generated * negDistributionPdfFcn(y) + w_random * randDistributionPdfFcn(y))
        probability_of_pos[idx] = w_correct * posDistributionPdfFcn(y)/sum_of_all_densities

    oi = overlappingIndex(pos_scores,neg_scores)

    return (ys,probability_of_pos,oi)

def overlappingIndex(points1,points2, numBins = 50):
    points1_vect = np.asarray(points1).ravel()
    points2_vect = np.asarray(points2).ravel()
    
    points_all = np.hstack((points1_vect,points2_vect))
        
    maxVal = np.max(points_all)
    minVal = np.min(points_all)
    
    p1hist, bins1 = np.histogram(points1_vect,bins=numBins,range=(minVal,maxVal),density=True)
    p2hist, bins2 = np.histogram(points2_vect,bins=numBins,range=(minVal,maxVal),density=True)
    

    histograms_stacked = np.vstack((p1hist,p2hist))

    binwidth = (maxVal-minVal)/numBins

    minimum_values_of_histograms = np.amin(histograms_stacked,axis=0)
    
    overlappingIndex = np.sum(minimum_values_of_histograms)*binwidth
    
    return overlappingIndex

if __name__ == '__main__':

    bitmapDatasetDescriptions_scoretoprob = cfg.data.path.scoretoprobtraining_bitmaps

    print("Loading datasets.")
    datasetToUseForScoreToProbabilityGeneration = getConcatenatedBitmapDataset(bitmapDatasetDescriptions_scoretoprob,verbose=True)
    print("Loading datasets completed.")

    device = torch.device(cfg.device)

    # Load network weights for learning model from checkpoint.
    branchmodel = BranchNet().to(device)
    decisionmodel = DecisionNet().to(device)
    print("Loading checkpoint {}".format(cfg.evaluation.model_checkpoint))
    checkpoint = torch.load(cfg.evaluation.model_checkpoint)
    branchmodel.load_state_dict(checkpoint['branchmodel_state_dict'])
    decisionmodel.load_state_dict(checkpoint['decisionmodel_state_dict'])

    # Initialize arrays for storing score values.
    positiveScores_learning = None
    negativeScores_learning = None

    positiveScores = dict()
    negativeScores = dict()

    dataloader = torch.utils.data.DataLoader(datasetToUseForScoreToProbabilityGeneration, batch_size=100)
    numSamples = len(dataloader)

    # Set network to evaluation mode.
    branchmodel.eval()
    decisionmodel.eval()
    with torch.no_grad():

        for i, data in enumerate(dataloader):
            print("Evaluating batch {} of {}".format(i+1,len(dataloader)))

            # Evaluate score with learning method.
            (anchorImage, positiveImage, negativeImage) = data
            anchorImage = anchorImage.to(device)
            positiveImage = positiveImage.to(device)
            negativeImage = negativeImage.to(device)

            anc = branchmodel(anchorImage)
            pos = branchmodel(positiveImage)
            neg = branchmodel(negativeImage)

            ancToPos = decisionmodel(anc,pos)
            ancToNeg = decisionmodel(anc,neg)

            ancToPos_np = ancToPos.cpu().data.numpy()
            ancToNeg_np = ancToNeg.cpu().data.numpy()

            if (positiveScores_learning is None or negativeScores_learning is None):
                positiveScores_learning = ancToPos_np
                negativeScores_learning = ancToNeg_np
            else:
                positiveScores_learning = np.append(positiveScores_learning, ancToPos_np)
                negativeScores_learning = np.append(negativeScores_learning, ancToNeg_np)
            
    print("Total number of samples: {}".format(len(positiveScores_learning)))

    eps = 1e-4
    (ys_learning, probs_learning, oi_learning) = getLookup(positiveScores_learning,negativeScores_learning,scoreInterpRangeMin=-eps, scoreInterpRangeMax=1+eps, titletext="learing model")
    print("Learning model overlapping index={}".format(oi_learning))
    (fig_hist, ax_hist, fig_lik, ax_lik) = plotHistogramsAndLikelihood(ys_learning,probs_learning,positiveScores_learning,negativeScores_learning,titletext="learning")

    print("positive scores: N={}, negative scores: N={}".format(len(positiveScores_learning), len(negativeScores_learning)))

    plt.show()