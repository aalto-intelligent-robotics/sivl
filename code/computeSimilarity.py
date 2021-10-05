import torch
import argparse
from config import cfg
from models.orthosimilarity import BranchNet, DecisionNet
from utils.utils import numpyImageToTensor
import numpy as np
from skimage import io

def loadImageAndCheckSize(path,dimension):
    image = io.imread(path)

    if (len(image.shape) < 3 or image.shape[2] != 3):
        raise RuntimeError("The images should have 3 channels (R,G,B).".format(path))

    (h, w, d) = image.shape

    if (h != dimension or w != dimension):
        raise RuntimeError("The image dimensions should be {} by {} pixels".format(dimension, dimension))
    
    return image

if __name__ == '__main__':
    sampleDimPx = cfg.sampledimensions.dimension_px

    parser = argparse.ArgumentParser(description='Compute similarity of given image patches using trained model.')
    imageHelpString = "RGB image with shape {} by {} px, at 1m/px resolution".format(sampleDimPx,sampleDimPx)
    parser.add_argument("image1", help=imageHelpString)
    parser.add_argument("image2", help=imageHelpString)
    args = parser.parse_args()

    image1 = loadImageAndCheckSize(args.image1, sampleDimPx)
    image2 = loadImageAndCheckSize(args.image2, sampleDimPx)

    image1 = numpyImageToTensor(image1)
    image2 = numpyImageToTensor(image2)

    with torch.no_grad():
        device = torch.device(cfg.device)
        print("Device={}".format(device))

        # Initialize branch and decision networks
        branchmodel = BranchNet().to(device)
        decisionmodel = DecisionNet().to(device)

        # Load model parameters from checkpoint file
        print("Loading checkpoint {}".format(cfg.evaluation.model_checkpoint))
        checkpoint = torch.load(cfg.evaluation.model_checkpoint)
        branchmodel.load_state_dict(checkpoint['branchmodel_state_dict'])
        decisionmodel.load_state_dict(checkpoint['decisionmodel_state_dict'])
        print("Checkpoint loaded")

        # Append batch dimension
        image1 = torch.unsqueeze(image1,0).to(device)
        image2 = torch.unsqueeze(image2,0).to(device)

        # Compute outputs from branch model for each image
        image1BranchOutput = branchmodel(image1)
        image2BranchOutput = branchmodel(image2)

        # Compute similarity using decision model
        c_torch = decisionmodel(image1BranchOutput,image2BranchOutput)

        # Transfer to cpu, convert to numpy and get rid of extra dimensions
        c_np = np.squeeze(c_torch.cpu().data.numpy())

        print("c={}".format(c_np))