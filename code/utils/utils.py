import torch
import numpy as np
import cv2

def numpyImageToTorch(image):
    image_torch = torch.from_numpy(image.astype(np.float32)).permute([2, 0, 1])
    return image_torch

def RotationMatrix2d(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def cropRectangularSectionFromImage(image,x,y,theta,sourceDim,targetDim,scale=1.0):
    dst_points = np.array([[0,0],[targetDim,0],[targetDim,targetDim],[0,targetDim]])
    src_points = np.array([[-sourceDim/2*scale,-sourceDim/2*scale],[sourceDim/2*scale,-sourceDim/2*scale],[sourceDim/2*scale,sourceDim/2*scale],[-sourceDim/2*scale,sourceDim/2*scale]])
    R = RotationMatrix2d(theta)

    # rotate source points
    src_points = np.matmul(R,src_points.T).T
    
    # translate source points
    src_points = src_points + np.array([[x,y]])

    h, _ = cv2.findHomography(src_points,dst_points)

    cropped = cv2.warpPerspective(image,h,(targetDim,targetDim))

    return cropped

def tensorToNumpy(tensor,changeAxisOrder=False):
    detached_tensor = tensor.detach().cpu()

    if (changeAxisOrder):
        detached_tensor = detached_tensor.permute(0, 2, 3, 1)

    return detached_tensor.numpy()

class ToTensor(object):
    """Convert ndarrays in sample to Tensors and convert type to given datatype."""
    def __init__(self, dtype=torch.float):
        self.dtype = dtype

    def __call__(self, sample):
        (anchor, positive, negative) = sample

        anchor = torch.from_numpy(anchor.transpose((2,0,1))).type(self.dtype)
        positive = torch.from_numpy(positive.transpose((2,0,1))).type(self.dtype)
        negative = torch.from_numpy(negative.transpose((2,0,1))).type(self.dtype)

        return (anchor, positive, negative)

def numpyImageToTensor(image):
    return torch.from_numpy(image.transpose((2,0,1))).type(torch.float)