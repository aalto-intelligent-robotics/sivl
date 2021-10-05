import os
import random
from skimage import io
from torch.utils.data import Dataset
import cv2
import numpy as np
import re

class BitmapDataset(Dataset):
    """Dataset for loading a set of bitmap images that are equally sized, aligned and have resolution 1 m/pixel."""
    
    def __init__(self, pathToAlignedPngImages, sampleDimension_px, marginBetweenSamples_px=0, homographyErrorStd=0, rotationErrorStd=0, translationErrorStd=0, transform=None, apShiftingStd=0.0, scaleStd=0.0, albTransform=None):
        """
        Parameters
        ----------
        pathToAlignedPngImages : str
            Path to a folder containing aligned images.
        sampleDimension_px : int
            Width and height of a sample image.
        marginBetweenSamples_px: float
            Margin between samples (to e.g. make sure that there is no overlap between samples when they are rotated.)
        homographyErrorStd: float
            Standard deviation for the zero-mean Gaussian noise added to coordinates of crop area coordinate cornerpoints, to introduce some projection error between the anchor and positive samples.
        rotationErrorStd: float
            Standard deviation (degrees) for the zero-mean Gaussian noise added to rotation between anchor and positive samples.
        translationErrorStd: float
            Standard deviation (pixels) for the zero-mean Gaussian noise added to translation between anchor and positive samples.
        transform: Transform object similar to torchvision.transforms
            Transform to apply to the samples.
        apShiftingStd: float
            Standard deviation (pixels) for the zero-mean Gaussian noise added to centerpoint coordinates. todo: Note! Only applied in __init__, not at each epoch!
        scaleStd: float
            Standard deviation (unitless) for the zero-mean Gaussian noise in selecting random scaling when drawing samples.
        albTransform: Albumentations-style transformation
            Albumentations-style transformation to apply to the samples.
        """
        
        extensions = ('.png', '.jpg', '.bmp', '.tif', '.tiff')

        self.homographyErrorStd = homographyErrorStd
        self.rotationErrorStd = rotationErrorStd
        self.translationErrorStd = translationErrorStd

        self.transform = transform
        self.albTransform = albTransform

        self.images = []
        self.image_width = None
        self.image_height = None
        self.image_dimensions = None

        self.months = []
        self.years = []

        imageIdxs = 0

        # Load images to memory.
        for subdir, dirs, files in os.walk(pathToAlignedPngImages):
           
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in extensions:
                    # For quantifying the spread of months and years in the dataset, we assume the filename is in format
                    # year_month_restoffilename.extension. If finding out year and month fails, just store None.
                    # The year/month information is not used for anything other than allowing to investigate statistics about
                    # the distribution of imaging times per dataset.
                    try:
                        filenameParts = re.split(r"_|\.", file)
                        year = int(filenameParts[0])
                        month = int(filenameParts[1])
                    except ValueError:
                        year = None
                        month = None

                    self.years.append(year)
                    self.months.append(month)

                    pathToImageFile = os.path.join(subdir, file)

                    image = io.imread(pathToImageFile)
                    
                    (h, w, d) = image.shape
                    
                    if (self.image_width is None or self.image_height is None or self.image_dimensions is None):
                        self.image_width = w
                        self.image_height = h
                        self.image_dimensions = d

                    assert (self.image_width == w), "Width of all images in dataset should be equal."
                    assert (self.image_height == h), "Height of all images in dataset should be equal."
                    assert (self.image_dimensions == d), "Number of dimensions in image should be equal."

                    self.images.append(image)

                    imageIdxs += 1

        # If no image files were found, raise an error.
        if (imageIdxs == 0):
            raise FileNotFoundError("No bitmap images were found in folder {}. Please download the images first, see instructions in README.md.".format(pathToAlignedPngImages))

        self.sampleDimension_px = sampleDimension_px

        self.apShiftingStd = apShiftingStd

        self.scaleStd = scaleStd

        # Create a list of centerpoints on dataset images.
        self.cropCenterpointXs = []
        self.cropCenterpointYs = []

        marginForRotation = sampleDimension_px/np.sqrt(2)

        currY = sampleDimension_px/2 + marginForRotation
        while currY + sampleDimension_px/2 + marginForRotation < self.image_height:
            currX = sampleDimension_px/2 + marginForRotation
            while currX + sampleDimension_px/2 + marginForRotation < self.image_width:               
                self.cropCenterpointXs.append(currX + np.random.normal(0,self.apShiftingStd))
                self.cropCenterpointYs.append(currY + + np.random.normal(0,self.apShiftingStd))
                currX += sampleDimension_px + marginBetweenSamples_px

            currY += sampleDimension_px + marginBetweenSamples_px

        self.numSamples = len(self.cropCenterpointXs)

        self.populationOfImageIndices = range(len(self.images))

    def __len__(self):
        return self.numSamples
    
    def __getitem__(self, idx):
        if len(self.populationOfImageIndices) >= 2:
            [anchorImageIdx, positiveImageIdx] = random.sample(self.populationOfImageIndices,2)
            [negativeImageIdx] = random.sample(self.populationOfImageIndices,1)
        else:
            anchorImageIdx = 0
            positiveImageIdx = 0
            negativeImageIdx = 0

        anchorMap = self.images[anchorImageIdx]
        positiveMap = self.images[positiveImageIdx]
        negativeMap = self.images[negativeImageIdx]

        center_x_ap = self.cropCenterpointXs[idx]
        center_y_ap = self.cropCenterpointYs[idx]

        # draw a negative sample randomly, making sure we are not drawing the same index
        while (True):
            negative_idx = np.random.randint(0,self.numSamples)
            if (negative_idx != idx):
                break

        center_x_n = self.cropCenterpointXs[negative_idx]
        center_y_n = self.cropCenterpointYs[negative_idx]

        # Define homography transformations for anchor, positive and negative samples

        # Randomly draw translation noise for positive sample
        pTranslationNoise_x = np.random.normal(0,self.translationErrorStd)
        pTranslationNoise_y = np.random.normal(0,self.translationErrorStd)

        # Randomly draw a rotation for anchor and positive samples, and separately for negative sample
        apRotation = np.random.uniform(0.0,360.0)
        nRotation = np.random.uniform(0.0,360.0)

        # Randomly draw a scale for anchor and positive samples, and separately for negative sample
        apScale = np.random.normal(1.0,self.scaleStd)
        nScale = np.random.normal(1.0,self.scaleStd)

        h_a = getNoisyHomographyWithRotationAndScaling(center_x_ap, center_y_ap, self.sampleDimension_px, self.sampleDimension_px, self.homographyErrorStd,apRotation,0.0,apScale,0.0)
        h_p = getNoisyHomographyWithRotationAndScaling(center_x_ap+pTranslationNoise_x, center_y_ap+pTranslationNoise_y, self.sampleDimension_px, self.sampleDimension_px, self.homographyErrorStd,apRotation,self.rotationErrorStd,apScale,0.0)
        h_n = getNoisyHomographyWithRotationAndScaling(center_x_n, center_y_n, self.sampleDimension_px, self.sampleDimension_px, self.homographyErrorStd,nRotation,0.0,nScale,0.0)

        dsize = (self.sampleDimension_px, self.sampleDimension_px)

        anchor = cv2.warpPerspective(anchorMap,h_a,dsize)
        positive = cv2.warpPerspective(positiveMap,h_p,dsize)
        negative = cv2.warpPerspective(negativeMap,h_n,dsize)

        # Apply Albumentations transforms.
        if (self.albTransform is not None):
            transformedSample = self.albTransform(image=anchor, positive=positive, negative=negative)
            anchor = transformedSample["image"]
            positive = transformedSample["positive"]
            negative = transformedSample["negative"]

        sample = (anchor, positive, negative)

        # Apply torch transforms.
        if (self.transform is not None):
            return self.transform(sample)

        return sample

    def getImages(self):
        return self.images

    def getYearsAndMonths(self):
        return self.years, self.months

    def getImageDimensionsPx(self):
        return (self.image_height, self.image_width)

    def plotSampleCenters(self, ax):        
        plotRectangle(ax,0,self.image_width,0,self.image_height,'k')

        for idx in range(self.numSamples):
            center_x = self.cropCenterpointXs[idx]
            center_y = self.cropCenterpointYs[idx]

            ax.plot(center_x,center_y,'kx')

def getNoisyHomographyWithRotationAndScaling(centerx,centery,dimx,dimy,cornerPxErrorStd,rotationAngle,rotationErrorStd,scalingFactor,scalingFactorErrorStd):

    srcPoints = np.zeros((4,2))
    dstPoints = np.zeros((4,2))

    # construct an array of source points
    srcPoints[0,:] = [centerx-dimx/2,centery-dimy/2]
    srcPoints[1,:] = [centerx+dimx/2,centery-dimy/2]
    srcPoints[2,:] = [centerx+dimx/2,centery+dimy/2]
    srcPoints[3,:] = [centerx-dimx/2,centery+dimy/2]

    # add random noise to source point coordinates with specified standard deviation
    for n in range(4):
        srcPoints[n,0] += np.random.normal(0.0,cornerPxErrorStd)
        srcPoints[n,1] += np.random.normal(0.0,cornerPxErrorStd)

    angle = rotationAngle + np.random.normal(0.0,rotationErrorStd)
    scale = scalingFactor + np.random.normal(0.0, scalingFactorErrorStd)

    # find a rotation matrix and rotate source points
    R = cv2.getRotationMatrix2D((centerx,centery), angle, scale)

    srcPoints = np.matmul(R[0:2,0:2],srcPoints.T)
    srcPoints[0,:] += R[0,2]
    srcPoints[1,:] += R[1,2]

    srcPoints = srcPoints.T

    # construct an array of destination points
    dstPoints[0,:] = [0,0]
    dstPoints[1,:] = [dimx,0]
    dstPoints[2,:] = [dimx,dimy]
    dstPoints[3,:] = [0,dimy]

    h, status = cv2.findHomography(srcPoints, dstPoints)

    return h

def plotRectangle(ax, left, right, top, bottom, color):
    ax.plot([left, right],[top,top],color)
    ax.plot([right, right], [top, bottom], color)
    ax.plot([right, left], [bottom, bottom], color)
    ax.plot([left,left],[bottom,top],color)
