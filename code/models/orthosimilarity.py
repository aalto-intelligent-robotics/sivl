import torch
import torch.nn as nn
import torch.nn.functional as F

class BranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        # structure similar to https://arxiv.org/abs/1504.03641
        # Zagoruyko, Sergey, and Nikos Komodakis. "Learning to compare image patches via convolutional neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015"
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7,stride=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5,stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3,stride=1)
        self.relu3 = nn.ReLU()
        self.adaptivemaxpool = nn.AdaptiveMaxPool2d(output_size=8)

    def forward(self, x,printShapes=False):
        if (printShapes): print("shape at input: {}".format(x.shape))
        x = self.conv1(x)
        if (printShapes): print("shape after conv1: {}".format(x.shape))
        x = self.relu1(x)
        if (printShapes): print("shape after relu1: {}".format(x.shape))
        x = self.maxpool1(x)
        if (printShapes): print("shape after maxpool1: {}".format(x.shape))
        x = self.conv2(x)
        if (printShapes): print("shape after conv2: {}".format(x.shape))
        x = self.relu2(x)
        if (printShapes): print("shape after relu2: {}".format(x.shape))
        x = self.maxpool2(x)
        if (printShapes): print("shape after maxpool2: {}".format(x.shape))
        x = self.conv3(x)
        if (printShapes): print("shape after conv3: {}".format(x.shape))
        x = self.relu3(x)
        if (printShapes): print("shape after relu3: {}".format(x.shape))
        x = self.adaptivemaxpool(x)
        if (printShapes): print("shape after adaptivemaxpool: {}".format(x.shape))
        x = torch.flatten(x, 1)
        if (printShapes): print("shape after flatten: {}".format(x.shape))

        return x

class DecisionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=8*8*256*2, out_features=512, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=512, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, x2, printShapes=False):
        x = torch.cat((x1,x2),dim=1)
        if (printShapes): print("shape after cat: {}".format(x.shape))
        x = self.linear1(x)
        if (printShapes): print("shape after linear1: {}".format(x.shape))
        x = self.relu(x)
        if (printShapes): print("shape after relu: {}".format(x.shape))
        x = self.linear2(x)
        if (printShapes): print("shape after linear2: {}".format(x.shape))
        x = self.sigmoid(x)
        if (printShapes): print("shape after sigmoid: {}".format(x.shape))
        return x
