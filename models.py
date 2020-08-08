## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # 1st structure (final feature map size = (224+2*0-5)/1 + 1 = 220, after poolling (220-2)/2 + 1 = 110)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd structure (final feature map size = (110+2*0-3)/1 + 1 = 108, after poolling (108-2)/2 + 1 = 54)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd structure (size input of flattened feature map = 64*54*54
        self.fc1 = nn.Linear(in_features=64 * 54 * 54, out_features=2048)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=512, out_features=136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x


class NetComplex(nn.Module):
    def __init__(self):
        super().__init__()

        # 1st structure (final feature map size = (224+2*0-5)/1 + 1 = 220, after poolling (220-2)/2 + 1 = 110)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd structure (final feature map size = (110+2*0-3)/1 + 1 = 108, after poolling (108-2)/2 + 1 = 54)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd structure (final feature map size = (54+2*0-3)/1 + 1 = 52, after poolling (52-2)/2 + 1 = 26)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4th structure (size input of flattened feature map = 128*26*26)
        self.fc1 = nn.Linear(in_features=128 * 26 * 26, out_features=2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(in_features=512, out_features=136)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(F.relu(self.bn2(self.conv2(x))))
        x = self.maxpool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout1(F.relu(self.bn4(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn5(self.fc2(x))))
        x = self.fc3(x)

        return x


class Resnet18_gray(nn.Module):
    def __init__(self):
        super().__init__()
        # using ResNet18 architecture (fine tuning - pretrained weights as initializations)
        # changing first layer to grayscale images and setting last layer to output 68 points in 2D
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        fc_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(fc_inputs, 2 * 68)

    def forward(self, x):
        return self.resnet18(x)


class Squeezenet10_gray(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeezenet = models.squeezenet1_0(pretrained=True)
        self.squeezenet.features[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2)
        self.squeezenet.classifier[1] = nn.Conv2d(
            512, 2 * 68, kernel_size=(1, 1), stride=(1, 1)
        )

    def forward(self, x):
        return self.squeezenet(x)


class Mobilenet2_gray(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.mobilenet.classifier[1] = nn.Linear(
            in_features=1280, out_features=2 * 68, bias=True
        )

    def forward(self, x):
        return self.mobilenet(x)
