import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_planes, out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.activation = nn.ReLU()

    def forward(self, X):

        out = self.conv(X)
        out = self.bn(out)
        out = self.activation(out)

        return out
        


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.block1 = ConvBlock(1, 16, kernel_size=3, stride=(2, 2), padding=1)
        self.block2 = ConvBlock(16, 32, kernel_size=3, stride=(2, 2), padding=1)
        self.block3 = ConvBlock(32, 64, kernel_size=3, stride=(2, 2), padding=1)
        self.block4 = ConvBlock(64, 128, kernel_size=3, stride=(2, 1), padding=1)
        self.block5 = ConvBlock(128, 256, kernel_size=3, stride=(2, 1), padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, X):

        out = self.block1(X)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        
        return out


def convnet():
    return ConvNet()