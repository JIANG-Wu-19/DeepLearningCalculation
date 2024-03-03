import torch
from torch import nn


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


class NiN(nn.Module):
    def __init__(self):
        super(NiN,self).__init__()
        self.nin1 = nin_block(1, 96, 11, 4, 0)
        self.nin2 = nin_block(96, 256, 5, 1, 2)
        self.nin3 = nin_block(256, 384, 3, 1, 1)
        self.nin4 = nin_block(384, 10, 3, 1, 1)
        self.maxPool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.adaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        X = self.nin1(X)
        X = self.maxPool2d(X)
        X = self.nin2(X)
        X = self.maxPool2d(X)
        X = self.nin3(X)
        X = self.maxPool2d(X)
        X = self.dropout(X)
        X = self.nin4(X)
        pred = self.flatten(X)
        return pred
