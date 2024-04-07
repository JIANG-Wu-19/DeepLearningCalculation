import torch 
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = self.conv1(X)
        X = self.sigmoid(X)
        # X = self.relu(X)
        X = self.avgPool(X)

        X = self.conv2(X)
        X = self.sigmoid(X)
        # X = self.relu(X)
        X = self.avgPool(X)

        X = self.flatten(X)

        X = self.fc1(X)
        X = self.sigmoid(X)
        # X = self.relu(X)

        X = self.fc2(X)
        X = self.sigmoid(X)
        # X = self.relu(X)

        pred = self.fc3(X)
        return pred