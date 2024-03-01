import torch
from torch import nn


# 这里设计的AlexNet测试的数据集为MNIST,单通道输入设计
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5,  padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.flatten=nn.Flatten()
        # 使用暂退法减轻过拟合
        self.fc1=nn.Sequential(
            nn.Linear(256*5*5,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2=nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc3=nn.Linear(4096,10)

    def forward(self, X):
        X=self.conv1(X)
        X=self.conv2(X)
        X=self.conv3(X)
        X=self.conv4(X)
        X=self.conv5(X)
        X=self.flatten(X)
        X=self.fc1(X)
        X = self.fc2(X)
        pred = self.fc3(X)
        return pred
