import torch
from MySequential import MySequential
from torch import nn
from FixedHiddenMLP import FixedHiddenMLP

X = torch.rand(2, 20)

# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

net=FixedHiddenMLP()
print(net(X))
