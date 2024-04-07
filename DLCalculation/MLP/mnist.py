import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

trans=transforms.ToTensor()
mnist_train= torchvision.datasets.MNIST(root="./data", train=True, transform=trans,download=False)

mnist_test= torchvision.datasets.MNIST(root="./data", train=False, transform=trans,download=False)




