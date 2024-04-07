import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,input_num):
        super(MLP,self).__init__()
        self.flatten=nn.Flatten()
        self.linear=nn.Sequential(
            nn.Linear(input_num,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
        
    def forward(self,x):
        x=self.flatten(x)
        pred=self.linear(x)
        return pred
    
    
        
        