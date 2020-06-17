import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.NLLLoss2d()
    def forward(self, a,targets):
        loss=self.loss(F.log_softmax(a), targets)
        return loss

class BCELoss(nn.Module):
    def __init__(self,weight=None):
        super().__init__()
        self.bceloss=nn.BCELoss()
    def forward(self,a,targets):
        loss=self.bceloss(F.sigmoid(a), targets.float())
        return loss


