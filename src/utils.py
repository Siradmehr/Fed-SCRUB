from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss as CELoss
import torch.nn.functional as F
eps = 1e-6
eps_4 = 1e-5
eps_2 = 5e-3
eps_3 = 5e-2

def torch_one_hot(arr, logits, nclass, delta=0):
    if len(arr.shape) == len(logits.shape) - 1:
        return F.one_hot(arr, nclass)
    return arr

class KL(torch.nn.Module):
    def __init__(self, nclass, T):
        super(KL, self).__init__()
        self.nclass = nclass
        self.distillKL = DistillKL(T)
        assert (nclass >= 2)

    def forward(self,logits, tg):
        if len(tg.shape) == len(logits.shape) - 1:
            loss = nn.CrossEntropyLoss()(logits, tg)
            return loss
        elif len(tg.shape) == len(logits.shape):
            return self.distillKL(logits, tg)
        

class JS(torch.nn.Module):
    def __init__(self, nclass, T):
        super(JS, self).__init__()
        self.nclass = nclass
        self.KL = KL(nclass, T)
        assert (nclass >= 2)

    def forward(self,logits, tg):
        if len(tg.shape) == len(logits.shape) - 1:
            targets = torch_one_hot(tg, logits, self.nclass).to(logits.device)
            logits = F.softmax(logits, dim=1)
        else:
            logits = F.softmax(logits, dim=1)
            targets = F.softmax(tg, dim=1)
            
        C = (logits + targets).to(logits.device)
        loss = - (torch.log(C + eps).to(targets.device) * C).to(targets.device) + (logits * torch.log(2 * logits + eps).to(targets.device)).to(targets.device)
        return loss.sum(dim=1).mean()

class X2(torch.nn.Module):
    def __init__(self, nclass, T):
        super(X2, self).__init__()
        self.nclass = nclass
        assert (nclass >= 2)

    def forward(self,logits, tg):
        if len(tg.shape) == len(logits.shape) - 1:
            targets = torch_one_hot(tg, logits, self.nclass).to(logits.device)
            logits = F.softmax(logits, dim=1)
        else:
            logits = F.softmax(logits, dim=1)
            targets = F.softmax(tg, dim=1)
            
        targets = targets.to(logits.device)
        loss = torch.pow((targets-logits), 2) / (eps_2 + logits)
        return torch.mean(torch.sum(loss, -1)).mul_(1 / 2)
def get_losses(loss, nclasses, T):
    if loss == "CE":
        return nn.CrossEntropyLoss()
    elif loss == "KL":
        return KL(nclass=nclasses, T= T)
    elif loss == "JS":
        return JS(nclass=nclasses, T= T)
    elif loss == "X2":
        return X2(nclass=nclasses, T= T)   
