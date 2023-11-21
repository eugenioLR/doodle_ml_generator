import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image


class RBM(nn.Module):
    """
    https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/
    """
    def __init__(self, n_visible, n_hidden, k=5):
        self.W = nn.Parameter(torch.randb(n_hidden, n_visible)*1e-2)
        self.v = nn.Parameter(torch.zeros(n_visible))
        self.h = nn.Parameter(torch.zeros(n_hidden))
    
    def sample(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
    def _visible_to_hidden(self, x):
        p_h = F.sigmoid(F.linear(x, self.W, bias=self.h))
        return p_h, self.sample(p_h)

    def _hidden_to_visible(self, x):
        p_v = F.sigmoid(F.linear(x, self.W.t(), bias=self.v))
        return p_v, self.sample(p_v)

    def forward(self, v):
        for _ in range(self.k):
            _, h = self.v_to_h(v)
            _, v = self.h_to_v(h)
        
        return v
    
    def free_energy(self, x):
        """
        gradient = energy[real] - energy[reconstructed]
        """

        wx_b = F.linear(x, self.W, self.h)

        hidden_term = torch.log(torch.exp(wx_b) + 1) + 1
        visible_term = torch.mv(x, self.v)
        return -(hidden_term + visible_term).mean()

