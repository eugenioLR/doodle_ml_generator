import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np

class EncoderDense(nn.Module):
    def __init__(self, input_size, code_dims, device="cpu"):
        super().__init__()
        self.layer1 = nn.Linear(input_size[0]*input_size[1], 100, device=device)
        self.layer2 = nn.Linear(100, code_dims, device=device)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))

        return x


class DecoderDense(nn.Module):
    def __init__(self, code_dims, output_size, device="cpu"):
        super().__init__()
        self.layer1 = nn.Linear(code_dims, 100, device=device)
        self.layer2 = nn.Linear(100, output_size[0]*output_size[1], device=device)
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        x = x.view(-1, *self.output_size)

        return x


class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
