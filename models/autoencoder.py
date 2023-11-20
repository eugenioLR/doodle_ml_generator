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
        self.input_size = input_size
        self.code_dims = code_dims

        self.layer1 = nn.Linear(input_size[0]*input_size[1], 400, device=device)
        self.layer2 = nn.Linear(400, 200, device=device)
        self.layer3 = nn.Linear(200, 100, device=device)
        self.layer4 = nn.Linear(100, code_dims, device=device)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))

        return x


class DecoderDense(nn.Module):
    def __init__(self, code_dims, output_size, device="cpu"):
        super().__init__()
        self.code_dims = code_dims
        self.output_size = output_size

        self.layer1 = nn.Linear(code_dims, 100, device=device)
        self.layer2 = nn.Linear(100, 200, device=device)
        self.layer3 = nn.Linear(200, 400, device=device)
        self.layer4 = nn.Linear(400, output_size[0]*output_size[1], device=device)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))

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
