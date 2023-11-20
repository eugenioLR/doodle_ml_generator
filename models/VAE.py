import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np


class EncoderDenseVAE(nn.Module):
    def __init__(self, input_size, code_dims, device="cpu"):
        super().__init__()

        self.layer1 = nn.Linear(input_size[0]*input_size[1], 400, device=device)
        self.layer2 = nn.Linear(400, 200, device=device)
        self.layer3 = nn.Linear(200, 100, device=device)
        self.layer4 = nn.Linear(100, code_dims, device=device)
        self.layer5 = nn.Linear(100, code_dims, device=device)
        # Sampling on GPU from https://avandekleut.github.io/vae/
        self.normal = torch.distributions.Normal(0,1)
        self.normal.loc = self.normal.loc.to(device)
        self.normal.scale = self.normal.scale.to(device)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.kl_loss = 0
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x_loc = self.layer4(x)
        x_scale = torch.exp(self.layer5(x))
        x = x_loc + x_scale * self.normal.sample(x_loc.shape)
        self.kl_loss = (x_scale.exp()**2 + x_loc**2 - x_scale - 0.5).sum()

        return x, x_loc, x_scale


class DecoderDenseVAE(nn.Module):
    def __init__(self, code_dims, output_size, device="cpu"):
        super().__init__()
        self.layer1 = nn.Linear(code_dims, 100, device=device)
        self.layer2 = nn.Linear(100, 200, device=device)
        self.layer3 = nn.Linear(200, 400, device=device)
        self.layer4 = nn.Linear(400, output_size[0]*output_size[1], device=device)
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))

        x = x.view(-1, *self.output_size)

        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder, beta=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
    
    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z 
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, z_loc, z_scale = self.encoder.forward(x)
        x = self.decoder.forward(z)

        return x, z_loc, z_scale
    