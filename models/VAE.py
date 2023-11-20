import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np


class EncoderVAE(nn.Module):
    def __init__(self, input_size, code_dims, std_scale):
        super().__init__()
        self.input_size = input_size
        self.code_dims = code_dims
        self.std_scale = std_scale

        # Sampling on GPU from https://avandekleut.github.io/vae/
        self.normal = torch.distributions.Normal(0,std_scale)
        self.normal.loc = self.normal.loc.to(device)
        self.normal.scale = self.normal.scale.to(device)

        self.kl_loss = 0

    def _sample_codes(self, loc, scale):
        self.kl_loss = 0.5*((scale/self.std_scale)**2 + loc**2 - (scale/self.std_scale).log() + self.code_dims).sum(dim=1)
        self.kl_loss = self.kl_loss.mean()

        return loc + scale * self.normal.sample(loc.shape)

class EncoderDenseVAE(EncoderVAE):
    def __init__(self, input_size, code_dims, std_scale=1, device="cpu"):
        super().__init__()

        self.layer1 = nn.Linear(input_size[0]*input_size[1], 400, device=device)
        self.layer2 = nn.Linear(400, 200, device=device)
        self.layer3 = nn.Linear(200, 100, device=device)
        self.layer4 = nn.Linear(100, code_dims, device=device)
        self.layer5 = nn.Linear(100, code_dims, device=device)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x_loc = self.layer4(x)
        x_scale = torch.exp(self.layer5(x))

        x = self._sample_codes(x_loc, x_scale)

        return x, x_loc, x_scale


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
    