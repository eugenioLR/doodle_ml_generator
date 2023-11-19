import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np


def normal_sample(loc, log_scale):
    return loc + torch.exp(0.5 * log_scale) * torch.randn_like(log_scale)


class EncoderDenseVAE(nn.Module):
    def __init__(self, input_size, code_dims, device="cpu"):
        super().__init__()
        self.layer1 = nn.Linear(input_size[0]*input_size[1], 100, device=device)
        self.layer2 = nn.Linear(100, code_dims, device=device)
        self.layer3 = nn.Linear(100, code_dims, device=device)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.layer1(x))
        x_loc = self.layer2(x)
        x_scale = self.layer3(x)
        x = normal_sample(x_loc, x_scale)

        return x, x_loc, x_scale


class DecoderDenseVAE(nn.Module):
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


class DenseVAE(nn.Module):
    def __init__(self, encoder, decoder, rec_loss=None, beta=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if rec_loss is None:
            rec_loss = F.mse_loss
        self.rec_loss = rec_loss
        self.beta = beta
    
    def forward(self, x):
        z, z_loc, z_scale = self.encoder(x)
        x = self.decoder(z)

        return x, z_loc, z_scale
    
    def update(self, batch):
        x, _ = batch
        x_flat = x.view(x.size(0), -1)

        y, z_loc, z_log_scale = self.forward(x_flat)

        rec_loss = self.rec_loss(x, y)
        kl_loss = (-0.5 * torch.sum(1 + z_log_scale - z_loc.pow(2) - z_log_scale.exp(), dim=1)).mean()
        loss = rec_loss + self.beta*kl_loss

        return loss






