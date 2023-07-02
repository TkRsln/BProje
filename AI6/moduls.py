# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:59:39 2023

@author: utkua
"""

import torch
import torch.nn as nn

# Define the generator network
class MINT_Generator(nn.Module):
    def __init__(self, latent_dim, label_dim, image_dim):
        super(MINT_Generator, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.image_dim = image_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, image_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        generated_image = self.model(x)
        return generated_image

# Define the discriminator network
class MINT_Discriminator(nn.Module):
    def __init__(self, image_dim, label_dim):
        super(MINT_Discriminator, self).__init__()
        self.image_dim = image_dim
        self.label_dim = label_dim

        self.model = nn.Sequential(
            nn.Linear(image_dim + label_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, labels):
        x = torch.cat((image, labels), dim=1)
        validity = self.model(x)
        return validity