# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:16:06 2023

@author: utkua
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_conditions=6, latent_dim=100,batch_size=64):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(262, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            #nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=(4,5), stride=2, padding=1,output_padding=(0,1)),
            nn.Tanh()
        )
        self.latent_dim=latent_dim
        self.condition_fc = nn.Linear(num_conditions, 221)
        self.noise_fc = nn.Linear(latent_dim, 1105)

    def forward(self, image, conditions):
        b_size=image.shape[0]
        print(f'[G-forward] im_shape:{image.shape}')
        encoded = self.encoder(image)
        print(f'[G-forward] encoded:{encoded.shape}')
        conditions = self.condition_fc(conditions)
        random = torch.rand(b_size,100)
        random = self.noise_fc(random)
        print(f'[G-forward] EN:{encoded.shape} / CON:{conditions.shape} / NO:{random.shape}')
        encoded = encoded.view(b_size,-1,17,13)
        conditions = conditions.view(b_size,-1,17,13)
        random = random.view(b_size,-1,17,13)
        print(f'[G-forward] 2 EN:{encoded.shape} / CON:{conditions.shape} / NO:{random.shape}')
        
        encoded_condition_noise = torch.cat((encoded, conditions, random), dim=1)
        print(f'[G-forward] concat:{encoded_condition_noise.shape}')
        encoded_condition_noise=encoded_condition_noise.view(b_size,-1,17,13)
        print(f'[G-forward] Final:{encoded_condition_noise.shape}')
    
        decoded = self.decoder(encoded_condition_noise)
        
        print(f'[G-forward] decoded:{decoded.shape}')
        return decoded
    

class Discriminator(nn.Module):
    def __init__(self, num_conditions):
        super(Discriminator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(6143 + num_conditions, 1),
            nn.Sigmoid()
            )

    def forward(self, img_out,img_in, conditions):
        
        print
        encoded = torch.cat((img_in, img_out), dim=1)
        
        encoded = self.encoder(encoded)
        encoded = encoded.view(encoded.size(0), -1)  # Flatten the feature map
        encoded_conditions = torch.cat((encoded, conditions), dim=1)
        print(f'[D-forward] {encoded_conditions.shape}')
        output = self.fc(encoded_conditions).squeeze(-1)
        print(f'[D-forward] {output.shape}')
        return output