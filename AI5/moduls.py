# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:59:39 2023

@author: utkua
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Generator_1(nn.Module):
    
    def __init__(self,input_channel=1,rand_channel=5):
        super(Generator_1,self).__init__()
        self.rand_channel = rand_channel
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 8, 3,padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, 3,padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3,padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2)
            )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32+rand_channel, 32, 4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 12, 4,stride=2,padding=1,output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(12, input_channel, 3,padding=1),
            nn.Tanh()
            )
        
    def forward(self,x):
        x=self.encoder(x)
        r=torch.rand((x.shape[0],self.rand_channel,x.shape[2],x.shape[3]))
        #print('[Generator] x:',x.shape,'r:',r.shape)
        x=torch.concat((x,r),-3)
        return self.decoder(x)
        
    
class Discriminator_1(nn.Module):
    
    def __init__(self,input_channel=1,input_size=(1090, 425)):
        super(Discriminator_1,self).__init__()
        self.input_channel=input_channel
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, 4, 3,padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 8, 3,padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, 3,padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2)
            )
        inp_size=(int(input_size[0]/8),int(input_size[1]/8))
        self.model2 = nn.Sequential(
                nn.Linear(inp_size[0]*inp_size[1]*16,32),
                nn.LeakyReLU(0.2),
                nn.Linear(32,1),
                nn.Sigmoid()
            )
        
    def forward(self,input_image,output_image):
        x = torch.concat((output_image,input_image),dim=-2)
        #print('[Discriminator] i:',input_image.shape,'o:',output_image.shape,' x:',x.shape)
        
        x = self.model(x)
        #print("--",x.shape)
        x = self.model2( x.view(x.shape[0],1,-1) ).view(-1)
        #print(f'[DISC]',x.shape)
        return x
        
    
class Discriminator_1_solo(nn.Module):
    
    def __init__(self,input_channel=1,input_size=(545, 425)):
        super(Discriminator_1_solo,self).__init__()
        self.input_channel=input_channel
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, 4, 3,padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 8, 3,padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, 3,padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2)
            )
        inp_size=(int(input_size[0]/8),int(input_size[1]/8))
        self.model2 = nn.Sequential(
                nn.Linear(inp_size[0]*inp_size[1]*16,32),
                nn.LeakyReLU(0.2),
                nn.Linear(32,1),
                nn.Sigmoid()
            )
        
    def forward(self,input_image,output_image):
        #x = torch.concat((output_image,input_image),dim=-2)
        #print('[Discriminator] i:',input_image.shape,'o:',output_image.shape,' x:',x.shape)
        
        x = self.model(output_image)
        #print("--",x.shape)
        x = self.model2( x.view(x.shape[0],1,-1) ).view(-1)
        #print(f'[DISC]',x.shape)
        return x
        
    
        
        
if __name__ == '__main__':
    
    from dataset import ModelBaseTrainDataset as data
    input_channel=1
    input_size=(545, 425)
    input_size=(int(input_size[0]/8),int(input_size[1]/8))
    print(input_channel*3*input_size[0]*input_size[1])
    
    r1 = torch.rand(1,545,425)
    r2 = torch.rand(1,545,425)
    fnl = torch.concat((r1,r2),dim=1)
    print(fnl.shape)
    
    
    d = data(scale=0.5,gry=True)
    x=d.get_tensor_data(55)[1]
    
    def generator():
        d = data(scale=0.5,gry=True)
        x=d.get_tensor_data(55)[1]
        input_channel=1
        
        print(x.shape)
        x=nn.Conv2d(input_channel, input_channel*3, 3,padding=1)(x)
        x=nn.MaxPool2d(2)(x)
        print(x.shape)
        x=nn.Conv2d(input_channel*3, input_channel*3*2, 3,padding=1)(x)
        x=nn.MaxPool2d(2)(x)
        print(x.shape)
        x=nn.Conv2d(input_channel*3*2, input_channel*3*2, 3,padding=1)(x)
        x=nn.MaxPool2d(2)(x)
        print(x.shape)
        
        r=torch.rand((1, x.shape[1],x.shape[2] ))
        print('RANDOM',r.shape)
        
        x=torch.concat((x,r),0)
        print(x.shape)
        
        
        
        print('RESHAPE',x.view(1,-1).shape)
    
    
    generator()
    