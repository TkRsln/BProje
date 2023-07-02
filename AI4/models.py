# -*- coding: utf-8 -*-
"""
Created on Tue May 16 22:19:52 2023

@author: utkua
"""



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Generator(nn.Module):
    def __init__(self,image_dim=545*425*3,condition_dim=6,noise_dim=100,device=None,train_id=None):
        super(Generator,self).__init__()
        self.tag='[Generator]'
        self.img_size=image_dim
        self.condition_size=condition_dim
        self.noise_dim=noise_dim
        """
        self.model = nn.Sequential(
            nn.Linear(image_dim+condition_dim+noise_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2), #1
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(1024),
            nn.Linear(1024, image_dim),
            nn.Tanh()
            )
        """
        self.model = nn.Sequential(
            nn.Linear(image_dim+condition_dim+noise_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2), #1
            nn.Linear(256,256),
            nn.LeakyReLU(256),
            nn.Linear(256, image_dim),
            nn.Tanh()
            )
        
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        self.to(self.device)
        
        
        if train_id != None:
            self.load_params(train_id)
            print(self.tag,'loaded params',train_id)
            
            
        
    def forward(self, image,conditions,noise):
        x = torch.cat((image.view(-1,self.img_size), conditions,noise), dim=1)
        return self.model(x)
    
    
    def save_params(self,train_id):
        torch.save(self.state_dict(), f"./saves/generator_{str(train_id)}")
           
    def load_params(self, train_id):    
        self.load_state_dict(torch.load(f"./saves/generator_{str(train_id)}"))
        self.eval()
            
        
class Discriminator(nn.Module):
    def __init__(self,image_size=545*425*3,condition_dim=6,device = None,train_id=None):
        super(Discriminator, self).__init__()
        self.image_size=image_size
        self.tag='[Discriminator]'
        self.model=nn.Sequential(
            nn.Linear(image_size+condition_dim,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,16),
            nn.LeakyReLU(0.2),
            nn.Linear(16,1),
            nn.Sigmoid()
            )
            
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        
        if train_id != None:
            self.load_params(train_id)
            print(self.tag,'loaded params',train_id)
            
    def forward(self,image,condition):
        x = torch.cat((image.view(-1,self.image_size), condition), dim=1)
        return self.model(x)
    
    def save_params(self,train_id):
        torch.save(self.state_dict(), f"./saves/discriminator_{str(train_id)}")
           
    def load_params(self, train_id):    
        self.load_state_dict(torch.load(f"./saves/discriminator_{str(train_id)}"))
        self.eval()
        
        
        

class Generator_1(nn.Module):
    def __init__(self,image_dim=545*425*3,condition_dim=6,noise_dim=100,device=None,train_id=None):
        super(Generator_1,self).__init__()
        self.tag='[Generator]'
        self.img_size=image_dim
        self.condition_size=condition_dim
        self.noise_dim=noise_dim
        """
        self.model = nn.Sequential(
            nn.Linear(image_dim+condition_dim+noise_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2), #1
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(1024),
            nn.Linear(1024, image_dim),
            nn.Tanh()
            )
        """
        self.model = nn.Sequential(
            nn.Linear(image_dim+condition_dim+noise_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2), #1
            nn.Linear(256,256),
            nn.LeakyReLU(256),
            nn.Linear(256, image_dim),
            nn.Tanh()
            )
        
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        self.to(self.device)
        
        
        if train_id != None:
            self.load_params(train_id)
            print(self.tag,'loaded params',train_id)
            
            
        
    def forward(self, image,conditions,noise):
        x = torch.cat((image.view(-1,self.img_size), conditions,noise), dim=1)
        return self.model(x)
    
    
    def save_params(self,train_id):
        torch.save(self.state_dict(), f"./saves/generator_1_{str(train_id)}")
           
    def load_params(self, train_id):    
        self.load_state_dict(torch.load(f"./saves/generator_1_{str(train_id)}"))
        self.eval()
            
        
class Discriminator_1(nn.Module):
    def __init__(self,image_size=545*425*3,condition_dim=6,device = None,train_id=None):
        super(Discriminator_1, self).__init__()
        self.image_size=image_size
        self.tag='[Discriminator]'
        self.model=nn.Sequential(
            nn.Linear(image_size+image_size+condition_dim,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
            )
            
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        if train_id != None:
            self.load_params(train_id)
            print(self.tag,'loaded params',train_id)
            
    def forward(self,image_out,image_shirt,condition):
        x = torch.cat((image_out.view(-1,self.image_size),image_shirt.view(-1,self.image_size) ,condition), dim=1)
        return self.model(x)
    
    def save_params(self,train_id):
        torch.save(self.state_dict(), f"./saves/discriminator_1_{str(train_id)}")
           
    def load_params(self, train_id):    
        self.load_state_dict(torch.load(f"./saves/discriminator_1_{str(train_id)}"))
        self.eval()
        
#CLASS SECOND
class Generator_2(nn.Module):
    def __init__(self,image_dim=545*425*3,condition_dim=6,noise_dim=100,device=None,train_id=None):
        super(Generator_2,self).__init__()
        self.tag='[Generator]'
        self.img_size=image_dim
        self.condition_size=condition_dim
        self.noise_dim=noise_dim
        """
        self.model = nn.Sequential(
            nn.Linear(image_dim+condition_dim+noise_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2), #1
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(1024),
            nn.Linear(1024, image_dim),
            nn.Tanh()
            )
        """
        self.model = nn.Sequential(
            nn.Linear(image_dim+condition_dim+noise_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2), #1
            nn.Linear(256,256),
            nn.LeakyReLU(256),
            nn.Linear(256, image_dim),
            nn.Tanh()
            )
        
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        self.to(self.device)
        
        
        if train_id != None:
            self.load_params(train_id)
            print(self.tag,'loaded params',train_id)
            
            
        
    def forward(self, image,conditions,noise):
        
        if len(image[0]) != 1:
            image=image.view(-1,self.image_size)
            
        x = torch.cat((image, conditions,noise), dim=1)
        return self.model(x)
    
    
    def save_params(self,train_id):
        torch.save(self.state_dict(), f"./saves/generator_1_{str(train_id)}")
           
    def load_params(self, train_id):    
        self.load_state_dict(torch.load(f"./saves/generator_1_{str(train_id)}"))
        self.eval()
            
        
class Discriminator_2(nn.Module):
    def __init__(self,image_size=545*425*3,condition_dim=6,device = None,train_id=None):
        super(Discriminator_2, self).__init__()
        self.image_size=image_size
        self.tag='[Discriminator]'
        self.model=nn.Sequential(
            nn.Linear(image_size+image_size+condition_dim,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
            )
            
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        if train_id != None:
            self.load_params(train_id)
            print(self.tag,'loaded params',train_id)
            
    def forward(self,image_out,image_shirt,condition):
        if len(image_out[0]) != 1:
            image_out=image_out.view(-1,self.image_size)
        
        if len(image_shirt[0]) != 1:
            image_shirt=image_shirt.view(-1,self.image_size)
            
        x = torch.cat((image_out,image_shirt ,condition), dim=1)
        return self.model(x)
    
    def save_params(self,train_id):
        torch.save(self.state_dict(), f"./saves/discriminator_1_{str(train_id)}")
           
    def load_params(self, train_id):    
        self.load_state_dict(torch.load(f"./saves/discriminator_1_{str(train_id)}"))
        self.eval()
        
            
            