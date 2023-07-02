# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:19:21 2023

@author: utkua
"""

import torch
import torch.nn as nn


class GAN(nn.Module):
    def __init__(self):
        super(GAN,self).__init__()
        self.encoder_img = ImgEncoder()
        self.encoder_text = TextEncoder()
        self.decoder=Decoder()
    
    def forward(self,img_tensor,text_tensor):
        image_data=self.encoder_img(img_tensor)
        text_data=self.encoder_text(text_tensor)
        x = torch.cat([text_data, image_data.view(image_data.size(0), -1)], dim=1)
        x = self.decoder(x)
        return x
        


class ImgEncoder(nn.Module):
        
    def __init__(self):
        super(ImgEncoder,self).__init__()
        # Convolutional encoder layers
        self.encoder_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
       
    def forward(self,x):
        conv_x = self.encoder_conv1(x)
        conv_x = self.relu(conv_x)
        conv_x = self.encoder_pool1(conv_x)
        conv_x = self.encoder_conv2(conv_x)
        conv_x = self.relu(conv_x)
        conv_x = self.encoder_pool2(conv_x)
        conv_x = self.encoder_conv3(conv_x)
        conv_x = self.relu(conv_x)
        conv_x = self.encoder_pool3(conv_x)
        return conv_x
        
        
class TextEncoder(nn.Module):
    def __init__(self,text_input=10):
        super(TextEncoder,self).__init__()# Linear encoder layers
        self.encoder_fc1 = nn.Linear(text_input, 128)
       
    def forward(self,x):
        linear_x = x.view(x.size(0), -1)
        linear_x = self.encoder_fc1(linear_x)
        linear_x = self.relu(linear_x)
        return linear_x
        
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()# Linear encoder layers
        # Convolutional decoder layers
        self.decoder_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self,x):
        # Decode
        x = self.decoder_conv1(x.view(x.size(0), 128, 3, 4))
        x = self.relu(x)
        x = self.decoder_conv2(x)
        x = self.relu(x)
        x = self.decoder_conv3(x)
        x = self.sigmoid(x)
        return x
         
        
        
    