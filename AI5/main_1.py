# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:18:00 2023

@author: utkua
"""

from moduls import Generator_1 as Generator
from moduls import Discriminator_1_solo as Discriminator
from dataset import ModelBaseTrainDataset as ModelData


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Display the generated samples
import matplotlib.pyplot as plt

def draw(output_images,input_images,s_img,show=False,id_=0):
    index=-1
    score=-1
    for i in range(len(s_img)):
        if score<s_img[i]:
            score=s_img[i]
            index=i
    #print('[DRAW]',output_images[index].shape,input_images[index].shape)
    if output_images[index].shape[0] == 1:
        image_out = output_images[index].squeeze()
        image_in = input_images[index].squeeze()
        img=torch.concat((image_out,image_in),dim=0)
    else:
        image_out = output_images[index]
        image_in = input_images[index]
        img=image_out#torch.concat((image_out,image_in),dim=2)
        img = img.permute(1, 2, 0)
    img=img.detach().numpy()
    # Display the image using PyPlot
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Remove axis labels
    if show:
        plt.show()
    else:
        plt.savefig('OUT_{str(id_)}.png')
    


# Hyperparameters
latent_dim = 100
label_dim = 10
image_dim = 28 * 28
batch_size = 64
num_epochs =  1000
lr = 0.002
gray=False
scale=0.125

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Models
generator = Generator(input_channel=1 if gray else 3)
discriminator = Discriminator(input_channel=1 if gray else 3,input_size=(850*scale,1090*scale))

#Load Dataset
dataset=ModelData(scale=scale,gry=gray,print_info=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Loss & Optim
adversarial_loss = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

o_img=None
i_img=None
s_img=None

print(f'[SIZE] {dataset.get_tensor_data(10)[0].shape}')
count=0
for epoch in range(num_epochs):
    for i,(input_img,input_txt,output_img) in enumerate(dataloader):
        #print(output_img.shape,input_img.shape)
        #o_img = output_img
        i_img = input_img
        #print(o_img.shape,i_img.shape)
        count+=1
        
        #Discriminator Train
        discriminator_optimizer.zero_grad()
        real_labels = torch.ones(output_img.shape[0]).to(device)
        fake_labels = torch.zeros(output_img.shape[0]).to(device)
        
        ## REAL-Value & Loss
        real_validity = discriminator(input_img, output_img)
        #print(f'real_val:{real_validity.shape} / real_labels:{real_labels.shape}')
        real_loss = adversarial_loss(real_validity, real_labels)
        
        ## FAKE-Value & Loss
        fake_images = generator(input_img)
        o_img = fake_images
        fake_validity = discriminator(input_img,fake_images)
        fake_loss = adversarial_loss(fake_validity, fake_labels)
        
        ## Optimizing        
        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()
        
        
        
        #Generator Train
        generator_optimizer.zero_grad()
        fake_images = generator(input_img)
        fake_validity = discriminator(input_img, fake_images)
        s_img = fake_validity
        generator_loss = adversarial_loss(fake_validity, real_labels)
        
        generator_loss.backward()
        generator_optimizer.step()

        # Print training progress
    
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
        f"Discriminator Loss: {discriminator_loss.item():.4f}, "
        f"Generator Loss: {generator_loss.item():.4f} "
        f"count: {count}"
        )
    
    if epoch%100==0:
        draw(o_img, i_img, s_img,id_=epoch)
        
    if epoch%500==0:
        torch.save(generator.state_dict(), f"./generator_{str(epoch)}")
        torch.save(discriminator.state_dict(), f"./discriminator_{str(epoch)}")
        
        
        
    
    









