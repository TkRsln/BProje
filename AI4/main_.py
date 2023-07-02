# -*- coding: utf-8 -*-
"""
Created on Wed May 17 01:07:34 2023

@author: utkua
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

from dataset import ModelBaseTrainDataset as modelset
from models import Generator_2 as Generator
from models import Discriminator_2 as Discriminator
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

tag='[MAIN]'

# Hyperparameters
#image_dim = 425*545*3 #3 * 900 * 850  # Image size with RGB channels
#noise_dim = 100



#Params
train_id=55
gryscale=True
dataset_scale=0.25
rand_size=100
text_dim = 10
batch_size = 64
num_epochs = 10
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_epoch=-1#54
bar_update=1
print(tag,'Hyperparameters setted.')

#Calculation
original_size=(1090,850)
image_dims=( 1 if gryscale else 3,int(original_size[0]*dataset_scale),int(original_size[1]*dataset_scale))#272*212*3
image_size = image_dims[0]*image_dims[1]*image_dims[2]
start_epoch= None if start_epoch<0 else start_epoch
#print(image_size,image_dims[0])


#DATASET
model_dataset = modelset(root_dir='./dataset_1/model_class/',gry=gryscale,scale=dataset_scale,print_info=False)
dataloader = DataLoader(model_dataset, batch_size=batch_size, shuffle=True)
print(tag,'datasets are ready.')

#MODELS
generator = Generator(train_id=start_epoch,image_dim=image_size)
discriminator = Discriminator(train_id=start_epoch,image_size=image_size)
start_epoch = 0 if start_epoch==None else start_epoch
print(tag,f'Models are created. train_id={44}')

#LOSS & OPTIMIZERS
adversarial_loss = nn.BCELoss()
generator_optim = optim.Adam(generator.parameters(),lr=lr)
discrim_optim = optim.Adam(discriminator.parameters(),lr=lr)
print(tag,'Loss and Optimizers are created.')


print(tag,'Starting training.')
#TQDM
total_process=num_epochs*len(dataloader)
bar = tqdm(total=total_process,desc='starting...')




def draw_4(generator,img,label,epoch,show=False):
    #label_dim=10, image_dim= 28*28, latent_dim=100,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sample_noise = torch.randn(1,rand_size).to(device)        
    gen=generator(img,label,sample_noise).detach().cpu()
    
    plt.imshow(gen.view(image_dims[1],image_dims[2]))
    plt.axis('off')
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(f'./AI4_OUT_{epoch}.png')

#TRAIN
for epoch in range(num_epochs):
    for i,(data_in,data_out) in enumerate(dataloader):
        
        real_images = data_out.to(device)
        #labels = torch.eye(label_dim)[labels].to(device)  # One-hot encoding
        # ...
        # Train the discriminator
        discrim_optim.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
        
        real_validity = discriminator(real_images, data_in[0] ,data_in[1])
        real_loss = adversarial_loss(real_validity, real_labels)
        
        noise = torch.randn(real_images.size(0), rand_size).to(device)
        fake_images = generator(data_in[0],data_in[1],noise)
        fake_validity = discriminator(fake_images, data_in[0] ,data_in[1])
        fake_loss = adversarial_loss(fake_validity, fake_labels)

        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        discrim_optim.step()
        
        # Train the generator
        generator_optim.zero_grad()
        noise = torch.randn(real_images.size(0), rand_size).to(device)
        fake_images = generator(data_in[0],data_in[1],noise)
        fake_validity = discriminator(fake_images, data_in[0] ,data_in[1])
        generator_loss = adversarial_loss(fake_validity, real_labels)
        
        generator_loss.backward()
        generator_optim.step()
        
        # Print training progress
        if (i + 1) % bar_update == 0:
            bar.update(bar_update)
            bar.desc=(
                f"Epoch:[{start_epoch+epoch+1}/{start_epoch+num_epochs}-({i+1}/{len(dataloader)})], Loss:[D:{discriminator_loss.item():.4f}, G:{generator_loss.item():.4f}]"
            )
    if (start_epoch + epoch + 1) % 10 == 0:
        generator.save_params(train_id=start_epoch+epoch+1)
        discriminator.save_params(train_id=start_epoch+epoch+1)
        #draw_4(generator,model_dataset.get_tensor_data(55)[0][0],model_dataset.get_tensor_data(20)[0][1],start_epoch+epoch)
            
        
        
        
        
        
        
        

def draw(generator,shirt_index=55,model_index=55,train_id=0):
    global rand_size,model_dataset,image_dims
    #DRAW
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    sample_noise = torch.randn(1,rand_size).to(device)     
    shirt_data=model_dataset.get_tensor_data(shirt_index)[0][0].unsqueeze(0)
    model_data_info=model_dataset.get_tensor_data(model_index)[0][1].unsqueeze(0)
    print(sample_noise.shape,shirt_data.shape,model_data_info.shape)
    
    gen=generator(shirt_data ,model_data_info,sample_noise).detach().cpu()
    
    # Reshape the tensor to the desired image dimensions
    tensor = gen.view( image_dims[0], image_dims[1], image_dims[2])
    
    # Convert the tensor to a PIL image
    image = TF.to_pil_image(tensor)
    
    # Convert PIL image to numpy array
    image_np = TF.to_tensor(image).numpy()
    
    # Transpose the dimensions to match matplotlib's expected format
    image_np = image_np.transpose(1, 2, 0)
    
    # Display the image using Matplotlib
    plt.imshow(image_np)
    plt.axis('off')
    show = True
    if show:
        plt.show()
    else:
        plt.savefig(f'./AI4_OUT_{train_id}.png')
        
draw(generator)
