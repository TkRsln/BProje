# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:57:42 2023

@author: utkua
"""


#import train_test as tool
import torch
from models import Discriminator
from models import Generator
from dataset import ModelBaseTrainDataset as modelset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Set random seed for reproducibility
torch.manual_seed(42)

#Params
train_id=55
gryscale=True
dataset_scale=0.25
rand_size=100

#Calculation
original_size=(1090,850)
image_dims=( 1 if gryscale else 3,int(original_size[0]*dataset_scale),int(original_size[1]*dataset_scale))#272*212*3
image_size = image_dims[0]*image_dims[1]*image_dims[2]
#print(image_size,image_dims[0])

#DEFINE GEN
generator = Generator(image_dim=image_size,train_id=train_id)
#dis = Discriminator(train_id=55)

#DATASET
model_dataset = modelset(root_dir='./dataset_1/model_class/',scale=1,print_info=False, gry=True)
print(model_dataset.get_tensor_data(10)[1].shape)
#dataloader = DataLoader(model_dataset, batch_size=batch_size, shuffle=True)
#return



#DRAW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
sample_noise = torch.randn(1,rand_size).to(device)        
gen=generator(model_dataset.get_tensor_data(55)[0][0].unsqueeze(0) ,model_dataset.get_tensor_data(55)[0][1].unsqueeze(0),sample_noise).detach().cpu()

# Reshape the tensor to the desired image dimensions
tensor = gen.view( image_dims[0],image_dims[1],image_dims[2])

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