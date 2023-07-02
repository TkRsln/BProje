# -*- coding: utf-8 -*-
"""
Created on Fri May  5 01:33:49 2023

@author: utkua
"""

from models import AI_2 as ai_model
from dataset import ModelBaseTrainDataset
import torch.nn as nn
import torch

from tqdm import tqdm


# Hyperparameters
num_epochs = 10
batch_size = 128
learning_rate=0.01

model = ai_model()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


my_dataset = ModelBaseTrainDataset()
my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)


main_bar = tqdm(total=num_epochs,desc='Main Process')
#sub_bar = tqdm(total=len(my_dataloader),desc='Sub Process')

for epoch in range(num_epochs):
    main_bar.update(1)
    #sub_bar.reset()
    for count,data in enumerate(my_dataloader):
        #sub_bar.update(1)
        input_data, output_img = data
        if len(input_data)!=2:
            print(input_data)
        out = model(input_data[0],input_data[1])
        loss = criterion(out, output_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1)%1==0:
        print('Epoch [{}/{}], Loss: {:.4f}'
          .format(epoch+1, num_epochs, loss.item()))
        
        

        
    
my_dataset = ModelBaseTrainDataset()
my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
for data in my_dataloader:
    input_data, output_img = data
    print(input_data[1].shape)
    break
print('done')    
