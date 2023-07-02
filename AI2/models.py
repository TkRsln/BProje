# -*- coding: utf-8 -*-
"""
Created on Thu May  4 01:36:20 2023

@author: utkua
"""
import torch.nn as nn
import torch



class AI_1(nn.Module):
    
    def __init__(self,img_weight=1090,img_height=850):
        super(AI_1, self).__init__()
        
        self.img_encoder=nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        
        self.txt_encoder=nn.Sequential(
                nn.Linear(6, 128),
                nn.ReLU(),
                nn.Linear(128, 545 * 425),
                nn.ReLU()
            )
        
        self.decoder=nn.Sequential(
                nn.Conv2d(33,64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4,stride=2,padding=1),
                nn.ReLU(),
                nn.Conv2d(16,3, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )
        
    def forward(self,input_img,input_text):
        x=self.img_encoder(input_img)
        y=self.txt_encoder(input_text)
        y = y.view(-1, 545, 425)
        
        # add a new dimension to tensor2 using unsqueeze
        y = y.unsqueeze(1)  # size: [101, 1, 545, 425]
        
        result = torch.cat((x, y), dim=1)
        # add the two tensors together
        #result = x + y
        print('[ERROR]',result.shape)
        #result = torch.cat((x, y), dim=0)
        return self.decoder(result)
        
class AI_2(nn.Module):
    
    def __init__(self):
        super(AI_2, self).__init__()
        self.img_encoder=nn.Sequential(
            #nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  
            #nn.ReLU(),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            )
        
        
        self.txt_encoder=nn.Sequential(
                nn.Linear(6, 128),
                nn.ReLU(),
                nn.Linear(128, 68 * 53),
                nn.ReLU()
            )
        
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(257, 128, kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4,stride=2,padding=0),
            #nn.ReLU(),
            #nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()            
            )
        
    def forward(self,input_img,input_text):
        x=self.img_encoder(input_img)
        y=self.txt_encoder(input_text)
        y = y.view(-1, 68, 53)
        
        # add a new dimension to tensor2 using unsqueeze
        y = y.unsqueeze(1)  # size: [101, 1, 545, 425]
        
        print('[ERROR]',y.shape,x.shape)
        result = torch.cat((x, y), dim=1)
        # add the two tensors together
        #result = x + y
        print('[ERROR]',result.shape)
        #result = torch.cat((x, y), dim=0)
        return self.decoder(result)
    
    
class AI_3(nn.Module):
    
    def __init__(self):
        super(AI_3, self).__init__()
        self.img_encoder=nn.Sequential(
            #nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  
            #nn.ReLU(),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
            )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4,stride=2,padding=0),
            #nn.ReLU(),
            #nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()            
            )
        
    def forward(self,input_img,input_text):
        x=self.img_encoder(input_img)
        
        return self.decoder(x)
        
    
class AI_4(nn.Module):
    """
    -Costs Memory
    """
    def __init__(self):
        super(AI_4, self).__init__()
        
        
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU()
            )
        
        self.img_decoder_linear=nn.Sequential(
            nn.Linear((64*68*53) + 3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, (64*68*53)),
            nn.ReLU()
        )
        
        self.img_decoder_cnn=nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4,stride=4,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4,stride=2,padding=2,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4,stride=2,padding=1),
            nn.sigmoid()
            )
        
    def forward(self,x,y): 
        #[hair,skin,shirt_type]
        y= torch.tensor(y).unsqueeze(0)
        
        x=self.img_encoder(x)
        x = x.view(1, -1)
        x =  torch.cat((sdata, x), dim=1)
        
        
#output_size = ((input_size - filter_size + 2*padding)/stride) + 1
#   
    
def test1():
    from dataset import ModelBaseTrainDataset
    dataset=ModelBaseTrainDataset()
    input_data,output_data=dataset.get_tensor_data(2)
    
    hair=0.0
    skin=0.0
    shirt_type=0.0
    
    sdata= torch.tensor([hair,skin,shirt_type]).unsqueeze(0)
    
    conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  
    conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  
    conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  
    max_pool1 = nn.MaxPool2d(2)
    
    linear_layer = nn.Linear((64*68*53) + 3, 2048)
    linear_layer2 = nn.Linear(2048, 4096)
    linear_layer3 = nn.Linear(4096, (64*68*53))
    
    
    
    tconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4,stride=4,output_padding=1)
    tconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4,stride=2,padding=2,output_padding=1)
    tconv3 = nn.ConvTranspose2d(16, 3, kernel_size=4,stride=2,padding=1)
    
    
    x=input_data[0]
    print('Raw:',x.shape)
    x=conv1(x)
    print('Conv1:',x.shape)
    x=conv2(x)
    x=max_pool1(x)
    print('Conv2',x.shape)
    x=conv3(x)
    print('Conv3',x.shape)
    
    #num_elements = x.size(1) * x.size(2)
    # Flatten the tensor using the view method
    x = x.view(1, -1)
    print('flat1',x.shape)
    x =  torch.cat((sdata, x), dim=1)
    #torch.cat((x1, x2), dim=1)
    print('flat2 concat',x.shape)
    
    
    x = linear_layer(x)
    print('linear',x.shape)
    x=linear_layer2(x)
    print('linear2',x.shape)
    x=linear_layer3(x)
    print('linear3',x.shape)
    
    
    x = x.view(64, 68, 53)
    print('shaped',x.shape)
    x=tconv1(x)
    print('tconv1',x.shape)
    x=tconv2(x)
    print('tconv2',x.shape)
    x=tconv3(x)
    print('tconv3 final',x.shape)
    

if __name__ == '__main__':
    test1()
    
        
"""
if __name__ == '__main__':
    #img=ImgEncoder()
    dataset=ModelBaseTrainDataset()
    input_data,output_data=dataset.get_tensor_data(2)
    #print(input_data,output_data)
    
    #x=conv3(x)
    #print('Conv3',x.shape)
    def test2():
        fc1 = nn.Linear(7, 128)
        fc2 = nn.Linear(128, 545 * 425)
    
        lst = [0,0,0,0,0,0,0]
        x = torch.Tensor(lst)
        
        x = fc1(x)
        x = torch.relu(x)
        x = fc2(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 545, 425)
        
        #print(x.shape)
        return x
"""
        
"""
    def test1():
        
        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  
        
        conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        
        conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        conv8 = nn.ConvTranspose2d(33, 16, kernel_size=4,stride=2,padding=1)
        
        x=input_data[0]
        print('Raw:',x.shape)
        x=conv1(x)
        print('Conv1:',x.shape)
        x=conv2(x)
        print('Conv2',x.shape)
        
        y=test2()
        
        
        
        result = torch.cat((x, y), dim=0)
        print('concat',result.shape)
        x = conv3(result)
        print('Conv3',x.shape)
        
        #####################################
        #img=ImgEncoder()
        from dataset import ModelBaseTrainDataset
        dataset=ModelBaseTrainDataset()
        input_data,output_data=dataset.get_tensor_data(2)
        
        
        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  
            
        conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
            
            
        conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        #conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        
        #conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        
        
        tconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4,stride=2,padding=1)
        tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4,stride=2,padding=1)
        tconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4,stride=2,padding=1)
        tconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4,stride=2,padding=0)
        conv_final= nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        
        
        
        x=input_data[0]
        print('Raw:',x.shape)
        x=conv1(x)
        print('Conv1:',x.shape)
        x=conv2(x)
        print('Conv2',x.shape)
        x=conv3(x)
        print('Conv3',x.shape)
        x=conv4(x)
        print('Conv4',x.shape)
        x=conv5(x)
        print('Conv5',x.shape)
        #x=conv6(x)
        #print('Conv6',x.shape)
        #x=conv7(x)
        #print('Conv6',x.shape)
        
        x=tconv1(x)
        print('T-Conv1',x.shape)
        x=tconv2(x)
        print('T-Conv2',x.shape)
        x=tconv3(x)
        print('T-Conv3',x.shape)
        x=tconv4(x)
        print('T-Conv4',x.shape)
        x=conv_final(x)
        print('Conv Final',x.shape)
            
"""
        
        
        #test1()
    
