# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:48:58 2023

@author: utkua
"""

import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image

#C:\Users\utkua\OneDrive\Masaüstü\AI\dataset_1\model_class
class ModelBaseTrainDataset(Dataset):
    def __init__(self, root_dir='C:\\Users\\utkua\\OneDrive\\Masaüstü\\AI\\dataset_1\\model_class\\',model_key='model_',cloth_key='tshirt',print_info:bool=False):
        super(ModelBaseTrainDataset,self).__init__()
        self.cloth_key=cloth_key
        self.info = self.find_info_files(root_dir,folder_key=model_key)
        self.info = self.get_info_from_files(self.info,cloth_key=cloth_key)
        self.size = self.find_size_from_info(self.info)
        if print_info:
            print(f'[ModelBaseTrainDataset] model_size:{len(self.info)}, cloth_size:{self.size}')
    
    def find_info_files(self,root_dir,folder_key='model_',info_txt='\\info.txt'):
        file_list = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        info_files=[]
        for f in file_list:
            if f.startswith(folder_key):
                if os.path.exists(root_dir+f+info_txt):
                    info_files.append(root_dir+f+info_txt)
        return info_files
    
    def get_info_from_files(self,info_files,cloth_key='tshirt'):
        info=[]
        for f in info_files:
            text_file = open(f,'r')
            text = text_file.read()
            text_file.close()
            dct=eval(text)
            dct['path']=f.replace('info.txt', '')
            dct['size']=len(dct[cloth_key].keys())
            info.append(dct)
        return info
    
    def find_size_from_info(self,infos):
        total=0
        for f in infos:
            total+=f['size']
        return total
    
    def img_to_tensor(self,path):
        target_image = Image.open(path)
        target_image = transforms.ToTensor()(target_image)
        return target_image
    
    def convert_text_to_tensor(self,cloth='oversize',hair='short',skin=1):
        lst=[
                1 if cloth=='oversize' else 0 ,
                1 if cloth=='normal' else 0 ,
                1 if cloth=='slim' else 0 ,
                1 if hair=='short' else 0 ,
                1 if hair=='medium' else 0 ,
                skin
            ]
        return torch.Tensor(lst)
    
    def get_data(self,index):
        total=0
        delta=0
        key=None#Cloth name/id
        for inf in self.info:
            if total+inf['size']>index:
                delta=index-total
                key=list(inf[self.cloth_key].keys())[delta]
                break
            else:
                total+=inf['size']
                
        input_img=inf['path']+f'input_{str(key)}.jpg'
        output_img=inf['path']+f'output_{str(key)}.jpg'
        cloth=inf[self.cloth_key][key]
        hair=inf['model']['hair'] 
        skin=inf['model']['skin']
        return (input_img,(cloth,hair,skin)),output_img
        
    def get_tensor_data(self,index):
        data=self.get_data(index)
        output_img=self.img_to_tensor(data[1])
        input_img=self.img_to_tensor(data[0][0])
        input_txt=self.convert_text_to_tensor(cloth=data[0][1][0],hair=data[0][1][1],skin=data[0][1][2])
        return (input_img,input_txt),output_img
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.get_tensor_data(index)
    
    
    
    
    
if __name__ == '__main__':
    #root_dir='C:\\Users\\utkua\\OneDrive\\Masaüstü\\AI\\dataset_1\\model_class\\'
    dataset=ModelBaseTrainDataset(print_info=True)
    #print(dataset.get_data(55)[0][1])
    #print(info)
            
"""

        # Load input image (64x64)
        input_image_path = self.root_dir+self.inputs[index] #os.path.join(self.root_dir, self.file_list[index])
        input_image = Image.open(input_image_path)
        input_image = self.transform_in(input_image)
        #display(input_image.shape)

        # Load target image (512x512)
        target_image_path = self.root_dir+self.outputs[index] #os.path.join(self.root_dir, self.file_list[index].replace('input', 'target'))
        target_image = Image.open(target_image_path)
        #target_image = self.transform_out(target_image)
        #display(target_image.shape)
        target_image = transforms.ToTensor()(target_image)

        return input_image, target_image
"""
            