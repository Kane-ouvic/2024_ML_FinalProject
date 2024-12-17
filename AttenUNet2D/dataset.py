# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:57:43 2024

@author: user
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Dataset
def get_data(root_dir, direction='x', data_type='train'):
    pkg_list = []
    path = os.path.join(root_dir, 'dataset_2d_{}'.format(direction))
    path = os.path.join(path, data_type)
    imgs_path, masks_path = os.path.join(path, 'imgs'), os.path.join(path, 'masks')
    
    for filename in os.listdir(imgs_path):
        img_path, mask_path = os.path.join(imgs_path, filename), os.path.join(masks_path, filename)
        pkg_list.append((img_path, mask_path))
    
    return pkg_list

class DarkSideDataset(Dataset):
    def __init__(self, pkg_list, direction='x'):
        self.pkg_list = pkg_list
        self.direction = direction
        
        if self.direction == 'x' or self.direction == 'y':
            self.transform = transforms.Compose([
                transforms.Resize((256, 1024), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Grayscale(),
                transforms.ToTensor(),
                ])
            
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Grayscale(),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.pkg_list)

    def __getitem__(self, idx):
        seismic_path, fault_path = self.pkg_list[idx]
        image = Image.open(seismic_path).convert("L")
        mask = Image.open(fault_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = (self.transform(mask) * 255).to(torch.long)
        
        return image, mask