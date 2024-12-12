from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL.Image as Image
import json, random
import numpy as np
import torch
import glob
class Weight_Matrix_Dataset(Dataset):
    def __init__(self, dataset, dataset_stats):
        # split = 'train' or 'val'

        self.dataset = dataset
        
        self.mean = dataset_stats['mean_channel']
        self.std = dataset_stats['std_channel']
        

    def __len__(self): 
        return len(self.dataset)
    
    def normalize_tensor(self, X):
        X = (X - self.mean) / self.std
        return X.requires_grad_(False)

    def __getitem__(self, idx): 
        
        img = self.dataset[idx]
        img = self.normalize_tensor(img)
        
        return img

def get_datasets(dataset_folder_path):
    data = torch.load(dataset_folder_path)
    
    # 질문. dataset_stats은 train 정보만 있는건지? 
    dataset_stats = torch.load(dataset_folder_path.replace('dataset.pt', 'dataset_stats.pt'))
    
    train_dataset = Weight_Matrix_Dataset(data['train'], dataset_stats['train'])
    valid_dataset = Weight_Matrix_Dataset(data['val'], dataset_stats['val'])  
    
    return train_dataset, valid_dataset