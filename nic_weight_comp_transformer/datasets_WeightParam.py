from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL.Image as Image
import json, random
import numpy as np
import torch
import glob
class WParam_dataset(Dataset):
    def __init__(self, dataset_folder, split='train', seed = 100, length=8, data_dim=512, normalize = True, normalize_channel = True):

        assert dataset_folder.endswith('.pt')
        
        self.dataset = torch.load(dataset_folder)[split]
        self.dataset_stats = torch.load(dataset_folder.replace('dataset.pt', 'dataset_stats.pt'))['train']
        self.length = length
        self.data_dim = data_dim
        assert self.dataset.shape[-1] == length*data_dim

        if normalize:
            if normalize_channel:
                self.mean = self.dataset_stats['mean_channel']
                self.std = self.dataset_stats['std_channel']
            else:
                self.mean = self.dataset_stats['mean']
                self.std = self.dataset_stats['std']

    def __len__(self): 
        return len(self.dataset)
    
    def normalize_tensor(self, X):
        X = (X - self.mean) / self.std
        return X.requires_grad_(False)

    def __getitem__(self, idx): 
        img = self.dataset[idx]
        img = self.normalize_tensor(img)
        img = img.reshape(self.length, self.data_dim)
        return img
        
        