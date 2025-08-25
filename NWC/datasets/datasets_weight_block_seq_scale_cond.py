import glob
import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Weight_Vector_Dataset(Dataset):
    def __init__(self, dataset, dataset_stats, input_size, args, uniform_scale = False, scale_max = -1):
        # split = 'train' or 'val'

        self.dataset = dataset

        self.uniform_scale = uniform_scale
        if self.uniform_scale:
            if scale_max <= 0:
                raise ValueError("If uniform_scale is True, scale_max must be a positive number.")
            self.scale_max = scale_max

        if args.dataset_stat_type == 'scaler':
            self.mean = torch.tensor(dataset_stats["mean"])
            self.std = torch.tensor(dataset_stats["std"])
        elif args.dataset_stat_type == 'channel':
            self.mean = torch.Tensor(dataset_stats["mean_channel"]).view(-1, input_size)
            self.std = torch.Tensor(dataset_stats["std_channel"]).view(-1, input_size)

        self.input_size = input_size
        
        if args.pre_normalize == True:
            self.dataset['weight'] = (self.dataset['weight'] - self.mean) / self.std
            self.mean = torch.tensor(0.0)
            self.std = torch.tensor(1.0)

    def __len__(self):
        return len(self.dataset['weight'])

    def __getitem__(self, idx):
        img = self.dataset['weight'][idx].view(-1, self.input_size)
        
        if self.uniform_scale:
            # random_scale_val = torch.rand(1) * self.scale_max
            # scale = torch.full_like(img, random_scale_val)
            scale = torch.rand_like(img) * self.scale_max
        else:
            scale = self.dataset['scale'][idx].view(-1, self.input_size)
        
        scale = self.dataset['scale'][idx].view(-1, self.input_size)
        return {'weight_block': img,
                'scale_cond': scale,
                }
        


def get_datasets_block_seq_scale_cond(dataset_pt_path, input_size, args, uniform_scale = False, scale_max = None):
    
    data = torch.load(dataset_pt_path)
    
    with open(dataset_pt_path.replace(".pt", "_dataset_stats.json"), "r", encoding="utf-8") as file:
        dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

    train_dataset = Weight_Vector_Dataset(
        data["train"], dataset_stats["train"], input_size, args,
        uniform_scale=uniform_scale, scale_max=scale_max
    )
    valid_dataset = Weight_Vector_Dataset(
        data["val"], dataset_stats["val"], input_size, args,
        uniform_scale=uniform_scale, scale_max=scale_max
    )

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]