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
    def __init__(self, dataset, dataset_stats, input_size, args, split='train', return_idx_ltype = False):
        self.dataset = dataset
        self.split = split

        if args.dataset_stat_type == 'scaler':
            self.mean = torch.tensor(dataset_stats["mean"])
            self.std = torch.tensor(dataset_stats["std"])
        elif args.dataset_stat_type == 'channel':
            self.mean = torch.Tensor(dataset_stats["mean_channel"]).view(-1, input_size)
            self.std = torch.Tensor(dataset_stats["std_channel"]).view(-1, input_size)

        self.input_size = input_size
        self.return_idx_ltype = return_idx_ltype
        
    def __len__(self):
        if self.return_idx_ltype:
            return len(self.dataset['weight'])
        return len(self.dataset)

    def __getitem__(self, i):
        if self.return_idx_ltype == False:
            img = self.dataset[i].view(-1, self.input_size)
            return {
                'weight_block': img,
                'qmap': torch.rand(1)
            }
        else:
            img = self.dataset['weight'][i].view(-1, self.input_size)
            return {
                'weight_block': img,  # (-1, 16)
                'qmap': torch.rand(1), # (1,)
                'depth': self.dataset['idx'][i].to(torch.long).reshape(1,), # (1, )
                'ltype': self.dataset['layer_type'][i].to(torch.long).reshape(1,), # (1, )
            }


# class Weight_Vector_Dataset(Dataset):
#     def __init__(self, dataset, dataset_stats, input_size, Q, args):
#         # split = 'train' or 'val'

#         self.dataset = dataset
#         self.Q = int(Q)

#         if args.dataset_stat_type == 'scaler':
#             self.mean = torch.tensor(dataset_stats["mean"])
#             self.std = torch.tensor(dataset_stats["std"])
#         elif args.dataset_stat_type == 'channel':
#             self.mean = torch.Tensor(dataset_stats["mean_channel"]).view(-1, input_size)
#             self.std = torch.Tensor(dataset_stats["std_channel"]).view(-1, input_size)

#         self.input_size = input_size
        
#         self.random_values = [torch.tensor(i, dtype=torch.long) for i in range(self.Q)]

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img = self.dataset[idx].view(-1, self.input_size)
#         q_level = random.choice(self.random_values)
#         return {'weight_block': img,
#                 'q_level': q_level.unsqueeze(0)}


def get_datasets_block_seq_qmap_uniform(dataset_pt_path, input_size, args, return_idx_ltype=False):
    
    data = torch.load(dataset_pt_path)
    
    with open(dataset_pt_path.replace(".pt", "_dataset_stats.json"), "r", encoding="utf-8") as file:
        dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

    train_dataset = Weight_Vector_Dataset(data["train"], dataset_stats["train"], input_size,  args, return_idx_ltype = return_idx_ltype)
    valid_dataset = Weight_Vector_Dataset(data["val"], dataset_stats["val"], input_size, args, split='val', return_idx_ltype=return_idx_ltype)

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]
