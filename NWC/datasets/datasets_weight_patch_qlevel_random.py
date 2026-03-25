import glob
import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Weight_Patch_Dataset(Dataset):
    def __init__(self, dataset, dataset_stats, input_size, Q, args, split='train', return_idx_ltype = False):
        self.dataset = dataset
        self.Q = int(Q)
        self.split = split

        if args.dataset_stat_type == 'scaler':
            self.mean = torch.tensor(dataset_stats["mean"])
            self.std = torch.tensor(dataset_stats["std"])
        elif args.dataset_stat_type == 'channel':
            self.mean = torch.Tensor(dataset_stats["mean_channel"]).view(-1, input_size)
            self.std = torch.Tensor(dataset_stats["std_channel"]).view(-1, input_size)

        self.input_size = input_size
        self.return_idx_ltype = return_idx_ltype
        
        assert dataset['weight'].shape[-1] == self.input_size, f"{dataset['weight'].shape}, must be matched {input_size}"

    def __len__(self):
        if 'weight' in self.dataset:
            return len(self.dataset['weight'])
        else:
            return len(self.dataset)

    def __getitem__(self, i):

        q_level = torch.randint(0, self.Q, (self.input_size,), dtype=torch.long)

        if self.return_idx_ltype == False:
            x = self.dataset['weight'][i]
            return {
                'weight_block': x,   # (B, L, I)
                'q_level': q_level  # (B, I)
            }
        else:
            x = self.dataset['weight'][i]
            return {
                'weight_block': x,  # (-1, 16)
                'q_level': q_level, # (1,)
                'depth': self.dataset['idx'][i].to(torch.long).reshape(1,), # (1, )
                'ltype': self.dataset['layer_type'][i].to(torch.long).reshape(1,), # (1, )
            }