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
    def __init__(self, dataset, dataset_stats, input_size, args, uniform_scale = False, return_idx_ltype = False, scale_max = -1,
        aug_scale=False,              # 랜덤 스케일링 on/off
        aug_scale_p=0.01,              # 적용 확률
        aug_scale_min=0.5,            # 스케일 하한
        aug_scale_max=2.0,            # 스케일 상한
        aug_scale_mode='block',       # 'block' | 'row' | 'element'
        aug_log_uniform=False,        # 로그-균등 분포(크기 변화에 더 자연스러움)
        aug_update_cond=False,         # 스케일링을 cond에도 반영할지
        ):
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
        self.return_idx_ltype = return_idx_ltype
        
        if getattr(args, "pre_normalize", False):
            self.dataset['weight'] = (self.dataset['weight'] - self.mean) / self.std
            self.mean = torch.tensor(0.0)
            self.std  = torch.tensor(1.0)

        self.aug_scale        = aug_scale
        self.aug_scale_p      = float(aug_scale_p)
        self.aug_scale_min    = float(aug_scale_min)
        self.aug_scale_max    = float(aug_scale_max)
        self.aug_scale_mode   = aug_scale_mode
        self.aug_log_uniform  = aug_log_uniform
        self.aug_update_cond  = aug_update_cond

    def __len__(self):
        return len(self.dataset['weight'])

    def _sample_scale_scalar(self):
        """스칼라 스케일 하나 샘플."""
        if self.aug_log_uniform:
            # log-uniform: exp(uniform(log(min), log(max)))
            lo, hi = np.log(self.aug_scale_min), np.log(self.aug_scale_max)
            s = float(np.exp(np.random.uniform(lo, hi)))
        else:
            s = float(np.random.uniform(self.aug_scale_min, self.aug_scale_max))
        return s

    def _sample_scale_tensor(self, shape):
        """주어진 shape로 스케일 텐서 샘플."""
        if self.aug_log_uniform:
            lo, hi = np.log(self.aug_scale_min), np.log(self.aug_scale_max)
            r = torch.empty(shape).uniform_(lo, hi).exp_()
        else:
            r = torch.empty(shape).uniform_(self.aug_scale_min, self.aug_scale_max)
        return r


    def __getitem__(self, idx):
        img = self.dataset['weight'][idx].view(-1, self.input_size)
        
        if self.uniform_scale:
            # random_scale_val = torch.rand(1) * self.scale_max
            # scale = torch.full_like(img, random_scale_val)
            scale = torch.rand_like(img) * self.scale_max + 1e-6  # 0 방지
        else:
            if self.dataset['scale'].shape[-1] != 1:
                scale = self.dataset['scale'][idx].view(-1, self.input_size)
            else :   ## dataset['scale'].shape == (len, 1)
                scale = self.dataset['scale'][idx] # shape: (1)
                    
        if self.aug_scale and (random.random() < self.aug_scale_p):
            if self.aug_scale_mode == 'block':
                s = self._sample_scale_scalar()
                img = img * s
                if self.aug_update_cond:
                    scale = scale * s
            # elif self.aug_scale_mode == 'row':
            #     # 각 row(=step/sequence 단위)마다 다른 스케일
            #     s = self._sample_scale_tensor((img.size(0), 1))
            #     img = img * s
            #     if self.aug_update_cond:
            #         scale = scale * s
            # elif self.aug_scale_mode == 'element':
            #     s = self._sample_scale_tensor(img.size())
            #     img = img * s
            #     if self.aug_update_cond:
            #         scale = scale * s
            else:
                raise ValueError(f"unknown aug_scale_mode: {self.aug_scale_mode}")
        if self.return_idx_ltype:
            return {'weight_block': img,
                    'scale_cond': scale,
                    'depth': self.dataset['idx'][idx].to(torch.long).reshape(1,), # (1, )
                    'ltype': self.dataset['layer_type'][idx].to(torch.long).reshape(1,), # (1, )
                }
        else :
            return {'weight_block': img,
                    'scale_cond': scale,
                }
        
def get_datasets_block_seq_scale_cond(dataset_pt_path, input_size, args, uniform_scale = False, scale_max = None):
# def get_datasets_block_seq_scale_cond(dataset_pt_path, input_size, args,
#                                       uniform_scale=False, scale_max=None,
#                                       # 아래부터는 augmentation 기본값 (필요 시 바꾸세요)
#                                       aug_scale=False, aug_scale_p=1.0,
#                                       aug_scale_min=0.5, aug_scale_max=2.0,
#                                       aug_scale_mode='block', aug_log_uniform=False,
#                                       aug_update_cond=True):

    data = torch.load(dataset_pt_path)
    with open(dataset_pt_path.replace(".pt", "_dataset_stats.json"), "r", encoding="utf-8") as f:
        dataset_stats = json.load(f)

    common_kwargs = dict(
        input_size=input_size, args=args,
        uniform_scale=uniform_scale, scale_max=scale_max,
        aug_scale=args.aug_scale, aug_scale_p=args.aug_scale_p,
        aug_scale_min=args.aug_scale_min, aug_scale_max=args.aug_scale_max,
        aug_scale_mode=args.aug_scale_mode, aug_log_uniform=args.aug_log_uniform,
        aug_update_cond=args.aug_update_cond,
        return_idx_ltype = args.use_pe
    )

    train_dataset = Weight_Vector_Dataset(data["train"], dataset_stats["train"], **common_kwargs)
    valid_dataset = Weight_Vector_Dataset(data["val"],   dataset_stats["val"],   **common_kwargs)

    return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]

# def get_datasets_block_seq_scale_cond(dataset_pt_path, input_size, args, uniform_scale = False, scale_max = None):
    
#     data = torch.load(dataset_pt_path)
    
#     with open(dataset_pt_path.replace(".pt", "_dataset_stats.json"), "r", encoding="utf-8") as file:
#         dataset_stats = json.load(file)  # JSON 파일을 Python 객체로 변환

#     train_dataset = Weight_Vector_Dataset(
#         data["train"], dataset_stats["train"], input_size, args,
#         uniform_scale=uniform_scale, scale_max=scale_max
#     )
#     valid_dataset = Weight_Vector_Dataset(
#         data["val"], dataset_stats["val"], input_size, args,
#         uniform_scale=uniform_scale, scale_max=scale_max
#     )

#     return train_dataset, valid_dataset, dataset_stats["train"]["std"], dataset_stats["val"]["std"]