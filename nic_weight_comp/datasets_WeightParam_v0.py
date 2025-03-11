import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class WParam_dataset(Dataset):
    def __init__(self, dataset_folder, split="train", param_type="attn", image_size=(256, 256), seed=100, slurm=False):
        path = os.path.join(dataset_folder, "path_json", f"{param_type}_tensor_path_{split}.json")

        with open(path, "r") as f:
            tensor_path_list = json.load(f)

        self.tmp_path = os.path.join(dataset_folder, f"tmp/{split}", f"{param_type}_tensor_path_{split}")
        os.makedirs(self.tmp_path, exist_ok=True)
        count = 0
        split_tensor_path_list = []
        for tensor_path in tensor_path_list:
            t = np.load(dataset_folder + tensor_path)

            if t.size % (image_size[0] * image_size[1]) != 0:
                continue
                # print(f'나누어 떨어지지 않습니다.')
            t = t.reshape(-1, image_size[0], image_size[1])
            l = t.shape[0]
            # l = int(t.size / (image_size[0] * image_size[1]))
            for i in range(l):
                path = os.path.join(self.tmp_path, f"{i+count}.npy")
                np.save(path, t[i])
                split_tensor_path_list.append(path)
            count += l

        self.split_tensor_path_list = split_tensor_path_list

        # random.seed(seed)
        # print(count)
        # random_idx_list = random.sample(range(count), 148200)

        # self.tensor_path_list = []
        # for idx in random_idx_list :
        #     self.tensor_path_list.append(split_tensor_path_list[idx])

        if split == "val":
            self.split_tensor_path_list = self.split_tensor_path_list[:50]

        # self.wp_mean = 8.708306e-07
        # self.wp_std = 0.023440132
        ## json 파일에 저장해서 거기서 읽어오게 수정

        # if split=='train' :
        #     self.transform = transforms.Compose(
        #         [transforms.RandomCrop(image_size), transforms.ToTensor()])
        # elif split=='valid' :
        #     self.transform = transforms.Compose(
        #         [transforms.CenterCrop(image_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.split_tensor_path_list)

    def load_tensor(self, tensor_path):
        X = np.load(tensor_path)
        X = (X - self.wp_mean) / self.wp_std
        X = np.repeat(np.expand_dims(X, axis=0), 3, axis=0)
        return torch.from_numpy(X).requires_grad_(False)

    def __getitem__(self, idx):
        img = self.load_tensor(self.split_tensor_path_list[idx])
        return img
