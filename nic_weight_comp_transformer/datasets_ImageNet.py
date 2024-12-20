import json
import random

import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageNet_dataset(Dataset):
    def __init__(self, dataset_folder="/data/ILSVRC2012", split="train", image_size=(256, 256), seed=100, slurm=False):

        if slurm == False:
            self.img_dir = f"{dataset_folder}/{split}"
        else:
            self.img_dir = dataset_folder

        if slurm == False:
            with open("./ImageNet_path_list_larger_than_256.json", "r") as f:
                img_path_list = json.load(f)
        else:  # 슬럼에서 돌리면
            with open("./ImageNet_path_list_larger_than_256_slurm.json", "r") as f:
                img_path_list = json.load(f)

        random.seed(seed)
        random_idx_list = random.sample(range(len(img_path_list)), 300000)

        self.img_path_list = []

        for idx in random_idx_list:
            self.img_path_list.append(img_path_list[idx])

        if split == "train":
            self.transform = transforms.Compose([transforms.RandomCrop(image_size), transforms.ToTensor()])
        elif split == "valid":
            self.transform = transforms.Compose([transforms.CenterCrop(image_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_path_list)

    def load_image(self, image_path):

        image = Image.open(self.img_dir + "/" + image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        return self.transform(image)

    def __getitem__(self, idx):
        img = self.load_image(self.img_path_list[idx])

        return img
