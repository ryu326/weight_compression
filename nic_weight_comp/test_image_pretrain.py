import os, random, sys, socket, lpips, shutil, operator

# 시간 측정해보기

import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets_Imagenet_best_worst import Imagenet_best_worst
from datasets_ImageNet import ImageNet_dataset

# from datasets_WeightParam import WParam_dataset
# from datasets_openimages_v6 import Openimages_v6_dataset

from pytorch_msssim import ms_ssim as ms_ssim_func

from models.TCM import TCM
from models.FTIC import FrequencyAwareTransFormer
from models.ELIC import ELIC, model_config

from utils.optimizers import *
from utils.util import *


from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL.Image as Image
import json, random
import numpy as np
import torch
import glob
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()


def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


imagenet_mean = 0.448
imagenet_std = 0.226

## attn 256 256
# wp_mean = 8.708306e-07
# wp_std = 0.023440132

## mlp 32 32
# wp_mean = -5.42295056421355e-06
# wp_std =  0.011819059083636133


def make_image_format(tensor, wp_mean, wp_std, normalize, imagelize):
    if normalize:
        tensor = (tensor - wp_mean) / wp_std
    if imagelize:
        assert normalize == True
        tensor = tensor * imagenet_std + imagenet_mean
    return tensor


# def reverse_image_format(tensor, wp_mean, wp_std):
#     tensor = tensor - imagenet_mean  # imagenet_mean을 빼기
#     tensor = tensor / (imagenet_std)  # (imagenet_std * c)로 나누기
#     tensor = tensor * wp_std + wp_mean  # wp_std로 곱하고 wp_mean을 더하기
#     return tensor


class WParam_dataset(Dataset):
    def __init__(self, dataset_folder, split="train", seed=100, normalize=True, imagelize=True, num=1000):

        directories = [
            os.path.join(dataset_folder, d)
            for d in os.listdir(dataset_folder)
            if os.path.isdir(os.path.join(dataset_folder, d))
        ]
        print(directories)
        self.wp_path_list = glob.glob(f"{directories[0]}/**/*.npy", recursive=True)
        print(len(self.wp_path_list))
        if split == "val":
            random.seed(seed)
            random_list = random.sample(self.wp_path_list, num)
            self.wp_path_list = random_list

        self.mean = np.load(dataset_folder + f"/mean_value.npy")
        self.std = np.load(dataset_folder + f"/std_value.npy")
        self.normalize = normalize
        self.imagelize = imagelize

        print(self.mean)
        print(self.std)

    def __len__(self):
        return len(self.wp_path_list)

    def load_tensor(self, tensor_path):
        X = np.load(tensor_path)
        # X = (X - self.mean) / self.std
        X = make_image_format(X, wp_mean=self.mean, wp_std=self.std, normalize=self.normalize, imagelize=self.normalize)
        X = np.repeat(np.expand_dims(X, axis=0), 3, axis=0)
        return torch.from_numpy(X).requires_grad_(False)

    def __getitem__(self, idx):
        img = self.load_tensor(self.wp_path_list[idx])
        return img


def test(test_dataset, model, save_path, logger):
    avg_bpp = 0.0
    mean_MSE = 0

    mse_func = nn.MSELoss()

    device = next(model.parameters()).device

    # for idx, (image_name, image) in enumerate(test_dataset) :
    for idx, image in enumerate(test_dataset):

        img = image.to(device)
        x = img.unsqueeze(0).to(device)
        # if idx == 100: break
        try:
            x_paddeimg, padding = pad(x, p=128)
            out_enc = model.compress(x_paddeimg.to(device))
        except:
            x_paddeimg, padding = pad(x, p=256)
            out_enc = model.compress(x_paddeimg.to(device))

        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

        num_pixels = x.size(0) * x.size(2) * x.size(3)
        # print(num_pixels)

        bpp = 0
        for s in out_enc["strings"]:
            if s != [0]:  #
                bpp += len(s[0]) * 8.0 / num_pixels

        x_hat = crop(out_dec["x_hat"], padding).clone().detach()
        mse = mse_func(x, x_hat).item()
        # mse = mse * (test_dataset.std ** 2) / (imagenet_std**2)
        mse = mse / (imagenet_std**2)

        avg_bpp += bpp
        mean_MSE += mse

        logger.info(f"File name: {idx}, MSE: {mse}, BPP: {bpp}")
        # print(f"File name: {idx}, MSE: {mse}, BPP: {bpp}")
        # output_path = os.path.join(save_path, f'{idx}')
        # torchvision.utils.save_image(x_hat, output_path, nrow=1)

    avg_bpp /= len(test_dataset)
    mean_MSE /= len(test_dataset)

    logger.info(f"Average_MSE: {mean_MSE}, Average_Bit-rate: {avg_bpp} bpp")

    return avg_bpp, mean_MSE


def logger_setup(log_file_name=None, log_file_folder_name=None, package_files=[]):
    formatter = logging.Formatter("%(asctime)s %(levelname)s - %(funcName)s: %(message)s", "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO".upper())

    stream = logging.StreamHandler()
    stream.setLevel("INFO".upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    info_file_handler = logging.FileHandler(log_file_folder_name + "/" + log_file_name, mode="a")
    info_file_handler.setLevel("INFO".upper())
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    # logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    return logger


def main(args):
    save_data = {}
    bpps = []
    mses = []
    # dataset_dir = "/workspace/jgryu/Weight_compression/Wparam_dataset/image_shape_wp/meta-llama-3-8b_mlp_val_json/256_256"
    dataset_dir = args.dataset_dir
    save_path = "/workspace/jgryu/Weight_compression/nic_weight_comp/test_wp_with_image_pt_model"

    test_dataset = WParam_dataset(
        dataset_folder=dataset_dir, split="val", seed=100, num=300, normalize=args.normalize, imagelize=args.imagelize
    )
    path = "/workspace/jgryu/Weight_compression/nic_weight_comp/checkpoints_image_pretrained"
    pt_list = os.listdir(path)

    log_path = os.path.join(save_path, "/".join(dataset_dir.split("/")[-2:]))
    if args.normalize:
        log_path += "_normalize"
    if args.imagelize:
        assert args.normalize
        log_path = log_path.replace("_normalize", "_imagelize")

    os.makedirs(log_path, exist_ok=True)

    # for pt, lm in zip(pt_list, ['0.05', '0.013', '0.025', '0.0025', '0.0035', '0.0067']):
    # [0.013, 0.0025, 0.0067, 0.05, 0.025, 0.0035]
    # for pt in pt_list:
    #     lm = pt.replace('.pth.tar', '')
    lmbdas = []
    for pt in pt_list:
        lm = pt.replace(".pth", "")
        lmbdas.append(float(lm))
    lmbdas = sorted(lmbdas)
    print(lmbdas)

    logging.basicConfig(level=logging.INFO)  # 로거 기본 설정
    root_logger = logging.getLogger()

    for lm in lmbdas:
        print(f"##### {lm} #####")
        pt = f"{lm}.pth"
        ck_path = f"/workspace/jgryu/Weight_compression/nic_weight_comp/checkpoints_image_pretrained/{lm}.pth"
        try:
            checkpoint = torch.load(ck_path, map_location=device)
            assert isinstance(checkpoint, dict), "Checkpoint is not a dictionary"
            assert "state_dict" in checkpoint, "Missing 'state_dict' in checkpoint"
            print(f"Checkpoint for {lm} loaded successfully.")
        except Exception as e:
            print(f"Failed to load checkpoint for {lm}: {e}")
            continue

        model = TCM(N=64)
        try:
            model.load_state_dict(checkpoint["state_dict"])
            print(f"Model state_dict loaded successfully for {lm}.")
        except RuntimeError as e:
            print(f"Failed to load model state_dict for {lm}: {e}")

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        logger = logger_setup(log_file_name=f"logs_lmbda{lm}", log_file_folder_name=log_path)

        model = model.eval().to(device)
        model.requires_grad_(False)
        model.update()

        avg_bpp, mean_MSE = test(test_dataset, model, log_path, logger)
        if not args.normalize:
            mean_MSE = mean_MSE / test_dataset.std**2
        elif args.imagelize:
            mean_MSE = mean_MSE / imagenet_std**2

        bpps.append(avg_bpp)
        mses.append(mean_MSE)

    save_data["mse"] = mses
    save_data["bpp"] = bpps

    print("bbp: ", bpps)
    print("mse: ", mses)

    json_path = log_path + "/data.json"
    with open(json_path, "w") as json_file:
        json.dump(save_data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main function with dataset directory.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--imagelize", action="store_true", default=False)

    args = parser.parse_args()
    main(args)


# python script_name.py --dataset_dir /path/to/dataset
