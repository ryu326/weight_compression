import os, argparse, logging, math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Tuple

def logger_setup(log_file_name=None, log_file_folder_name = None, filepath=os.path.abspath(__file__), package_files=[]):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', 
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO'.upper())

    stream = logging.StreamHandler()
    stream.setLevel('INFO'.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    info_file_handler = logging.FileHandler(log_file_folder_name + '/' + log_file_name , mode="a")
    info_file_handler.setLevel('INFO'.upper())
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
            
    return logger

#######################################################################################################################################
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        
        self.lmbda = lmbda
        
    def forward(self, output):
        
        out = {}
        
        b, s, d = output["x"].size()
        num_pixels = s * d
        
        out["mse_loss"] = self.mse(output["x"], output["x_hat"])
        
        # BPP
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        # 전체 loss
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]

        return out

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    
    parser.add_argument(
        "--dist_port", type=int, default=6006, required=True, help="dist_port(default: %(default)s)"
    )
    parser.add_argument(
        "--iter",
        default=200000,
        type=int,
        help="Number of iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--u-length",
        type=int,
        default=4
    )
    
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
   
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
   
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    
    parser.add_argument(
        "--slurm", action="store_true", default=False
    )
   
    parser.add_argument(
        "--dataset-path", type=str, default="/home/jgryu/Weight_compression/Wparam_dataset/dataset_2d/models--meta-llama--Meta-Llama-3-8B/mlp_attn__512_512_dataset.pt"
    )
    parser.add_argument(
        "--data_dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--length",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lmbda",
        type=float,
        default=1.,
    )
    args = parser.parse_args(argv)
    return args

def create_exp_folder(save_path) :
    try:
        os.mkdir(save_path)
        os.mkdir(f"{save_path}/figures")
    except:
        os.makedirs(save_path)
        os.makedirs(f"{save_path}/figures")
