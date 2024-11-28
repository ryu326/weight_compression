import os, argparse, logging, math, lpips

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
#######################################################################################################################################
# 이미지 <-> 주파수 왔다리 갔다리 (내가 짠 것보다 코드가 깔끔해서 대체함. 킹받네. 출처: https://github.com/CassiniHuy/image-low-pass-filters-pytorch)

### Ideal low-pass
def _get_center_distance(size: Tuple[int], device: str = 'cpu') -> Tensor:
    """Compute the distance of each matrix element to the center.

    Args:
        size (Tuple[int]): [m, n].
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [m, n].
    """    
    m, n = size
    i_ind = torch.tile(
                torch.tensor([[[i]] for i in range(m)], device=device),
                dims=[1, n, 1]).float()  # [m, n, 1]
    j_ind = torch.tile(
                torch.tensor([[[i] for i in range(n)]], device=device),
                dims=[m, 1, 1]).float()  # [m, n, 1]
    ij_ind = torch.cat([i_ind, j_ind], dim=-1)  # [m, n, 2]
    ij_ind = ij_ind.reshape([m * n, 1, 2])  # [m * n, 1, 2]
    center_ij = torch.tensor(((m - 1) / 2, (n - 1) / 2), device=device).reshape(1, 2)
    center_ij = torch.tile(center_ij, dims=[m * n, 1, 1])
    dist = torch.cdist(ij_ind, center_ij, p=2).reshape([m, n])
    return dist


def _get_ideal_weights(size: Tuple[int], D0: int, lowpass: bool = True, device: str = 'cpu') -> Tensor:
    """Get H(u, v) of ideal bandpass filter.

    Args:
        size (Tuple[int]): [H, W].
        D0 (int): The cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """    
    center_distance = _get_center_distance(size, device)
    center_distance[center_distance > D0] = -1
    center_distance[center_distance != -1] = 1
    if lowpass is True:
        center_distance[center_distance == -1] = 0
    else:
        center_distance[center_distance == 1] = 0
        center_distance[center_distance == -1] = 1
    return center_distance


def _to_freq(image: Tensor) -> Tensor:
    """Convert from spatial domain to frequency domain.

    Args:
        image (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W]
    """    
    img_fft = torch.fft.fft2(image)
    img_fft_shift = torch.fft.fftshift(img_fft)
    return img_fft_shift

def _to_space(image_fft: Tensor) -> Tensor:
    """Convert from frequency domain to spatial domain.

    Args:
        image_fft (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W].
    """    
    img_ifft_shift = torch.fft.ifftshift(image_fft)
    img_ifft = torch.fft.ifft2(img_ifft_shift)
    img = img_ifft.real.clamp(0, 1)
    return img

def _to_space_feature(feature_fft: Tensor) -> Tensor:
    """Convert from frequency domain to spatial domain.

    Args:
        image_fft (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W].
    """    
    feature_ifft_shift = torch.fft.ifftshift(feature_fft)
    feature_ifft = torch.fft.ifft2(feature_ifft_shift).real
    return feature_ifft

def ideal_bandpass(image: Tensor, D0: int, lowpass: bool = True) -> Tensor:
    """Low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.

    Returns:
        Tensor: [B, C, H, W].
    """    
    img_fft = _to_freq(image)
    weights = _get_ideal_weights(img_fft.shape[-2:], D0=D0, lowpass=lowpass, device=image.device)
    img_fft = img_fft * weights
    img = _to_space(img_fft)
    
    return img

def ideal_bandpass_freq_domain(image: Tensor, D0: int, lowpass: bool = True) -> Tensor:
    """Low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.

    Returns:
        Tensor: [B, C, H, W].
    """    
    img_fft = _to_freq(image)
    weights = _get_ideal_weights(img_fft.shape[-2:], D0=D0, lowpass=lowpass, device=image.device)
    img_fft = img_fft * weights
    
    return img_fft

#######################################################################################################################################
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, radius_denominator=8):
        super().__init__()
        self.mse = nn.MSELoss()
        
        self.lmbda = lmbda
        
    def forward(self, output):
        
        out = {}
        
        b, _, h, w = output["x"].size()
        num_pixels = b * h * w
        
        out["mse_loss"] = self.mse(output["x"], output["x_hat"])
        
        # BPP
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        # 전체 loss
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
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
        "--model_name",
        type=str,
        default='TCM',
        help="pre-trained neural codec for processing low-freq. regions (default: %(default)s)",
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
        "--image_quality",
        type=int,
        default=1,
        help="Image quality level (default: %(default)s)",
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
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
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
        "--radius_denominator", type=int, default=8
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="../Wparam_dataset"
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
