import argparse
import logging
import math
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# def logger_setup(log_file_name=None, log_file_folder_name=None, filepath=os.path.abspath(__file__), package_files=[]):
#     formatter = logging.Formatter("%(asctime)s %(levelname)s - %(funcName)s: %(message)s", "%H:%M:%S")
#     logger = logging.getLogger(__name__)
#     logger.setLevel("INFO".upper())

#     stream = logging.StreamHandler()
#     stream.setLevel("INFO".upper())
    
#     stream.setFormatter(formatter)
#     logger.addHandler(stream)

#     info_file_handler = logging.FileHandler(log_file_folder_name + "/" + log_file_name, mode="a")
#     info_file_handler.setLevel("INFO".upper())
#     info_file_handler.setFormatter(formatter)
#     logger.addHandler(info_file_handler)

#     logger.info(filepath)

#     for f in package_files:
#         logger.info(f)
#         with open(f, "r") as package_f:
#             logger.info(package_f.read())

#     return logger


def logger_setup(log_file_name=None, log_file_folder_name=None,
                 filepath=os.path.abspath(__file__), package_files=[]):
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s - %(funcName)s: %(message)s", "%H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 중복 핸들러 방지
    if logger.handlers:
        return logger

    # 스트림 핸들러 (항상 추가)
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    # 파일 핸들러 (둘 다 값이 있을 때만)
    if log_file_name and log_file_folder_name:
        os.makedirs(log_file_folder_name, exist_ok=True)
        info_file_handler = logging.FileHandler(
            os.path.join(log_file_folder_name, log_file_name), mode="a"
        )
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        try:
            with open(f, "r") as package_f:
                logger.info(package_f.read())
        except FileNotFoundError:
            logger.warning(f"File not found: {f}")

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
    size = out_net["x_hat"].size()
    num_pixels = size[0] * size[2]
    return sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels) for likelihoods in out_net["likelihoods"].values()
    ).item()


def create_exp_folder(save_path):
    try:
        os.mkdir(save_path)
        os.mkdir(f"{save_path}/figures")
    except:
        os.makedirs(save_path)
        os.makedirs(f"{save_path}/figures")
