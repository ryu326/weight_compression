"""Misc utilities: AverageMeter, checkpoint save/load, simple logger."""
import logging
import os
import sys
from typing import Optional

import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def make_logger(name: str = "nwc_v2", log_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
