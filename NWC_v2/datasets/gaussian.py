"""Synthetic Gaussian block dataset.

Each `__getitem__(i)` draws a fresh `torch.randn(seq_len, input_size) * std`
sample. With `num_blocks` items per epoch, behaves like a streaming dataset.
"""
import torch
from torch.utils.data import Dataset


class GaussianBlockDataset(Dataset):
    def __init__(
        self,
        num_blocks: int = 100_000,
        seq_len: int = 1024,
        input_size: int = 16,
        std: float = 1.0,
        seed: int = 0,
    ):
        self.num_blocks = int(num_blocks)
        self.seq_len = int(seq_len)
        self.input_size = int(input_size)
        self.std = float(std)
        self.seed = int(seed)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        # Per-sample seed so DataLoader workers stay deterministic-ish across
        # restarts (the dataset object survives but workers re-fork).
        g = torch.Generator()
        g.manual_seed(self.seed + int(idx))
        x = torch.randn(self.seq_len, self.input_size, generator=g) * self.std
        return {"weight_block": x}


def get_dataset_gaussian(args):
    std = float(getattr(args, "gaussian_std", 1.0))
    train = GaussianBlockDataset(
        num_blocks=int(getattr(args, "num_blocks", 100_000)),
        seq_len=int(getattr(args, "seq_len", 1024)),
        input_size=int(getattr(args, "input_size", 16)),
        std=std,
        seed=int(getattr(args, "seed", 0)),
    )
    val = GaussianBlockDataset(
        num_blocks=max(64, int(getattr(args, "num_blocks", 100_000)) // 200),
        seq_len=int(getattr(args, "seq_len", 1024)),
        input_size=int(getattr(args, "input_size", 16)),
        std=std,
        seed=int(getattr(args, "seed", 0)) + 1,
    )
    # 6-tuple: (train, val, train_std, val_std, train_mean, val_mean)
    return train, val, std, std, 0.0, 0.0
