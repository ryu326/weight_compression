"""Single nn.Linear transform — bias-free projection in_dim → out_dim."""
import torch.nn as nn


def build(in_dim: int, out_dim: int, **kwargs) -> nn.Module:
    return nn.Linear(in_dim, out_dim, bias=False)
