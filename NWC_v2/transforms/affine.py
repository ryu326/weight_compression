"""Per-element learnable affine transform: y = a · x + b.

Shape-preserving (in_dim == out_dim).  At init, `a = 1` and `b = 0` so the
transform is the identity; both drift during training.
"""
import torch
import torch.nn as nn


class AffineTransform(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if in_dim != out_dim:
            raise ValueError(
                f"affine transform requires in_dim == out_dim, "
                f"got in_dim={in_dim} out_dim={out_dim}"
            )
        self.dim = in_dim
        self.a = nn.Parameter(torch.ones(in_dim))
        self.b = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        return x * self.a + self.b

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


def build(in_dim: int, out_dim: int, **kwargs) -> nn.Module:
    return AffineTransform(in_dim, out_dim)
