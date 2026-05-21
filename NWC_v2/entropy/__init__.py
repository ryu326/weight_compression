"""Entropy model factory.  All builds return an `nn.Module` whose forward
takes `(N, C, *)` and returns `(y_hat, likelihoods)` of the same shape.

Modules also expose `loss()` (aux) and `update(force=...)` for API parity
with compressai's EntropyBottleneck.
"""
from typing import Any

import torch.nn as nn

from . import compressai_eb, parametric, lattice

_BUILDERS = {
    "compressai": compressai_eb.build,
    "parametric": parametric.build,
    "lattice": lattice.build,
}


def get_entropy_model(name: str, channels: int, **kwargs: Any) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(
            f"unknown entropy model '{name}'; choose from {sorted(_BUILDERS.keys())}"
        )
    return _BUILDERS[name](channels=channels, **kwargs)


__all__ = ["get_entropy_model"]
