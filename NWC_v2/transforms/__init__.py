"""Transform factory.  All builds return an `nn.Module` whose forward maps
`(..., in_dim) -> (..., out_dim)`."""
from typing import Any

import torch.nn as nn

from . import affine, linear, rht, resblock

_BUILDERS = {
    "affine": affine.build,
    "rht": rht.build,
    "linear": linear.build,
    "resblock": resblock.build,
}


def get_transform(name: str, in_dim: int, out_dim: int, **kwargs: Any) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(
            f"unknown transform '{name}'; choose from {sorted(_BUILDERS.keys())}"
        )
    return _BUILDERS[name](in_dim=in_dim, out_dim=out_dim, **kwargs)


__all__ = ["get_transform"]
