"""LTC `EntropyBottleneckLattice` wrapper, with Square (Z^d) lattice quantizer.

Adapts the LTC interface to our codec convention:
    Input:  y_perm with shape (N, C, *)
    Output: (y_hat, likelihoods) both with same shape (N, C, *)

The likelihood is a Monte-Carlo estimate of the convolution of the underlying
density with the lattice's Voronoi region — for the Square (Z^d) lattice this
is uniform on [-0.5, 0.5]^C.

Quantization uses straight-through estimator: y_hat = round(y) + (y - y.detach()).
"""
from typing import Optional

import torch
import torch.nn as nn

# Add NWC root to sys.path so we can import LTC submodule.  Append so we
# don't shadow NWC_v2's own top-level modules.
import sys
_LTC_ROOT = "/home/jgryu/workspace/weight_compression/NWC"
if _LTC_ROOT not in sys.path:
    sys.path.append(_LTC_ROOT)

from lattice_transform_coding.LTC.entropy_models import EntropyBottleneckLattice  # noqa: E402


class LatticeEntropyModel(nn.Module):
    """Square (Z^d) lattice quantization + LTC's EntropyBottleneckLattice density.

    Args:
        channels: dimension of the lattice (== last spatial axis after permute,
                 i.e. the codec's M).
        n_voronoi: number of MC samples drawn from the Voronoi region per call
                  for likelihood estimation.
        filters, init_scale, tail_mass: forwarded to EntropyBottleneckLattice.
    """

    def __init__(
        self,
        channels: int,
        n_voronoi: int = 16,
        filters: tuple = (3, 3, 3, 3),
        init_scale: float = 10.0,
        tail_mass: float = 1e-9,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_voronoi = int(n_voronoi)
        self.eb = EntropyBottleneckLattice(
            channels=self.channels,
            tail_mass=tail_mass,
            init_scale=init_scale,
            filters=tuple(filters),
        )

    def _sample_voronoi(self, n: int, device, dtype):
        """Square lattice (Z^d) Voronoi region = unit cube [-0.5, 0.5]^d."""
        return (torch.rand(n, self.channels, device=device, dtype=dtype) - 0.5)

    def forward(self, x: torch.Tensor, training: Optional[bool] = None):
        if training is None:
            training = self.training

        # x shape: (N, C, *).  Move channels to last, flatten the rest.
        N, C, *spatial = x.shape
        if C != self.channels:
            raise ValueError(
                f"LatticeEntropyModel expects channels={self.channels}, got C={C}"
            )
        # (N, C, *) -> (N, *, C) -> (N * prod(*), C)
        perm = (0,) + tuple(range(2, x.ndim)) + (1,)
        x_perm = x.permute(*perm).contiguous()
        flat_shape = x_perm.shape  # (N, *, C)
        x_flat = x_perm.reshape(-1, C)  # (B, C) where B = N * prod(*)

        # quantization: round to integer lattice with STE
        y_hard = torch.round(x_flat)
        if training:
            # match LTC training: add Voronoi noise so EB sees a smoothed dist
            noise_train = self._sample_voronoi(1, device=x.device, dtype=x.dtype).squeeze(0)
            y_flat = x_flat + noise_train.unsqueeze(0)  # (B, C)
        else:
            y_flat = y_hard + (x_flat - x_flat.detach())  # STE round

        # likelihoods via MC over Voronoi region
        noise_mc = self._sample_voronoi(self.n_voronoi, device=x.device, dtype=x.dtype)
        lik_flat = self.eb(x_flat, noise_mc, train=training)  # (B, C)
        lik_flat = lik_flat.clamp_min(1e-9)

        # reshape back: (B, C) -> (N, *, C) -> (N, C, *)
        y_hat = y_flat.reshape(flat_shape)
        likelihoods = lik_flat.reshape(flat_shape)
        inv_perm = (0, x.ndim - 1) + tuple(range(1, x.ndim - 1))
        y_hat = y_hat.permute(*inv_perm).contiguous()
        likelihoods = likelihoods.permute(*inv_perm).contiguous()
        return y_hat, likelihoods

    def loss(self) -> torch.Tensor:
        # EBL has no separate quantile aux loss in this codepath.
        device = next(self.eb.parameters()).device
        return torch.zeros((), device=device)

    def update(self, force: bool = False, **kwargs) -> bool:
        return True


def build(channels: int, n_voronoi: int = 16, **kwargs) -> nn.Module:
    return LatticeEntropyModel(channels=channels, n_voronoi=n_voronoi)
