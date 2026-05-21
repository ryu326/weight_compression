"""Mixture-of-Gaussian + Laplacian entropy model.

Per-element shared mixture (channels argument is ignored — same parameters
across all elements; analogous to compressai EntropyBottleneck(channels=1)).
Forward signature mirrors compressai:
    (y_perm: (N, C, *)) -> (y_hat, likelihoods)
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_LOG2E = math.log2(math.e)


class MixtureEntropyModel(nn.Module):
    """K = num_gaussian + num_laplacian shared 1-D mixture components.

    During training: y_hat = y + uniform_noise(-0.5, 0.5).
    During eval:     y_hat = round(y - median) + median.
    Likelihood:      p(y_hat) = sum_k w_k * (CDF_k(y_hat + 0.5) - CDF_k(y_hat - 0.5)).
    """

    def __init__(
        self,
        num_gaussian: int = 3,
        num_laplacian: int = 3,
        scale_init: float = 1.0,
        likelihood_bound: float = 1e-9,
    ) -> None:
        super().__init__()
        K = num_gaussian + num_laplacian
        if K < 1:
            raise ValueError("Need at least 1 mixture component")
        self.num_gaussian = num_gaussian
        self.num_laplacian = num_laplacian
        self.K = K
        self.likelihood_bound = float(likelihood_bound)

        self._logits = nn.Parameter(torch.zeros(K))
        self._means = nn.Parameter(torch.linspace(-1.0, 1.0, K))
        self._log_scales = nn.Parameter(torch.full((K,), math.log(scale_init)))

        # buffers for compatibility with `update()` API (not used here, but
        # configure_optimizers checks for `.quantiles` substring in param names —
        # since we have none, all params go into the main optimizer).
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    @property
    def weights(self) -> Tensor:
        return F.softmax(self._logits, dim=0)

    def _get_median(self) -> Tensor:
        # Mean of means weighted by mixture weights — close enough to median for centering.
        return (self.weights * self._means).sum()

    def _component_cdf(self, x: Tensor, k: int) -> Tensor:
        mu = self._means[k]
        s = self._log_scales[k].exp().clamp_min(1e-6)
        centered = x - mu
        if k < self.num_gaussian:
            return 0.5 * (1.0 + torch.erf(centered / (s * math.sqrt(2.0))))
        return 0.5 + 0.5 * torch.sign(centered) * (1.0 - torch.exp(-centered.abs() / s))

    def log_likelihood_quantized(self, x: Tensor) -> Tensor:
        """log2 P(round(x)) under the mixture, with bin-width = 1."""
        half = 0.5
        w = self.weights
        prob = torch.zeros_like(x)
        for k in range(self.K):
            prob = prob + w[k] * (
                self._component_cdf(x + half, k) - self._component_cdf(x - half, k)
            )
        return torch.log2(prob.clamp_min(self.likelihood_bound))

    def forward(self, x: Tensor, training: Optional[bool] = None):
        """Quantize x and return (y_hat, likelihoods).

        Args:
            x: (N, C, *) — convention from compressai EntropyBottleneck.
            training: if True (or self.training), use additive uniform noise;
                otherwise round.
        """
        if training is None:
            training = self.training

        median = self._get_median()
        if training:
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            y_hat = x + noise
        else:
            y_hat = torch.round(x - median) + median

        log2_lik = self.log_likelihood_quantized(y_hat)
        likelihoods = (2.0 ** log2_lik).clamp_min(self.likelihood_bound)
        return y_hat, likelihoods

    def loss(self) -> Tensor:
        # No quantile-fitting aux loss; return zero so aux_optimizer pass is harmless.
        return torch.zeros((), device=self._logits.device)

    def update(self, force: bool = False, **kwargs) -> bool:
        # No CDF tables to bake; provide method for API parity with compressai EB.
        return True


def build(channels: int, num_gaussian: int = 3, num_laplacian: int = 3, **kwargs) -> nn.Module:
    # `channels` is ignored — mixture is shared across all elements.
    return MixtureEntropyModel(
        num_gaussian=int(num_gaussian),
        num_laplacian=int(num_laplacian),
    )
