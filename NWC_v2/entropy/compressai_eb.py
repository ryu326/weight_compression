"""Wrapper around `compressai.entropy_models.EntropyBottleneck`.

Forward signature (matching our codec convention):
    (y_perm: (N, C, *)) -> (y_hat, likelihoods)
"""
from compressai.entropy_models import EntropyBottleneck


def build(channels: int, **kwargs) -> EntropyBottleneck:
    return EntropyBottleneck(channels=int(channels))
