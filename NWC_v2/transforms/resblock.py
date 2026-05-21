"""Residual MLP transform — clone of NWC's `Encoder_without_q_embedding`.

Linear(in_dim → dim_encoder) → [Linear_ResBlock(dim_encoder)] × n_resblock
                              → Linear(dim_encoder → out_dim)
"""
import torch.nn as nn


class Linear_ResBlock(nn.Module):
    """Single residual block: Linear → (LayerNorm) → ReLU → +identity."""

    def __init__(self, in_ch: int, norm: bool = True):
        super().__init__()
        if norm:
            self.lin_1 = nn.Sequential(
                nn.Linear(in_ch, in_ch),
                nn.LayerNorm(in_ch),
                nn.ReLU(),
            )
        else:
            self.lin_1 = nn.Sequential(
                nn.Linear(in_ch, in_ch),
                nn.ReLU(),
            )

    def forward(self, x):
        return x + self.lin_1(x)


class ResBlockTransform(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_resblock: int = 4,
        dim_encoder: int = 32,
        norm: bool = True,
    ):
        super().__init__()
        self.weight_in = nn.Linear(in_dim, dim_encoder)
        self.weight_stack = nn.ModuleList(
            [Linear_ResBlock(dim_encoder, norm) for _ in range(n_resblock)]
        )
        self.out = nn.Linear(dim_encoder, out_dim)

    def forward(self, x):
        x = self.weight_in(x)
        for layer in self.weight_stack:
            x = layer(x)
        return self.out(x)


def build(
    in_dim: int,
    out_dim: int,
    n_resblock: int = 4,
    dim_encoder: int = 32,
    norm: bool = True,
    **kwargs,
) -> nn.Module:
    return ResBlockTransform(
        in_dim=in_dim,
        out_dim=out_dim,
        n_resblock=n_resblock,
        dim_encoder=dim_encoder,
        norm=norm,
    )
