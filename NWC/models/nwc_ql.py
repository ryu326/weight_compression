import math
import warnings

from typing import cast
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel


__all__ = [
    "CompressionModel",
    "SimpleVAECompressionModel",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
]


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def ste_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x) - x.detach() + x

class Linear_ResBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.lin_1 = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.LayerNorm(in_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        res = self.lin_1(x)
        out = identity + res

        return out


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([Linear_ResBlock(in_dim)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_dim, n_res_layers, dim_encoder, dim_encoder_out):
        super(Encoder, self).__init__()
        
        
        self.weight_in = nn.Linear(in_dim, dim_encoder)
        self.weight_stack = nn.ModuleList([Linear_ResBlock(dim_encoder)] * n_res_layers)
        self.out = nn.Linear(dim_encoder, dim_encoder_out)

    def forward(self, x, q_embedding):
        x = self.weight_in(x)
        # import ipdb; ipdb.set_trace()
        for i, layer in enumerate(self.weight_stack):
            x = layer(x)
            # print(q_embedding.shape)
            x = x * q_embedding
        return self.out(x)



def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class NWC_ql(CompressionModel):
    """Simple VAE model with arbitrary latent codec.

    .. code-block:: none

               ┌───┐  y  ┌────┐ y_hat ┌───┐
        x ──►──┤g_a├──►──┤ lc ├───►───┤g_s├──►── x_hat
               └───┘     └────┘       └───┘
    """
    
    def __init__(self, input_size, dim_encoder, n_resblock, Q, scale, shift):
        super().__init__()
            
        # self.register_buffer('scale', scale)    
        # self.register_buffer('shift', shift)
        
        ## quality level 개수
        self.Q = Q
        
        self.scale = scale
        self.shift = shift    
        
        self.input_size = input_size
        self.dim_encoder_out = input_size
        self.dim_encoder = dim_encoder
        
        self.quality_embedding = nn.Embedding(Q, dim_encoder)
        
        self.g_a = Encoder(input_size, n_resblock, dim_encoder, self.dim_encoder_out)
        self.g_s = Encoder(self.dim_encoder_out, n_resblock, dim_encoder, input_size)
        
        self.entropy_bottleneck = EntropyBottleneck(self.dim_encoder_out)

    # def __getitem__(self, key: str) -> LatentCodec:
    #     return self.latent_codec[key]

    def forward(self, data):
        x = data['weight_block']
        q_level = data['q_level']
        
        q_embed = self.quality_embedding(q_level)
        # print(q_embed.shape)
        # assert q_embed.dim() == 2
        # q_embed = self.quality_embedding(q_level).unsqueeze(0).unsqueeze(0)
        q_embed = self.quality_embedding(q_level).unsqueeze(1)
        # import ipdb; ipdb.set_trace()
        x_shift = (x - self.shift) / self.scale

        y = self.g_a(x_shift, q_embed)
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        y_hat = y_hat.permute(*perm).contiguous()
        
        # y_offset = self.entropy_bottleneck._get_medians()
        # y_tmp = y - y_offset
        # y_hat = ste_round(y_tmp) + y_offset
        
        x_hat = self.g_s(y_hat, q_embed)
        x_hat = self.scale * x_hat + self.shift
        
        return {
            "x": x,
            "x_hat": x_hat,
            "likelihoods": y_likelihoods,
            "embedding_loss": None,
            "y": y,
            "y_hat": y_hat
        }

    def compress(self, data):
        x = data['weight_block']    
        x_shift = (x - self.shift) / self.scale
        
        q_level = data['q_level']
        
        q_embed = self.quality_embedding(q_level)
        # q_embed = self.quality_embedding(q_level).unsqueeze(1)
        
        y = self.g_a(x_shift, q_embed)      

        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()

        # shape = torch.Size([])
        shape = y.size()[2:]
        y_strings = self.entropy_bottleneck.compress(y)
        
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        # import ipdb; ipdb.set_trace()
        return {"strings": [y_strings], "shape": shape, "y_hat": y_hat, "q_level": q_level}

    # def decompress(self, strings: List[List[bytes]], shape, q_level, **kwargs):
    def decompress(self, strings, shape, q_level, **kwargs):
        
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape, **kwargs)
        
        perm = list(range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()
        
        # q_embed = self.quality_embedding(q_level)
        q_embed = self.quality_embedding(q_level).unsqueeze(1)
        
        # x_hat = self.g_s(y_hat).clamp_(0, 1)
        x_hat = self.g_s(y_hat, q_embed)
        
        x_hat = self.scale * x_hat + self.shift
        
        return {
            "x_hat": x_hat,
        }