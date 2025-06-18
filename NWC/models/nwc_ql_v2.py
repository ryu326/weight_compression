import math
import warnings

from typing import cast
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import numpy as np
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
    def __init__(self, in_ch, norm = True):
        super().__init__()

        if norm == True:
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
        identity = x
        res = self.lin_1(x)
        out = identity + res

        return out

# class ResidualStack(nn.Module):
#     """
#     A stack of residual layers inputs:
#     - in_dim : the input dimension
#     - h_dim : the hidden layer dimension
#     - res_h_dim : the hidden dimension of the residual block
#     - n_res_layers : number of layers to stack
#     """

#     def __init__(self, in_dim, n_res_layers):
#         super(ResidualStack, self).__init__()
#         self.n_res_layers = n_res_layers
#         self.stack = nn.ModuleList([Linear_ResBlock(in_dim)] * n_res_layers)

#     def forward(self, x):
#         for layer in self.stack:
#             x = layer(x)
#         return x



class Encoder(nn.Module):
    def __init__(self, in_dim, n_res_layers, dim_encoder, dim_encoder_out, norm = True):
        super(Encoder, self).__init__()
        
        self.weight_in = nn.Linear(in_dim, dim_encoder)
        self.weight_stack = nn.ModuleList([Linear_ResBlock(dim_encoder, norm)] * n_res_layers)
        self.out = nn.Linear(dim_encoder, dim_encoder_out)

    def forward(self, x, q_embedding):
        x = self.weight_in(x)
        for i, layer in enumerate(self.weight_stack):
            x = layer(x)
            # print(q_embedding.shape)
            x = x * q_embedding
        return self.out(x)

    def reset_parameters(self):
        """
        Encoder 내부 모듈들의 파라미터를 재초기화합니다.
        """
        def reset_fn(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.apply(reset_fn)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class NWC_ql_learnable_scale(CompressionModel):    
    def __init__(self, input_size, dim_encoder, n_resblock, Q, M, scale, shift, norm = False):
        super().__init__()
            
        self.register_buffer('scale', scale)    
        self.register_buffer('shift', shift)
        
        self.Q = Q  ## quality level 개수
        self.M = M
        self.n_resblock = n_resblock
        
        self.input_size = input_size
        self.dim_encoder = dim_encoder
        
        self.quality_embedding = nn.Embedding(Q, dim_encoder)
        
        self.g_a = Encoder(input_size + 1, n_resblock, dim_encoder, M, norm = norm)
        self.g_s = Encoder(M, n_resblock, dim_encoder, input_size + 1, norm = norm)
        
        self.entropy_bottleneck = EntropyBottleneck(M)

    def forward(self, data):
        B = data['weight_block'].size(0)
        x = data['weight_block'].view(B, -1) ## (B, dim)
        sig = x.std(dim=-1)   ## (B,)
        q_level = data['q_level'] ## (B,)
        
        x_norm = x.view(B, -1, self.input_size) ## (B, -1, input_size)
        sig = sig.view(B, 1, 1) ## (B, 1, 1)
        q_level = q_level.view(B, 1) ## (B, 1)

        q_embed = self.quality_embedding(q_level) ## (B, 1, encdim)
        assert q_embed.dim() == 3
        x_norm = x_norm / sig
        sig_exp = sig.expand(-1, x_norm.size(1), -1)  # (B, T, 1)
        x_norm = torch.cat([x_norm, sig_exp], dim=-1) ## (B, -1, input_size + 1)

        y = self.g_a(x_norm, q_embed)
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        # y_offset = self.entropy_bottleneck._get_medians()
        # print(y.shape, y_offset.shape)        
        # y_tmp = y - y_offset
        # y_hat = ste_round(y_tmp) + y_offset
        
        y_hat = y_hat.permute(*perm).contiguous()
        
        x_hat = self.g_s(y_hat, q_embed)
        scale = x_hat[:, :, -1:]
        x_hat = x_hat[:,:,:-1]

        x_hat = (x_hat*scale).contiguous().view(B, -1) ## (B, dim)
        
        return {
            "x": x,
            "x_hat": x_hat,
            "likelihoods": y_likelihoods,
            "embedding_loss": None,
            "y": y,
            "y_hat": y_hat,
            "sig": sig,
        }

    def compress(self, data):
        x = data['weight_block'] ## (B, dim)
        sig = x.std(dim=-1)   ## (B,)
        q_level = data['q_level'] ## (B,)
        B = x.size(0)
        
        x_norm = x.view(B, -1, self.input_size) ## (B, -1, input_size)
        sig = sig.view(B, 1, 1) ## (B, 1, 1)
        q_level = q_level.view(B, 1) ## (B, 1)

        q_embed = self.quality_embedding(q_level) ## (B, 1, encdim)
        assert q_embed.dim() == 3
        x_norm = x_norm / sig
        sig_exp = sig.expand(-1, x_norm.size(1), -1)  # (B, T, 1)
        x_norm = torch.cat([x_norm, sig_exp], dim=-1) ## (B, -1, input_size + 1)
        
        y = self.g_a(x_norm, q_embed)      

        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()

        # shape = torch.Size([])
        shape = y.size()[2:]
        y_strings = self.entropy_bottleneck.compress(y)
        
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        return {"strings": [y_strings], "shape": shape, "y_hat": y_hat, "q_level": q_level}

    def decompress(self, enc_data, **kwargs):
        strings = enc_data["strings"]
        shape = enc_data["shape"]
        q_level = enc_data["q_level"]
        
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape, **kwargs)
        
        perm = list(range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()
        
        q_level = q_level.view(q_level.numel(), 1)
        q_embed = self.quality_embedding(q_level)

        x_hat = self.g_s(y_hat, q_embed)
        scale = x_hat[:, :, -1:]
        x_hat = x_hat[:,:,:-1]

        x_hat = (x_hat*scale).contiguous().view(x_hat.size(0), -1) ## (B, dim)
        
        return {
            "x_hat": x_hat,
        }