import math
import warnings

from typing import cast
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
import time
from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional, GaussianMixtureConditional
from compressai.models import CompressionModel

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
nwc_root = os.path.abspath(os.path.join(current_dir, '..'))  # /NWC
sys.path.append(nwc_root)

from lattice_transform_coding.LTC.entropy_models import EntropyBottleneckLattice
from lattice_transform_coding.LTC.quantizers import get_lattice

# from sga import EntropyBottleneckNoQuant

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

# class Linear_ResBlock(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()

#         self.lin_1 = nn.Sequential(
#             nn.Linear(in_ch, in_ch),
#             nn.LayerNorm(in_ch),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         identity = x
#         res = self.lin_1(x)
#         out = identity + res

#         return out
    
#     def reset_parameters(self):
#         for layer in self.lin_1:
#             if hasattr(layer, 'reset_parameters'):
#                 layer.reset_parameters()

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

# === 추가: 공통 유틸 / 믹스인 ===
class AsinhCompandingMixin:
    def _init_companding(self, input_size, compand=True, learnable_s=True, per_feature_s=True,
                        s_init=1.0, s_min=0.1, s_max=10.0, comp_clip=8.0, soft_clip=True):
        self.compand = compand
        self._comp_s_min = float(s_min)
        self._comp_s_max = float(s_max)
        self._comp_clip  = float(comp_clip)
        self._soft_clip  = bool(soft_clip)
        shape = (1,1,input_size) if (compand and per_feature_s) else (1,1,1)
        self.register_buffer("_comp_s_buf", torch.ones(shape))
        if not compand: return
        log_s_init = math.log(max(float(s_init), 1e-6))
        if learnable_s:
            self.log_s = nn.Parameter(torch.full(shape, log_s_init))
        else:
            self.register_buffer("log_s", torch.full(shape, log_s_init))

    @property
    def comp_s(self):
        if hasattr(self, "log_s"):
            s = self.log_s.exp()
        else:
            s = self._comp_s_buf
        return torch.clamp(s, min=self._comp_s_min, max=self._comp_s_max)
    
    def _compand(self, x_shift):
        if not self.compand:
            return x_shift
        # y = asinh(x/s)
        return torch.asinh(x_shift / self.comp_s)

    def _inv_compand(self, y):
        if not self.compand:
            return y
        # y 소프트 클립: |y|→큰 값에서 점진 포화(기울기도 안정)
        limit = self._comp_clip
        y_stable = limit * torch.tanh(y / limit)  # 권장
        return self.comp_s * torch.sinh(y_stable)


class Encoder(nn.Module):
    def __init__(self, in_dim, n_res_layers, dim_encoder, dim_encoder_out, norm):
        super(Encoder, self).__init__()
        
        self.weight_in = nn.Linear(in_dim, dim_encoder)
        # self.weight_stack = nn.ModuleList([Linear_ResBlock(dim_encoder, norm)] * n_res_layers)
        self.weight_stack = nn.ModuleList([Linear_ResBlock(dim_encoder, norm) for _ in range(n_res_layers)]) ## debug
        self.out = nn.Linear(dim_encoder, dim_encoder_out)

    def forward(self, x, q_embedding):
        x = self.weight_in(x)
        for i, layer in enumerate(self.weight_stack):
            x = layer(x)
            # print(q_embedding.shape)
            x = x * q_embedding
        return self.out(x)

    def reset_parameters(self):
        def reset_fn(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.apply(reset_fn)

class Encoder_without_q_embedding(nn.Module):
    def __init__(self, in_dim, n_res_layers, dim_encoder, dim_encoder_out, norm):
        super(Encoder_without_q_embedding, self).__init__()
        
        self.weight_in = nn.Linear(in_dim, dim_encoder)
        # self.weight_stack = nn.ModuleList([Linear_ResBlock(dim_encoder, norm)] * n_res_layers)
        self.weight_stack = nn.ModuleList([Linear_ResBlock(dim_encoder, norm) for _ in range(n_res_layers)])
        self.out = nn.Linear(dim_encoder, dim_encoder_out)

    def forward(self, x):
        x = self.weight_in(x)
        for i, layer in enumerate(self.weight_stack):
            x = layer(x)
        return self.out(x)

class IdxEmbedding(nn.Module):
    def __init__(self, num_layers=42, hidden_dim=16, out_dim = 16):
        super().__init__()
        
        self.out_dim = out_dim
        self.embedding = nn.Embedding(num_layers, hidden_dim)  # layer_idx 임베딩
        self.ltype_embedding = nn.Embedding(7, hidden_dim)    # wtype 임베딩
        
    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        layer_idx_emb = self.embedding(x[:, 0])
        ltype_emb = self.ltype_embedding(x[:, 1])
        wtype_emb = self.wtype_embedding(x[:, 2])
        
        combined = torch.cat([layer_idx_emb, ltype_emb, wtype_emb], dim=-1)
        hidden = self.fc(combined)
        output = self.output_layer(hidden)
        return output

class ContinuousEmbedding(nn.Module):
    def __init__(self, in_dim=1, dim=256, hidden=128, n_freq=16, rand_std=1.0):
        super().__init__()
        if n_freq > 0:
            B = torch.randn(1, n_freq) * rand_std  # 랜덤 Fourier 주파수
            self.register_buffer("B", B)
            feat_dim = 1 + 2 * n_freq
            self.use_ff = True
        else:
            feat_dim = 1
            self.use_ff = False
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):  # x: (..., in_dim), float
        x = x.unsqueeze(-1)
        if self.use_ff:
            proj = 2 * math.pi * (x @ self.B)                  # (..., n_freq)
            enc = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
        else:
            enc = x
        y = self.mlp(enc)                                      # (..., dim)
        return y

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class NWC_id(CompressionModel, AsinhCompandingMixin):
    """Simple VAE model with arbitrary latent codec.

    .. code-block:: none

               ┌───┐  y  ┌────┐ y_hat ┌───┐
        x ──►──┤g_a├──►──┤ lc ├───►───┤g_s├──►── x_hat
               └───┘     └────┘       └───┘
    """
    
    def __init__(self, input_size, dim_encoder, n_resblock, Q, M, scale, shift, 
                 norm=True, mode = 'aun', pe = False, pe_n_depth = 42, pe_n_ltype = 7, use_hyper = False,
                 use_companding=False, learnable_s=True, per_feature_s=True, comp_s_init=1.0, scale_cond = False, continuous = False):
        super().__init__()
            
        self.register_buffer('scale', scale)    
        self.register_buffer('shift', shift)
        
        ## quality level 개수
        self.Q = Q
        self.M = M
        self.n_resblock = n_resblock
        
        self.mode = mode
        assert self.mode in ['aun', 'ste', 'sga']
        
        self.input_size = input_size
        self.dim_encoder = dim_encoder
        
        self.scale_cond = scale_cond
        self.continuous = continuous
        if not scale_cond and not continuous:
            self.quality_embedding = nn.Embedding(Q, dim_encoder)
        else:
            self.quality_embedding = ContinuousEmbedding(dim = dim_encoder)
            
        # self.g_a = Encoder(input_size, n_resblock, dim_encoder, M, norm)
        # self.g_s = Encoder(M, n_resblock, dim_encoder, input_size, norm)

        self.entropy_bottleneck = EntropyBottleneck(M)
        # self.pe = pe
        # self.use_hyper = use_hyper

        # if pe:
        #     self.depth_embedding = nn.Embedding(pe_n_depth, input_size)
        #     self.ltype_embedding = nn.Embedding(pe_n_ltype, input_size)

        # if use_hyper == True:
        #     self.h_a = Encoder_without_q_embedding(M, n_resblock//2, dim_encoder//4, M//2, norm)
        #     self.h_s_means = Encoder_without_q_embedding(M//2, n_resblock//2, dim_encoder//4, M, norm)
        #     self.h_s_scales = Encoder_without_q_embedding(M//2, n_resblock//2, dim_encoder//4, M, norm)
            
        #     # self.gaussian_conditional = GaussianMixtureConditional(None)
        #     self.gaussian_conditional = GaussianConditional(None)
        #     self.entropy_bottleneck = EntropyBottleneck(M//2)

        self._init_companding(input_size, compand=use_companding, 
                              learnable_s=learnable_s, per_feature_s=per_feature_s, 
                              s_init=comp_s_init)


    def forward(self, data, scale = None, shift = None, y_in = None):
        if y_in is not None:
            y = y_in
        else:
            x = data['weight_block']  # (B, -1, input_size)            
            if not self.scale_cond:
                # q_level = data['q_level'] # (B, -1)
                # q_embed = self.quality_embedding(q_level) # (B, -1, encdim)
                scale = scale if scale is not None else self.scale # scalar or (B, -1, 1)
                shift = shift if shift is not None else self.shift # scalar or (B, -1, 1)
                x_shift = (x - shift) / scale
            else : 
                # scale_cond = data['scale_cond'] # (B, -1), dataset이면 (B, 1)
                # q_embed = self.quality_embedding(scale_cond) # (B, -1, encdim)
                # x_shift = x / scale_cond.unsqueeze(-1)
                assert True
            
            # if self.pe:
            #     d_embed = self.depth_embedding(data['depth']) # (B, 1, 16)
            #     l_embed = self.ltype_embedding(data['ltype']) # (B, 1, 16)
            #     x_shift = x_shift + d_embed + l_embed
            
            # x_shift = self._compand(x_shift)
                        
            # y = self.g_a(x_shift, q_embed)
            y = x_shift

            perm = list(range(y.dim()))
            perm[-1], perm[1] = perm[1], perm[-1]
            y_hat = y.permute(*perm).contiguous()
            
            y_hat, y_likelihoods = self.entropy_bottleneck(y_hat)
            
            # ####### STE quant ########
            if self.mode == 'ste':
                perm_ = np.arange(len(y_hat.shape))
                perm_[0], perm_[1] = perm_[1], perm_[0]
                inv_perm = np.arange(len(y_hat.shape))[np.argsort(perm_)]
                y_hat = y_hat.permute(*perm_).contiguous()
                shape = y_hat.size()
                y_hat = y_hat.reshape(y_hat.size(0), 1, -1)        
                y_offset = self.entropy_bottleneck._get_medians()
                y_tmp = y_hat - y_offset
                y_hat = ste_round(y_tmp) + y_offset        
                y_hat = y_hat.reshape(shape)
                y_hat = y_hat.permute(*inv_perm).contiguous()
            # ####################
            
            y_hat = y_hat.permute(*perm).contiguous()        
            
            # x_hat = self.g_s(y_hat, q_embed)
            x_hat = y_hat            
            x_hat = self._inv_compand(x_hat)
            
            if not self.scale_cond:
                x_hat = scale * x_hat + shift
            else:
                # x_hat = x_hat * scale_cond.unsqueeze(-1)        
                assert True
                
            return {
                # "x": x,
                "x_hat": x_hat,
                "likelihoods": y_likelihoods,
                # "embedding_loss": None,
                # "y": y,
                # "y_hat": y_hat
            }

    def compress(self, data, scale= None, shift= None):
        x = data['weight_block']  # (B, -1, input_size)       
        scale = scale if scale is not None else self.scale # scalar or (B, -1, 1)
        shift = shift if shift is not None else self.shift # scalar or (B, -1, 1)
        x_shift = (x - shift) / scale

        # y = self.g_a(x_shift, q_embed)
        y = x_shift

        enc_data = self.compress_without_hyperprior(y)
            
        return enc_data
    
    def compress_without_hyperprior(self, y):
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()
        
        shape = y.size()[2:]
        y_strings = self.entropy_bottleneck.compress(y)
        
        return {"strings": [y_strings], "shape": shape}

    def decompress(self, enc_data, scale= None, shift= None):
        strings = enc_data["strings"]
        shape = enc_data["shape"]
        scale = scale if scale is not None else self.scale # scalar or (B, -1, 1)
        shift = shift if shift is not None else self.shift # scalar or (B, -1, 1)
            
        x_hat = self.decompress_without_hyperprior(strings, shape, None)

        x_hat = self._inv_compand(x_hat)

        x_hat = scale * x_hat + shift
 
        return {"x_hat": x_hat}
    
    def decompress_without_hyperprior(self, strings, shape, q_embed):
        
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        
        perm = list[int](range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()
        
        # x_hat = self.g_s(y_hat, q_embed)
        x_hat = y_hat
        
        return x_hat
        