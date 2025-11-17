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

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)
        z.shape = (batch, P, channel)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class NWC_vq(nn.Module):
    def __init__(self, input_size, dim_encoder, n_resblock, M, scale, shift, beta, K, e_dim,
                 norm=True, mode = 'aun'):
        super().__init__()
            
        self.register_buffer('scale', scale)    
        self.register_buffer('shift', shift)
        
        self.M = M
        self.K = K
        self.e_dim = e_dim
        self.n_resblock = n_resblock
        self.bits = (M / e_dim) * math.log2(K) / input_size
        
        
        self.mode = mode
        assert self.mode in ['aun', 'ste', 'sga']
        
        self.input_size = input_size
        self.dim_encoder = dim_encoder
        
        self.g_a = Encoder_without_q_embedding(input_size, n_resblock, dim_encoder, M, norm)
        self.g_s = Encoder_without_q_embedding(M, n_resblock, dim_encoder, input_size, norm)
        
        self.vector_quantization = VectorQuantizer(self.K, self.e_dim, beta)

    def _flatten(self, y):
        # y: [B, L, M]
        y_shape = y.size()
        
        assert y_shape[-1] % self.e_dim == 0, \
            f"M ({y_shape[-1]}) must be a multiple of quantizer code size ({self.e_dim})"
            
        # [B, L, M] -> [B*L*(M/8), 8]
        y = y.reshape(-1, self.e_dim) 
        return y, y_shape
    
    def _unflatten(self, y_hat, y_shape):
        # y_hat: [B*L*(M/8), 8]
        # y_shape: [B, L, M]
        y_hat = y_hat.reshape(y_shape) # [B, L, M]
        return y_hat

    def forward(self, data, scale = None, shift = None, y_in = None):
        
        x = data['weight_block']  # (B, -1, input_size)            
        scale = scale if scale is not None else self.scale # scalar or (B, -1, 1)
        shift = shift if shift is not None else self.shift # scalar or (B, -1, 1)
        x_shift = (x - shift) / scale
                        
        y = self.g_a(x_shift)
        
        y_hat, y_shape = self._flatten(y)
        embedding_loss, y_hat, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(y_hat)
        y_hat = self._unflatten(y_hat, y_shape)
        
        x_hat = self.g_s(y_hat)
        
        x_hat = scale * x_hat + shift
            
        return {
            "embedding_loss": embedding_loss,
            "x_hat": x_hat,
            "perplexity": perplexity,
            "z_q": y_hat,
            "x": x,
            'bits': self.bits
            # "min_encodings": min_encodings,
            # "min_encoding_indices": min_encoding_indices,
        }