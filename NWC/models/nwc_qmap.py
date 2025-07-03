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

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, in_dim, n_res_layers, dim_encoder, dim_encoder_out, norm, dec = False):
        super(Encoder, self).__init__()
        
        self.weight_in = nn.Linear(in_dim, dim_encoder)
        self.weight_stack = nn.ModuleList([Linear_ResBlock(dim_encoder, norm)] * n_res_layers)
        self.out = nn.Linear(dim_encoder, dim_encoder_out)

        self.dec = dec
        if dec:
            self.qm_in_dec = TwoLayerMLP(in_dim, in_dim, in_dim)
        else:
            self.qm_in = TwoLayerMLP(in_dim+1, in_dim, in_dim)
        self.qm_stack = nn.ModuleList([nn.Linear(in_dim, dim_encoder)]* n_res_layers)
       
        
    def forward_enc(self, x, qmap):
        qm = self.qm_in(torch.cat([x, qmap], dim = -1))
        x = self.weight_in(x)
        for i, layer in enumerate(self.weight_stack):
            x = layer(x)
            q_cond = self.qm_stack[i](qm)
            x = x * q_cond
        return self.out(x)

    def forward_dec(self, x):
        qm = self.qm_in_dec(x)
        x = self.weight_in(x)
        for i, layer in enumerate(self.weight_stack):
            x = layer(x)
            q_cond = self.qm_stack[i](qm)
            x = x * q_cond
        return self.out(x)

    def forward(self, x, qmap=None):
        if self.dec:
            return self.forward_dec(x)
        else:
            return self.forward_enc(x, qmap)
    
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

class NWC_qmap(CompressionModel):
    def __init__(self, input_size, dim_encoder, n_resblock, M, scale, shift, norm=True, mode = 'aun', pe = False, pe_n_depth = 42, pe_n_ltype = 7):
        super().__init__()
            
        self.register_buffer('scale', scale)    
        self.register_buffer('shift', shift)
        
        self.M = M
        self.n_resblock = n_resblock
        
        self.mode = mode
        assert self.mode in ['aun', 'ste', 'sga']
        
        self.input_size = input_size
        self.dim_encoder = dim_encoder
                
        self.g_a = Encoder(input_size, n_resblock, dim_encoder, M, norm)
        self.g_s = Encoder(M, n_resblock, dim_encoder, input_size, norm, dec=True)
        
        self.entropy_bottleneck = EntropyBottleneck(M)
        
        self.pe = pe
        if pe:
            self.depth_embedding = nn.Embedding(pe_n_depth, input_size)
            self.ltype_embedding = nn.Embedding(pe_n_ltype, input_size)

    def forward(self, data):
        x = data['weight_block']  # (B, T, input_size)
        qmap = data['qmap'] # (B, T)
        qmap = qmap.reshape(x.shape[0], x.shape[1], 1) # (B, T, 1)
        # qmap = qmap.unsqueeze(1).repeat(1, x.shape[1], 1)  # (B, -1, 1)
        # qmap = qmap.unsqueeze(-1).expand(x.shape[0], x.shape[1], 1)  # (B, -1, 1)

        x_shift = (x - self.shift) / self.scale
        
        if self.pe:
            d_embed = self.depth_embedding(data['depth']) # (B, 1, 16)
            l_embed = self.ltype_embedding(data['ltype']) # (B, 1, 16)
            x_shift = x_shift + d_embed + l_embed
        
        y = self.g_a(x_shift, qmap)
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        # ####### STE quant ########
        if self.mode == 'ste':
            perm_ = np.arange(len(y.shape))
            perm_[0], perm_[1] = perm_[1], perm_[0]
            inv_perm = np.arange(len(y.shape))[np.argsort(perm_)]
            y_hat = y.permute(*perm_).contiguous()
            shape = y_hat.size()
            y_hat = y_hat.reshape(y_hat.size(0), 1, -1)        
            y_offset = self.entropy_bottleneck._get_medians()
            y_tmp = y_hat - y_offset
            y_hat = ste_round(y_tmp) + y_offset        
            y_hat = y_hat.reshape(shape)
            y_hat = y_hat.permute(*inv_perm).contiguous()
        # ####################
        
        y_hat = y_hat.permute(*perm).contiguous()        
        
        x_hat = self.g_s(y_hat)
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
        x = data['weight_block']  # (B, -1. input_size)
        qmap = data['qmap'] # (B, 1)
        # qmap = qmap.unsqueeze(1).repeat(1, x.shape[1], 1)  # (B, -1, 1)
        qmap = qmap.unsqueeze(1).repeat(x.shape[0], x.shape[1], 1)  # (B, -1, 1)

        x_shift = (x - self.shift) / self.scale
        
        if self.pe:
            d_embed = self.depth_embedding(data['depth']) # (B, 1, 16)
            l_embed = self.ltype_embedding(data['ltype']) # (B, 1, 16)
            x_shift = x_shift + d_embed + l_embed
        
        y = self.g_a(x_shift, qmap)

        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()

        shape = y.size()[2:]
        y_strings = self.entropy_bottleneck.compress(y)
        
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        # import ipdb; ipdb.set_trace()
        return {"strings": [y_strings], "shape": shape, "y_hat": y_hat}

    # def decompress(self, strings: List[List[bytes]], shape, q_level, **kwargs):
    # def decompress(self, strings, shape, q_level, **kwargs):
    def decompress(self, enc_data, **kwargs):
        strings = enc_data["strings"]
        shape = enc_data["shape"]
        
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape, **kwargs)
        
        perm = list(range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()
        
        x_hat = self.g_s(y_hat)
        
        x_hat = self.scale * x_hat + self.shift
        
        return {
            "x_hat": x_hat,
        }


class Encoder2(nn.Module):
    def __init__(self, in_dim, n_res_layers, dim_encoder, dim_encoder_out, norm, dec = False):
        super(Encoder2, self).__init__()
        
        self.weight_in = nn.Linear(in_dim, dim_encoder)
        self.weight_stack = nn.ModuleList([Linear_ResBlock(dim_encoder, norm)] * n_res_layers)
        self.out = nn.Linear(dim_encoder, dim_encoder_out)

        self.dec = dec
        if dec:
            self.qm_in_dec = TwoLayerMLP(in_dim, in_dim, in_dim)
        else:
            self.qm_in = TwoLayerMLP(1, in_dim, in_dim)
        self.qm_stack = nn.ModuleList([nn.Linear(in_dim, dim_encoder)]* n_res_layers)
       
        
    def forward_enc(self, x, qmap):
        qm = self.qm_in(qmap)
        x = self.weight_in(x)
        for i, layer in enumerate(self.weight_stack):
            x = layer(x)
            q_cond = self.qm_stack[i](qm)
            x = x * q_cond
        return self.out(x)

    def forward_dec(self, x):
        qm = self.qm_in_dec(x)
        x = self.weight_in(x)
        for i, layer in enumerate(self.weight_stack):
            x = layer(x)
            q_cond = self.qm_stack[i](qm)
            x = x * q_cond
        return self.out(x)

    def forward(self, x, qmap=None):
        if self.dec:
            return self.forward_dec(x)
        else:
            return self.forward_enc(x, qmap)
    
    def reset_parameters(self):
        """
        Encoder 내부 모듈들의 파라미터를 재초기화합니다.
        """
        def reset_fn(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.apply(reset_fn)

class NWC_qmap2(CompressionModel):
    def __init__(self, input_size, dim_encoder, n_resblock, M, scale, shift, norm=True, mode = 'aun', pe = False, pe_n_depth = 42, pe_n_ltype = 7):
        super().__init__()
            
        self.register_buffer('scale', scale)    
        self.register_buffer('shift', shift)
        
        self.M = M
        self.n_resblock = n_resblock
        
        self.mode = mode
        assert self.mode in ['aun', 'ste', 'sga']
        
        self.input_size = input_size
        self.dim_encoder = dim_encoder
                
        self.g_a = Encoder2(input_size, n_resblock, dim_encoder, M, norm)
        self.g_s = Encoder2(M, n_resblock, dim_encoder, input_size, norm, dec=True)
        
        self.entropy_bottleneck = EntropyBottleneck(M)
        
        self.pe = pe
        if pe:
            self.depth_embedding = nn.Embedding(pe_n_depth, input_size)
            self.ltype_embedding = nn.Embedding(pe_n_ltype, input_size)

    def forward(self, data):
        x = data['weight_block']  # (B, T, input_size)
        qmap = data['qmap'] # (B, T)
        # import ipdb; ipdb.set_trace()
        # qmap = qmap.reshape(x.shape[0], x.shape[1], 1) # (B, T, 1)
        # qmap = qmap.unsqueeze(1).repeat(1, x.shape[1], 1)  # (B, -1, 1)
        qmap = qmap.unsqueeze(-1).expand(x.shape[0], x.shape[1], 1)  # (B, -1, 1)

        x_shift = (x - self.shift) / self.scale
        
        if self.pe:
            d_embed = self.depth_embedding(data['depth']) # (B, 1, 16)
            l_embed = self.ltype_embedding(data['ltype']) # (B, 1, 16)
            x_shift = x_shift + d_embed + l_embed
        
        y = self.g_a(x_shift, qmap)
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        # ####### STE quant ########
        if self.mode == 'ste':
            perm_ = np.arange(len(y.shape))
            perm_[0], perm_[1] = perm_[1], perm_[0]
            inv_perm = np.arange(len(y.shape))[np.argsort(perm_)]
            y_hat = y.permute(*perm_).contiguous()
            shape = y_hat.size()
            y_hat = y_hat.reshape(y_hat.size(0), 1, -1)        
            y_offset = self.entropy_bottleneck._get_medians()
            y_tmp = y_hat - y_offset
            y_hat = ste_round(y_tmp) + y_offset        
            y_hat = y_hat.reshape(shape)
            y_hat = y_hat.permute(*inv_perm).contiguous()
        # ####################
        
        y_hat = y_hat.permute(*perm).contiguous()        
        
        x_hat = self.g_s(y_hat)
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
        x = data['weight_block']  # (B, -1. input_size)
        qmap = data['qmap'] # (B, 1)
        # qmap = qmap.unsqueeze(1).repeat(1, x.shape[1], 1)  # (B, -1, 1)
        # qmap = qmap.unsqueeze(1).repeat(x.shape[0], x.shape[1], 1)  # (B, -1, 1)
        qmap = qmap.unsqueeze(-1).expand(x.shape[0], x.shape[1], 1)  # (B, -1, 1)

        x_shift = (x - self.shift) / self.scale
        
        if self.pe:
            d_embed = self.depth_embedding(data['depth']) # (B, 1, 16)
            l_embed = self.ltype_embedding(data['ltype']) # (B, 1, 16)
            x_shift = x_shift + d_embed + l_embed
        
        y = self.g_a(x_shift, qmap)

        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()

        shape = y.size()[2:]
        y_strings = self.entropy_bottleneck.compress(y)
        
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        # import ipdb; ipdb.set_trace()
        return {"strings": [y_strings], "shape": shape, "y_hat": y_hat}

    # def decompress(self, strings: List[List[bytes]], shape, q_level, **kwargs):
    # def decompress(self, strings, shape, q_level, **kwargs):
    def decompress(self, enc_data, **kwargs):
        strings = enc_data["strings"]
        shape = enc_data["shape"]
        
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape, **kwargs)
        
        perm = list(range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()
        
        x_hat = self.g_s(y_hat)
        
        x_hat = self.scale * x_hat + self.shift
        
        return {
            "x_hat": x_hat,
        }
    

class Encoder3(nn.Module):
    def __init__(self, in_dim, n_res_layers, dim_encoder, dim_encoder_out, norm):
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

class NWC_qmap3(CompressionModel):
    def __init__(self, input_size, dim_encoder, n_resblock, M, scale, shift, norm=True, mode = 'aun', pe = False, pe_n_depth = 42, pe_n_ltype = 7):
        super().__init__()
            
        self.register_buffer('scale', scale)    
        self.register_buffer('shift', shift)
        
        self.M = M
        self.n_resblock = n_resblock
        
        self.mode = mode
        assert self.mode in ['aun', 'ste', 'sga']
        
        self.input_size = input_size
        self.dim_encoder = dim_encoder
        
        self.g_a = Encoder3(input_size, n_resblock, dim_encoder, M, norm)
        self.g_s = Encoder3(M, n_resblock, dim_encoder, input_size, norm)

        self.entropy_bottleneck = EntropyBottleneck(M)

        self.qmap_in = TwoLayerMLP(1, dim_encoder, dim_encoder)
        
        self.pe = pe
        if pe:
            self.depth_embedding = nn.Embedding(pe_n_depth, input_size)
            self.ltype_embedding = nn.Embedding(pe_n_ltype, input_size)

    def forward(self, data):
        x = data['weight_block']  # (B, T, input_size)
        qmap = data['qmap'] # (B, T)
        # import ipdb; ipdb.set_trace()
        # qmap = qmap.reshape(x.shape[0], x.shape[1], 1) # (B, T, 1)
        # qmap = qmap.unsqueeze(1).repeat(1, x.shape[1], 1)  # (B, -1, 1)
        qmap = qmap.unsqueeze(-1).expand(x.shape[0], x.shape[1], 1)  # (B, T, 1)

        x_shift = (x - self.shift) / self.scale
        
        if self.pe:
            d_embed = self.depth_embedding(data['depth']) # (B, 1, 16)
            l_embed = self.ltype_embedding(data['ltype']) # (B, 1, 16)
            x_shift = x_shift + d_embed + l_embed
        
        qmap_emb = self.qmap_in(qmap)
        y = self.g_a(x_shift, qmap_emb)
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        # ####### STE quant ########
        if self.mode == 'ste':
            perm_ = np.arange(len(y.shape))
            perm_[0], perm_[1] = perm_[1], perm_[0]
            inv_perm = np.arange(len(y.shape))[np.argsort(perm_)]
            y_hat = y.permute(*perm_).contiguous()
            shape = y_hat.size()
            y_hat = y_hat.reshape(y_hat.size(0), 1, -1)        
            y_offset = self.entropy_bottleneck._get_medians()
            y_tmp = y_hat - y_offset
            y_hat = ste_round(y_tmp) + y_offset        
            y_hat = y_hat.reshape(shape)
            y_hat = y_hat.permute(*inv_perm).contiguous()
        # ####################
        
        y_hat = y_hat.permute(*perm).contiguous()        
        
        x_hat = self.g_s(y_hat, qmap_emb)
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
        x = data['weight_block']  # (B, -1. input_size)
        qmap = data['qmap'] # (B, 1)
        # qmap = qmap.unsqueeze(1).repeat(1, x.shape[1], 1)  # (B, -1, 1)
        # qmap = qmap.unsqueeze(1).repeat(x.shape[0], x.shape[1], 1)  # (B, -1, 1)
        qmap = qmap.unsqueeze(-1).expand(x.shape[0], x.shape[1], 1)  # (B, -1, 1)

        x_shift = (x - self.shift) / self.scale
        
        if self.pe:
            d_embed = self.depth_embedding(data['depth']) # (B, 1, 16)
            l_embed = self.ltype_embedding(data['ltype']) # (B, 1, 16)
            x_shift = x_shift + d_embed + l_embed
        
        qmap_emb = self.qmap_in(qmap)
        y = self.g_a(x_shift, qmap_emb)

        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()

        shape = y.size()[2:]
        y_strings = self.entropy_bottleneck.compress(y)
        
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        # import ipdb; ipdb.set_trace()
        return {"strings": [y_strings], "shape": shape, "y_hat": y_hat, "qmap": qmap}

    # def decompress(self, strings: List[List[bytes]], shape, q_level, **kwargs):
    # def decompress(self, strings, shape, q_level, **kwargs):
    def decompress(self, enc_data, **kwargs):
        strings = enc_data["strings"]
        shape = enc_data["shape"]
        qmap = enc_data["qmap"]

        y_hat = self.entropy_bottleneck.decompress(strings[0], shape, **kwargs)
        
        perm = list(range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()
        
        qmap_emb = self.qmap_in(qmap)
        x_hat = self.g_s(y_hat, qmap_emb)
        
        x_hat = self.scale * x_hat + self.shift
        
        return {
            "x_hat": x_hat,
        }