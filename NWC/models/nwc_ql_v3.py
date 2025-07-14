import math
import warnings

from typing import cast
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional, GaussianMixtureConditional
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

class Encoder(nn.Module):
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

    def reset_parameters(self):
        def reset_fn(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.apply(reset_fn)

class Encoder_without_q_embedding(nn.Module):
    def __init__(self, in_dim, n_res_layers, dim_encoder, dim_encoder_out, norm):
        super(Encoder_without_q_embedding, self).__init__()
        
        self.weight_in = nn.Linear(in_dim, dim_encoder)
        self.weight_stack = nn.ModuleList([Linear_ResBlock(dim_encoder, norm)] * n_res_layers)
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


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class FiLMLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.generator = nn.Linear(d_model, d_model * 2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, a_feat, b_feat):
        # (B, N, 2*D) -> (B, N, D), (B, N, D)
        gamma, beta = self.generator(b_feat).chunk(2, dim=-1)
        
        # a에 FiLM 적용: a_new = a * scale + shift
        modulated_a = a_feat * gamma + beta
        
        return self.norm(a_feat + modulated_a)

class NWC_ql_v3(CompressionModel):
    """Simple VAE model with arbitrary latent codec.

    .. code-block:: none

               ┌───┐  y  ┌────┐ y_hat ┌───┐
        x ──►──┤g_a├──►──┤ lc ├───►───┤g_s├──►── x_hat
               └───┘     └────┘       └───┘
    """
    
    def __init__(self, input_size, dim_encoder, n_resblock, Q, M, scale, shift, dim_proj = None, n_clayers = 2,
                 norm=True, mode = 'aun', pe = False, pe_n_depth = 42, pe_n_ltype = 7, use_hyper = False):
        super().__init__()
            
        self.register_buffer('scale', scale)    
        self.register_buffer('shift', shift)
        
        self.Q = Q
        self.M = M
        self.n_resblock = n_resblock
        
        self.mode = mode
        assert self.mode in ['aun', 'ste', 'sga']
        
        self.input_size = input_size
        self.dim_encoder = dim_encoder
                
        dim_proj = dim_encoder//input_size if dim_proj is None else dim_proj
        self.dim_proj = dim_proj
        self.proj_x = nn.Linear(1, dim_proj)
        self.proj_s = nn.Linear(1, dim_proj)
        self.cond_layers = nn.ModuleList([FiLMLayer(dim_proj) for _ in range(n_clayers)])
        
        # dim_proj = dim_encoder//M if dim_proj is None else dim_proj
        self.proj_y = nn.Linear(dim_proj, 1)
        self.proj_s_dec = nn.Linear(1, dim_proj)
        self.cond_layers_dec = nn.ModuleList([FiLMLayer(dim_proj) for _ in range(n_clayers)])
        
        self.g_a = Encoder_without_q_embedding(dim_encoder, n_resblock, dim_encoder, M, norm)
        self.g_s = Encoder_without_q_embedding(M, n_resblock, dim_encoder, dim_encoder, norm)
        
        self.quality_embedding = nn.Embedding(Q, dim_encoder)
        
        self.entropy_bottleneck = EntropyBottleneck(M)
        self.pe = pe
        self.use_hyper = use_hyper

        if pe:
            self.depth_embedding = nn.Embedding(pe_n_depth, input_size)
            self.ltype_embedding = nn.Embedding(pe_n_ltype, input_size)

        if use_hyper == True:
            self.h_a = Encoder_without_q_embedding(M, n_resblock//2, dim_encoder//4, M//2, norm)
            self.h_s_means = Encoder_without_q_embedding(M//2, n_resblock//2, dim_encoder//4, M, norm)
            self.h_s_scales = Encoder_without_q_embedding(M//2, n_resblock//2, dim_encoder//4, M, norm)
            
            # self.gaussian_conditional = GaussianMixtureConditional(None)
            self.gaussian_conditional = GaussianConditional(None)
            self.entropy_bottleneck = EntropyBottleneck(M//2)

    def forward(self, data, scale = None, shift = None):
        x = data['weight_block']  # (B, -1, input_size)
        scale_cond = data['scale_cond'] # (B, -1, input_size)
        
        x_shift = x / scale_cond
        
        if self.pe:
            d_embed = self.depth_embedding(data['depth']) # (B, 1, 16)
            l_embed = self.ltype_embedding(data['ltype']) # (B, 1, 16)
            x_shift = x_shift + d_embed + l_embed
         
        x_shift = self.proj_x(x_shift.unsqueeze(-1)) # (B, -1, input_size, 1) -> (B, -1, input_size, dim_proj)
        s_enc = self.proj_s(scale_cond.unsqueeze(-1)) # (B, -1, input_size, 1) -> (B, -1, input_size, dim_proj)
        
        for layer in self.cond_layers:
            x_shift = layer(x_shift, s_enc)
        
        x_shift = torch.flatten(x_shift, start_dim=-2)  # (B, -1, dim_encoder)
        y = self.g_a(x_shift)
        
        if self.use_hyper == True:
            z = self.h_a(y)
            
            perm = list(range(z.dim()))
            perm[-1], perm[1] = perm[1], perm[-1]
            z = z.permute(*perm).contiguous()
            
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            z_hat = z_hat.permute(*perm).contiguous()
            
            means_hat = self.h_s_means(z_hat)
            scales_hat = self.h_s_scales(z_hat)
            
            perm = list(range(y.dim()))
            perm[-1], perm[1] = perm[1], perm[-1]
            y = y.permute(*perm).contiguous()
            
            scales_hat = scales_hat.permute(*perm).contiguous()
            means_hat = means_hat.permute(*perm).contiguous()
            
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
            y_hat = y_hat.permute(*perm).contiguous()
            
            y_hat = self.g_s(y_hat)
            
            y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[1], self.input_size, self.dim_proj)  # (B, -1, input_size, dim_proj)
            s_dec = self.proj_s_dec(scale_cond.unsqueeze(-1)) # (B, -1, input_size, 1) -> (B, -1, input_size, dim_proj)
            
            for layer in self.cond_layers_dec:
                y_hat = layer(y_hat, s_dec)
            
            x_hat = self.proj_y(y_hat).squeeze(-1)  # (B, -1, input_size, dim_proj) --> (B, -1, input_size, 1) --> (B, -1, input_size)
            
            # x_hat = scale * x_hat + shift
            x_hat = x_hat * scale_cond
            
            return {
                "x": x,
                "x_hat": x_hat,
                "likelihoods": {'y': y_likelihoods, 'z': z_likelihoods},
                "embedding_loss": None,
                "y": y,
                "y_hat": y_hat
            }

        else:
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
            y_hat = self.g_s(y_hat)
            
            y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[1], self.input_size, self.dim_proj)  # (B, -1, input_size, dim_proj)
            s_dec = self.proj_s_dec(scale_cond.unsqueeze(-1)) # (B, -1, input_size, 1) -> (B, -1, input_size, dim_proj)
            
            for layer in self.cond_layers_dec:
                y_hat = layer(y_hat, s_dec)
            
            x_hat = self.proj_y(y_hat).squeeze(-1)  # (B, -1, input_size, dim_proj) --> (B, -1, input_size, 1) --> (B, -1, input_size)
            
            # x_hat = scale * x_hat + shift
            x_hat = x_hat * scale_cond
            
            return {
                "x": x,
                "x_hat": x_hat,
                "likelihoods": y_likelihoods,
                "embedding_loss": None,
                "y": y,
                "y_hat": y_hat
            }

    def compress(self, data, scale= None, shift= None):
        x = data['weight_block']  # (B, -1, input_size)
        scale_cond = data['scale_cond'] # (B, -1, input_size)
        
        x_shift = x / scale_cond
        
        if self.pe:
            d_embed = self.depth_embedding(data['depth']) # (B, 1, 16)
            l_embed = self.ltype_embedding(data['ltype']) # (B, 1, 16)
            x_shift = x_shift + d_embed + l_embed
         
        x_shift = self.proj_x(x_shift.unsqueeze(-1)) # (B, -1, input_size, 1) -> (B, -1, input_size, dim_proj)
        s_enc = self.proj_s(scale_cond.unsqueeze(-1)) # (B, -1, input_size, 1) -> (B, -1, input_size, dim_proj)
        
        for layer in self.cond_layers:
            x_shift = layer(x_shift, s_enc)
        
        x_shift = torch.flatten(x_shift, start_dim=-2)  # (B, -1, dim_encoder)
        y = self.g_a(x_shift)

        if self.use_hyper:
            enc_data = self.compress_with_hyperprior(y)
        else:
            enc_data = self.compress_without_hyperprior(y)
        enc_data["scale_cond"] = scale_cond
        return enc_data        

    def compress_with_hyperprior(self, y):

        z = self.h_a(y)
        
        perm = list(range(z.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        z = z.permute(*perm).contiguous()
        
        shape = z.size()[2:]
        z_strings = self.entropy_bottleneck.compress(z)
        
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        z_hat = z_hat.permute(*perm).contiguous()
        
        means_hat = self.h_s_means(z_hat)
        scales_hat = self.h_s_scales(z_hat)
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()
        
        scales_hat = scales_hat.permute(*perm).contiguous()
        means_hat = means_hat.permute(*perm).contiguous()
        
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        
        return {"strings": [y_strings, z_strings], "shape": shape}
    
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
        scale_cond = enc_data["scale_cond"]  # (B, -1, input_size)

        if self.use_hyper == True:
            y_hat = self.decompress_with_hyperprior(strings, shape, scale_cond)
        else:
            y_hat = self.decompress_without_hyperprior(strings, shape, scale_cond)
        
        y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[1], self.input_size, self.dim_proj)  # (B, -1, input_size, dim_proj)
        s_dec = self.proj_s_dec(scale_cond.unsqueeze(-1)) # (B, -1, input_size, 1) -> (B, -1, input_size, dim_proj)
        
        for layer in self.cond_layers_dec:
            y_hat = layer(y_hat, s_dec)
        
        x_hat = self.proj_y(y_hat).squeeze(-1)  # (B, -1, input_size, dim_proj) --> (B, -1, input_size, 1) --> (B, -1, input_size)
        
        # x_hat = scale * x_hat + shift
        x_hat = x_hat * scale_cond
        
        
        x_hat = x_hat * scale_cond
        return {"x_hat": x_hat}

    def decompress_with_hyperprior(self, strings, shape, scale_cond):
        
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        
        perm = list(range(z_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        z_hat = z_hat.permute(*perm).contiguous()
        
        means_hat = self.h_s_means(z_hat)
        scales_hat = self.h_s_scales(z_hat)
        
        scales_hat = scales_hat.permute(*perm).contiguous()
        means_hat = means_hat.permute(*perm).contiguous()
        
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        
        perm = list(range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()
        
        y_hat = self.g_s(y_hat)
                
        return y_hat        
    
    def decompress_without_hyperprior(self, strings, shape, scale_cond):
        
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        
        perm = list(range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()        
        y_hat = self.g_s(y_hat)
        
        return y_hat