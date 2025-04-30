import math
import warnings

from typing import cast
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor

import torch
import torch.nn as nn
import numpy as np
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

    def forward(self, x):
        ## x : (B, Channel, Length)
        ## out : (B, Channel, Length)
        
        perm = list(range(x.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        x = x.permute(*perm).contiguous()
        x = self.weight_in(x)
        for i, layer in enumerate(self.weight_stack):
            x = layer(x)
        x = self.out(x)
        
        x = x.permute(*perm).contiguous()
        
        return x


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class SimpleVAECompressionModel(CompressionModel):
    """Simple VAE model with arbitrary latent codec.

    .. code-block:: none

               ┌───┐  y  ┌────┐ y_hat ┌───┐
        x ──►──┤g_a├──►──┤ lc ├───►───┤g_s├──►── x_hat
               └───┘     └────┘       └───┘
    """
    
    def __init__(self, input_size, dim_encoder, n_resblock, M, scale, shift):
        super().__init__()
            
        self.register_buffer('scale', scale)    
        self.register_buffer('shift', shift)   
        
        # self.scale = scale
        # self.shift = shift        
        
        self.input_size = input_size
        self.M = M
        self.dim_encoder = dim_encoder
        self.n_resblock = n_resblock
        
        self.g_a =  nn.Sequential(
            nn.Linear(input_size, dim_encoder),
            ResidualStack(dim_encoder, n_resblock),
            nn.Linear(dim_encoder, M),
        )
        
        self.g_s = nn.Sequential(
            nn.Linear(M, dim_encoder),
            ResidualStack(dim_encoder, n_resblock),
            nn.Linear(dim_encoder, input_size),
        )
        
        self.entropy_bottleneck = EntropyBottleneck(M)

    # def __getitem__(self, key: str) -> LatentCodec:
    #     return self.latent_codec[key]

    def forward(self, data):
        x = data['weight_block']
        assert not 'q_level' in data.keys()
        x_shift = (x - self.shift) / self.scale
        # if 'pe' in data:
        #     x_shift = data['pe2']*x_shift + data['pe']
        y = self.g_a(x_shift)
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)       
        
        ######## STE quant ########
        ## v2 이걸 하는게 맞다 
        # perm_ = np.arange(len(x.shape))
        # perm_[0], perm_[1] = perm_[1], perm_[0]
        # inv_perm = np.arange(len(x.shape))[np.argsort(perm_)]
        # y_hat = y.permute(*perm_).contiguous()
        # shape = y_hat.size()
        # y_hat = y_hat.reshape(x.size(0), 1, -1)        
        # y_offset = self.entropy_bottleneck._get_medians()
        # y_tmp = y_hat - y_offset
        # y_hat = ste_round(y_tmp) + y_offset        
        # y_hat = y_hat.reshape(shape)
        # y_hat = y_hat.permute(*inv_perm).contiguous()
        ######################
        
    
        y_hat = y_hat.permute(*perm).contiguous()
        
        x_hat = self.g_s(y_hat)
        x_hat = self.scale * x_hat + self.shift
        
        return {
            "x": x,
            "x_hat": x_hat,
            "likelihoods": {'y': y_likelihoods},
            "embedding_loss": None,
            "y": y,
            "y_hat": y_hat
        }

    def compress(self, data):
        x = data['weight_block']    
        x_shift = (x - self.shift) / self.scale
        if 'pe' in data:
            x_shift = data['pe2']*x_shift + data['pe']
        y = self.g_a(x_shift)
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()

        # shape = torch.Size([])
        shape = y.size()[2:]
        y_strings = self.entropy_bottleneck.compress(y)
        
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        # import ipdb; ipdb.set_trace()
        return {"strings": [y_strings], "shape": shape, "y_hat": y_hat}

    # def decompress(self, strings: List[List[bytes]], shape, **kwargs):
    # def decompress(self, strings, shape, **kwargs):
    def decompress(self, enc_data, **kwargs):
        strings = enc_data["strings"]
        shape = enc_data["shape"]
        
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape, **kwargs)
        
        perm = list(range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()
        
        # x_hat = self.g_s(y_hat).clamp_(0, 1)
        x_hat = self.g_s(y_hat)
        
        x_hat = self.scale * x_hat + self.shift
        
        return {
            "x_hat": x_hat,
        }
    

class NWC_without_encoder(CompressionModel):
    """Simple VAE model with arbitrary latent codec.

    .. code-block:: none

               ┌───┐  y  ┌────┐ y_hat ┌───┐
        x ──►──┤g_a├──►──┤ lc ├───►───┤g_s├──►── x_hat
               └───┘     └────┘       └───┘
    """
    
    def __init__(self, input_size, dim_encoder, n_resblock, M, scale, shift):
        super().__init__()
            
        self.register_buffer('scale', scale)    
        self.register_buffer('shift', shift)   
        
        # self.scale = scale
        # self.shift = shift        
        
        self.input_size = input_size
        self.M = M
        self.dim_encoder = dim_encoder

        self.g_a =  nn.Sequential(
            nn.Linear(input_size, dim_encoder),
            ResidualStack(dim_encoder, n_resblock),
            nn.Linear(dim_encoder, M),
        )
        
        self.g_s = nn.Sequential(
            nn.Linear(M, dim_encoder),
            ResidualStack(dim_encoder, n_resblock),
            nn.Linear(dim_encoder, input_size),
        )
        
        self.entropy_bottleneck = EntropyBottleneck(M)

    # def __getitem__(self, key: str) -> LatentCodec:
    #     return self.latent_codec[key]

    def forward(self, data):
        x = data['weight_block']
        y = x
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        # ####### STE quant ########
        # # v2 이걸 하는게 맞다 
        # perm_ = np.arange(len(y.shape))
        # perm_[0], perm_[1] = perm_[1], perm_[0]
        # inv_perm = np.arange(len(y.shape))[np.argsort(perm_)]
        # y_hat = y.permute(*perm_).contiguous()
        # shape = y_hat.size()
        # y_hat = y_hat.reshape(y_hat.size(0), 1, -1)        
        # y_offset = self.entropy_bottleneck._get_medians()
        # y_tmp = y_hat - y_offset
        # y_hat = ste_round(y_tmp) + y_offset        
        # y_hat = y_hat.reshape(shape)
        # y_hat = y_hat.permute(*inv_perm).contiguous()
        # ####################
        
        y_hat = y_hat.permute(*perm).contiguous()
        
        x_hat = self.g_s(y_hat)
        x_hat = self.scale * x_hat + self.shift
        
        return {
            "x": x,
            "x_hat": x_hat,
            "likelihoods": {'y': y_likelihoods},
            "embedding_loss": None,
            "y": y,
            "y_hat": y_hat
        }
        
    def forward_encoder(self, data):
        x = data['weight_block']
        assert not 'q_level' in data.keys()
        x_shift = (x - self.shift) / self.scale
        # if 'pe' in data:
        #     x_shift = data['pe2']*x_shift + data['pe']
        y = self.g_a(x_shift)
        
        return {
            "y": y
        }

    def compress(self, data):
        x = data['weight_block']    
        y = x
        
        perm = list(range(y.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y = y.permute(*perm).contiguous()

        # shape = torch.Size([])
        shape = y.size()[2:]
        y_strings = self.entropy_bottleneck.compress(y)
        
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        # import ipdb; ipdb.set_trace()
        return {"strings": [y_strings], "shape": shape, "y_hat": y_hat}

    # def decompress(self, strings: List[List[bytes]], shape, **kwargs):
    # def decompress(self, strings, shape, **kwargs):
    def decompress(self, enc_data, **kwargs):
        strings = enc_data["strings"]
        shape = enc_data["shape"]
        
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape, **kwargs)
        
        perm = list(range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()
        
        # x_hat = self.g_s(y_hat).clamp_(0, 1)
        x_hat = self.g_s(y_hat)
        
        x_hat = self.scale * x_hat + self.shift
        
        return {
            "x_hat": x_hat,
        }


class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, input_size, dim_encoder, n_resblock, M, N, scale, shift, **kwargs):
        super().__init__(**kwargs)

        self.register_buffer('scale', scale)    
        self.register_buffer('shift', shift)   
        
        self.input_size = input_size
        self.dim_encoder = dim_encoder

        self.entropy_bottleneck = EntropyBottleneck(M)

        self.g_a = Encoder(input_size, n_resblock, dim_encoder, M)
        self.g_s = Encoder(M, n_resblock, dim_encoder, input_size)

        self.h_a = Encoder(M, n_resblock//2, dim_encoder, N)
        self.h_s = Encoder(N, n_resblock//2, dim_encoder, M)

        # self.g_a = nn.Sequential(
        #     conv(3, N),
        #     GDN(N),
        #     conv(N, N),
        #     GDN(N),
        #     conv(N, N),
        #     GDN(N),
        #     conv(N, M),
        # )

        # self.g_s = nn.Sequential(
        #     deconv(M, N),
        #     GDN(N, inverse=True),
        #     deconv(N, N),
        #     GDN(N, inverse=True),
        #     deconv(N, N),
        #     GDN(N, inverse=True),
        #     deconv(N, 3),
        # )

        # self.h_a = nn.Sequential(
        #     conv(M, N, stride=1, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     conv(N, N),
        #     nn.ReLU(inplace=True),
        #     conv(N, N),
        # )

        # self.h_s = nn.Sequential(
        #     deconv(N, N),
        #     nn.ReLU(inplace=True),
        #     deconv(N, N),
        #     nn.ReLU(inplace=True),
        #     conv(N, M, stride=1, kernel_size=3),
        #     nn.ReLU(inplace=True),
        # )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    # @property
    # def downsampling_factor(self) -> int:
    #     return 2 ** (4 + 2)

    # def forward(self, x):
    #     y = self.g_a(x)
    #     z = self.h_a(torch.abs(y))
    #     z_hat, z_likelihoods = self.entropy_bottleneck(z)
    #     scales_hat = self.h_s(z_hat)
    #     y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
    #     x_hat = self.g_s(y_hat)

    #     return {
    #         "x_hat": x_hat,
    #         "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
    #     }

    # @classmethod
    # def from_state_dict(cls, state_dict):
    #     """Return a new model instance from `state_dict`."""
    #     N = state_dict["g_a.0.weight"].size(0)
    #     M = state_dict["g_a.6.weight"].size(0)
    #     net = cls(N, M)
    #     net.load_state_dict(state_dict)
    #     return net

    # def compress(self, x):
    #     y = self.g_a(x)
    #     z = self.h_a(torch.abs(y))

    #     z_strings = self.entropy_bottleneck.compress(z)
    #     z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    #     scales_hat = self.h_s(z_hat)
    #     indexes = self.gaussian_conditional.build_indexes(scales_hat)
    #     y_strings = self.gaussian_conditional.compress(y, indexes)
    #     return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    # def decompress(self, strings, shape):
    #     assert isinstance(strings, list) and len(strings) == 2
    #     z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
    #     scales_hat = self.h_s(z_hat)
    #     indexes = self.gaussian_conditional.build_indexes(scales_hat)
    #     y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
    #     x_hat = self.g_s(y_hat).clamp_(0, 1)
    #     return {"x_hat": x_hat}


class MeanScaleHyperprior(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, input_size, dim_encoder, n_resblock, M, N, scale, shift, **kwargs):
        super().__init__(input_size, dim_encoder, n_resblock, M, N, scale, shift, **kwargs)

        # self.h_a = nn.Sequential(
        #     conv(M, N, stride=1, kernel_size=3),
        #     nn.LeakyReLU(inplace=True),
        #     conv(N, N),
        #     nn.LeakyReLU(inplace=True),
        #     conv(N, N),
        # )

        # self.h_s = nn.Sequential(
        #     deconv(N, M),
        #     nn.LeakyReLU(inplace=True),
        #     deconv(M, M * 3 // 2),
        #     nn.LeakyReLU(inplace=True),
        #     conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        # )
        
        self.h_a = Encoder(M, n_resblock, M, N)
        self.h_s = Encoder(N, n_resblock, M*2, M*2)

    def forward(self, data):
        x = data['weight_block']    
        x_shift = (x - self.shift) / self.scale
        if 'pe' in data:
            x_shift = x_shift + data['pe']
        
        perm = list(range(x_shift.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        x_shift = x_shift.permute(*perm).contiguous()
        y = self.g_a(x_shift)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        x_hat = x_hat.permute(*perm).contiguous()
        x_hat = self.scale * x_hat + self.shift

        return {
            "x": x,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, data):
        x = data['weight_block']
        x_shift = (x - self.shift) / self.scale
        if 'pe' in data:
            x_shift = x_shift + data['pe']
            
        perm = list(range(x_shift.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        x_shift = x_shift.permute(*perm).contiguous()
        
        y = self.g_a(x_shift)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        # x_hat = self.g_s(y_hat).clamp_(0, 1)
        x_hat = self.g_s(y_hat)        
        
        perm = list(range(x_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]        
        x_hat = x_hat.permute(*perm).contiguous()
        x_hat = self.scale * x_hat + self.shift
        
        return {"x_hat": x_hat}


# dim_encoder은 'nhead' 의 배수로 하기를 바람.
class SimpleVAECompressionModel_with_Transformer(SimpleVAECompressionModel):
    def __init__(self, input_size, dim_encoder, n_layer, M, scale, shift, use_hyper = False):
        super().__init__(input_size, dim_encoder, 1, M, scale, shift)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_encoder, nhead=8, batch_first = True) 
        g_a = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.g_a = nn.Sequential(nn.Linear(input_size, dim_encoder), 
                                g_a, 
                                nn.Linear(dim_encoder, M))
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=dim_encoder, nhead=8, batch_first = True) 
        g_s = nn.TransformerEncoder(decoder_layer, num_layers=n_layer)
        
        self.g_s = nn.Sequential(nn.Linear(M, dim_encoder), 
                                 g_s, 
                                 nn.Linear(dim_encoder, input_size))
        
        self.use_hyper = use_hyper
        if use_hyper == True:
            
            h_a_layer = nn.TransformerEncoderLayer(d_model=dim_encoder//2, nhead=8, batch_first = True) 
            h_a = nn.TransformerEncoder(h_a_layer, num_layers=n_layer//2)
            
            self.h_a = nn.Sequential(nn.Linear(M, dim_encoder//2), h_a)
            
            h_s_means_layer = nn.TransformerEncoderLayer(d_model=dim_encoder//2, nhead=8, batch_first = True) 
            h_s_means = nn.TransformerEncoder(h_s_means_layer, num_layers=n_layer//2)
            
            self.h_s_means = nn.Sequential(h_s_means, nn.Linear(dim_encoder//2, M))
            
            h_s_scales_layer = nn.TransformerEncoderLayer(d_model=dim_encoder//2, nhead=8, batch_first = True) 
            h_s_scales = nn.TransformerEncoder(h_s_scales_layer, num_layers=n_layer//2)
            
            self.h_s_scales = nn.Sequential(h_s_scales, nn.Linear(dim_encoder//2, M))
            
            self.gaussian_conditional = GaussianConditional(None)
            self.entropy_bottleneck = EntropyBottleneck(dim_encoder//2)
            
    
    def forward(self, data):
        
        x = data['weight_block']    
        
        self.shift = self.shift.to(x.device)
        self.scale = self.scale.to(x.device)
        
        x_shift = (x - self.shift) / self.scale

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
            
            x_hat = self.g_s(y_hat)
            x_hat = self.scale * x_hat + self.shift
            
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
            y_hat = y_hat.permute(*perm).contiguous()
            
            x_hat = self.g_s(y_hat)
            x_hat = self.scale * x_hat + self.shift
            
            return {
                "x": x,
                "x_hat": x_hat,
                "likelihoods": {'y': y_likelihoods},
                "embedding_loss": None,
                "y": y,
                "y_hat": y_hat
            }
    
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
        
    def compress(self, data):
        
        x = data['weight_block']    
        x_shift = (x - self.shift) / self.scale

        y = self.g_a(x_shift)
        
        if self.use_hyper == True:
            
            return self.compress_with_hyperprior(y)
            
        else:
            return self.compress_without_hyperprior(y)    
    
    def decompress_with_hyperprior(self, strings, shape):
        
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
        
        x_hat = self.g_s(y_hat)
        x_hat = self.scale * x_hat + self.shift
        
        return {"x_hat": x_hat}
        
    
    def decompress_without_hyperprior(self, strings, shape):
        
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        
        perm = list(range(y_hat.dim()))
        perm[-1], perm[1] = perm[1], perm[-1]
        y_hat = y_hat.permute(*perm).contiguous()
        
        # x_hat = self.g_s(y_hat).clamp_(0, 1)
        x_hat = self.g_s(y_hat)
        
        x_hat = self.scale * x_hat + self.shift
        
        return {"x_hat": x_hat}
        
    
    def decompress(self, strings, shape):
        
        if self.use_hyper == True:
            
            return self.decompress_with_hyperprior(strings, shape)
        
        else:
            return self.decompress_without_hyperprior(strings, shape)
