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
# from entropybottleneck import EntropyBottleneck_with_conditional_Delta
import sys
sys.path.append('/workspace/Weight_compression')
from NWC.models.entropybottleneck import EntropyBottleneck_with_conditional_Delta

class LightweightDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_sizes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_sizes[0], padding='same'),
            nn.SELU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_sizes[1], padding='same'),
            nn.SELU(inplace=True)
        )

    def forward(self, x):
        return self.decoder(x)

# class LightweightDecoder(nn.Module):
#     def __init__(self, in_channels=64, hidden_channels=32, out_channels=1):
#         super().__init__()
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=4, stride=1, padding=1),  # upsample ×1
#             nn.SELU(inplace=True),
#             nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=1, padding=1),  # upsample ×1
#             nn.SELU(inplace=True)
#         )

#     def forward(self, x):
#         return self.decoder(x)

def ste_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x) - x.detach() + x

class NWCC_dec_only(CompressionModel):    
    def __init__(self, in_channels = 4, hidden_channels = 4, out_channels = 1, kernel_sizes = [5, 11], M = 4):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.M = M

        self.config = {
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'out_channels': out_channels,
            'kernel_sizes': kernel_sizes,
            'M': M
        }

        self.g_s = LightweightDecoder(in_channels, hidden_channels, out_channels, kernel_sizes)        
        self.entropy_bottleneck = EntropyBottleneck_with_conditional_Delta(M)
        
    def forward(self, y, delta):
        # W : (B, C, H, W)
        # Qmap : (B, 1, H, W)
        
        _y_hat, y_likelihoods = self.entropy_bottleneck(y, delta)
        
        ####### STE quant ########
        # v2 이걸 하는게 맞다 
        perm_ = np.arange(len(y.shape))
        perm_[0], perm_[1] = perm_[1], perm_[0]
        inv_perm = np.arange(len(y.shape))[np.argsort(perm_)]
        y_hat = y.permute(*perm_).contiguous()
        delta = delta.permute(*perm_).contiguous()
        
        shape = y_hat.size()
        y_hat = y_hat.reshape(y_hat.size(0), 1, -1)
        delta = delta.reshape(delta.size(0), 1, -1)        
        
        y_offset = self.entropy_bottleneck._get_medians()
        y_tmp = y_hat - y_offset
        y_hat =  ste_round(y_tmp/delta)*delta + y_offset        
        y_hat = y_hat.reshape(shape)
        delta = delta.reshape(shape)
        
        y_hat = y_hat.permute(*inv_perm).contiguous()
        delta = delta.permute(*inv_perm).contiguous()
        
        ####################
        
        W_hat = self.g_s(y_hat)
        
        return {
            "W_hat": W_hat,
            "likelihoods": {'y': y_likelihoods},
            "y_hat": y_hat
        }
        
    def compress(self, y, delta):   
        shape = y.size()[2:]
        y_strings = self.entropy_bottleneck.compress(y)

        return {"strings": [y_strings], "shape": shape}

    # def decompress(self, strings: List[List[bytes]], shape, **kwargs):
    # def decompress(self, strings, shape, **kwargs):
    def decompress(self, enc_data, **kwargs):
        strings = enc_data["strings"]
        shape = enc_data["shape"]
        
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape, **kwargs)
        
        x_hat = self.g_s(y_hat)        
        
        return {
            "W_hat": x_hat,
        }
        