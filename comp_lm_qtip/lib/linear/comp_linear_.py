import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module

from lib.algo.nwc import model_foward_one_batch, block_LDL

class CompLinear2(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        ## my
        self.row_norm = Parameter(torch.empty((out_features, 1), **factory_kwargs))
        # self.reset_parameters()
        
        self.model = None
        self.scale_cond = Parameter(torch.empty((out_features, 1), **factory_kwargs))
        self.args = None
            
    def forward(self, input: Tensor) -> Tensor:
        W_hat, bpp_loss_sum, num_pixels = self.comp_W()
        W = W_hat * self.row_norm
        self.bpp_loss_sum = bpp_loss_sum
        self.num_pixels = num_pixels
        return F.linear(input, W, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def comp_W(self):
        W = self.weight
        args = self.args
        row_norm = self.row_norm
        
        bs = min(W.shape[1], 4096*4096 // W.shape[0])
        # bs = min(W.shape[1], 4096*4096 // W.shape[0]) if args.comp_batch_size == -1 else args.comp_batch_size(m, n) = W.shape
        (m, n) = W.shape
        W_hat = torch.zeros_like(W)
        num_pixels = 0
        bpp_loss_sum = 0
        bpp_sum = 0
        codes = []        
        
        if args.ldlq:
            bs = 128 if args.comp_batch_size == -1 else args.comp_batch_size
            # assert args.direction == 'col'
            L, D = block_LDL(H, bs)
            assert n % bs == 0

        for i,e in enumerate(range(n, 0, -bs)):
            s = max(0, e - bs)
            if args.ldlq:
                w = W[:, s:e] + (W[:, e:] - W_hat[:, e:]) @ L[e:, s:e]
            else:
                w = W[:, s:e]        
            
            x_hat, n_pixels, bpp_loss_, out, out_enc, nbits = model_foward_one_batch(w.clone(), self.model, args, rnorm = row_norm)

            codes.append(out_enc)
            bpp_sum += nbits
            W_hat[:, s:e] = x_hat
            num_pixels += n_pixels
            # bpp_loss_sum += bpp_loss_.item()
            bpp_loss_sum += bpp_loss_

        return W_hat, bpp_loss_sum, num_pixels