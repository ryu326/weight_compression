import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module

from lib.algo.nwc_refactory import model_foward_one_batch, block_LDL, comp_W
from lib import utils

class CompLinear2(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    Wr: Tensor

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
        self.Wr = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        ## my
        # self.reset_parameters()
        
        self.model = None
        self.args = None
        self.metadata = None
        self.qlevel = None
        self.hatWr = None 
        self.out = None

    def forward(self, input: Tensor) -> Tensor:
        if self.hatWr == None:
            self.args.comp_batch_size = 1024
            self.out = self.compress_linear()
            W_hat = utils.de_standardize_Wr(self.out['hatWr'], self.metadata, self.args)
            self.bpp_loss = self.out['bpp_loss_for_train']
        else:
            W_hat = utils.de_standardize_Wr(self.hatWr, self.metadata, self.args)
        assert any([not torch.isnan(W_hat).any(), not torch.isinf(W_hat).any()]), "W_hat has NaN or Inf values"
        return F.linear(input, W_hat, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
    
    def compress_linear(self):
        comp_model = self.model
        metadata = self.metadata
        
        res = comp_W(self.Wr, None, comp_model, self.args, qlevel = self.qlevel, **metadata)

        total_metadata_bpp = utils.calculate_metadata_bpp(metadata, self.Wr.shape, self.args)
        bpp_keys = ['bpp_loss_sum', 'bpp_sum']
        for key in bpp_keys:
            if res.get(key) is not None:
                res[key] += total_metadata_bpp
        
        # {'hatWr': W_hat,
        #     'Wr_ldlq': W_ldl,
        #     'bpp_loss_sum': bpp_loss_sum.item(),
        #     'bpp_loss': bpp_loss_sum.item() / num_pixels,
        #     'num_pixels': num_pixels,
        #     'bpp_sum': bpp_sum,
        #     'bpp': bpp_sum / num_pixels,
        #     'codes': codes,
        #     'bpp_loss_for_train': bpp_loss_sum / num_pixels,
        #     }   
   
        
        return res