from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from typing import Optional, Tuple
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

class EntropyBottleneck_with_conditional_Delta(EntropyBottleneck):
    def __init__(
        self,
        channels: int,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        super().__init__(
            channels=channels,
            tail_mass=tail_mass,
            init_scale=init_scale,
            filters=filters,
            **kwargs,
        )

    def _likelihood(
        self, inputs: Tensor, Delta:Tensor, stop_gradient: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # half = float(0.5)
        half = Delta/2
        # half = half.view(Delta.size(0), 1, 1, 1)

        lower = self._logits_cumulative(inputs - half, stop_gradient=stop_gradient)
        upper = self._logits_cumulative(inputs + half, stop_gradient=stop_gradient)
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        return likelihood, lower, upper
    
    def forward(
        self, x: Tensor, Delta:Tensor = torch.tensor(1.0), training: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            raise NotImplementedError()
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            # perm = (1, 2, 3, 0)
            # inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        ##
        Delta = Delta.permute(*perm).contiguous()
        Delta = Delta.reshape(Delta.size(0), 1, -1)
        
        # Add noise or quantize
        outputs = self.quantize(
            values, Delta, "noise" if training else "dequantize", self._get_medians()
        )

        if not torch.jit.is_scripting():
            # likelihood, _, _ = self._likelihood(outputs)
            likelihood, _, _ = self._likelihood(outputs, Delta)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
            # TorchScript not yet supported
            # likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood
    

    def quantize(
        self, inputs: Tensor, Delta:Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            # half = float(Delta/2)
            # noise = torch.empty_like(inputs).uniform_(-half, half)
            half = Delta.detach()/2
            noise = (torch.rand_like(inputs) - 0.5) * 2 * half
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        # Delta = Delta.view(Delta.size(0), 1, 1, 1)
        
        # outputs = torch.round(outputs)
        outputs = torch.round(outputs / Delta) 
        outputs = outputs * Delta
        
        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()

        return outputs

    def _quantize(
        self, inputs: Tensor, Delta:Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        return self.quantize(inputs, Delta, mode, means)
