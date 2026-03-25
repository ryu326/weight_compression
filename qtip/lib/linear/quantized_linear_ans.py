from __future__ import annotations

import torch
import torch.nn as nn

from lib.codebook import ans_uniform


class QuantizedLinearANS(nn.Module):
    """
    ANS-uniform quantized linear layer.

    - Uses qtip/my-kernal ANS fused decode+matvec kernel when possible.
    - Keeps SU/SV scaling buffers similar to QuantizedLinear.
    - Existing files remain untouched; this is an additive implementation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        step_size: float = 0.02,
        prob_bits: int = 9,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        grad_ckpt: bool = False,
        use_kernel: bool = True,
        cache_hatW: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grad_ckpt = bool(grad_ckpt)
        self.dtype = dtype

        self.codebook_class = ans_uniform.ANSUniformLinear(
            in_features=in_features,
            out_features=out_features,
            prob_bits=prob_bits,
            step_size=step_size,
            dtype=dtype,
            use_kernel=use_kernel,
            cache_hatw=cache_hatW,
        )

        self.register_buffer("SU", torch.ones(in_features, dtype=self.dtype))
        self.register_buffer("SV", torch.ones(out_features, dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @torch.no_grad()
    def encode_weight(
        self,
        weight: torch.Tensor,
        step_size: float | None = None,
        prob_bits: int | None = None,
    ) -> None:
        """
        Encode dense [out_features, in_features] weight into ANS buffers.
        """
        self.codebook_class.encode_weight(weight, step_size=step_size, prob_bits=prob_bits)

    @torch.no_grad()
    def load_from_linear(
        self,
        linear: nn.Linear,
        step_size: float | None = None,
        prob_bits: int | None = None,
        copy_bias: bool = True,
    ) -> None:
        if linear.weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"linear.weight shape mismatch: expected {(self.out_features, self.in_features)}, "
                f"got {tuple(linear.weight.shape)}"
            )
        self.encode_weight(linear.weight, step_size=step_size, prob_bits=prob_bits)
        if copy_bias and self.bias is not None and linear.bias is not None:
            self.bias.copy_(linear.bias.detach().to(self.bias.device, dtype=self.bias.dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.grad_ckpt:
            return torch.utils.checkpoint.checkpoint(self._no_ckpt_forward, input, use_reentrant=True)
        return self._no_ckpt_forward(input)

    def _no_ckpt_forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"Input feature mismatch: expected {self.in_features}, got {input.shape[-1]}"
            )

        flat_x = input.view(-1, self.in_features).to(torch.float32)
        flat_x = flat_x * self.SU

        out = self.codebook_class(flat_x).to(torch.float32)
        out = out * self.SV

        if self.bias is not None:
            out = out + self.bias

        return out.view(*input.shape[:-1], self.out_features).to(input.dtype)

