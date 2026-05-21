import math
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from lib.utils.matmul_had import get_hadK, matmul_hadU_cuda, matmul_hadUt_cuda


_LOG2E = math.log2(math.e)
_LOG2 = math.log(2.0)


def _real_hartley_transform(x: Tensor, dim: int = -1) -> Tensor:
    spectrum = torch.fft.fft(x, dim=dim, norm="ortho")
    return spectrum.real - spectrum.imag


class MixtureEntropyModel(nn.Module):
    """Mixture of Gaussian and/or Laplacian components for entropy modeling.

    All latent elements share the same mixture parameters (analogous to
    channels=1 EntropyBottleneck).  Rate computation is pure element-wise
    — no channel expansion, so memory = O(N) regardless of component count.
    """

    def __init__(
        self,
        num_gaussian: int = 2,
        num_laplacian: int = 2,
        scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        K = num_gaussian + num_laplacian
        if K < 1:
            raise ValueError("Need at least 1 component")
        self.num_gaussian = num_gaussian
        self.num_laplacian = num_laplacian
        self.K = K

        self._logits = nn.Parameter(torch.zeros(K))
        self._means = nn.Parameter(torch.linspace(-1.0, 1.0, K))
        self._log_scales = nn.Parameter(torch.full((K,), math.log(scale_init)))

        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    @property
    def weights(self) -> Tensor:
        return F.softmax(self._logits, dim=0)

    @property
    def means(self) -> Tensor:
        return self._means

    @property
    def scales(self) -> Tensor:
        return self._log_scales.exp().clamp_min(1e-6)

    def _component_log_prob(self, x: Tensor, k: int) -> Tensor:
        """Log-probability of x under component k (continuous density)."""
        mu = self._means[k]
        s = self._log_scales[k].exp().clamp_min(1e-6)
        centered = x - mu
        if k < self.num_gaussian:
            return -0.5 * math.log(2 * math.pi) - self._log_scales[k] - 0.5 * (centered / s) ** 2
        else:
            return -math.log(2.0) - self._log_scales[k] - centered.abs() / s

    def _component_cdf(self, x: Tensor, k: int) -> Tensor:
        """CDF of component k evaluated at x."""
        mu = self._means[k]
        s = self._log_scales[k].exp().clamp_min(1e-6)
        centered = x - mu
        if k < self.num_gaussian:
            return 0.5 * (1.0 + torch.erf(centered / (s * math.sqrt(2.0))))
        else:
            return 0.5 + 0.5 * torch.sign(centered) * (1.0 - torch.exp(-centered.abs() / s))

    def log_likelihood_continuous(self, x: Tensor) -> Tensor:
        """log2 p(x) under the mixture — memory O(N), loops over K."""
        log_w = F.log_softmax(self._logits, dim=0)
        acc = torch.full_like(x, -float("inf"))
        for k in range(self.K):
            lp = log_w[k] + self._component_log_prob(x, k)
            acc = torch.logaddexp(acc, lp)
        return acc * _LOG2E

    def log_likelihood_quantized(self, x: Tensor, qs: Tensor) -> Tensor:
        """log2 P(Q(x)) = log2(CDF(x+qs/2) - CDF(x-qs/2))."""
        half_qs = 0.5 * qs
        w = self.weights
        prob = torch.zeros_like(x)
        for k in range(self.K):
            prob = prob + w[k] * (self._component_cdf(x + half_qs, k) - self._component_cdf(x - half_qs, k))
        return torch.log2(prob.clamp_min(1e-9))

    def forward(
        self,
        x: Tensor,
        training: Optional[bool] = None,
        qs: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Quantize + compute likelihoods.

        Args:
            x: input latent [1, 1, H, W]
            training: if True add uniform noise, else round (dequantize)
            qs: quantization step size (scalar tensor or None)

        Returns:
            (quantized, likelihoods) — likelihoods in (0, 1]
        """
        if training is None:
            training = self.training

        median = self._get_median()

        if qs is not None:
            if training:
                noise = torch.empty_like(x).uniform_(-0.5, 0.5) * qs
                quantized = x + noise
            else:
                quantized = torch.round((x - median) / qs) * qs + median
            log2_lik = self.log_likelihood_quantized(quantized, qs)
        else:
            if training:
                noise = torch.empty_like(x).uniform_(-0.5, 0.5)
                quantized = x + noise
            else:
                quantized = torch.round(x - median) + median
            log2_lik = self.log_likelihood_quantized(quantized, torch.ones((), device=x.device, dtype=x.dtype))

        likelihoods = (2.0 ** log2_lik).clamp(1e-9, 1.0)
        return quantized, likelihoods

    def _get_median(self) -> Tensor:
        """Weighted median (approximated by weighted mean for simplicity)."""
        w = self.weights
        return (w * self._means).sum()

    def _get_medians(self) -> Tensor:
        """CompressAI-compatible shape: [1, 1, 1]."""
        return self._get_median().reshape(1, 1, 1)

    def _likelihood(self, outputs: Tensor, qs: Optional[Tensor] = None) -> tuple[Tensor, None, None]:
        """Direct likelihood for STE path (skip quantization)."""
        if qs is None:
            qs = torch.ones((), device=outputs.device, dtype=outputs.dtype)
        log2_lik = self.log_likelihood_quantized(outputs, qs)
        likelihoods = (2.0 ** log2_lik).clamp(1e-9, 1.0)
        return likelihoods, None, None

    @property
    def use_likelihood_bound(self) -> bool:
        return False

    def likelihood_lower_bound(self, likelihood: Tensor) -> Tensor:
        return likelihood.clamp_min(1e-9)

    def loss(self) -> Tensor:
        return torch.tensor(0.0, device=self._logits.device)

    def update(self, force: bool = False, **kwargs) -> bool:
        self._offset = torch.zeros(1, dtype=torch.int32, device=self._logits.device)
        self._quantized_cdf = torch.zeros(1, dtype=torch.int32, device=self._logits.device)
        self._cdf_length = torch.zeros(1, dtype=torch.int32, device=self._logits.device)
        return True

    # ── DietGPU ANS (byte-level rANS) for real bitstream encoding ──────────
    _dietgpu_loaded = False
    _dietgpu_temp_mem: dict = {}

    @classmethod
    def _ensure_dietgpu(cls, device: torch.device, need_bytes: int = 0,
                        lib_path: Optional[str] = None) -> None:
        if not cls._dietgpu_loaded:
            import ctypes
            path = lib_path or os.environ.get(
                "DIETGPU_LIB", "/tmp/dietgpu/build/lib/libdietgpu.so"
            )
            if not os.path.exists(path):
                raise RuntimeError(
                    f"dietgpu shared library not found at {path}; "
                    "build dietgpu and/or set DIETGPU_LIB"
                )
            tp = os.path.join(os.path.dirname(torch.__file__), "lib",
                              "libtorch_python.so")
            if os.path.exists(tp):
                ctypes.CDLL(tp, mode=ctypes.RTLD_GLOBAL)
            torch.ops.load_library(path)
            cls._dietgpu_loaded = True

        desired = max(64 * 1024 * 1024, need_bytes)
        key = str(device)
        cur = cls._dietgpu_temp_mem.get(key)
        if cur is None or cur.numel() < desired:
            cls._dietgpu_temp_mem[key] = torch.empty(
                [desired], dtype=torch.uint8, device=device,
            )

    def _symbols_for_bitstream(self, x: Tensor, qs: Optional[Tensor]):
        """Quantize + shift to smallest int dtype for byte-level compression.

        Returns (shifted_contig_tensor, offset:int, dtype).
        """
        median = self._get_median()
        if qs is not None:
            symbols = torch.round((x - median) / qs).to(torch.int64)
        else:
            symbols = torch.round(x - median).to(torch.int64)
        actual_min = int(symbols.min().item())
        actual_max = int(symbols.max().item())
        K = actual_max - actual_min + 1
        dtype = torch.uint8 if K <= 256 else torch.int16
        shifted = (symbols - actual_min).to(dtype).contiguous()
        return shifted, actual_min, dtype

    def compress(self, x: Tensor, qs: Optional[Tensor] = None) -> dict:
        """Compress x via dietgpu.  Returns dict with strings/shape/meta/num_bits."""
        self._ensure_dietgpu(x.device, need_bytes=int(x.numel() * 4))
        shifted, offset, dtype = self._symbols_for_bitstream(x, qs)
        temp_mem = type(self)._dietgpu_temp_mem[str(x.device)]
        comp, sizes, _ = torch.ops.dietgpu.compress_data(
            False, [shifted], False, temp_mem,
        )
        out_bytes = int(sizes[0].item())
        compressed = comp[0].narrow(0, 0, out_bytes).clone()
        return {
            "strings": [compressed],
            "shape": tuple(x.shape),
            "meta": {
                "offset": offset,
                "dtype": str(dtype).replace("torch.", ""),
                "numel": int(shifted.numel()),
            },
            "num_bits": out_bytes * 8,
        }

    def prepare_for_inference(
        self, strings: list, meta: dict, qs: Optional[Tensor] = None,
        device=None,
    ) -> None:
        """Pre-compute decompress buffers (called from ECLinear.prepare_for_inference)."""
        dev = device if device is not None else next(iter(self.parameters())).device
        self._ensure_dietgpu(dev, need_bytes=int(meta["numel"] * 4))

        dtype_map = {"uint8": torch.uint8, "int16": torch.int16}
        dtype = dtype_map[meta["dtype"]]

        comp_buf = strings[0]
        if not isinstance(comp_buf, torch.Tensor):
            comp_buf = torch.as_tensor(bytearray(comp_buf), dtype=torch.uint8)
        self._inf_comp_buf = comp_buf.to(dev, dtype=torch.uint8).contiguous()
        self._inf_out_buf = torch.empty([meta["numel"]], dtype=dtype, device=dev)

        median = float(self._get_median().item())
        offset = int(meta["offset"])
        qs_val = float(qs) if qs is not None else 1.0

        self._inf_bias_noscale = float(offset) + median / qs_val
        self._inf_qs = qs_val
        self._inf_ready = True

    def decompress(self, strings: list[bytes], size=None,
                   meta: Optional[dict] = None,
                   qs: Optional[Tensor] = None,
                   device=None) -> Tensor:
        if meta is None:
            raise ValueError("decompress requires meta dict from compress()")

        # Fast path: qs fused into right_diag, decompress is add-only (no multiply)
        if getattr(self, "_inf_ready", False):
            dev = self._inf_comp_buf.device
            temp_mem = type(self)._dietgpu_temp_mem[str(dev)]
            torch.ops.dietgpu.decompress_data(
                False, [self._inf_comp_buf], [self._inf_out_buf], False, temp_mem,
            )
            latent = self._inf_out_buf.float() + self._inf_bias_noscale
            if size is not None:
                latent = latent.reshape(size)
            return latent

        # Slow path: first call / no prepare
        dev = device if device is not None else next(iter(self.parameters())).device
        self._ensure_dietgpu(dev, need_bytes=int(meta["numel"] * 4))
        temp_mem = type(self)._dietgpu_temp_mem[str(dev)]

        dtype_map = {"uint8": torch.uint8, "int16": torch.int16}
        dtype = dtype_map[meta["dtype"]]
        comp_buf = strings[0]
        if not isinstance(comp_buf, torch.Tensor):
            comp_buf = torch.as_tensor(
                bytearray(comp_buf), dtype=torch.uint8,
            )
        comp_buf = comp_buf.to(dev, dtype=torch.uint8).contiguous()
        out_buf = torch.empty([meta["numel"]], dtype=dtype, device=dev)
        torch.ops.dietgpu.decompress_data(
            False, [comp_buf], [out_buf], False, temp_mem,
        )
        symbols = out_buf.to(torch.int64) + int(meta["offset"])
        median = self._get_median().to(device=dev, dtype=torch.float32)
        if qs is not None:
            latent = symbols.to(torch.float32) * float(qs) + median
        else:
            latent = symbols.to(torch.float32) + median
        if size is not None:
            latent = latent.reshape(size)
        return latent


class EntropyConstrainedLinear(Module):

    __constants__ = ["in_features", "out_features", "decoder_type"]
    in_features: int
    out_features: int
    latent: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        decoder_type: str = "rht",
        entropy_bottleneck_kwargs: Optional[dict[str, Any]] = None,
        rht_seed: Optional[int] = 0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if decoder_type not in {"rht", "dft", "identity"}:
            raise ValueError(
                "decoder_type must be 'rht', 'dft', or 'identity', "
                f"got {decoder_type!r}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.decoder_type = decoder_type
        self.rht_seed = rht_seed
        self.quantize_mode: str = "noise"
        self.register_buffer("qs", None, persistent=True)

        if decoder_type == "rht":
            hadK_in, K_in = get_hadK(in_features)
            hadK_out, K_out = get_hadK(out_features)
            self.register_buffer("_hadK_in", hadK_in, persistent=False)
            self.register_buffer("_hadK_out", hadK_out, persistent=False)
            self._K_in = K_in
            self._K_out = K_out

        self.latent = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        sign_dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.left_diag = Parameter(
            self._make_random_sign(out_features, sign_dtype, device, rht_seed)
        )
        self.right_diag = Parameter(
            self._make_random_sign(
                in_features,
                sign_dtype,
                device,
                None if rht_seed is None else rht_seed + 1,
            )
        )

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        eb_kwargs = entropy_bottleneck_kwargs or {}
        num_gaussian = int(eb_kwargs.get("num_gaussian", 2))
        num_laplacian = int(eb_kwargs.get("num_laplacian", 2))
        scale_init = float(eb_kwargs.get("scale_init", 1.0))
        self.entropy_bottleneck = MixtureEntropyModel(
            num_gaussian=num_gaussian,
            num_laplacian=num_laplacian,
            scale_init=scale_init,
        )
        if device is not None:
            self.entropy_bottleneck = self.entropy_bottleneck.to(device=device)
        self.entropy_bottleneck = self.entropy_bottleneck.float()
        self._last_rate_loss: Optional[Tensor] = None

        self.reset_parameters()

    @staticmethod
    def _make_random_sign(
        size: int,
        dtype: torch.dtype,
        device: Optional[torch.device],
        seed: Optional[int],
    ) -> Tensor:
        if seed is None:
            sign = torch.randint(0, 2, (size,))
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            sign = torch.randint(0, 2, (size,), generator=generator)
        sign = sign.mul(2).sub(1)
        if device is None:
            return sign.to(dtype=dtype)
        return sign.to(device=device, dtype=dtype)

    @staticmethod
    def _tensor_num_bits(tensor: Optional[Tensor]) -> int:
        if tensor is None:
            return 0
        return int(tensor.numel() * tensor.element_size() * 8)

    def _normalize_left_scale(self, row_scale: Tensor) -> Tensor:
        scale = row_scale.to(device=self.latent.device, dtype=self.latent.dtype)
        if scale.ndim == 2:
            if scale.shape == (self.out_features, 1):
                scale = scale[:, 0]
            elif scale.shape == (1, self.out_features):
                scale = scale[0]
        if scale.ndim != 1 or scale.numel() != self.out_features:
            raise ValueError(
                f"row_scale must have {self.out_features} elements, got shape {tuple(row_scale.shape)}"
            )
        return scale

    def _normalize_right_scale(self, col_scale: Tensor) -> Tensor:
        scale = col_scale.to(device=self.latent.device, dtype=self.latent.dtype)
        if scale.ndim == 2:
            if scale.shape == (self.in_features, 1):
                scale = scale[:, 0]
            elif scale.shape == (1, self.in_features):
                scale = scale[0]
        if scale.ndim != 1 or scale.numel() != self.in_features:
            raise ValueError(
                f"col_scale must have {self.in_features} elements, got shape {tuple(col_scale.shape)}"
            )
        return scale

    def _apply(self, fn):
        super()._apply(fn)
        self.entropy_bottleneck = self.entropy_bottleneck.to(device=self.latent.device)
        self.entropy_bottleneck = self.entropy_bottleneck.float()
        return self

    def _reset_diag_parameters(self) -> None:
        with torch.no_grad():
            if self.decoder_type == "identity":
                self.left_diag.copy_(torch.ones(self.out_features, dtype=self.left_diag.dtype, device=self.left_diag.device))
                self.right_diag.copy_(torch.ones(self.in_features, dtype=self.right_diag.dtype, device=self.right_diag.device))
            else:
                self.left_diag.copy_(self._make_random_sign(self.out_features, self.left_diag.dtype, self.left_diag.device, self.rht_seed))
                self.right_diag.copy_(self._make_random_sign(self.in_features, self.right_diag.dtype, self.right_diag.device, None if self.rht_seed is None else self.rht_seed + 1))

    @property
    def weight(self) -> Tensor:
        return self.decode_latent(self.latent)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.latent, a=math.sqrt(5))
        self._reset_diag_parameters()
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.latent)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _latent_to_entropy_input(self, latent: Tensor) -> Tensor:
        return latent.unsqueeze(0).unsqueeze(0)

    def _entropy_input_to_latent(self, latent: Tensor) -> Tensor:
        return latent.squeeze(0).squeeze(0)

    def _entropy_device(self) -> torch.device:
        return self.entropy_bottleneck._logits.device

    def _entropy_dtype(self) -> torch.dtype:
        return self.entropy_bottleneck._logits.dtype

    def _orthogonal_transform(self, x: Tensor) -> Tensor:
        if self.decoder_type == "rht":
            return matmul_hadU_cuda(x, self._hadK_in, self._K_in)
        if self.decoder_type == "identity":
            return x
        return _real_hartley_transform(x, dim=-1)

    def _decode_with_rht(self, latent: Tensor) -> Tensor:
        x = matmul_hadU_cuda(latent, self._hadK_in, self._K_in)
        x = x * self.right_diag.unsqueeze(0)
        x = matmul_hadU_cuda(x.T, self._hadK_out, self._K_out).T
        x = x * self.left_diag.unsqueeze(1)
        return x

    def _encode_with_rht(self, weight: Tensor) -> Tensor:
        x = weight / self.left_diag.unsqueeze(1)
        x = x / self.right_diag.unsqueeze(0)
        x = matmul_hadUt_cuda(x.T, self._hadK_out, self._K_out).T
        x = matmul_hadUt_cuda(x, self._hadK_in, self._K_in)
        return x

    def _decode_with_dft(self, latent: Tensor) -> Tensor:
        x = _real_hartley_transform(latent, dim=-1)
        x = x * self.right_diag.unsqueeze(0)
        x = _real_hartley_transform(x.T, dim=-1).T
        x = x * self.left_diag.unsqueeze(1)
        return x

    def _encode_with_dft(self, weight: Tensor) -> Tensor:
        x = weight / self.left_diag.unsqueeze(1)
        x = _real_hartley_transform(x.T, dim=-1).T
        x = x / self.right_diag.unsqueeze(0)
        x = _real_hartley_transform(x, dim=-1)
        return x

    def _decode_with_identity(self, latent: Tensor) -> Tensor:
        x = latent * self.right_diag.unsqueeze(0)
        x = x * self.left_diag.unsqueeze(1)
        return x

    def _encode_with_identity(self, weight: Tensor) -> Tensor:
        x = weight / self.left_diag.unsqueeze(1)
        x = x / self.right_diag.unsqueeze(0)
        return x

    def encode_weight(self, weight: Tensor) -> Tensor:
        if weight.shape != self.latent.shape:
            raise ValueError(f"weight shape must be {tuple(self.latent.shape)}, got {tuple(weight.shape)}")
        weight = weight.to(device=self.latent.device, dtype=self.latent.dtype)
        if self.decoder_type == "rht":
            return self._encode_with_rht(weight)
        if self.decoder_type == "identity":
            return self._encode_with_identity(weight)
        return self._encode_with_dft(weight)

    def decode_latent(self, latent: Tensor) -> Tensor:
        latent = latent.to(device=self.latent.device, dtype=self.latent.dtype)
        if self.decoder_type == "rht":
            return self._decode_with_rht(latent)
        if self.decoder_type == "identity":
            return self._decode_with_identity(latent)
        return self._decode_with_dft(latent)

    def initialize_from_weight(self, weight: Tensor, bias: Optional[Tensor] = None) -> None:
        with torch.no_grad():
            weight = weight.to(device=self.latent.device, dtype=self.latent.dtype)
            self.latent.copy_(self.encode_weight(weight))
            if self.bias is not None and bias is not None:
                self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))

    def fuse_decoder_diagonal(
        self,
        row_scale: Optional[Tensor] = None,
        col_scale: Optional[Tensor] = None,
        inverse: bool = True,
    ) -> None:
        row = self._normalize_left_scale(row_scale) if row_scale is not None else None
        col = self._normalize_right_scale(col_scale) if col_scale is not None else None
        with torch.no_grad():
            if inverse:
                if col is not None:
                    self.right_diag.div_(col)
                if row is not None:
                    self.left_diag.div_(row)
            else:
                if col is not None:
                    self.right_diag.mul_(col)
                if row is not None:
                    self.left_diag.mul_(row)

    @classmethod
    def from_linear(
        cls,
        linear: Module,
        *,
        decoder_type: str = "rht",
        entropy_bottleneck_kwargs: Optional[dict[str, Any]] = None,
        rht_seed: Optional[int] = 0,
    ) -> "EntropyConstrainedLinear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            decoder_type=decoder_type,
            entropy_bottleneck_kwargs=entropy_bottleneck_kwargs,
            rht_seed=rht_seed,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        layer.initialize_from_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None,
        )
        return layer

    def aux_loss(self) -> Tensor:
        return self.entropy_bottleneck.loss()

    def update_entropy_model(self, force: bool = False, update_quantiles: bool = False) -> bool:
        return self.entropy_bottleneck.update(force=force)

    def _quantize_and_rate_chunk(
        self, chunk: Tensor, mode: str, training: bool, effective_qs: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Process one row-chunk: quantize + compute rate sum. Used inside grad_checkpoint."""
        eb = self.entropy_bottleneck
        if mode == "ste":
            median = eb._get_median()
            if effective_qs is not None:
                hard = torch.round((chunk - median) / effective_qs) * effective_qs + median
            else:
                hard = torch.round(chunk - median) + median
            quantized = hard.detach() + chunk - chunk.detach()
            likelihoods, _, _ = eb._likelihood(quantized, qs=effective_qs)
        else:
            if effective_qs is not None:
                quantized, likelihoods = eb(chunk, training=training, qs=effective_qs)
            else:
                quantized, likelihoods = eb(chunk, training=training)
        rate_sum = (-torch.log2(likelihoods.float().clamp_min(1e-9))).sum()
        return quantized, rate_sum

    def quantize_latent(
        self,
        training: Optional[bool] = None,
        qs=None,
        mode: Optional[str] = None,
        return_likelihoods: bool = True,
    ) -> tuple[Tensor, Optional[Tensor]]:
        mode = mode if mode is not None else getattr(self, "quantize_mode", "noise")
        if mode not in ("noise", "ste"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')
        if training is None:
            training = self.training

        effective_qs = qs if qs is not None else getattr(self, "qs", None)
        latent = self.latent.to(device=self._entropy_device(), dtype=self._entropy_dtype())

        chunk_rows = int(getattr(self, "entropy_chunk_rows", 0) or 0)
        H = latent.shape[0]
        use_chunking = chunk_rows > 0 and H > chunk_rows

        if use_chunking:
            use_ckpt = bool(getattr(self, "entropy_grad_ckpt", True))
            quantized_chunks = []
            rate_sum = torch.zeros((), device=latent.device, dtype=torch.float32)
            total_numel = 0
            for start in range(0, H, chunk_rows):
                end = min(start + chunk_rows, H)
                chunk = latent[start:end]
                if use_ckpt:
                    q_chunk, r_chunk = grad_checkpoint(
                        self._quantize_and_rate_chunk,
                        chunk, mode, training, effective_qs,
                        use_reentrant=False,
                    )
                else:
                    q_chunk, r_chunk = self._quantize_and_rate_chunk(
                        chunk, mode, training, effective_qs,
                    )
                quantized_chunks.append(q_chunk)
                rate_sum = rate_sum + r_chunk
                total_numel += chunk.numel()
            quantized = torch.cat(quantized_chunks, dim=0)
            self._last_rate_loss = rate_sum / float(max(total_numel, 1))
        else:
            q, rate_sum = self._quantize_and_rate_chunk(latent, mode, training, effective_qs)
            quantized = q
            self._last_rate_loss = rate_sum / float(max(latent.numel(), 1))

        quantized = quantized.to(device=self.latent.device, dtype=self.latent.dtype)
        return quantized, None

    def reconstruct_weight(
        self,
        training: Optional[bool] = None,
        quantized: bool = True,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if quantized:
            latent, likelihoods = self.quantize_latent(training=training)
        else:
            latent, likelihoods = self.latent, None
        return self.decode_latent(latent), likelihoods

    def estimated_bits(self, likelihoods: Optional[Tensor] = None) -> Tensor:
        if likelihoods is None:
            _, likelihoods = self.quantize_latent(training=self.training)
        assert likelihoods is not None
        return (-torch.log2(likelihoods.float().clamp_min(1e-9))).sum()

    def estimated_bits_per_parameter(self, likelihoods: Optional[Tensor] = None) -> Tensor:
        return self.estimated_bits(likelihoods) / self.latent.numel()

    def decoding_parameter_bit_breakdown(
        self,
        include_entropy_medians: bool = True,
        force_update_entropy_tables: bool = False,
    ) -> dict[str, int]:
        if force_update_entropy_tables and self.entropy_bottleneck._offset.numel() == 0:
            self.update_entropy_model(force=False, update_quantiles=False)
        eb = self.entropy_bottleneck
        bits: dict[str, int] = {
            "left_diag": self._tensor_num_bits(self.left_diag),
            "right_diag": self._tensor_num_bits(self.right_diag),
            "mixture_params": self._tensor_num_bits(eb._logits) + self._tensor_num_bits(eb._means) + self._tensor_num_bits(eb._log_scales),
        }
        bits["total"] = int(sum(bits.values()))
        return bits

    def decoding_parameter_bits(
        self,
        include_entropy_medians: bool = True,
        force_update_entropy_tables: bool = False,
    ) -> int:
        return self.decoding_parameter_bit_breakdown(
            include_entropy_medians=include_entropy_medians,
            force_update_entropy_tables=force_update_entropy_tables,
        )["total"]

    def compress_latent(
        self,
        force_update: bool = True,
        update_quantiles: bool = False,
        qs: Optional[Tensor] = None,
    ) -> dict[str, Any]:
        """Compress latent via dietgpu (byte-level rANS).

        Returns dict with:
          - strings: [torch.uint8 tensor on CPU] — compressed bitstream
          - shape: latent shape
          - meta: {offset, dtype, numel} — needed for decompress
          - num_bits: actual bitstream size in bits
          - num_bits_estimated: theoretical rate under mixture CDF (for reference)
        """
        if force_update or self.entropy_bottleneck._offset.numel() == 0:
            self.update_entropy_model(force=force_update, update_quantiles=update_quantiles)

        effective_qs = qs if qs is not None else getattr(self, "qs", None)
        eb = self.entropy_bottleneck
        latent = self.latent.detach().to(device=self._entropy_device(), dtype=self._entropy_dtype())

        # estimate (theoretical) rate for comparison
        median = eb._get_median()
        if effective_qs is not None:
            quantized = torch.round((latent - median) / effective_qs) * effective_qs + median
            log2_lik = eb.log_likelihood_quantized(quantized, effective_qs)
        else:
            quantized = torch.round(latent - median) + median
            log2_lik = eb.log_likelihood_quantized(
                quantized, torch.ones((), device=latent.device, dtype=latent.dtype),
            )
        estimated_bits = int((-log2_lik).sum().item())

        # actual dietgpu bitstream
        pack = eb.compress(latent, qs=effective_qs)
        pack["shape"] = tuple(self.latent.shape)
        pack["num_bits_estimated"] = estimated_bits
        return pack

    def decompress_latent(
        self,
        strings: list[bytes],
        shape: Optional[tuple[int, int]] = None,
        device=None,
        dtype=None,
        qs: Optional[Tensor] = None,
        meta: Optional[dict] = None,
    ) -> Tensor:
        """Decompress dietgpu bitstream back to latent tensor."""
        if meta is None:
            raise ValueError("decompress_latent requires meta dict from compress_latent()")
        target_dev = device if device is not None else self.latent.device
        target_dtype = dtype if dtype is not None else self.latent.dtype
        effective_qs = qs if qs is not None else getattr(self, "qs", None)
        shape = shape or tuple(self.latent.shape)

        latent = self.entropy_bottleneck.decompress(
            strings, size=shape, meta=meta, qs=effective_qs, device=target_dev,
        )
        return latent.to(device=target_dev, dtype=target_dtype)

    def decompress_weight(
        self,
        strings: list[bytes],
        shape: Optional[tuple[int, int]] = None,
        device=None,
        dtype=None,
        meta: Optional[dict] = None,
    ) -> Tensor:
        latent = self.decompress_latent(
            strings, shape=shape, device=device, dtype=dtype, meta=meta,
        )
        return self.decode_latent(latent)

    def prepare_for_inference(self, pack: dict) -> None:
        """Call once after compress_latent. Fuses qs into right_diag,
        pre-allocates decompress buffers. Subsequent forward_from_bitstream
        calls skip qs multiply (58M fewer ops) and buffer allocation."""
        dev = self.latent.device
        qs = getattr(self, "qs", None)
        self.entropy_bottleneck.prepare_for_inference(
            pack["strings"], pack["meta"], qs=qs, device=dev,
        )
        qs_val = float(qs) if qs is not None else 1.0
        self._inf_right_diag = (self.right_diag * qs_val).to(dev, dtype=torch.float32)
        self._inf_pack = pack

    def forward_from_bitstream(
        self,
        input: Tensor,
        strings: list[bytes] = None,
        shape: Optional[tuple[int, int]] = None,
        qs: Optional[Tensor] = None,
        meta: Optional[dict] = None,
    ) -> Tensor:
        # If prepared, use cached pack
        if strings is None and hasattr(self, "_inf_pack"):
            p = self._inf_pack
            strings, shape, meta = p["strings"], p["shape"], p["meta"]
            qs = getattr(self, "qs", None)
        latent = self.decompress_latent(
            strings, shape=shape, device=input.device,
            dtype=self.latent.dtype, qs=qs, meta=meta,
        )
        return self._forward_with_latent(input, latent)

    def _forward_with_latent(self, input: Tensor, latent: Tensor) -> Tensor:
        original_shape = input.shape[:-1]
        work_dtype = torch.float32
        x = input.reshape(-1, self.in_features).to(device=input.device, dtype=work_dtype)
        latent = latent.to(device=input.device, dtype=work_dtype)

        # Use qs-fused right_diag if prepared for inference
        rd = getattr(self, "_inf_right_diag", None)
        if rd is None:
            rd = self.right_diag.to(device=input.device, dtype=work_dtype)
        x = x * rd
        if self.decoder_type == "rht":
            x = matmul_hadUt_cuda(x, self._hadK_in, self._K_in)
        else:
            x = self._orthogonal_transform(x)
        x = x @ latent.T
        x = self._orthogonal_transform(x)
        x = x * self.left_diag.to(device=input.device, dtype=work_dtype)

        output = x.reshape(*original_shape, self.out_features).to(dtype=input.dtype)
        if self.bias is not None:
            output = output + self.bias.to(device=input.device, dtype=input.dtype)
        return output

    def forward(
        self,
        input: Tensor,
        return_likelihoods: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        latent, likelihoods = self.quantize_latent(
            training=self.training,
            return_likelihoods=True,
        )
        output = self._forward_with_latent(input, latent)
        if return_likelihoods:
            assert likelihoods is not None
            return output, likelihoods
        return output

    def extra_repr(self) -> str:
        eb = self.entropy_bottleneck
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, decoder_type={self.decoder_type}, "
            f"mixture=G{eb.num_gaussian}+L{eb.num_laplacian}"
        )


entropy_constrained_linear = EntropyConstrainedLinear
Linear = EntropyConstrainedLinear
