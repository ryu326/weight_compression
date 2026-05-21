import math
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from compressai.entropy_models import EntropyBottleneck
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from lib.utils.matmul_had import get_hadK, matmul_hadU, matmul_hadUt


def _real_hartley_transform(x: Tensor, dim: int = -1) -> Tensor:
    spectrum = torch.fft.fft(x, dim=dim, norm="ortho")
    return spectrum.real - spectrum.imag


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
            # Match qtip's Hadamard path and fail early on unsupported sizes.
            get_hadK(in_features)
            get_hadK(out_features)

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

        entropy_bottleneck_kwargs = entropy_bottleneck_kwargs or {}
        self.entropy_bottleneck = EntropyBottleneck(1, **entropy_bottleneck_kwargs)
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
                self.left_diag.copy_(
                    torch.ones(
                        self.out_features,
                        dtype=self.left_diag.dtype,
                        device=self.left_diag.device,
                    )
                )
                self.right_diag.copy_(
                    torch.ones(
                        self.in_features,
                        dtype=self.right_diag.dtype,
                        device=self.right_diag.device,
                    )
                )
            else:
                self.left_diag.copy_(
                    self._make_random_sign(
                        self.out_features,
                        self.left_diag.dtype,
                        self.left_diag.device,
                        self.rht_seed,
                    )
                )
                self.right_diag.copy_(
                    self._make_random_sign(
                        self.in_features,
                        self.right_diag.dtype,
                        self.right_diag.device,
                        None if self.rht_seed is None else self.rht_seed + 1,
                    )
                )

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
        return next(self.entropy_bottleneck.parameters()).device

    def _entropy_dtype(self) -> torch.dtype:
        return next(self.entropy_bottleneck.parameters()).dtype

    def _orthogonal_transform(self, x: Tensor) -> Tensor:
        if self.decoder_type == "rht":
            return matmul_hadU(x)
        if self.decoder_type == "identity":
            return x
        return _real_hartley_transform(x, dim=-1)

    def _decode_with_rht(self, latent: Tensor) -> Tensor:
        x = matmul_hadU(latent)
        x = x * self.right_diag.unsqueeze(0)
        x = matmul_hadU(x.T).T
        x = x * self.left_diag.unsqueeze(1)
        return x

    def _encode_with_rht(self, weight: Tensor) -> Tensor:
        x = weight / self.left_diag.unsqueeze(1)
        x = x / self.right_diag.unsqueeze(0)
        x = matmul_hadUt(x.T).T
        x = matmul_hadUt(x)
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
            raise ValueError(
                f"weight shape must be {tuple(self.latent.shape)}, got {tuple(weight.shape)}"
            )
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

    def initialize_from_weight(
        self,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> None:
        with torch.no_grad():
            weight = weight.to(device=self.latent.device, dtype=self.latent.dtype)
            self.latent.copy_(self.encode_weight(weight))

            if self.bias is not None and bias is not None:
                self.bias.copy_(
                    bias.to(device=self.bias.device, dtype=self.bias.dtype)
                )

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

    def update_entropy_model(
        self, force: bool = False, update_quantiles: bool = False
    ) -> bool:
        return self.entropy_bottleneck.update(
            force=force, update_quantiles=update_quantiles
        )

    def _ste_forward_entropy(self, x: Tensor) -> tuple[Tensor, Tensor]:
        eb = self.entropy_bottleneck
        perm = np.arange(len(x.shape))
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)]

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        medians = eb._get_medians()
        hard = torch.round(values - medians).detach() + medians
        outputs = hard + (values - values.detach())

        likelihood, _, _ = eb._likelihood(outputs)
        if eb.use_likelihood_bound:
            likelihood = eb.likelihood_lower_bound(likelihood)

        outputs = outputs.reshape(shape).permute(*inv_perm).contiguous()
        likelihood = likelihood.reshape(shape).permute(*inv_perm).contiguous()
        return outputs, likelihood

    def _eb_forward_chunk(self, chunk: Tensor, training: bool) -> tuple[Tensor, Tensor]:
        return self.entropy_bottleneck(chunk, training=training)

    def _ste_chunk(self, v_chunk: Tensor) -> tuple[Tensor, Tensor]:
        eb = self.entropy_bottleneck
        medians = eb._get_medians()
        hard = torch.round(v_chunk - medians).detach() + medians
        out = hard + (v_chunk - v_chunk.detach())
        lik, _, _ = eb._likelihood(out)
        if eb.use_likelihood_bound:
            lik = eb.likelihood_lower_bound(lik)
        return out, lik

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

        effective_qs = qs if qs is not None else getattr(self, "qs", None)

        latent = self.latent
        if effective_qs is not None:
            latent = latent / effective_qs

        latent = self._latent_to_entropy_input(latent).to(
            device=self._entropy_device(), dtype=self._entropy_dtype()
        )

        chunk_rows = int(getattr(self, "entropy_chunk_rows", 0) or 0)
        use_chunking = chunk_rows > 0 and latent.ndim == 4 and latent.shape[2] > chunk_rows

        if use_chunking:
            quantized_chunks = []
            rate_bits_sum = torch.zeros((), device=latent.device, dtype=torch.float32)
            total_numel = 0
            for start in range(0, latent.shape[2], chunk_rows):
                end = min(start + chunk_rows, latent.shape[2])
                chunk = latent[:, :, start:end, :]
                if mode == "ste":
                    perm = np.arange(len(chunk.shape))
                    perm[0], perm[1] = perm[1], perm[0]
                    inv_perm = np.arange(len(chunk.shape))[np.argsort(perm)]
                    c = chunk.permute(*perm).contiguous()
                    c_shape = c.size()
                    values = c.reshape(c.size(0), 1, -1)
                    q_chunk, l_chunk = grad_checkpoint(
                        self._ste_chunk, values, use_reentrant=False,
                    )
                    q_chunk = q_chunk.reshape(c_shape).permute(*inv_perm).contiguous()
                    l_chunk = l_chunk.reshape(c_shape).permute(*inv_perm).contiguous()
                else:
                    q_chunk, l_chunk = grad_checkpoint(
                        self._eb_forward_chunk, chunk, training,
                        use_reentrant=False,
                    )
                quantized_chunks.append(q_chunk)
                rate_bits_sum = rate_bits_sum + (
                    -torch.log2(l_chunk.float().clamp_min(1e-9))
                ).sum()
                total_numel += int(l_chunk.numel())
            quantized = torch.cat(quantized_chunks, dim=2)
            self._last_rate_loss = rate_bits_sum / float(max(total_numel, 1))
            likelihoods = None
        else:
            if mode == "ste":
                quantized, likelihoods = self._ste_forward_entropy(latent)
            else:
                quantized, likelihoods = self.entropy_bottleneck(latent, training=training)
            self._last_rate_loss = (
                -torch.log2(likelihoods.float().clamp_min(1e-9))
            ).mean()

        if not return_likelihoods:
            likelihoods = None

        quantized = self._entropy_input_to_latent(quantized).to(
            device=self.latent.device, dtype=self.latent.dtype
        )
        if likelihoods is not None:
            likelihoods = self._entropy_input_to_latent(likelihoods).to(
                device=self.latent.device
            )

        if effective_qs is not None:
            quantized = quantized * effective_qs

        return quantized, likelihoods

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

    def estimated_bits_per_parameter(
        self, likelihoods: Optional[Tensor] = None
    ) -> Tensor:
        return self.estimated_bits(likelihoods) / self.latent.numel()

    def decoding_parameter_bit_breakdown(
        self,
        include_entropy_medians: bool = True,
        force_update_entropy_tables: bool = False,
    ) -> dict[str, int]:
        if force_update_entropy_tables and self.entropy_bottleneck._offset.numel() == 0:
            self.update_entropy_model(force=False, update_quantiles=False)

        entropy = self.entropy_bottleneck
        bits: dict[str, int] = {
            "left_diag": self._tensor_num_bits(self.left_diag),
            "right_diag": self._tensor_num_bits(self.right_diag),
            "entropy_quantized_cdf": self._tensor_num_bits(
                getattr(entropy, "_quantized_cdf", None)
            ),
            "entropy_cdf_length": self._tensor_num_bits(
                getattr(entropy, "_cdf_length", None)
            ),
            "entropy_offset": self._tensor_num_bits(getattr(entropy, "_offset", None)),
        }
        if include_entropy_medians:
            bits["entropy_quantiles_for_medians"] = self._tensor_num_bits(
                getattr(entropy, "quantiles", None)
            )
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
        if force_update or self.entropy_bottleneck._offset.numel() == 0:
            self.update_entropy_model(
                force=force_update, update_quantiles=update_quantiles
            )

        effective_qs = qs if qs is not None else getattr(self, "qs", None)
        latent = self.latent.detach()
        if effective_qs is not None:
            latent = latent / effective_qs

        latent = self._latent_to_entropy_input(latent).to(
            device=self._entropy_device(), dtype=self._entropy_dtype()
        )
        strings = self.entropy_bottleneck.compress(latent)
        return {
            "strings": strings,
            "shape": tuple(self.latent.shape),
            "num_bits": sum(len(string) * 8 for string in strings),
        }

    def decompress_latent(
        self,
        strings: list[bytes],
        shape: Optional[tuple[int, int]] = None,
        device=None,
        dtype=None,
        qs: Optional[Tensor] = None,
    ) -> Tensor:
        effective_qs = qs if qs is not None else getattr(self, "qs", None)
        shape = shape or tuple(self.latent.shape)
        latent = self.entropy_bottleneck.decompress(strings, shape)
        latent = self._entropy_input_to_latent(latent)
        if effective_qs is not None:
            latent = latent * effective_qs
        if device is None:
            device = self.latent.device
        if dtype is None:
            dtype = self.latent.dtype
        return latent.to(device=device, dtype=dtype)

    # def compress_weight(
    #     self,
    #     force_update: bool = True,
    #     update_quantiles: bool = False,
    # ) -> dict[str, Any]:
    #     out = self.compress_latent(
    #         force_update=force_update,
    #         update_quantiles=update_quantiles,
    #     )
    #     out["weight"] = self.decompress_weight(out["strings"], out["shape"])
    #     return out

    def decompress_weight(
        self,
        strings: list[bytes],
        shape: Optional[tuple[int, int]] = None,
        device=None,
        dtype=None,
    ) -> Tensor:
        latent = self.decompress_latent(
            strings,
            shape=shape,
            device=device,
            dtype=dtype,
        )
        return self.decode_latent(latent)

    def forward_from_bitstream(
        self,
        input: Tensor,
        strings: list[bytes],
        shape: Optional[tuple[int, int]] = None,
        qs: Optional[Tensor] = None,
    ) -> Tensor:
        latent = self.decompress_latent(
            strings,
            shape=shape,
            device=input.device,
            dtype=self.latent.dtype,
            qs=qs,
        )
        return self._forward_with_latent(input, latent)

    # ── nvcomp GPU codec path ──────────────────────────────────────────────
    # Compress-time: quantize latent to integer symbols, shift to unsigned,
    # encode with nvcomp (GPU).  Forward-time: nvcomp-decode → affine back to
    # float latent → normal _forward_with_latent.  qs is fused into right_diag
    # at prepare time (same trick as the dietgpu path) so decode is add-only.
    _nvcomp_codec_cache: dict[str, Any] = {}

    @classmethod
    def _get_nvcomp_codec(cls, algorithm: str):
        cache = cls._nvcomp_codec_cache
        if algorithm not in cache:
            from nvidia import nvcomp  # lazy import — only base env has it
            cache[algorithm] = nvcomp.Codec(algorithm=algorithm)
        return cache[algorithm]

    def _quantize_to_unsigned_symbols(
        self, qs: Optional[Tensor] = None,
    ) -> tuple[Tensor, int, torch.dtype, float]:
        """Quantize latent to integer EB-symbols and shift to unsigned dtype.

        Matches the pipeline in quantize_latent / compress_latent:
          x_eb = latent / qs ; symbols_eb = round(x_eb - median).
        Dequantized = (symbols_eb + median) * qs.

        Returns (shifted, offset_min, sym_dtype, median_float)."""
        effective_qs = qs if qs is not None else getattr(self, "qs", None)
        latent = self.latent.detach().to(
            device=self._entropy_device(), dtype=self._entropy_dtype()
        )
        if effective_qs is not None:
            latent = latent / float(effective_qs)
        # compressai EntropyBottleneck(channels=1) → single scalar median
        median = float(self.entropy_bottleneck._get_medians().reshape(()).item())
        symbols = torch.round(latent - median).to(torch.int64)
        mn = int(symbols.min().item())
        mx = int(symbols.max().item())
        K = mx - mn + 1
        sym_dtype = torch.uint8 if K <= 256 else torch.int16
        shifted = (symbols - mn).to(sym_dtype).contiguous()
        return shifted, mn, sym_dtype, median

    def compress_latent_nvcomp(
        self,
        algorithm: str = "Zstd",
        qs: Optional[Tensor] = None,
        force_update: bool = True,
        update_quantiles: bool = False,
    ) -> dict[str, Any]:
        """Quantize → nvcomp-encode on GPU.  Returns a pack with the encoded
        nvcomp.Array kept on-device for fast forward."""
        if force_update or self.entropy_bottleneck._offset.numel() == 0:
            self.update_entropy_model(force=force_update, update_quantiles=update_quantiles)

        from nvidia import nvcomp



        shifted, offset_min, sym_dtype, median = self._quantize_to_unsigned_symbols(qs=qs)
        flat = shifted.flatten().contiguous()
        # nvcomp.Array consumes uint8 via __cuda_array_interface__; for int16
        # we view bytes and restore the dtype at decode time.
        if sym_dtype == torch.int16:
            bytes_view = flat.view(torch.uint8)
        else:
            bytes_view = flat
        src_array = nvcomp.Array(bytes_view)
        codec = self._get_nvcomp_codec(algorithm)
        encoded = codec.encode(src_array)

        # Try to record encoded size (best-effort; codec APIs vary).
        num_bytes = -1
        try:
            num_bytes = int(encoded.buffer_size)
        except Exception:
            try:
                num_bytes = int(encoded.cpu().shape[0])
            except Exception:
                pass

        effective_qs = qs if qs is not None else getattr(self, "qs", None)
        qs_val = float(effective_qs) if effective_qs is not None else 1.0

        return {
            "algorithm": algorithm,
            "encoded": encoded,            # nvcomp.Array (GPU)
            "shape": tuple(self.latent.shape),
            "offset": offset_min,
            "median": median,
            "qs": qs_val,
            "sym_dtype": "uint8" if sym_dtype == torch.uint8 else "int16",
            "numel": int(shifted.numel()),
            "num_bytes": num_bytes,
            "num_bits": num_bytes * 8 if num_bytes >= 0 else None,
        }

    def decompress_latent_nvcomp(
        self,
        pack: dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Slow path: nvcomp-decode → recover the float latent tensor.
        Used for round-trip verification — forward_from_bitstream_nvcomp is
        the fast path that fuses this with the matmul."""
        codec = self._get_nvcomp_codec(pack["algorithm"])
        decoded = codec.decode(pack["encoded"])
        dev = device if device is not None else self.latent.device
        # nvcomp.Array → torch via __cuda_array_interface__
        buf = torch.as_tensor(decoded, device=dev).contiguous()
        if pack["sym_dtype"] == "int16":
            buf = buf.view(torch.int16)
        buf = buf[: pack["numel"]].to(torch.float32)
        # Dequantize: symbols = shifted + offset_min, then (symbols + median) * qs
        qs = float(pack["qs"])
        latent = (buf + float(pack["offset"]) + float(pack["median"])) * qs
        return latent.reshape(pack["shape"]).to(dtype=self.latent.dtype)

    def prepare_for_inference_nvcomp(self, pack: dict[str, Any]) -> None:
        """Cache codec + qs-fused right_diag so forward_from_bitstream_nvcomp
        does: decode → add-only bias → matmul (no qs multiply)."""
        dev = self.latent.device
        qs = float(pack["qs"])
        self._nvc_pack = pack
        self._nvc_codec = self._get_nvcomp_codec(pack["algorithm"])
        # Latent in "u-space" uses bias = offset_min + median (no qs); x side
        # carries the qs factor. Then x_rd @ (u + bias).T == x @ (u + bias) * qs
        # == x @ dequantized_latent.T.
        self._nvc_bias_noscale = float(pack["offset"]) + float(pack["median"])
        self._nvc_right_diag = (self.right_diag * qs).to(dev, dtype=torch.float32).contiguous()
        self._nvc_left_diag = self.left_diag.to(dev, dtype=torch.float32).contiguous()
        self._nvc_shape = pack["shape"]
        self._nvc_numel = int(pack["numel"])
        self._nvc_sym_dtype = torch.uint8 if pack["sym_dtype"] == "uint8" else torch.int16

    def forward_from_bitstream_nvcomp(
        self,
        input: Tensor,
        pack: Optional[dict[str, Any]] = None,
        skip_diag: bool = False,
    ) -> Tensor:
        """Fast path. Compute dtype follows `input.dtype` — fp16 input stays
        fp16 through decode+matmul (TensorCore eligible on A6000/H100),
        fp32 input stays fp32. Call after prepare_for_inference_nvcomp() to use
        the cached pack, or pass pack= explicitly.

        `skip_diag=True` drops the `x * right_diag` and `out * left_diag`
        element-wise multiplications (breaks numerical correctness — for
        latency profiling only)."""
        work_dtype = input.dtype
        if pack is None:
            if not hasattr(self, "_nvc_pack"):
                raise RuntimeError(
                    "call prepare_for_inference_nvcomp(pack) first, or pass pack= explicitly"
                )
            codec = self._nvc_codec
            encoded = self._nvc_pack["encoded"]
            sym_dtype = self._nvc_sym_dtype
            numel = self._nvc_numel
            shape = self._nvc_shape
            bias_noscale = self._nvc_bias_noscale
            rd = self._nvc_right_diag.to(dtype=work_dtype)
            ld = self._nvc_left_diag.to(dtype=work_dtype)
        else:
            codec = self._get_nvcomp_codec(pack["algorithm"])
            encoded = pack["encoded"]
            sym_dtype = torch.uint8 if pack["sym_dtype"] == "uint8" else torch.int16
            numel = int(pack["numel"])
            shape = pack["shape"]
            qs = float(pack["qs"])
            bias_noscale = float(pack["offset"]) + float(pack["median"])
            rd = (self.right_diag * qs).to(input.device, dtype=work_dtype).contiguous()
            ld = self.left_diag.to(input.device, dtype=work_dtype).contiguous()

        decoded = codec.decode(encoded)
        buf = torch.as_tensor(decoded, device=input.device).contiguous()
        if sym_dtype == torch.int16:
            buf = buf.view(torch.int16)
        u = buf[:numel].to(work_dtype)
        latent = (u + bias_noscale).reshape(shape)  # work_dtype

        original_shape = input.shape[:-1]
        x = input.reshape(-1, self.in_features)
        if not skip_diag:
            x = x * rd  # qs-fused
        if self.decoder_type == "rht":
            x = matmul_hadUt(x)
        else:
            x = self._orthogonal_transform(x)
        x = x @ latent.T
        x = self._orthogonal_transform(x)
        if not skip_diag:
            x = x * ld

        output = x.reshape(*original_shape, self.out_features)
        if self.bias is not None:
            output = output + self.bias.to(device=input.device, dtype=work_dtype)
        return output

    def _forward_with_latent(self, input: Tensor, latent: Tensor) -> Tensor:
        original_shape = input.shape[:-1]
        work_dtype = torch.float32
        x = input.reshape(-1, self.in_features).to(device=input.device, dtype=work_dtype)
        latent = latent.to(device=input.device, dtype=work_dtype)

        x = x * self.right_diag.to(device=input.device, dtype=work_dtype)
        if self.decoder_type == "rht":
            x = matmul_hadUt(x)
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
        # Always materialize likelihoods to keep behavior consistent with
        # pre-ecft_decoder versions where forward computed both outputs.
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
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, decoder_type={self.decoder_type}"
        )


entropy_constrained_linear = EntropyConstrainedLinear
Linear = EntropyConstrainedLinear
