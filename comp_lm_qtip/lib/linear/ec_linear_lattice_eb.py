import math
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

from compressai.entropy_models import EntropyBottleneck

from lib.utils.matmul_had import get_hadK, matmul_hadU_cuda, matmul_hadUt_cuda


def _real_hartley_transform(x: Tensor, dim: int = -1) -> Tensor:
    spectrum = torch.fft.fft(x, dim=dim, norm="ortho")
    return spectrum.real - spectrum.imag


class EntropyConstrainedLinear(Module):
    """Entropy-constrained Linear with OLVQ-style lattice quantization and
    compressai's EntropyBottleneck (channels=lattice_dim) for rate modeling.

    Babai's rounding technique (OLVQ, NeurIPS 2024):
        m   = v · B^{-T}          (flat latent → index in learnable lattice basis)
        m_q = round(m)            (or + uniform noise during training)
        q   = m_q · B^T           (reconstruct in original basis)

    The n integer coordinates in each vector are modeled by a per-dimension
    cumulative-logistic density (compressai `EntropyBottleneck(channels=n)`),
    and compressed via compressai's range coder. An orthogonality regularizer
    on B's columns (OLVQ Eq. 9) is exposed via `orthogonality_loss()` and
    scaled into `aux_loss()` by `lambda_ortho`.
    """

    __constants__ = ["in_features", "out_features", "decoder_type", "lattice_dim"]
    in_features: int
    out_features: int
    lattice_dim: int
    latent: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        decoder_type: str = "rht",
        lattice_dim: int = 4,
        entropy_bottleneck_kwargs: Optional[dict[str, Any]] = None,
        rht_seed: Optional[int] = 0,
        lambda_ortho: float = 0.0,
        B_init: str = "identity",
        B_init_scale: Optional[float] = None,
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
        total = in_features * out_features
        if total % lattice_dim != 0:
            raise ValueError(
                f"latent numel ({out_features}×{in_features}={total}) must be "
                f"divisible by lattice_dim ({lattice_dim})"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.decoder_type = decoder_type
        self.lattice_dim = lattice_dim
        self.rht_seed = rht_seed
        self.lambda_ortho = float(lambda_ortho)
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

        # Learnable lattice basis B ∈ R^{n×n}
        self.B = Parameter(
            self._init_B(lattice_dim, B_init, B_init_scale, dtype, device)
        )

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        eb_kwargs = entropy_bottleneck_kwargs or {}
        # compressai EB channels = lattice_dim (one univariate density per lattice axis)
        self.entropy_bottleneck = EntropyBottleneck(lattice_dim, **eb_kwargs)
        if device is not None:
            self.entropy_bottleneck = self.entropy_bottleneck.to(device=device)
        self.entropy_bottleneck = self.entropy_bottleneck.float()
        self._last_rate_loss: Optional[Tensor] = None

        self.reset_parameters()

    @staticmethod
    def _init_B(
        n: int,
        mode: str,
        scale: Optional[float],
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
    ) -> Tensor:
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        if mode == "identity":
            B = torch.eye(n, dtype=dtype, device=device)
            if scale is not None:
                B = B * float(scale)
            return B
        if mode == "orthogonal":
            # Random orthogonal matrix via QR: preserves volume (|det B|=1),
            # so lattice granularity matches identity (Z^n rotated).
            A = torch.randn(n, n, dtype=dtype, device=device)
            Q, R = torch.linalg.qr(A)
            # Ensure Q is a true rotation (det=+1 up to sign) by absorbing R diag signs
            Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
            if scale is not None:
                Q = Q * float(scale)
            return Q
        if mode == "uniform":
            # OLVQ Eq. 8 bound = 1/(n · S^{1/n}); user passes bound via `scale`.
            # Note: default bound=1/n makes det(B) very small → fine grid → high rate.
            # Prefer `orthogonal` unless reproducing paper semantics.
            bound = float(scale) if scale is not None else (1.0 / n)
            B = (torch.rand(n, n, dtype=dtype, device=device) * 2.0 - 1.0) * bound
            # Ensure non-singular start
            B = B + torch.eye(n, dtype=dtype, device=device) * 1e-3
            return B
        raise ValueError(f"Unknown B_init: {mode!r}")

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
                f"row_scale must have {self.out_features} elements, "
                f"got shape {tuple(row_scale.shape)}"
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
                f"col_scale must have {self.in_features} elements, "
                f"got shape {tuple(col_scale.shape)}"
            )
        return scale

    def _apply(self, fn):
        super()._apply(fn)
        self.entropy_bottleneck = self.entropy_bottleneck.to(
            device=self.latent.device
        )
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

    def _entropy_device(self) -> torch.device:
        return next(self.entropy_bottleneck.parameters()).device

    def _entropy_dtype(self) -> torch.dtype:
        return next(self.entropy_bottleneck.parameters()).dtype

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
            raise ValueError(
                f"weight shape must be {tuple(self.latent.shape)}, "
                f"got {tuple(weight.shape)}"
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
        self, weight: Tensor, bias: Optional[Tensor] = None
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
        row = (
            self._normalize_left_scale(row_scale) if row_scale is not None else None
        )
        col = (
            self._normalize_right_scale(col_scale)
            if col_scale is not None
            else None
        )
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
        lattice_dim: int = 4,
        entropy_bottleneck_kwargs: Optional[dict[str, Any]] = None,
        rht_seed: Optional[int] = 0,
        lambda_ortho: float = 0.0,
        B_init: str = "identity",
        B_init_scale: Optional[float] = None,
    ) -> "EntropyConstrainedLinear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            decoder_type=decoder_type,
            lattice_dim=lattice_dim,
            entropy_bottleneck_kwargs=entropy_bottleneck_kwargs,
            rht_seed=rht_seed,
            lambda_ortho=lambda_ortho,
            B_init=B_init,
            B_init_scale=B_init_scale,
        )
        layer.initialize_from_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None,
        )
        return layer

    def orthogonality_loss(self) -> Tensor:
        """Normalized off-diagonal Gram loss over B's columns (OLVQ Eq. 9)."""
        n = self.lattice_dim
        if n <= 1:
            return torch.tensor(0.0, device=self.B.device, dtype=self.B.dtype)
        B = self.B
        col_norms = B.norm(dim=0).clamp_min(1e-9)
        Bn = B / col_norms
        gram = Bn.T @ Bn
        off = gram - torch.diag(torch.diagonal(gram))
        return off.abs().sum() / float(n * (n - 1))

    def aux_loss(self) -> Tensor:
        # Only EB's quantile-fitting loss — this goes to aux_optimizer which
        # updates `.quantiles` params only.  `orthogonality_loss()` is consumed
        # separately by the main training loop (see ec_linear_ft.py).
        return self.entropy_bottleneck.loss()

    def update_entropy_model(
        self, force: bool = False, update_quantiles: bool = False
    ) -> bool:
        return self.entropy_bottleneck.update(
            force=force, update_quantiles=update_quantiles,
        )

    def _effective_B(self, qs: Optional[Tensor] = None) -> Tensor:
        """B scaled by qs (overall step scale) — on entropy device/dtype."""
        B = self.B.to(device=self._entropy_device(), dtype=self._entropy_dtype())
        effective_qs = qs if qs is not None else getattr(self, "qs", None)
        if effective_qs is not None:
            B = B * effective_qs
        return B

    def _latent_to_vectors(self, latent: Tensor) -> Tensor:
        """Flatten (out, in) → (M, n), M = out·in / n."""
        return latent.reshape(-1, self.lattice_dim)

    def _vectors_to_latent(self, v: Tensor) -> Tensor:
        return v.reshape(self.out_features, self.in_features)

    def _vectors_to_eb_input(self, v: Tensor) -> Tensor:
        """(M, n) → (1, n, M) for compressai EB (channels=n)."""
        return v.T.unsqueeze(0).contiguous()

    def _eb_output_to_vectors(self, x: Tensor) -> Tensor:
        """(1, n, M) → (M, n)."""
        return x.squeeze(0).T.contiguous()

    def _ste_forward_eb(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """STE-through-round with per-channel medians. x shape (1, n, M)."""
        eb = self.entropy_bottleneck
        # Replicate EB's internal permute: (N, C, ...) → (C, N, ...)
        perm = np.arange(len(x.shape))
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)]

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)  # (C, 1, N·...)

        medians = eb._get_medians()
        hard = torch.round(values - medians).detach() + medians
        outputs = hard + (values - values.detach())

        likelihood, _, _ = eb._likelihood(outputs)
        if eb.use_likelihood_bound:
            likelihood = eb.likelihood_lower_bound(likelihood)

        outputs = outputs.reshape(shape).permute(*inv_perm).contiguous()
        likelihood = likelihood.reshape(shape).permute(*inv_perm).contiguous()
        return outputs, likelihood

    def _quantize_and_rate_chunk(
        self,
        chunk_v: Tensor,
        B: Tensor,
        B_inv_T: Tensor,
        mode: str,
        training: bool,
    ) -> tuple[Tensor, Tensor]:
        """Chunk-level Babai + EB. chunk_v shape (chunk_rows, n)."""
        m = chunk_v @ B_inv_T  # (chunk_rows, n)
        m_eb = self._vectors_to_eb_input(m)  # (1, n, chunk_rows)

        if mode == "ste":
            m_q_eb, likelihoods_eb = self._ste_forward_eb(m_eb)
        else:  # "noise"
            m_q_eb, likelihoods_eb = self.entropy_bottleneck(m_eb, training=training)

        m_q = self._eb_output_to_vectors(m_q_eb)  # (chunk_rows, n)
        q_rec = m_q @ B.T  # (chunk_rows, n)
        rate_sum = (-torch.log2(likelihoods_eb.float().clamp_min(1e-9))).sum()
        return q_rec, rate_sum

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

        B = self._effective_B(qs)
        # B^{-T} = solve(B^T, I) — numerically more stable than inv(B).T
        I_n = torch.eye(B.size(0), device=B.device, dtype=B.dtype)
        B_inv_T = torch.linalg.solve(B.T, I_n)  # tiny (n, n)
        latent = self.latent.to(
            device=self._entropy_device(), dtype=self._entropy_dtype()
        )
        v = self._latent_to_vectors(latent)  # (M, n)

        chunk_rows = int(getattr(self, "entropy_chunk_rows", 0) or 0)
        M = v.shape[0]
        use_chunking = chunk_rows > 0 and M > chunk_rows

        if use_chunking:
            use_ckpt = bool(getattr(self, "entropy_grad_ckpt", True))
            q_chunks = []
            rate_sum = torch.zeros((), device=v.device, dtype=torch.float32)
            total_numel = 0
            for start in range(0, M, chunk_rows):
                end = min(start + chunk_rows, M)
                chunk = v[start:end]
                if use_ckpt:
                    q_chunk, r_chunk = grad_checkpoint(
                        self._quantize_and_rate_chunk,
                        chunk, B, B_inv_T, mode, training,
                        use_reentrant=False,
                    )
                else:
                    q_chunk, r_chunk = self._quantize_and_rate_chunk(
                        chunk, B, B_inv_T, mode, training,
                    )
                q_chunks.append(q_chunk)
                rate_sum = rate_sum + r_chunk
                total_numel += chunk.numel()
            q_vecs = torch.cat(q_chunks, dim=0)
            self._last_rate_loss = rate_sum / float(max(total_numel, 1))
        else:
            q_vecs, rate_sum = self._quantize_and_rate_chunk(
                v, B, B_inv_T, mode, training
            )
            self._last_rate_loss = rate_sum / float(max(v.numel(), 1))

        quantized = self._vectors_to_latent(q_vecs).to(
            device=self.latent.device, dtype=self.latent.dtype
        )
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
            self.quantize_latent(training=self.training)
            assert self._last_rate_loss is not None
            return self._last_rate_loss * float(self.latent.numel())
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

        eb = self.entropy_bottleneck
        bits: dict[str, int] = {
            "left_diag": self._tensor_num_bits(self.left_diag),
            "right_diag": self._tensor_num_bits(self.right_diag),
            "B": self._tensor_num_bits(self.B),
            "entropy_quantized_cdf": self._tensor_num_bits(
                getattr(eb, "_quantized_cdf", None)
            ),
            "entropy_cdf_length": self._tensor_num_bits(
                getattr(eb, "_cdf_length", None)
            ),
            "entropy_offset": self._tensor_num_bits(getattr(eb, "_offset", None)),
        }
        if include_entropy_medians:
            bits["entropy_quantiles_for_medians"] = self._tensor_num_bits(
                getattr(eb, "quantiles", None)
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
        """Babai-round → (1, n, M) EB input → compressai range coder."""
        if force_update or self.entropy_bottleneck._offset.numel() == 0:
            self.update_entropy_model(
                force=force_update, update_quantiles=update_quantiles,
            )

        B = self._effective_B(qs)
        I_n = torch.eye(B.size(0), device=B.device, dtype=B.dtype)
        B_inv_T = torch.linalg.solve(B.T, I_n)       # tiny (n, n)
        latent = self.latent.detach().to(
            device=self._entropy_device(), dtype=self._entropy_dtype()
        )
        v = self._latent_to_vectors(latent)          # (M, n)
        m = v @ B_inv_T                              # (M, n)
        m_eb = self._vectors_to_eb_input(m)          # (1, n, M)

        strings = self.entropy_bottleneck.compress(m_eb)
        M = m.shape[0]
        return {
            "strings": strings,
            "shape": tuple(self.latent.shape),
            "m_shape": (M, self.lattice_dim),
            "num_bits": sum(len(s) * 8 for s in strings),
        }

    def decompress_latent(
        self,
        strings: list[bytes],
        shape: Optional[tuple[int, int]] = None,
        device=None,
        dtype=None,
        qs: Optional[Tensor] = None,
        m_shape: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        shape = shape or tuple(self.latent.shape)
        if m_shape is None:
            m_shape = (shape[0] * shape[1] // self.lattice_dim, self.lattice_dim)
        M = m_shape[0]

        # compressai decompress: size = spatial dims (without batch/channel).
        # Our EB input was (1, n, M) → spatial = (M,).
        m_eb = self.entropy_bottleneck.decompress(strings, [M])  # (1, n, M)
        m_q = self._eb_output_to_vectors(m_eb)                   # (M, n)

        target_dev = device if device is not None else self.latent.device
        target_dtype = dtype if dtype is not None else self.latent.dtype
        B = self._effective_B(qs).to(device=target_dev, dtype=torch.float32)
        v = m_q.to(device=target_dev, dtype=torch.float32) @ B.T  # (M, n)
        latent = v.reshape(shape)
        return latent.to(device=target_dev, dtype=target_dtype)

    def decompress_weight(
        self,
        strings: list[bytes],
        shape: Optional[tuple[int, int]] = None,
        device=None,
        dtype=None,
        m_shape: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        latent = self.decompress_latent(
            strings, shape=shape, device=device, dtype=dtype, m_shape=m_shape,
        )
        return self.decode_latent(latent)

    def prepare_for_inference(self, pack: dict) -> None:
        """Pre-compute B^T (with qs fused) and cache the bitstream pack."""
        dev = self.latent.device
        qs = getattr(self, "qs", None)
        qs_val = float(qs) if qs is not None else 1.0
        self._inf_BT = (self.B.T * qs_val).to(dev, dtype=torch.float32).contiguous()
        self._inf_right_diag = self.right_diag.to(dev, dtype=torch.float32)
        self._inf_pack = pack
        self._inf_m_shape = pack.get(
            "m_shape",
            (self.latent.numel() // self.lattice_dim, self.lattice_dim),
        )
        self._inf_shape = pack.get("shape", tuple(self.latent.shape))

    def forward_from_bitstream(
        self,
        input: Tensor,
        strings: list[bytes] = None,
        shape: Optional[tuple[int, int]] = None,
        qs: Optional[Tensor] = None,
        m_shape: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        if strings is None and hasattr(self, "_inf_BT"):
            # Fast path: qs pre-fused into B^T, use cached pack
            M = self._inf_m_shape[0]
            m_eb = self.entropy_bottleneck.decompress(
                self._inf_pack["strings"], [M],
            )  # (1, n, M)
            m_q = self._eb_output_to_vectors(m_eb).to(
                device=input.device, dtype=torch.float32
            )
            v = m_q @ self._inf_BT
            latent = v.reshape(self._inf_shape)
            return self._forward_with_latent(input, latent)

        if strings is None and hasattr(self, "_inf_pack"):
            p = self._inf_pack
            strings, shape = p["strings"], p["shape"]
            m_shape = p.get("m_shape", self._inf_m_shape)
            qs = getattr(self, "qs", None)
        latent = self.decompress_latent(
            strings, shape=shape, device=input.device,
            dtype=self.latent.dtype, qs=qs, m_shape=m_shape,
        )
        return self._forward_with_latent(input, latent)

    def _forward_with_latent(self, input: Tensor, latent: Tensor) -> Tensor:
        original_shape = input.shape[:-1]
        work_dtype = torch.float32
        x = input.reshape(-1, self.in_features).to(
            device=input.device, dtype=work_dtype
        )
        latent = latent.to(device=input.device, dtype=work_dtype)

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
            output = output + self.bias.to(
                device=input.device, dtype=input.dtype
            )
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
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, decoder_type={self.decoder_type}, "
            f"lattice_dim={self.lattice_dim}, "
            f"eb=EntropyBottleneck(channels={self.lattice_dim}), "
            f"lambda_ortho={self.lambda_ortho}"
        )


entropy_constrained_linear = EntropyConstrainedLinear
Linear = EntropyConstrainedLinear
