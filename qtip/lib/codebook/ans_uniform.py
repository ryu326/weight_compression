from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch import nn


def _ensure_my_kernel_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    my_kernel_dir = repo_root / "my-kernal"
    if not my_kernel_dir.exists():
        raise ImportError(
            f"Cannot find my-kernal directory at {my_kernel_dir}. "
            "Expected qtip/my-kernal to exist."
        )
    if str(my_kernel_dir) not in sys.path:
        sys.path.insert(0, str(my_kernel_dir))
    return my_kernel_dir


def _load_ans_runtime() -> Tuple[object, Callable]:
    _ensure_my_kernel_path()

    try:
        import my_kernels  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Failed to import my_kernels. Build it first:\n"
            "cd qtip/my-kernal && python setup.py build_ext --inplace"
        ) from exc

    try:
        from ans_uniform_encoder import encode_uniform_2d_to_ans  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Failed to import ans_uniform_encoder from qtip/my-kernal. "
            "Check that qtip/my-kernal/ans_uniform_encoder.py exists."
        ) from exc

    return my_kernels, encode_uniform_2d_to_ans


class ans_uniform_codebook(nn.Module):
    """
    Host-side ANS container + GPU fused matvec launcher.

    Workflow:
    1) `encode_weight(weight)` once
    2) call `forward(x)` for inference
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prob_bits: int = 9,
        step_size: float = 0.02,
        dtype: torch.dtype = torch.float16,
        use_kernel: bool = True,
        cache_hatw: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.internal_dtype = dtype
        self.use_kernel = bool(use_kernel)
        self.cache_hatw = bool(cache_hatw)

        self.register_buffer("prob_bits", torch.tensor(int(prob_bits), dtype=torch.int32))
        self.register_buffer("step_size", torch.tensor(float(step_size), dtype=torch.float32))
        self.register_buffer("symbol_offset", torch.tensor(0, dtype=torch.int32))

        self.register_buffer("ans_words", torch.empty(0, dtype=torch.int16))
        self.register_buffer("ans_states", torch.empty(0, dtype=torch.int32))
        self.register_buffer("ans_stream_starts", torch.empty(0, dtype=torch.int32))
        self.register_buffer("ans_stream_words", torch.empty(0, dtype=torch.int32))
        self.register_buffer("ans_lookup_u16", torch.empty(0, dtype=torch.int64))

        # Optional dense fallback for gradient or non-kernel execution.
        self.register_buffer("hatW", torch.empty(0, dtype=self.internal_dtype))

        self._my_kernels = None
        self._encode_uniform_2d_to_ans = None

    def _lazy_runtime(self) -> None:
        if self._my_kernels is None or self._encode_uniform_2d_to_ans is None:
            self._my_kernels, self._encode_uniform_2d_to_ans = _load_ans_runtime()
            try:
                from lib.codebook import ans_ops

                ans_ops.register_ans_ops()
            except Exception:
                # torch.ops path is an optimization; direct my_kernels path still works.
                pass

    def _resolve_torch_op(self, prob_bits: int) -> Optional[Callable]:
        op_name = (
            f"decompress_matvec_ans_uniform_"
            f"{self.out_features}_1_{self.in_features}_{int(prob_bits)}"
        )
        ns = getattr(torch.ops, "quip_lib", None)
        if ns is None or not hasattr(ns, op_name):
            return None
        return getattr(ns, op_name)

    @property
    def is_encoded(self) -> bool:
        return self.ans_states.numel() > 0 and self.ans_lookup_u16.numel() > 0

    @torch.no_grad()
    def encode_weight(
        self,
        weight: torch.Tensor,
        step_size: Optional[float] = None,
        prob_bits: Optional[int] = None,
    ) -> None:
        """
        Encode dense weight [out_features, in_features] to ANS buffers.
        """
        if weight.dim() != 2:
            raise ValueError(f"weight must be 2D, got {tuple(weight.shape)}")
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape mismatch: expected {(self.out_features, self.in_features)}, "
                f"got {tuple(weight.shape)}"
            )

        pb = int(self.prob_bits.item() if prob_bits is None else prob_bits)
        ss = float(self.step_size.item() if step_size is None else step_size)
        if pb not in (9, 10, 11):
            raise ValueError(f"prob_bits must be in {{9,10,11}}, got {pb}")
        if ss <= 0:
            raise ValueError(f"step_size must be > 0, got {ss}")

        self._lazy_runtime()

        target_device = weight.device if weight.is_cuda else torch.device("cuda")
        encoded = self._encode_uniform_2d_to_ans(
            weight_2d=weight,
            step_size=ss,
            prob_bits=pb,
            device=target_device,
        )

        self.prob_bits = torch.tensor(pb, device=target_device, dtype=torch.int32)
        self.step_size = torch.tensor(ss, device=target_device, dtype=torch.float32)
        self.symbol_offset = torch.tensor(
            int(encoded.symbol_offset), device=target_device, dtype=torch.int32
        )

        self.ans_words = encoded.ans_words
        self.ans_states = encoded.ans_states
        self.ans_stream_starts = encoded.ans_stream_starts
        self.ans_stream_words = encoded.ans_stream_words
        self.ans_lookup_u16 = encoded.ans_lookup_u16

        if self.cache_hatw:
            # Differentiable fallback path.
            hatw = (torch.round(weight.detach().to(torch.float32) / ss) * ss).to(self.internal_dtype)
            self.hatW = hatw.contiguous().to(target_device)
        else:
            self.hatW = torch.empty(0, device=target_device, dtype=self.internal_dtype)

    def _kernel_matvec(self, x2d: torch.Tensor) -> torch.Tensor:
        """
        x2d: [batch, in_features], float32/float16 on CUDA
        returns: [batch, out_features], float32
        """
        if not self.is_encoded:
            raise RuntimeError("ANS buffers are empty. Call encode_weight() first.")
        if not x2d.is_cuda:
            raise RuntimeError("Kernel path requires CUDA input.")

        pb = int(self.prob_bits.item())
        ss = float(self.step_size.item())
        so = int(self.symbol_offset.item())
        x2d_fp16 = x2d.to(torch.float16).contiguous()

        op = self._resolve_torch_op(pb)
        if op is not None:
            return op(
                self.ans_words,
                self.ans_states,
                self.ans_stream_starts,
                self.ans_stream_words,
                self.ans_lookup_u16,
                x2d_fp16,
                ss,
                so,
            )

        self._lazy_runtime()

        batch = x2d.shape[0]
        out = torch.empty((batch, self.out_features), dtype=torch.float32, device=x2d.device)

        if hasattr(self._my_kernels, "decompress_matvec_ans_uniform_batched"):
            self._my_kernels.decompress_matvec_ans_uniform_batched(
                out,
                self.ans_words,
                self.ans_states,
                self.ans_stream_starts,
                self.ans_stream_words,
                self.ans_lookup_u16,
                x2d_fp16,
                ss,
                so,
                pb,
            )
            return out

        for i in range(batch):
            x_col = x2d_fp16[i : i + 1].T.contiguous()
            out_col = torch.empty((self.out_features, 1), dtype=torch.float32, device=x2d.device)
            self._my_kernels.decompress_matvec_ans_uniform(
                out_col,
                self.ans_words,
                self.ans_states,
                self.ans_stream_starts,
                self.ans_stream_words,
                self.ans_lookup_u16,
                x_col,
                ss,
                so,
                pb,
            )
            out[i].copy_(out_col[:, 0])

        return out

    def forward(self, x2d: torch.Tensor) -> torch.Tensor:
        """
        x2d: [batch, in_features]
        """
        if x2d.dim() != 2:
            raise ValueError(f"x2d must be 2D, got {tuple(x2d.shape)}")
        if x2d.shape[-1] != self.in_features:
            raise ValueError(
                f"Input feature mismatch: expected {self.in_features}, got {x2d.shape[-1]}"
            )

        # Prefer dense fallback when available for autograd / non-CUDA.
        if self.hatW.numel() > 0 and (x2d.requires_grad or not self.use_kernel or not x2d.is_cuda):
            return (x2d.to(self.hatW.dtype) @ self.hatW.T).to(torch.float32)

        if not self.use_kernel:
            raise RuntimeError(
                "use_kernel=False but no cached dense weight available. "
                "Set cache_hatw=True before encode_weight()."
            )

        return self._kernel_matvec(x2d)


class ANSUniformLinear(nn.Module):
    """
    BitshiftLinear-like wrapper around `ans_uniform_codebook`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prob_bits: int = 9,
        step_size: float = 0.02,
        dtype: torch.dtype = torch.float16,
        use_kernel: bool = True,
        cache_hatw: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.cb = ans_uniform_codebook(
            in_features=in_features,
            out_features=out_features,
            prob_bits=prob_bits,
            step_size=step_size,
            dtype=dtype,
            use_kernel=use_kernel,
            cache_hatw=cache_hatw,
        )

    @torch.no_grad()
    def encode_weight(
        self,
        weight: torch.Tensor,
        step_size: Optional[float] = None,
        prob_bits: Optional[int] = None,
    ) -> None:
        self.cb.encode_weight(weight, step_size=step_size, prob_bits=prob_bits)

    def forward(self, x2d: torch.Tensor) -> torch.Tensor:
        return self.cb(x2d)
