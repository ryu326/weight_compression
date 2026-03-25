from __future__ import annotations

import sys
from pathlib import Path

import torch


ANS_UNIFORM_SHAPES = (
    (256, 256),
    (1024, 3072),
    (3072, 3072),
    (3072, 8192),
    (4096, 4096),
    (4096, 11008),
    (11008, 4096),
    (8192, 8192),
    (16384, 16384),
)
ANS_PROB_BITS = (9, 10, 11)
ANS_SCHEMA = (
    "(Tensor ans_words, Tensor ans_states, Tensor ans_stream_starts, "
    "Tensor ans_stream_words, Tensor ans_lookup_u16, Tensor x, float step_size, int symbol_offset) -> Tensor"
)
_ANS_OPS_REGISTERED = False


def _ensure_my_kernel_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    my_kernel_dir = repo_root / "my-kernal"
    if str(my_kernel_dir) not in sys.path:
        sys.path.insert(0, str(my_kernel_dir))


def _load_my_kernels():
    _ensure_my_kernel_path()
    try:
        import my_kernels  # type: ignore
    except Exception:
        return None
    return my_kernels


def _register_one(my_kernels, m: int, k: int, prob_bits: int) -> None:
    op_name = f"decompress_matvec_ans_uniform_{m}_1_{k}_{prob_bits}"
    qualname = f"quip_lib::{op_name}"
    ns = getattr(torch.ops, "quip_lib", None)
    if ns is not None and hasattr(ns, op_name):
        return
    torch.library.define(qualname, ANS_SCHEMA)

    @torch.library.register_fake(qualname)
    def _fake(
        ans_words: torch.Tensor,
        ans_states: torch.Tensor,
        ans_stream_starts: torch.Tensor,
        ans_stream_words: torch.Tensor,
        ans_lookup_u16: torch.Tensor,
        x: torch.Tensor,
        step_size: float,
        symbol_offset: int,
        _m: int = m,
    ) -> torch.Tensor:
        del ans_words, ans_states, ans_stream_starts, ans_stream_words, ans_lookup_u16, step_size, symbol_offset
        batch = int(x.shape[0]) if x.dim() == 2 else 1
        return torch.zeros((batch, _m), dtype=torch.float32, device=x.device)

    @torch.library.impl(qualname, "cuda")
    def _cuda(
        ans_words: torch.Tensor,
        ans_states: torch.Tensor,
        ans_stream_starts: torch.Tensor,
        ans_stream_words: torch.Tensor,
        ans_lookup_u16: torch.Tensor,
        x: torch.Tensor,
        step_size: float,
        symbol_offset: int,
        _m: int = m,
        _k: int = k,
        _prob_bits: int = prob_bits,
    ) -> torch.Tensor:
        x2d = x.view(1, -1) if x.dim() == 1 else x
        if x2d.dim() != 2:
            raise RuntimeError(f"x must be 1D or 2D, got shape {tuple(x.shape)}")
        if int(x2d.shape[1]) != _k:
            raise RuntimeError(
                f"Input feature mismatch for {op_name}: expected {_k}, got {int(x2d.shape[1])}"
            )

        x2d_fp16 = x2d.to(torch.float16).contiguous()
        out = torch.empty((x2d_fp16.shape[0], _m), dtype=torch.float32, device=x2d_fp16.device)
        my_kernels.decompress_matvec_ans_uniform_batched(
            out,
            ans_words,
            ans_states,
            ans_stream_starts,
            ans_stream_words,
            ans_lookup_u16,
            x2d_fp16,
            float(step_size),
            int(symbol_offset),
            int(_prob_bits),
        )
        return out


def register_ans_ops() -> None:
    global _ANS_OPS_REGISTERED
    if _ANS_OPS_REGISTERED:
        return

    my_kernels = _load_my_kernels()
    if my_kernels is None:
        return

    for m, k in ANS_UNIFORM_SHAPES:
        for prob_bits in ANS_PROB_BITS:
            _register_one(my_kernels, m, k, prob_bits)
    _ANS_OPS_REGISTERED = True


register_ans_ops()
