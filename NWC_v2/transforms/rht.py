"""RHT transform: 2D RHT over the last two dims.

Reshapes the last two dims `(L, I)` of the input to `(rht_rows, rht_cols)`
(both powers of 2; default 128×128 for the standard `(1024, 16)` sample
shape) and applies an independent ±1 sign diagonal + 1D Hadamard on each
of the row and column axes:

    Y = scale · H_R · D_R · X · D_C · H_C  +  shift

where `D_R, D_C ∈ {-1, +1}^{rows, cols}` are learnable per-element diags
(init ±1) and `scale, shift` are learnable scalars (init 1, 0).
Output shape matches input.
"""
from typing import Optional

import torch
import torch.nn as nn


def _try_fast_hadamard():
    try:
        from fast_hadamard_transform import hadamard_transform  # type: ignore
        return hadamard_transform
    except Exception:
        return None


def _try_qtip_hadU():
    try:
        from lib.utils.matmul_had import matmul_hadU  # type: ignore
        return matmul_hadU
    except Exception:
        try:  # alternate path when comp_lm_qtip is on sys.path differently
            import sys
            sys.path.insert(0, "/home/jgryu/workspace/weight_compression/comp_lm_qtip")
            from lib.utils.matmul_had import matmul_hadU  # type: ignore
            return matmul_hadU
        except Exception:
            return None


_FAST_HAD = _try_fast_hadamard()
_QTIP_HAD = None  # lazy: don't import until first call (avoids module-load failures)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class RHTTransform(nn.Module):
    """2D RHT applied on the last two dims, reshaped to (rows, cols)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rht_rows: int = 128,
        rht_cols: int = 128,
        seed: Optional[int] = 0,
    ):
        super().__init__()
        if in_dim != out_dim:
            raise ValueError(
                f"rht transform requires in_dim == out_dim, "
                f"got in_dim={in_dim} out_dim={out_dim}"
            )
        if not _is_pow2(rht_rows) or not _is_pow2(rht_cols):
            raise ValueError(
                f"rht requires rht_rows and rht_cols to be powers of 2, "
                f"got rows={rht_rows} cols={rht_cols}"
            )
        self.in_dim = in_dim
        self.rows = rht_rows
        self.cols = rht_cols

        if seed is None:
            sign_r = torch.randint(0, 2, (rht_rows,))
            sign_c = torch.randint(0, 2, (rht_cols,))
        else:
            g = torch.Generator(device="cpu").manual_seed(seed)
            sign_r = torch.randint(0, 2, (rht_rows,), generator=g)
            sign_c = torch.randint(0, 2, (rht_cols,), generator=g)
        sign_r = sign_r.mul(2).sub(1).to(torch.float32)
        sign_c = sign_c.mul(2).sub(1).to(torch.float32)
        # Independent learnable per-element diags on row and col axes.
        self.diag_row = nn.Parameter(sign_r)
        self.diag_col = nn.Parameter(sign_c)
        # Learnable scalar scale + shift on the output.  Identity at init.
        self.scale = nn.Parameter(torch.ones(()))
        self.shift = nn.Parameter(torch.zeros(()))

    def _hadamard_lastdim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Hadamard transform along the last dim (size = `dim`)."""
        global _QTIP_HAD
        if _FAST_HAD is not None:
            scale = dim ** (-0.5)
            return _FAST_HAD(x.contiguous(), scale)
        if _QTIP_HAD is None:
            _QTIP_HAD = _try_qtip_hadU()
        if _QTIP_HAD is not None:
            orig_shape = x.shape
            x_flat = x.reshape(-1, orig_shape[-1])
            y = _QTIP_HAD(x_flat)
            return y.reshape(orig_shape)
        raise RuntimeError(
            "RHT requires either `fast_hadamard_transform` package or qtip's "
            "registered `hadamard::hadamard` CUDA op; neither is available."
        )

    def _check_2d_shape(self, x: torch.Tensor):
        if x.dim() < 2:
            raise ValueError(f"rht expects at least 2D input, got shape {tuple(x.shape)}")
        L, I = x.shape[-2], x.shape[-1]
        if L * I != self.rows * self.cols:
            raise ValueError(
                f"rht: last two dims (L={L}, I={I}) → L*I={L*I} must equal "
                f"rows*cols={self.rows*self.cols}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = scale · H_R · D_R · X · D_C · H_C  +  shift  (last 2 dims as rows×cols)."""
        self._check_2d_shape(x)
        orig_shape = x.shape
        X = x.reshape(*orig_shape[:-2], self.rows, self.cols)
        # apply both diags
        X = X * self.diag_row.view(self.rows, 1) * self.diag_col.view(1, self.cols)
        # Hadamard on cols (last dim)
        X = self._hadamard_lastdim(X, self.cols)
        # Hadamard on rows (transpose → last dim → transpose back)
        X = X.transpose(-1, -2).contiguous()
        X = self._hadamard_lastdim(X, self.rows)
        X = X.transpose(-1, -2).contiguous()
        # scalar affine
        X = self.scale * X + self.shift
        return X.reshape(orig_shape)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Exact inverse of `forward`.  Hadamards (1/√N normalized) are
        self-inverse; diags are inverted element-wise; affine is inverted as
        `(y − shift) / scale`."""
        self._check_2d_shape(y)
        orig_shape = y.shape
        Y = y.reshape(*orig_shape[:-2], self.rows, self.cols)
        # undo scalar affine
        Y = (Y - self.shift) / self.scale
        # undo Hadamard on rows
        Y = Y.transpose(-1, -2).contiguous()
        Y = self._hadamard_lastdim(Y, self.rows)
        Y = Y.transpose(-1, -2).contiguous()
        # undo Hadamard on cols
        Y = self._hadamard_lastdim(Y, self.cols)
        # undo diags (learnable → element-wise inverse)
        Y = Y / (self.diag_row.view(self.rows, 1) * self.diag_col.view(1, self.cols))
        return Y.reshape(orig_shape)

    def extra_repr(self) -> str:
        return f"rows={self.rows}, cols={self.cols}, in_dim={self.in_dim}"


class RHTInverse(nn.Module):
    """Decoder side of a tied RHT pair: shares parameters with the encoder
    (`base`) and its forward calls `base.inverse(y)`.  Constructed by the
    codec when both encoder_transform and decoder_transform are 'rht'."""

    def __init__(self, base: RHTTransform):
        super().__init__()
        self.base = base  # shared params (deduped by optimizer.parameters())

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.base.inverse(y)

    def extra_repr(self) -> str:
        return "tied (decoder = encoder.inverse)"


def build(
    in_dim: int,
    out_dim: int,
    rht_seed: Optional[int] = 0,
    rht_rows: int = 128,
    rht_cols: int = 128,
    **kwargs,
) -> nn.Module:
    return RHTTransform(
        in_dim=in_dim, out_dim=out_dim,
        rht_rows=rht_rows, rht_cols=rht_cols, seed=rht_seed,
    )
