"""ECFT (Entropy-Constrained Fine-Tuning) helpers extracted from finetune.py.

Covers:
  - scalar/formatter utilities used by ECFT logs
  - decoder-layer log writers (meta/step lines) for --ecft_decoder
  - rate/aux/cache helpers that iterate EntropyConstrainedLinear submodules
"""

import math
import os
from typing import Any, Iterable, Optional

import torch


GLOBAL_STD = 0.012528747320175171


def to_scalar(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.tolist()
    if hasattr(x, "item") and callable(x.item):
        try:
            return x.item()
        except Exception:
            return x
    return x


def fmt_value(x: Any, digits: int = 6) -> str:
    x = to_scalar(x)
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.{digits}f}"
    return str(x)


def scale_mse_for_log(x: Any) -> Any:
    x = to_scalar(x)
    if x is None:
        return None
    if isinstance(x, (int, float)):
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return xf
        return xf / GLOBAL_STD
    return x


def decoder_totals_from_out(out: Any) -> tuple[Optional[float], Optional[float]]:
    if not isinstance(out, dict):
        return None, None
    decoder_bits = out.get("decoder_bits", {})
    num_params = to_scalar(out.get("num_pixels"))
    decoder_total_raw = None
    if isinstance(decoder_bits, dict):
        decoder_total_raw = to_scalar(decoder_bits.get("total"))
    decoder_total: Any = None
    if (
        isinstance(decoder_total_raw, (int, float))
        and isinstance(num_params, (int, float))
        and float(num_params) > 0
    ):
        decoder_total = float(decoder_total_raw) / float(num_params)
    else:
        decoder_total = decoder_total_raw
    return decoder_total, decoder_total_raw


def dec_log_path(args, name: str) -> Optional[str]:
    save_root = getattr(args, "res_path", None) or getattr(args, "save_path", None)
    if not save_root:
        return None
    log_dir = os.path.join(save_root, "ecft_dec_log")
    os.makedirs(log_dir, exist_ok=True)
    safe_name = str(name).replace("/", "_")
    return os.path.join(log_dir, f"{safe_name}.log")


def dec_log_init(
    path: Optional[str],
    *,
    epochs: int,
    lmbda: float,
    adaptive: bool,
    target_rate: Any,
) -> None:
    if path is None:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "META "
            f"epochs={int(epochs)} "
            f"lmbda={float(lmbda):.8f} "
            f"adaptive={1 if adaptive else 0} "
            f"target_rate={fmt_value(target_rate)}\n"
        )
        f.write("STEP stage epoch loss mse_loss rate_loss aux_loss lambda_rate\n")


def dec_log_append(path: Optional[str], row: dict) -> None:
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            "STEP "
            f"{row.get('stage', '-')} "
            f"{fmt_value(row.get('epoch'))} "
            f"{fmt_value(row.get('loss'))} "
            f"{fmt_value(row.get('mse_loss'))} "
            f"{fmt_value(row.get('rate_loss'))} "
            f"{fmt_value(row.get('aux_loss'))} "
            f"{fmt_value(row.get('lambda_rate'))}\n"
        )


def rate_loss_from_modules(
    ec_modules: Iterable,
    *,
    training: bool,
    device,
) -> torch.Tensor:
    """Run quantize_latent on each module and return mean rate (bits/param)."""
    total_bits = torch.zeros((), device=device, dtype=torch.float32)
    total_numel = 0
    for module in ec_modules:
        module.quantize_latent(training=training, return_likelihoods=False)
        rate_loss = getattr(module, "_last_rate_loss", None)
        if rate_loss is None:
            _, likelihoods = module.quantize_latent(training=training)
            total_bits = total_bits + (
                -torch.log2(likelihoods.float().clamp_min(1e-9))
            ).sum()
            total_numel += int(likelihoods.numel())
        else:
            n = int(module.latent.numel())
            total_bits = total_bits + rate_loss.float() * float(n)
            total_numel += n
    if total_numel <= 0:
        return torch.zeros((), device=device, dtype=torch.float32)
    return total_bits / float(total_numel)


def rate_loss_from_cache(
    ec_modules: Iterable,
    *,
    device,
) -> torch.Tensor:
    """Read cached `_last_rate_loss` after a forward pass; recompute if missing."""
    total_bits = torch.zeros((), device=device, dtype=torch.float32)
    total_numel = 0
    for module in ec_modules:
        rate_loss = getattr(module, "_last_rate_loss", None)
        if rate_loss is None:
            module.quantize_latent(training=True, return_likelihoods=False)
            rate_loss = getattr(module, "_last_rate_loss", None)
        if rate_loss is None:
            _, likelihoods = module.quantize_latent(training=True)
            total_bits = total_bits + (
                -torch.log2(likelihoods.float().clamp_min(1e-9))
            ).sum()
            total_numel += int(likelihoods.numel())
        else:
            n = int(module.latent.numel())
            total_bits = total_bits + rate_loss.float() * float(n)
            total_numel += n
    if total_numel <= 0:
        return torch.zeros((), device=device, dtype=torch.float32)
    return total_bits / float(total_numel)


def aux_loss_from_modules(ec_modules: Iterable, *, device) -> torch.Tensor:
    aux = torch.zeros((), device=device, dtype=torch.float32)
    for module in ec_modules:
        aux = aux + module.aux_loss()
    return aux


def clear_forward_cache(ec_modules: Iterable) -> None:
    for module in ec_modules:
        module._last_rate_loss = None
