#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
COMP_LM_ROOT = WORKSPACE_ROOT / "comp_lm_qtip"

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(COMP_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(COMP_LM_ROOT))

from NWC.models import get_model
from lib import utils as nwc_utils
from lib.algo import nwc_refactory


LOGGER = logging.getLogger("compress_resnet_nwc")

WEIGHT_ENUMS = {
    "resnet18": models.ResNet18_Weights,
    "resnet34": models.ResNet34_Weights,
    "resnet50": models.ResNet50_Weights,
    "resnet101": models.ResNet101_Weights,
    "resnet152": models.ResNet152_Weights,
}


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


@dataclass
class PadInfo:
    original_rows: int
    original_cols: int
    padded_rows: int
    padded_cols: int
    pad_rows: int
    pad_cols: int


@dataclass
class HessianStats:
    sample_count: int
    batch_count: int
    input_dim: int


def _setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


def _expand_checkpoint_patterns(patterns: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            expanded.extend(matches)
            continue
        if any(char in pattern for char in "*?[]"):
            raise FileNotFoundError(
                f"No checkpoint matched pattern: {pattern}. "
                "If you used '...', replace it with the full real path."
            )
        if not os.path.isfile(pattern):
            raise FileNotFoundError(f"Checkpoint file does not exist: {pattern}")
        expanded.append(pattern)
    return expanded


def _validate_checkpoint_path(checkpoint_path: str) -> None:
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json next to checkpoint: {config_path}")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _load_resnet(arch: str, pretrained: bool) -> Tuple[nn.Module, object | None]:
    builder = getattr(models, arch)
    weights = WEIGHT_ENUMS[arch].DEFAULT if pretrained else None
    try:
        model = builder(weights=weights)
    except TypeError:
        model = builder(pretrained=pretrained)
    model.eval()
    return model, weights


def _should_use_identity_scale_shift(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, False)
        for name in (
            "layer_normalize",
            "row_normalize",
            "row_normalize2",
            "col_normalize",
            "global_normalize",
            "scaleH",
            "scaleHinv",
            "normalization_search",
        )
    )


def _build_eval_transform(weights) -> transforms.Compose:
    if weights is not None and hasattr(weights, "transforms"):
        return weights.transforms()
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _resolve_eval_dir(dataset_dir: str) -> str:
    val_dir = os.path.join(dataset_dir, "val")
    return val_dir if os.path.isdir(val_dir) else dataset_dir


def _build_image_loader(
    dataset_dir: str,
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> torch.utils.data.DataLoader:
    eval_dir = _resolve_eval_dir(dataset_dir)
    dataset = datasets.ImageFolder(eval_dir, transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _get_module_input_dim(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    if isinstance(module, nn.Conv2d):
        return int((module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1])
    raise TypeError(f"Unsupported module type for Hessian accumulation: {type(module)}")


def _accumulate_linear_hessian(
    hessian: torch.Tensor,
    module: nn.Linear,
    inputs: Tuple[torch.Tensor, ...],
) -> int:
    input_tensor = inputs[0].detach().reshape(-1, module.in_features)
    input_tensor = input_tensor.to(device=hessian.device, dtype=hessian.dtype)
    hessian.addmm_(input_tensor.t(), input_tensor)
    return int(input_tensor.shape[0])


def _accumulate_conv_hessian(
    hessian: torch.Tensor,
    module: nn.Conv2d,
    inputs: Tuple[torch.Tensor, ...],
) -> int:
    if module.groups != 1:
        raise NotImplementedError(
            f"Hessian accumulation currently supports only groups=1 Conv2d, got groups={module.groups}."
        )
    input_tensor = inputs[0].detach().to(device=hessian.device, dtype=hessian.dtype)
    unfolded = F.unfold(
        input_tensor,
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )
    unfolded = unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])
    hessian.addmm_(unfolded.t(), unfolded)
    return int(unfolded.shape[0])


def _regularize_hessian(hessian: torch.Tensor, damping: float) -> torch.Tensor:
    hessian = 0.5 * (hessian + hessian.t())
    if damping <= 0:
        return hessian
    diag_mean = torch.diag(hessian).mean().abs().item()
    damp = diag_mean * damping if diag_mean > 0 else damping
    idx = torch.arange(hessian.shape[0], device=hessian.device)
    hessian = hessian.clone()
    hessian[idx, idx] += damp
    return hessian


def _evaluate_model(
    model: nn.Module,
    dataset_dir: str,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
) -> Dict[str, float]:
    loader = _build_image_loader(
        dataset_dir=dataset_dir,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = model.to(device)
    model.eval()
    top1 = 0
    top5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            total += labels.numel()
            top1 += (topk[:, 0] == labels).sum().item()
            top5 += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()

    model.cpu()

    if total == 0:
        raise RuntimeError("Evaluation loader produced zero samples.")

    return {
        "top1": 100.0 * top1 / total,
        "top5": 100.0 * top5 / total,
        "num_samples": float(total),
    }


def _compile_patterns(include_regex: str | None, exclude_regex: str | None):
    include_pattern = re.compile(include_regex) if include_regex else None
    exclude_pattern = re.compile(exclude_regex) if exclude_regex else None
    return include_pattern, exclude_pattern


def _iter_target_modules(
    model: nn.Module,
    include_conv: bool,
    include_linear: bool,
    include_pattern,
    exclude_pattern,
) -> Iterator[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        is_target = (
            (include_conv and isinstance(module, nn.Conv2d))
            or (include_linear and isinstance(module, nn.Linear))
        )
        if not is_target:
            continue
        if include_pattern and not include_pattern.search(name):
            continue
        if exclude_pattern and exclude_pattern.search(name):
            continue
        yield name, module


def _compute_weight_stats(
    model: nn.Module,
    include_conv: bool,
    include_linear: bool,
    include_pattern,
    exclude_pattern,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_count = 0
    total_sum = 0.0
    total_sq_sum = 0.0

    for _, module in _iter_target_modules(
        model, include_conv, include_linear, include_pattern, exclude_pattern
    ):
        weight = module.weight.detach().to(torch.float64)
        total_count += weight.numel()
        total_sum += weight.sum().item()
        total_sq_sum += torch.square(weight).sum().item()

    if total_count == 0:
        raise RuntimeError("No target Conv2d/Linear modules were found for weight stats.")

    mean = total_sum / total_count
    var = max(total_sq_sum / total_count - mean * mean, 0.0)
    std = math.sqrt(var)
    return (
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


def _load_comp_model_for_resnet(
    args: argparse.Namespace,
    model: nn.Module,
    include_pattern,
    exclude_pattern,
) -> nn.Module:
    config_path = os.path.join(os.path.dirname(args.comp_model_path), "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = Config(**json.load(handle))

    if args.ql and "ql" not in str(config.architecture):
        raise ValueError(
            f"--ql requires a qlevel-aware codec checkpoint, but "
            f"{config_path} declares architecture='{config.architecture}'."
        )

    if config.architecture == "nwc_ql" and not hasattr(config, "Q"):
        config.Q = 4
    if not hasattr(config, "no_layernorm"):
        config.no_layernorm = False

    state = torch.load(args.comp_model_path, map_location="cpu", weights_only=False)
    state_dict = copy.deepcopy(state["state_dict"])
    comp_model = get_model(
        config.architecture,
        config,
        scale=torch.empty(()),
        shift=torch.empty(()),
    )
    comp_model.config = config

    if _should_use_identity_scale_shift(args):
        scale = torch.ones(1, dtype=torch.float32)
        shift = torch.zeros(1, dtype=torch.float32)
        LOGGER.info(
            "Using identity scale/shift because normalization-related options are enabled."
        )
    elif args.use_train_scale:
        scale = state_dict.get("scale")
        shift = state_dict.get("shift")
        if scale is None or shift is None:
            raise RuntimeError(
                "--use_train_scale was set, but checkpoint does not contain scale/shift."
            )
    else:
        state_dict.pop("scale", None)
        state_dict.pop("shift", None)
        shift, scale = _compute_weight_stats(
            model,
            include_conv=not args.skip_conv,
            include_linear=not args.skip_linear,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
        )

    if args.scale_std is not None and not _should_use_identity_scale_shift(args):
        scale = scale * args.scale_std
    elif args.scale_std is not None:
        LOGGER.info(
            "Ignoring --scale_std because normalization-related options force scale/shift to 1/0."
        )

    if not args.initialize_codec:
        comp_model.load_state_dict(state_dict, strict=False)

    try:
        comp_model.scale.copy_(torch.as_tensor(scale, dtype=torch.float32))
        comp_model.shift.copy_(torch.as_tensor(shift, dtype=torch.float32))
    except Exception:
        comp_model.scale = torch.as_tensor(scale, dtype=torch.float32)
        comp_model.shift = torch.as_tensor(shift, dtype=torch.float32)

    comp_model.eval()
    if hasattr(comp_model, "update") and callable(getattr(comp_model, "update", None)):
        comp_model.update()
    if getattr(args, "ste", False):
        comp_model.mode = "ste"
    return comp_model


def _weight_to_matrix(weight: torch.Tensor, module: nn.Module) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if isinstance(module, nn.Conv2d):
        return weight.reshape(weight.shape[0], -1), tuple(weight.shape)
    if isinstance(module, nn.Linear):
        return weight.reshape(weight.shape[0], weight.shape[1]), tuple(weight.shape)
    raise TypeError(f"Unsupported module type for weight flattening: {type(module)}")


def _matrix_to_weight(matrix: torch.Tensor, original_shape: Sequence[int]) -> torch.Tensor:
    return matrix.reshape(*original_shape)


def _get_required_col_multiple(args: argparse.Namespace, block_size: int) -> int:
    col_multiple = block_size
    if getattr(args, "ldlq", False):
        col_multiple = math.lcm(col_multiple, int(args.comp_batch_size))
    return col_multiple


def _pad_matrix_for_direction(
    matrix: torch.Tensor,
    args: argparse.Namespace,
    block_size: int,
    direction: str,
) -> Tuple[torch.Tensor, PadInfo]:
    rows, cols = matrix.shape
    pad_rows = 0
    pad_cols = 0
    col_multiple = _get_required_col_multiple(args, block_size)

    if direction == "row":
        pad_cols = (-cols) % col_multiple
    elif direction == "col":
        pad_cols = (-cols) % col_multiple
        pad_rows = (-rows) % block_size
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    padded = matrix
    if pad_rows or pad_cols:
        padded = nn.functional.pad(matrix, (0, pad_cols, 0, pad_rows))

    pad_info = PadInfo(
        original_rows=rows,
        original_cols=cols,
        padded_rows=padded.shape[0],
        padded_cols=padded.shape[1],
        pad_rows=pad_rows,
        pad_cols=pad_cols,
    )
    return padded, pad_info


def _crop_padded_matrix(matrix: torch.Tensor, pad_info: PadInfo) -> torch.Tensor:
    return matrix[: pad_info.original_rows, : pad_info.original_cols]


def _pad_hessian_for_matrix(hessian: torch.Tensor, pad_info: PadInfo) -> torch.Tensor:
    if hessian.shape != (pad_info.original_cols, pad_info.original_cols):
        raise ValueError(
            f"Hessian shape {tuple(hessian.shape)} does not match matrix input dimension "
            f"{pad_info.original_cols}."
        )
    if pad_info.pad_cols == 0:
        return hessian

    padded_dim = pad_info.original_cols + pad_info.pad_cols
    padded = torch.zeros(
        padded_dim,
        padded_dim,
        dtype=hessian.dtype,
        device=hessian.device,
    )
    padded[: pad_info.original_cols, : pad_info.original_cols] = hessian
    eps = max(torch.finfo(hessian.dtype).eps, 1e-12)
    idx = torch.arange(pad_info.original_cols, padded_dim, device=hessian.device)
    padded[idx, idx] = eps
    return padded


def _compute_module_input_hessian(
    model: nn.Module,
    module_name: str,
    module: nn.Module,
    dataset_dir: str,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    hessian_dtype: torch.dtype,
    hessian_damping: float,
) -> Tuple[torch.Tensor, HessianStats]:
    if num_batches <= 0:
        raise ValueError("--hessian_num_batches must be positive when Hessian accumulation is enabled.")

    loader = _build_image_loader(
        dataset_dir=dataset_dir,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    available_batches = len(loader)
    effective_batches = min(num_batches, available_batches)
    if effective_batches <= 0:
        raise RuntimeError("Hessian calibration loader produced zero batches.")
    if effective_batches < num_batches:
        LOGGER.warning(
            "Requested --hessian_num_batches=%d for %s, but loader has only %d batches. Using %d instead.",
            num_batches,
            module_name,
            available_batches,
            effective_batches,
        )

    input_dim = _get_module_input_dim(module)
    hessian = torch.zeros(input_dim, input_dim, dtype=hessian_dtype, device=device)
    sample_count = 0

    def _hook(current_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        nonlocal sample_count
        if isinstance(current_module, nn.Linear):
            sample_count += _accumulate_linear_hessian(hessian, current_module, inputs)
            return
        if isinstance(current_module, nn.Conv2d):
            sample_count += _accumulate_conv_hessian(hessian, current_module, inputs)
            return
        raise TypeError(f"Unsupported module type for Hessian accumulation: {type(current_module)}")

    handle = module.register_forward_pre_hook(_hook)
    model = model.to(device)
    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(loader):
                if batch_idx >= effective_batches:
                    break
                images = images.to(device, non_blocking=True)
                _ = model(images)
    finally:
        handle.remove()
        model.cpu()

    if sample_count == 0:
        raise RuntimeError(f"Hessian accumulation collected zero samples for {module_name}.")

    hessian = _regularize_hessian(hessian / float(sample_count), hessian_damping)
    return hessian, HessianStats(
        sample_count=sample_count,
        batch_count=effective_batches,
        input_dim=input_dim,
    )


def _ensure_compression_args_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "Q": 4,
        "ql": False,
        "ql_invH": False,
        "ql_random_uniform": False,
        "ql_search": False,
        "ql_search_layer_idx": "",
        "ql_search_layer_name": "",
        "ql_search_value": None,
        "ql_search_r": None,
        "ldlq": False,
        "ft_y": False,
        "patch": False,
        "scaleH": False,
        "smooth_scaleH_alpha": None,
        "lb_scaleH": None,
        "scaleHinv": False,
        "scale_cond0": False,
        "scale_cond": False,
        "scale_cond_ub": None,
        "whiten": False,
        "global_normalize": False,
        "row_normalize2": False,
        "fp_iter": False,
        "fp_iter_max": None,
        "fp_tol": 1e-5,
        "incoh_mode": "none",
        "code_optim": False,
        "normalization_search": False,
        "perlayer_ft_bs": 128,
        "ft_comp_learning_rate": 1e-4,
        "ft_comp_aux_learning_rate": 1e-3,
        "loss": "rdloss",
        "layer_idx": -1,
        "layer_name": "",
        "use_train_scale": False,
        "initialize_codec": False,
        "ste": False,
        "scale_std": None,
        "sigma_reg": 1e-2,
        "in_hess_eig_path": None,
        "in_hess_name": None,
        "hessian_num_batches": None,
        "hessian_batch_size": None,
        "hessian_damping": 1e-6,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def _compress_single_module(
    module_name: str,
    module: nn.Module,
    comp_model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    hessian: torch.Tensor | None = None,
    hessian_stats: HessianStats | None = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    original_weight = module.weight.detach().to(
        torch.float64 if args.use_fp64 else torch.float32
    )
    matrix, original_shape = _weight_to_matrix(original_weight, module)
    padded_matrix, pad_info = _pad_matrix_for_direction(
        matrix, args, comp_model.input_size, args.direction
    )
    padded_hessian = _pad_hessian_for_matrix(hessian, pad_info) if hessian is not None else None

    start = time.time()
    out = nwc_refactory.compress_linear(
        padded_matrix,
        padded_hessian,
        comp_model,
        None,
        args,
        device=str(device),
    )
    elapsed_sec = time.time() - start

    hat_wr = out["hatWr"].to(padded_matrix.dtype)
    reconstructed_padded = nwc_utils.de_standardize_Wr(
        hat_wr, out["metadata"], args, comp_model
    )
    reconstructed_matrix = _crop_padded_matrix(reconstructed_padded, pad_info)
    reconstructed_weight = _matrix_to_weight(reconstructed_matrix, original_shape).to(
        device=original_weight.device,
        dtype=original_weight.dtype,
    )

    mse = torch.mean(torch.square(original_weight - reconstructed_weight)).item()
    rel_l2 = (
        torch.linalg.norm(original_weight - reconstructed_weight)
        / torch.linalg.norm(original_weight).clamp_min(1e-12)
    ).item()
    num_params = int(module.weight.numel())
    estimated_bits = float(out["bpp_loss_sum"])
    actual_bits = float(out["bpp_sum"]) if args.use_codes else None
    metadata_bits = float(
        nwc_utils.calculate_metadata_bpp(out["metadata"], padded_matrix.shape, args)
    )

    stats = {
        "module_name": module_name,
        "module_type": type(module).__name__,
        "original_shape": list(original_shape),
        "matrix_shape": list(matrix.shape),
        "padded_shape": [pad_info.padded_rows, pad_info.padded_cols],
        "used_hessian": hessian is not None,
        "hessian_input_dim": hessian_stats.input_dim if hessian_stats else None,
        "hessian_sample_count": hessian_stats.sample_count if hessian_stats else None,
        "hessian_batch_count": hessian_stats.batch_count if hessian_stats else None,
        "pad_rows": pad_info.pad_rows,
        "pad_cols": pad_info.pad_cols,
        "num_params": num_params,
        "metadata_bits": metadata_bits,
        "estimated_bits": estimated_bits,
        "estimated_bpp": float(out["bpp_loss"]),
        "estimated_bits_per_param": (
            estimated_bits / num_params if num_params > 0 else None
        ),
        "actual_bits": actual_bits,
        "actual_bpp": float(out["bpp"]) if args.use_codes else None,
        "actual_bits_per_param": (
            actual_bits / num_params if actual_bits is not None and num_params > 0 else None
        ),
        "mse": float(mse),
        "relative_l2": float(rel_l2),
        "mse_normed": float(out.get("mse_normed", 0.0)),
        "elapsed_sec": round(elapsed_sec, 4),
    }
    return reconstructed_weight.to(module.weight.dtype), stats


def _write_results(output_dir: Path, layer_stats: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    json_path = output_dir / "layer_stats.json"
    csv_path = output_dir / "layer_stats.csv"
    summary_path = output_dir / "summary.json"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(layer_stats, handle, indent=2)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if layer_stats:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(layer_stats[0].keys()))
            writer.writeheader()
            writer.writerows(layer_stats)


def _build_summary(
    layer_stats: List[Dict[str, object]],
    eval_before: Dict[str, float] | None,
    eval_after: Dict[str, float] | None,
    use_codes: bool,
) -> Dict[str, object]:
    total_params = sum(int(row["num_params"]) for row in layer_stats)
    total_metadata_bits = sum(float(row["metadata_bits"]) for row in layer_stats)
    total_estimated_bits = sum(float(row["estimated_bits"]) for row in layer_stats)
    total_actual_bits = (
        sum(float(row["actual_bits"]) for row in layer_stats if row["actual_bits"] is not None)
        if use_codes
        else None
    )
    total_sq_err = sum(
        float(row["mse"]) * int(row["num_params"]) for row in layer_stats
    )

    summary = {
        "compressed_layers": len(layer_stats),
        "compressed_params": total_params,
        "fp32_bits": 32.0 * total_params,
        "use_codes": use_codes,
        "metadata_bits": total_metadata_bits,
        "estimated_bits": total_estimated_bits,
        "actual_bits": total_actual_bits,
        "estimated_bits_per_param": (
            total_estimated_bits / total_params if total_params > 0 else None
        ),
        "actual_bits_per_param": (
            total_actual_bits / total_params
            if total_actual_bits is not None and total_params > 0
            else None
        ),
        "estimated_compression_ratio": (
            (32.0 * total_params) / total_estimated_bits
            if total_estimated_bits > 0
            else None
        ),
        "actual_compression_ratio": (
            (32.0 * total_params) / total_actual_bits
            if total_actual_bits is not None and total_actual_bits > 0
            else None
        ),
        "weighted_mse": total_sq_err / total_params if total_params > 0 else None,
    }
    if eval_before is not None:
        summary["eval_before"] = eval_before
    if eval_after is not None:
        summary["eval_after"] = eval_after
    return summary


def _single_run(args: argparse.Namespace) -> None:
    _ensure_compression_args_defaults(args)

    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / "run.log")

    include_pattern, exclude_pattern = _compile_patterns(
        args.include_regex, args.exclude_regex
    )
    device = _resolve_device(args.device)
    model, weights = _load_resnet(args.arch, pretrained=not args.no_pretrained)
    compressed_model = copy.deepcopy(model)
    hessian_model = copy.deepcopy(model) if args.hessian_num_batches else None
    transform = _build_eval_transform(weights)

    eval_before = None
    if args.dataset_dir:
        LOGGER.info("Evaluating FP32 model")
        eval_before = _evaluate_model(
            model,
            dataset_dir=args.dataset_dir,
            transform=transform,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            max_batches=args.eval_batches,
        )
        LOGGER.info("FP32 top1=%.4f top5=%.4f", eval_before["top1"], eval_before["top5"])

    LOGGER.info("Loading NWC compression model from %s", args.comp_model_path)
    comp_model = _load_comp_model_for_resnet(args, model, include_pattern, exclude_pattern)

    target_modules = list(
        _iter_target_modules(
            model,
            include_conv=not args.skip_conv,
            include_linear=not args.skip_linear,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
        )
    )

    if not target_modules:
        raise RuntimeError("No target modules matched the current include/exclude settings.")

    layer_stats: List[Dict[str, object]] = []
    compressed_modules = dict(compressed_model.named_modules())

    for layer_idx, (module_name, module) in enumerate(target_modules):
        args.layer_idx = layer_idx
        args.layer_name = module_name.replace(".", "_")
        LOGGER.info("Compressing [%d/%d] %s", layer_idx + 1, len(target_modules), module_name)

        hessian = None
        hessian_stats = None
        if args.hessian_num_batches:
            hessian_module = dict(hessian_model.named_modules())[module_name]
            LOGGER.info(
                "Accumulating Hessian for %s using %d calibration batches",
                module_name,
                args.hessian_num_batches,
            )
            hessian, hessian_stats = _compute_module_input_hessian(
                model=hessian_model,
                module_name=module_name,
                module=hessian_module,
                dataset_dir=args.dataset_dir,
                transform=transform,
                device=device,
                batch_size=args.hessian_batch_size or args.eval_batch_size,
                num_workers=args.num_workers,
                num_batches=args.hessian_num_batches,
                hessian_dtype=torch.float64 if args.use_fp64 else torch.float32,
                hessian_damping=args.hessian_damping,
            )

        reconstructed_weight, stats = _compress_single_module(
            module_name=module_name,
            module=module,
            comp_model=comp_model,
            args=args,
            device=device,
            hessian=hessian,
            hessian_stats=hessian_stats,
        )

        with torch.no_grad():
            compressed_modules[module_name].weight.copy_(
                reconstructed_weight.to(compressed_modules[module_name].weight.device)
            )

        layer_stats.append(stats)
        LOGGER.info(
            "%s mse=%.6e rel_l2=%.6e est_bpp=%.4f est_bpp_per_param=%.4f metadata_bits=%.1f",
            module_name,
            stats["mse"],
            stats["relative_l2"],
            stats["estimated_bpp"],
            stats["estimated_bits_per_param"],
            stats["metadata_bits"],
        )

    eval_after = None
    if args.dataset_dir:
        LOGGER.info("Evaluating compressed model")
        eval_after = _evaluate_model(
            compressed_model,
            dataset_dir=args.dataset_dir,
            transform=transform,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            max_batches=args.eval_batches,
        )
        LOGGER.info(
            "Compressed top1=%.4f top5=%.4f",
            eval_after["top1"],
            eval_after["top5"],
        )

    summary = _build_summary(
        layer_stats,
        eval_before=eval_before,
        eval_after=eval_after,
        use_codes=args.use_codes,
    )
    _write_results(output_dir, layer_stats, summary)
    if args.use_codes:
        LOGGER.info(
            "Summary params=%d est_bits=%.1f act_bits=%.1f meta_bits=%.1f est_bpp_per_param=%.4f act_bpp_per_param=%.4f",
            summary["compressed_params"],
            summary["estimated_bits"],
            summary["actual_bits"],
            summary["metadata_bits"],
            summary["estimated_bits_per_param"],
            summary["actual_bits_per_param"],
        )
    else:
        LOGGER.info(
            "Summary params=%d est_bits=%.1f meta_bits=%.1f est_bpp_per_param=%.4f actual_bits disabled without --use_codes",
            summary["compressed_params"],
            summary["estimated_bits"],
            summary["metadata_bits"],
            summary["estimated_bits_per_param"],
        )

    torch.save(
        {
            "arch": args.arch,
            "state_dict": compressed_model.state_dict(),
            "layer_stats": layer_stats,
            "summary": summary,
            "args": vars(args),
        },
        output_dir / "compressed_model.pt",
    )

    LOGGER.info("Saved compressed model to %s", output_dir / "compressed_model.pt")
    LOGGER.info("Saved summary to %s", output_dir / "summary.json")


def _append_cli_arg(cmd: List[str], name: str, value) -> None:
    if value is None:
        return
    cmd.extend([name, str(value)])


def _append_bool_flag(cmd: List[str], name: str, enabled: bool) -> None:
    if enabled:
        cmd.append(name)


def _extract_lambda_tag(checkpoint_path: str) -> str:
    match = re.search(r"lmbda([^_/]+)", checkpoint_path)
    if match:
        return f"lmbda{match.group(1)}"
    return Path(checkpoint_path).parent.name


def _sanitize_run_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _build_parallel_worker_command(
    args: argparse.Namespace,
    checkpoint_path: str,
    worker_save_path: str,
    worker_device: str,
) -> List[str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--arch",
        args.arch,
        "--comp_model_path",
        checkpoint_path,
        "--save_path",
        worker_save_path,
        "--device",
        worker_device,
        "--direction",
        args.direction,
        "--comp_batch_size",
        str(args.comp_batch_size),
        "--Q",
        str(args.Q),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--parallel_worker",
    ]
    _append_bool_flag(cmd, "--ql", args.ql)
    _append_bool_flag(cmd, "--ldlq", args.ldlq)
    _append_cli_arg(cmd, "--ql_search_value", args.ql_search_value)
    _append_bool_flag(cmd, "--use_codes", args.use_codes)
    _append_bool_flag(cmd, "--row_normalize", args.row_normalize)
    _append_bool_flag(cmd, "--col_normalize", args.col_normalize)
    _append_bool_flag(cmd, "--layer_normalize", args.layer_normalize)
    _append_bool_flag(cmd, "--use_train_scale", args.use_train_scale)
    _append_cli_arg(cmd, "--scale_std", args.scale_std)
    _append_cli_arg(cmd, "--sigma_reg", args.sigma_reg)
    _append_cli_arg(cmd, "--perlayer_ft_epochs", args.perlayer_ft_epochs)
    _append_cli_arg(cmd, "--perlayer_ft_bs", args.perlayer_ft_bs)
    _append_cli_arg(cmd, "--ft_comp_learning_rate", args.ft_comp_learning_rate)
    _append_cli_arg(cmd, "--ft_comp_aux_learning_rate", args.ft_comp_aux_learning_rate)
    _append_cli_arg(cmd, "--include_regex", args.include_regex)
    _append_cli_arg(cmd, "--exclude_regex", args.exclude_regex)
    _append_bool_flag(cmd, "--skip_conv", args.skip_conv)
    _append_bool_flag(cmd, "--skip_linear", args.skip_linear)
    _append_bool_flag(cmd, "--no_pretrained", args.no_pretrained)
    _append_bool_flag(cmd, "--use_fp64", args.use_fp64)
    _append_cli_arg(cmd, "--dataset_dir", args.dataset_dir)
    _append_cli_arg(cmd, "--eval_batches", args.eval_batches)
    _append_cli_arg(cmd, "--hessian_num_batches", args.hessian_num_batches)
    _append_cli_arg(cmd, "--hessian_batch_size", args.hessian_batch_size)
    _append_cli_arg(cmd, "--hessian_damping", args.hessian_damping)
    return cmd


def _build_sweep_row(checkpoint_path: str, worker_save_path: str) -> Dict[str, object]:
    summary_path = Path(worker_save_path) / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    eval_before = summary.get("eval_before") or {}
    eval_after = summary.get("eval_after") or {}
    before_top1 = eval_before.get("top1")
    after_top1 = eval_after.get("top1")
    before_top5 = eval_before.get("top5")
    after_top5 = eval_after.get("top5")

    return {
        "lambda_tag": _extract_lambda_tag(checkpoint_path),
        "checkpoint_path": checkpoint_path,
        "save_path": worker_save_path,
        "fp32_top1": before_top1,
        "compressed_top1": after_top1,
        "top1_drop": (before_top1 - after_top1) if before_top1 is not None and after_top1 is not None else None,
        "fp32_top5": before_top5,
        "compressed_top5": after_top5,
        "top5_drop": (before_top5 - after_top5) if before_top5 is not None and after_top5 is not None else None,
        "estimated_bits_per_param": summary.get("estimated_bits_per_param"),
        "actual_bits_per_param": summary.get("actual_bits_per_param"),
        "estimated_compression_ratio": summary.get("estimated_compression_ratio"),
        "actual_compression_ratio": summary.get("actual_compression_ratio"),
        "weighted_mse": summary.get("weighted_mse"),
    }


def _write_sweep_results(output_dir: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    json_path = output_dir / "sweep_results.json"
    csv_path = output_dir / "sweep_results.csv"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _parallel_run(args: argparse.Namespace) -> None:
    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / "run.log")

    gpu_queue = list(args.gpu_ids)
    pending_paths = list(args.comp_model_paths)
    active_jobs = []
    sweep_rows: List[Dict[str, object]] = []

    while pending_paths or active_jobs:
        while pending_paths and gpu_queue:
            gpu_id = gpu_queue.pop(0)
            checkpoint_path = pending_paths.pop(0)
            run_name = _sanitize_run_name(_extract_lambda_tag(checkpoint_path))
            worker_save_path = str(output_dir / run_name)
            os.makedirs(worker_save_path, exist_ok=True)
            worker_cmd = _build_parallel_worker_command(
                args=args,
                checkpoint_path=checkpoint_path,
                worker_save_path=worker_save_path,
                worker_device="cuda" if args.device != "cpu" else args.device,
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            stdout_handle = open(
                os.path.join(worker_save_path, "stdout.log"), "w", encoding="utf-8"
            )
            stderr_handle = open(
                os.path.join(worker_save_path, "stderr.log"), "w", encoding="utf-8"
            )
            process = subprocess.Popen(
                worker_cmd,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            active_jobs.append(
                {
                    "gpu_id": gpu_id,
                    "checkpoint_path": checkpoint_path,
                    "worker_save_path": worker_save_path,
                    "process": process,
                    "stdout_handle": stdout_handle,
                    "stderr_handle": stderr_handle,
                }
            )
            LOGGER.info(
                "Launched %s on GPU %s -> %s",
                checkpoint_path,
                gpu_id,
                worker_save_path,
            )

        if not active_jobs:
            break

        time.sleep(5)
        still_active = []
        for job in active_jobs:
            return_code = job["process"].poll()
            if return_code is None:
                still_active.append(job)
                continue

            job["stdout_handle"].close()
            job["stderr_handle"].close()
            gpu_queue.append(job["gpu_id"])

            if return_code != 0:
                for other_job in active_jobs:
                    if other_job is job:
                        continue
                    if other_job["process"].poll() is None:
                        other_job["process"].terminate()
                    other_job["stdout_handle"].close()
                    other_job["stderr_handle"].close()
                raise RuntimeError(
                    f"Worker for checkpoint {job['checkpoint_path']} on GPU {job['gpu_id']} failed "
                    f"with exit code {return_code}. Check {job['worker_save_path']}/stderr.log"
                )

            sweep_rows.append(
                _build_sweep_row(job["checkpoint_path"], job["worker_save_path"])
            )
            sweep_rows.sort(key=lambda item: item["lambda_tag"])
            _write_sweep_results(output_dir, sweep_rows)
            LOGGER.info(
                "Completed %s on GPU %s",
                job["checkpoint_path"],
                job["gpu_id"],
            )
        active_jobs = still_active

    LOGGER.info("Saved sweep results to %s", output_dir)


def run(args: argparse.Namespace) -> None:
    if args.comp_model_paths and not args.parallel_worker:
        _parallel_run(args)
        return
    _single_run(args)


def _parse_args() -> argparse.Namespace:
    default_save_path = os.path.join(
        os.getcwd(),
        "resnet_nwc_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )

    parser = argparse.ArgumentParser(
        description=(
            "Apply NWC weight compression to torchvision ResNet models without "
            "requiring Hessian inputs. Conv2d/Linear weights are flattened, padded "
            "to the codec block size, compressed layer-by-layer, then restored."
        )
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=sorted(WEIGHT_ENUMS.keys()),
        help="Torchvision ResNet architecture to compress.",
    )
    parser.add_argument(
        "--comp_model_path",
        type=str,
        default=None,
        help="Path to the trained NWC checkpoint (.pt).",
    )
    parser.add_argument(
        "--comp_model_paths",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of NWC checkpoints for lambda/checkpoint sweep execution.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=default_save_path,
        help="Output directory for compressed model and layer statistics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compression/evaluation device. Examples: auto, cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="row",
        choices=["row", "col"],
        help="Compression direction expected by the NWC checkpoint.",
    )
    parser.add_argument(
        "--comp_batch_size",
        type=int,
        default=2048,
        help="Column-chunk size passed to nwc_refactory.comp_W.",
    )
    parser.add_argument(
        "--Q",
        type=int,
        default=4,
        help="Quantization-level alphabet size used when qlevel metadata is enabled.",
    )
    parser.add_argument(
        "--ql",
        action="store_true",
        help="Enable qlevel metadata. With H=None, fill qlevel from --ql_search_value.",
    )
    parser.add_argument(
        "--ldlq",
        action="store_true",
        help="Enable LDLQ refinement. Requires online Hessian accumulation via --hessian_num_batches.",
    )
    parser.add_argument(
        "--ql_search_value",
        type=int,
        default=None,
        help="Fixed qlevel value used when --ql is enabled without a Hessian matrix.",
    )
    parser.add_argument(
        "--use_codes",
        action="store_true",
        help="Use entropy coding path to collect actual bits instead of only likelihood-based bpp.",
    )
    parser.add_argument(
        "--row_normalize",
        action="store_true",
        help="Enable row normalization before compression.",
    )
    parser.add_argument(
        "--col_normalize",
        action="store_true",
        help="Enable column normalization before compression.",
    )
    parser.add_argument(
        "--layer_normalize",
        action="store_true",
        help="Enable layer-wise normalization before compression.",
    )
    parser.add_argument(
        "--use_train_scale",
        action="store_true",
        help="Reuse scale/shift stored in the NWC checkpoint instead of recomputing them from ResNet weights.",
    )
    parser.add_argument(
        "--scale_std",
        type=float,
        default=None,
        help="Optional multiplier applied to the computed global std.",
    )
    parser.add_argument(
        "--sigma_reg",
        type=float,
        default=1e-2,
        help="Regularization strength passed to utils.regularize_H2 before H-dependent compression.",
    )
    parser.add_argument(
        "--perlayer_ft_epochs",
        type=int,
        default=0,
        help="Optional per-layer fine-tuning epochs for the codec on each standardized weight matrix.",
    )
    parser.add_argument(
        "--perlayer_ft_bs",
        type=int,
        default=128,
        help="Batch size used for optional per-layer codec fine-tuning.",
    )
    parser.add_argument(
        "--ft_comp_learning_rate",
        type=float,
        default=1e-4,
        help="Main optimizer LR for optional per-layer codec fine-tuning.",
    )
    parser.add_argument(
        "--ft_comp_aux_learning_rate",
        type=float,
        default=1e-3,
        help="Aux optimizer LR for optional per-layer codec fine-tuning.",
    )
    parser.add_argument(
        "--include_regex",
        type=str,
        default=None,
        help="Only compress modules whose full name matches this regex.",
    )
    parser.add_argument(
        "--exclude_regex",
        type=str,
        default=None,
        help="Skip modules whose full name matches this regex.",
    )
    parser.add_argument(
        "--skip_conv",
        action="store_true",
        help="Do not compress Conv2d layers.",
    )
    parser.add_argument(
        "--skip_linear",
        action="store_true",
        help="Do not compress Linear layers.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Instantiate the ResNet with random weights instead of torchvision pretrained weights.",
    )
    parser.add_argument(
        "--use_fp64",
        action="store_true",
        help="Run compression in float64 instead of float32.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Optional ImageNet directory for before/after evaluation. Can be the root or the val folder.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Batch size for optional evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader workers for optional evaluation.",
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=None,
        help="Optional max number of validation batches for quick evaluation.",
    )
    parser.add_argument(
        "--hessian_num_batches",
        type=int,
        default=None,
        help="Optional number of calibration batches used to accumulate per-layer input Hessians online.",
    )
    parser.add_argument(
        "--hessian_batch_size",
        type=int,
        default=None,
        help="Optional calibration batch size for Hessian accumulation. Defaults to --eval_batch_size.",
    )
    parser.add_argument(
        "--hessian_damping",
        type=float,
        default=1e-6,
        help="Relative diagonal damping applied after online Hessian accumulation.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of GPU ids for checkpoint-level parallel execution.",
    )
    parser.add_argument(
        "--parallel_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.comp_model_path is None and not args.comp_model_paths:
        parser.error("Provide either --comp_model_path or --comp_model_paths.")
    if args.comp_model_path is not None and args.comp_model_paths:
        parser.error("Use either --comp_model_path or --comp_model_paths, not both.")
    if args.comp_model_paths and not args.gpu_ids and not args.parallel_worker:
        parser.error("--comp_model_paths requires --gpu_ids for parallel checkpoint sweep.")
    if args.gpu_ids and args.device == "cpu":
        parser.error("--gpu_ids requires a CUDA device mode, not --device cpu.")

    if args.skip_conv and args.skip_linear:
        parser.error("At least one of Conv2d or Linear must remain enabled.")
    if args.comp_batch_size <= 0:
        parser.error("--comp_batch_size must be positive.")
    if args.perlayer_ft_epochs < 0:
        parser.error("--perlayer_ft_epochs must be non-negative.")
    if args.perlayer_ft_bs <= 0:
        parser.error("--perlayer_ft_bs must be positive.")
    if args.eval_batch_size <= 0:
        parser.error("--eval_batch_size must be positive.")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative.")
    if args.eval_batches is not None and args.eval_batches <= 0:
        parser.error("--eval_batches must be positive or omitted.")
    if args.hessian_num_batches is not None and args.hessian_num_batches <= 0:
        parser.error("--hessian_num_batches must be positive or omitted.")
    if args.hessian_batch_size is not None and args.hessian_batch_size <= 0:
        parser.error("--hessian_batch_size must be positive or omitted.")
    if args.hessian_damping < 0:
        parser.error("--hessian_damping must be non-negative.")
    if args.sigma_reg < 0:
        parser.error("--sigma_reg must be non-negative.")
    if args.hessian_num_batches is not None and not args.dataset_dir:
        parser.error("--hessian_num_batches requires --dataset_dir for calibration data.")
    if args.ldlq and args.hessian_num_batches is None:
        parser.error("--ldlq requires --hessian_num_batches because LDLQ needs a Hessian matrix.")

    try:
        if args.comp_model_path is not None:
            expanded = _expand_checkpoint_patterns([args.comp_model_path])
            if len(expanded) != 1:
                parser.error("--comp_model_path must resolve to exactly one checkpoint.")
            args.comp_model_path = expanded[0]
            _validate_checkpoint_path(args.comp_model_path)
        if args.comp_model_paths:
            args.comp_model_paths = _expand_checkpoint_patterns(args.comp_model_paths)
            for checkpoint_path in args.comp_model_paths:
                _validate_checkpoint_path(checkpoint_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    return args


if __name__ == "__main__":
    run(_parse_args())
