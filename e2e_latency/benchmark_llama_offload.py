#!/usr/bin/env python3
"""
Benchmark script for comparing:
1. Baseline: 16-bit Llama-3 8B with forced CPU offloading under an 8 GB VRAM cap
2. Ours: full-VRAM 16-bit runtime + analytical "compression + on-the-fly decoding" overhead

Power-limit emulation is intentionally left outside this script. Run one of the following
commands before launching the benchmark if you want to emulate a lower-power GPU envelope:

    # Example: 150 W power limit
    sudo nvidia-smi -i 0 -pl 150

    # Example: 100 W power limit
    sudo nvidia-smi -i 0 -pl 100

Recommended on multi-GPU hosts:

    CUDA_VISIBLE_DEVICES=0 python benchmark_llama_offload.py ...

The script has three modes:
    compare       : end-to-end driver. Spawns child processes for baseline/full-VRAM runs.
    baseline-run  : 8 GB VRAM limit + accelerate CPU offload + PCIe delay hooks.
    fullvram-run  : pure 16-bit run with the whole model on one GPU.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.cache_utils import Cache, DynamicCache
except Exception:
    Cache = None
    DynamicCache = None


GB = 1024**3
REFERENCE_MATRIX_PARAMS = 4096 * 4096


@dataclass
class TransferStats:
    bandwidth_gbps: float
    native_bandwidth_gbps: Optional[float]
    hooked_module_names: List[str] = field(default_factory=list)
    offloaded_parameter_bytes: int = 0
    module_calls: int = 0
    total_transfer_bytes: int = 0
    injected_sleep_s: float = 0.0

    def reset_runtime_counters(self) -> None:
        self.module_calls = 0
        self.total_transfer_bytes = 0
        self.injected_sleep_s = 0.0


@dataclass
class GenerationMetrics:
    ttft_ms: float
    tpot_ms: Optional[float]
    decode_step_ms: List[float]
    generated_token_count: int
    generated_text_preview: str


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence.")
    return sum(values) / len(values)


def stddev(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute stddev of an empty sequence.")
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return mean(present)


def stddev_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return stddev(present)


def mean_per_position(series: Sequence[Sequence[float]]) -> List[float]:
    if not series:
        return []
    expected_length = len(series[0])
    if any(len(values) != expected_length for values in series):
        raise ValueError("All per-token decode measurements must have the same length.")
    return [
        mean([values[index] for values in series])
        for index in range(expected_length)
    ]


def summarize_generation_runs(run_metrics: Sequence[GenerationMetrics]) -> Dict[str, Any]:
    if not run_metrics:
        raise ValueError("At least one generation run is required.")

    return {
        "ttft_ms": mean([metric.ttft_ms for metric in run_metrics]),
        "ttft_std_ms": stddev([metric.ttft_ms for metric in run_metrics]),
        "tpot_ms": mean_optional([metric.tpot_ms for metric in run_metrics]),
        "tpot_std_ms": stddev_optional([metric.tpot_ms for metric in run_metrics]),
        "per_token_decode_ms": mean_per_position(
            [metric.decode_step_ms for metric in run_metrics]
        ),
        "generated_token_count": run_metrics[0].generated_token_count,
        "generated_text_preview": run_metrics[0].generated_text_preview,
        "runs": [
            {
                "run_index": index + 1,
                "ttft_ms": metric.ttft_ms,
                "tpot_ms": metric.tpot_ms,
                "per_token_decode_ms": metric.decode_step_ms,
                "generated_token_count": metric.generated_token_count,
                "generated_text_preview": metric.generated_text_preview,
            }
            for index, metric in enumerate(run_metrics)
        ],
    }


def format_stat(
    value: Optional[float],
    std: Optional[float],
    num_runs: int,
    unit: str = "",
    precision: int = 3,
) -> str:
    if value is None:
        return "N/A"
    suffix = f" {unit}" if unit else ""
    formatted = f"{value:.{precision}f}{suffix}"
    if num_runs <= 1 or std is None:
        return formatted
    return f"{formatted} (std {std:.{precision}f}{suffix})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Llama-3 8B offloading vs compression benchmark"
    )
    parser.add_argument(
        "--mode",
        choices=("compare", "baseline-run", "fullvram-run"),
        default="compare",
        help="compare launches both sub-runs and prints the final comparison table.",
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Meta-Llama-3-8B",
        help="HF model id or local model path.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Model dtype used for baseline/full-VRAM measurement.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Single GPU index used for the emulation.",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=1024,
        help="Input prompt length in tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Total generated tokens. TPOT is averaged over tokens 2..N.",
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        nargs="+",
        default=[12.0],
        help="Target PCIe bandwidth(s) to emulate for the baseline run.",
    )
    parser.add_argument(
        "--native-bandwidth-gbps",
        type=float,
        default=None,
        help=(
            "Optional native host<->GPU bandwidth. If set, only the extra delay needed "
            "to slow down to the target bandwidth is injected."
        ),
    )
    parser.add_argument(
        "--baseline-vram-gb",
        type=float,
        default=8.0,
        help="Forced GPU memory cap for the baseline run.",
    )
    parser.add_argument(
        "--cpu-memory-gb",
        type=float,
        default=256.0,
        help="CPU memory budget exposed to accelerate in the baseline run.",
    )
    parser.add_argument(
        "--baseline-runtime-reserve-gb",
        type=float,
        default=1.5,
        help=(
            "Extra GPU headroom reserved during baseline device-map planning so "
            "offloaded modules can be brought back to GPU at runtime."
        ),
    )
    parser.add_argument(
        "--decode-ms-per-matrix",
        type=float,
        default=1.07,
        help="Analytical decode cost in ms per 4096x4096 parameters.",
    )
    parser.add_argument(
        "--decode-overhead-scale",
        type=float,
        default=1.0,
        help="Extra multiplier applied to analytical decode overhead.",
    )
    parser.add_argument(
        "--compressed-bitwidth",
        type=float,
        default=4.0,
        help="Only used for reporting the assumption behind the analytical method.",
    )
    parser.add_argument(
        "--include-non-layer-params-in-decode",
        action="store_true",
        help="If set, embed/lm_head params are also counted in analytical decode overhead.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention backend passed to transformers.from_pretrained.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic greedy decoding setup.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Disable a short warmup generation before measurement.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help=(
            "Number of repeated measurement runs per configuration. "
            "Reported TTFT/TPOT values are averaged over these runs."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders if needed.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def bytes_to_gb(value: int) -> float:
    return value / float(GB)


def ensure_cuda(gpu_id: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this benchmark requires a GPU.")
    torch.cuda.set_device(gpu_id)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_gib(value_gb: float) -> str:
    return f"{value_gb:.2f}GiB"


def build_dummy_prompt_inputs(
    tokenizer: AutoTokenizer,
    prompt_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if prompt_length < 1:
        raise ValueError("--prompt-length must be >= 1")

    seed_text = (
        "Latency benchmarking input for Llama. "
        "This synthetic prompt is repeated to reach a target token count. "
    )
    seed_ids = tokenizer(
        seed_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
    if seed_ids.numel() == 0:
        raise RuntimeError("Tokenizer produced an empty seed prompt.")

    prefix_ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        prefix_ids.append(int(tokenizer.bos_token_id))

    remaining = prompt_length - len(prefix_ids)
    if remaining < 0:
        raise ValueError("Prompt length is smaller than the BOS prefix.")

    repeats = max(1, math.ceil(max(remaining, 1) / seed_ids.numel()))
    body_ids = seed_ids.repeat(repeats)[:remaining]
    full_ids = torch.tensor(prefix_ids, dtype=torch.long)
    if remaining > 0:
        full_ids = torch.cat([full_ids, body_ids.to(dtype=torch.long)], dim=0)

    input_ids = full_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    preview = tokenizer.decode(input_ids[0][: min(prompt_length, 64)], skip_special_tokens=True)
    return input_ids, attention_mask, preview


def infer_input_device(model: torch.nn.Module, fallback_gpu_id: int) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for mapped in device_map.values():
            if isinstance(mapped, int):
                return torch.device(f"cuda:{mapped}")
            mapped_str = str(mapped)
            if mapped_str.startswith("cuda:"):
                return torch.device(mapped_str)
    return torch.device(f"cuda:{fallback_gpu_id}")


def calculate_emulated_delay_s(
    transfer_bytes: int,
    target_bandwidth_gbps: float,
    native_bandwidth_gbps: Optional[float] = None,
) -> float:
    if target_bandwidth_gbps <= 0:
        raise ValueError("Target bandwidth must be > 0 GB/s.")

    target_time_s = transfer_bytes / (target_bandwidth_gbps * 1e9)
    if native_bandwidth_gbps is None:
        return target_time_s

    native_time_s = transfer_bytes / (native_bandwidth_gbps * 1e9)
    return max(0.0, target_time_s - native_time_s)


def count_module_bytes(module: torch.nn.Module) -> int:
    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=True))
    buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers(recurse=True))
    return int(param_bytes + buffer_bytes)


def extract_top_level_offloaded_modules(device_map: Dict[str, Any]) -> List[str]:
    cpu_like = []
    for module_name, placement in device_map.items():
        placement_str = str(placement)
        if placement_str in {"cpu", "disk"}:
            cpu_like.append(module_name)

    cpu_like.sort(key=lambda name: (name.count("."), name))
    selected: List[str] = []
    for module_name in cpu_like:
        if module_name == "":
            selected = [module_name]
            break
        if any(module_name.startswith(parent + ".") for parent in selected if parent):
            continue
        selected.append(module_name)
    return selected


def register_offload_delay_hooks(
    model: torch.nn.Module,
    bandwidth_gbps: float,
    native_bandwidth_gbps: Optional[float],
) -> Tuple[List[Any], TransferStats]:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return [], TransferStats(
            bandwidth_gbps=bandwidth_gbps,
            native_bandwidth_gbps=native_bandwidth_gbps,
        )

    handles: List[Any] = []
    stats = TransferStats(
        bandwidth_gbps=bandwidth_gbps,
        native_bandwidth_gbps=native_bandwidth_gbps,
    )

    for module_name in extract_top_level_offloaded_modules(device_map):
        module = model if module_name == "" else model.get_submodule(module_name)
        module_bytes = count_module_bytes(module)
        if module_bytes == 0:
            continue

        stats.hooked_module_names.append(module_name or "<root>")
        stats.offloaded_parameter_bytes += module_bytes

        def pre_hook(
            _module: torch.nn.Module,
            _inputs: Tuple[Any, ...],
            module_bytes: int = module_bytes,
            hook_stats: TransferStats = stats,
        ) -> None:
            delay_s = calculate_emulated_delay_s(
                transfer_bytes=module_bytes,
                target_bandwidth_gbps=hook_stats.bandwidth_gbps,
                native_bandwidth_gbps=hook_stats.native_bandwidth_gbps,
            )
            if delay_s > 0:
                time.sleep(delay_s)
                hook_stats.injected_sleep_s += delay_s
            hook_stats.module_calls += 1
            hook_stats.total_transfer_bytes += module_bytes

        handles.append(module.register_forward_pre_hook(pre_hook))

    return handles, stats


def get_generation_forward_extra_kwargs(model: torch.nn.Module) -> Dict[str, Any]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return {}

    if "num_logits_to_keep" in signature.parameters:
        return {"num_logits_to_keep": 1}
    if "logits_to_keep" in signature.parameters:
        return {"logits_to_keep": 1}
    return {}


@torch.inference_mode()
def run_greedy_decode_measurement(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    input_device: torch.device,
    sync_cuda_device: torch.device,
) -> GenerationMetrics:
    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")

    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)
    forward_extra_kwargs = get_generation_forward_extra_kwargs(model)

    sync_device(sync_cuda_device)
    ttft_start = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        **forward_extra_kwargs,
    )
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    sync_device(sync_cuda_device)
    ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

    decode_step_ms: List[float] = []
    generated_tokens = [next_token]
    past_key_values = outputs.past_key_values
    if (
        DynamicCache is not None
        and past_key_values is not None
        and (Cache is None or not isinstance(past_key_values, Cache))
    ):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    current_input = next_token
    current_attention_mask = torch.cat(
        [attention_mask, torch.ones_like(next_token, device=input_device)],
        dim=-1,
    )

    for _ in range(max_new_tokens - 1):
        sync_device(sync_cuda_device)
        step_start = time.perf_counter()
        outputs = model(
            input_ids=current_input,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            **forward_extra_kwargs,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        sync_device(sync_cuda_device)
        decode_step_ms.append((time.perf_counter() - step_start) * 1000.0)

        generated_tokens.append(next_token)
        past_key_values = outputs.past_key_values
        if (
            DynamicCache is not None
            and past_key_values is not None
            and (Cache is None or not isinstance(past_key_values, Cache))
        ):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        current_input = next_token
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones_like(next_token, device=input_device)],
            dim=-1,
        )

    generated_token_ids = torch.cat(generated_tokens, dim=-1)[0]
    preview = tokenizer.decode(generated_token_ids[: min(max_new_tokens, 32)], skip_special_tokens=True)
    tpot_ms = None if not decode_step_ms else sum(decode_step_ms) / len(decode_step_ms)
    return GenerationMetrics(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        decode_step_ms=decode_step_ms,
        generated_token_count=max_new_tokens,
        generated_text_preview=preview,
    )


def locate_decoder_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    candidates = (
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
    )

    for path in candidates:
        current: Any = model
        valid = True
        for attr in path.split("."):
            if not hasattr(current, attr):
                valid = False
                break
            current = getattr(current, attr)
        if valid and isinstance(current, (torch.nn.ModuleList, list, tuple)) and len(current) > 0:
            return current
    raise RuntimeError("Could not locate decoder layers for analytical overhead estimation.")


def estimate_decode_overhead_from_model(
    model: torch.nn.Module,
    config: Any,
    decode_ms_per_matrix: float,
    decode_overhead_scale: float,
    include_non_layer_params: bool,
) -> Dict[str, Any]:
    decoder_layers = locate_decoder_layers(model)
    per_layer_params = [sum(p.numel() for p in layer.parameters()) for layer in decoder_layers]
    total_decoder_layer_params = int(sum(per_layer_params))
    average_decoder_layer_params = float(total_decoder_layer_params) / len(per_layer_params)
    total_model_params = int(sum(p.numel() for p in model.parameters()))

    counted_params = total_model_params if include_non_layer_params else total_decoder_layer_params
    overhead_ms = (
        counted_params / REFERENCE_MATRIX_PARAMS * decode_ms_per_matrix * decode_overhead_scale
    )
    decode_param_scope = (
        "full_model_parameters_including_embeddings_and_lm_head"
        if include_non_layer_params
        else "decoder_layers_only_excluding_embeddings_and_lm_head"
    )

    return {
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", len(per_layer_params))),
        "decoder_layer_params_avg": average_decoder_layer_params,
        "decoder_layer_params_total": total_decoder_layer_params,
        "total_model_params": total_model_params,
        "counted_decode_params": counted_params,
        "decode_param_scope": decode_param_scope,
        "decode_ms_per_4096x4096": decode_ms_per_matrix,
        "decode_overhead_scale": decode_overhead_scale,
        "include_non_layer_params_in_decode": include_non_layer_params,
        "per_forward_decode_overhead_ms": overhead_ms,
    }


def cleanup_model(model: Optional[torch.nn.Module]) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_load_kwargs(args: argparse.Namespace, torch_dtype: torch.dtype) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


def get_no_split_module_classes(config: Any) -> List[str]:
    architectures = set(getattr(config, "architectures", []) or [])
    model_type = getattr(config, "model_type", None)

    if "LlamaForCausalLM" in architectures or model_type == "llama":
        return ["LlamaDecoderLayer"]
    if "MistralForCausalLM" in architectures or model_type == "mistral":
        return ["MistralDecoderLayer"]
    if "MixtralForCausalLM" in architectures or model_type == "mixtral":
        return ["MixtralDecoderLayer"]
    return []


def infer_baseline_device_map(
    args: argparse.Namespace,
    config: Any,
    torch_dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    reserve_gb = min(args.baseline_runtime_reserve_gb, max(0.0, args.baseline_vram_gb - 0.5))
    planning_gpu_gb = max(0.5, args.baseline_vram_gb - reserve_gb)

    planning_max_memory = {
        args.gpu_id: format_gib(planning_gpu_gb),
        "cpu": format_gib(args.cpu_memory_gb),
    }
    runtime_max_memory = {
        args.gpu_id: format_gib(args.baseline_vram_gb),
        "cpu": format_gib(args.cpu_memory_gb),
    }

    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=args.trust_remote_code,
        )

    infer_kwargs: Dict[str, Any] = {
        "max_memory": planning_max_memory,
        "no_split_module_classes": get_no_split_module_classes(config),
        "dtype": torch_dtype,
        "offload_buffers": True,
    }
    try:
        infer_signature = inspect.signature(infer_auto_device_map)
        if "fallback_allocation" in infer_signature.parameters:
            infer_kwargs["fallback_allocation"] = True
    except (TypeError, ValueError):
        pass

    device_map = infer_auto_device_map(
        empty_model,
        **infer_kwargs,
    )

    del empty_model
    gc.collect()

    planning_info = {
        "planning_gpu_gb": f"{planning_gpu_gb:.3f}",
        "runtime_gpu_gb": f"{args.baseline_vram_gb:.3f}",
        "runtime_reserve_gb": f"{reserve_gb:.3f}",
    }
    return device_map, planning_info


def load_tokenizer(args: argparse.Namespace) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def run_warmup_if_needed(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_device: torch.device,
    sync_cuda_device: torch.device,
) -> None:
    if args.skip_warmup:
        return

    warmup_length = max(8, min(args.prompt_length, 32))
    warmup_tokens = max(2, min(args.max_new_tokens, 4))
    warmup_input_ids, warmup_attention_mask, _ = build_dummy_prompt_inputs(
        tokenizer=tokenizer,
        prompt_length=warmup_length,
    )
    run_greedy_decode_measurement(
        model=model,
        tokenizer=tokenizer,
        input_ids=warmup_input_ids,
        attention_mask=warmup_attention_mask,
        max_new_tokens=warmup_tokens,
        input_device=input_device,
        sync_cuda_device=sync_cuda_device,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_baseline(args: argparse.Namespace, bandwidth_gbps: float) -> Dict[str, Any]:
    ensure_cuda(args.gpu_id)
    torch_dtype = get_torch_dtype(args.dtype)
    total_vram_bytes = torch.cuda.get_device_properties(args.gpu_id).total_memory
    memory_fraction = min(0.99, (args.baseline_vram_gb * GB) / float(total_vram_bytes))
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=args.gpu_id)
    set_seed(args.seed)

    tokenizer = load_tokenizer(args)
    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    max_memory = {
        args.gpu_id: format_gib(args.baseline_vram_gb),
        "cpu": format_gib(args.cpu_memory_gb),
    }

    model = None
    handles: List[Any] = []
    try:
        device_map, planning_info = infer_baseline_device_map(
            args=args,
            config=config,
            torch_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=device_map,
            max_memory=max_memory,
            offload_buffers=True,
            **model_load_kwargs(args, torch_dtype),
        )
        model.eval()
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        input_device = infer_input_device(model, args.gpu_id)
        sync_cuda_device = torch.device(f"cuda:{args.gpu_id}")
        handles, transfer_stats = register_offload_delay_hooks(
            model=model,
            bandwidth_gbps=bandwidth_gbps,
            native_bandwidth_gbps=args.native_bandwidth_gbps,
        )
        run_warmup_if_needed(
            args=args,
            model=model,
            tokenizer=tokenizer,
            input_device=input_device,
            sync_cuda_device=sync_cuda_device,
        )

        input_ids, attention_mask, prompt_preview = build_dummy_prompt_inputs(
            tokenizer=tokenizer,
            prompt_length=args.prompt_length,
        )
        generation_runs: List[GenerationMetrics] = []
        injected_sleep_runs_ms: List[float] = []
        hook_call_runs: List[float] = []
        transfer_runs_gb: List[float] = []
        run_details: List[Dict[str, Any]] = []

        for run_index in range(args.num_runs):
            transfer_stats.reset_runtime_counters()
            metrics = run_greedy_decode_measurement(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                input_device=input_device,
                sync_cuda_device=sync_cuda_device,
            )
            injected_sleep_ms = transfer_stats.injected_sleep_s * 1000.0
            emulated_transfer_gb = bytes_to_gb(transfer_stats.total_transfer_bytes)

            generation_runs.append(metrics)
            injected_sleep_runs_ms.append(injected_sleep_ms)
            hook_call_runs.append(float(transfer_stats.module_calls))
            transfer_runs_gb.append(emulated_transfer_gb)
            run_details.append(
                {
                    "run_index": run_index + 1,
                    "ttft_ms": metrics.ttft_ms,
                    "tpot_ms": metrics.tpot_ms,
                    "per_token_decode_ms": metrics.decode_step_ms,
                    "generated_token_count": metrics.generated_token_count,
                    "generated_text_preview": metrics.generated_text_preview,
                    "injected_sleep_ms": injected_sleep_ms,
                    "hook_call_count": transfer_stats.module_calls,
                    "emulated_transfer_gb": emulated_transfer_gb,
                }
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        aggregated_metrics = summarize_generation_runs(generation_runs)

        result = {
            "mode": "baseline-run",
            "model_id": args.model_id,
            "dtype": args.dtype,
            "gpu_id": args.gpu_id,
            "prompt_length": args.prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "num_runs": args.num_runs,
            "baseline_vram_gb": args.baseline_vram_gb,
            "baseline_runtime_reserve_gb": args.baseline_runtime_reserve_gb,
            "memory_fraction": memory_fraction,
            "bandwidth_gbps": bandwidth_gbps,
            "native_bandwidth_gbps": args.native_bandwidth_gbps,
            "ttft_ms": aggregated_metrics["ttft_ms"],
            "ttft_std_ms": aggregated_metrics["ttft_std_ms"],
            "tpot_ms": aggregated_metrics["tpot_ms"],
            "tpot_std_ms": aggregated_metrics["tpot_std_ms"],
            "per_token_decode_ms": aggregated_metrics["per_token_decode_ms"],
            "generated_text_preview": aggregated_metrics["generated_text_preview"],
            "prompt_preview": prompt_preview,
            "hooked_offloaded_modules": transfer_stats.hooked_module_names,
            "offloaded_parameter_gb": bytes_to_gb(transfer_stats.offloaded_parameter_bytes),
            "injected_sleep_ms": mean(injected_sleep_runs_ms),
            "injected_sleep_std_ms": stddev(injected_sleep_runs_ms),
            "hook_call_count": mean(hook_call_runs),
            "hook_call_count_std": stddev(hook_call_runs),
            "emulated_transfer_gb": mean(transfer_runs_gb),
            "emulated_transfer_std_gb": stddev(transfer_runs_gb),
            "hf_device_map": getattr(model, "hf_device_map", None),
            "baseline_device_map_planning": planning_info,
            "model_architectures": getattr(config, "architectures", None),
            "runs": run_details,
        }
        print_baseline_summary(result)
        return result
    finally:
        for handle in handles:
            handle.remove()
        cleanup_model(model)


def run_fullvram(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_cuda(args.gpu_id)
    torch_dtype = get_torch_dtype(args.dtype)
    set_seed(args.seed)
    tokenizer = load_tokenizer(args)
    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map={"": args.gpu_id},
            **model_load_kwargs(args, torch_dtype),
        )
        model.eval()
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        input_device = torch.device(f"cuda:{args.gpu_id}")
        sync_cuda_device = input_device
        run_warmup_if_needed(
            args=args,
            model=model,
            tokenizer=tokenizer,
            input_device=input_device,
            sync_cuda_device=sync_cuda_device,
        )

        input_ids, attention_mask, prompt_preview = build_dummy_prompt_inputs(
            tokenizer=tokenizer,
            prompt_length=args.prompt_length,
        )
        generation_runs: List[GenerationMetrics] = []
        run_details: List[Dict[str, Any]] = []
        for run_index in range(args.num_runs):
            metrics = run_greedy_decode_measurement(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                input_device=input_device,
                sync_cuda_device=sync_cuda_device,
            )
            generation_runs.append(metrics)
            run_details.append(
                {
                    "run_index": run_index + 1,
                    "ttft_ms": metrics.ttft_ms,
                    "tpot_ms": metrics.tpot_ms,
                    "per_token_decode_ms": metrics.decode_step_ms,
                    "generated_token_count": metrics.generated_token_count,
                    "generated_text_preview": metrics.generated_text_preview,
                }
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        aggregated_metrics = summarize_generation_runs(generation_runs)
        analytical = estimate_decode_overhead_from_model(
            model=model,
            config=config,
            decode_ms_per_matrix=args.decode_ms_per_matrix,
            decode_overhead_scale=args.decode_overhead_scale,
            include_non_layer_params=args.include_non_layer_params_in_decode,
        )

        result = {
            "mode": "fullvram-run",
            "model_id": args.model_id,
            "dtype": args.dtype,
            "gpu_id": args.gpu_id,
            "prompt_length": args.prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "num_runs": args.num_runs,
            "ttft_ms": aggregated_metrics["ttft_ms"],
            "ttft_std_ms": aggregated_metrics["ttft_std_ms"],
            "tpot_ms": aggregated_metrics["tpot_ms"],
            "tpot_std_ms": aggregated_metrics["tpot_std_ms"],
            "per_token_decode_ms": aggregated_metrics["per_token_decode_ms"],
            "generated_text_preview": aggregated_metrics["generated_text_preview"],
            "prompt_preview": prompt_preview,
            "analytical_decode": analytical,
            "runs": run_details,
        }
        print_fullvram_summary(result, args.compressed_bitwidth)
        return result
    finally:
        cleanup_model(model)


def make_child_command(
    script_path: Path,
    args: argparse.Namespace,
    mode: str,
    json_output: Path,
    bandwidth_gbps: Optional[float] = None,
) -> List[str]:
    command = [
        sys.executable,
        str(script_path),
        "--mode",
        mode,
        "--model-id",
        args.model_id,
        "--dtype",
        args.dtype,
        "--gpu-id",
        str(args.gpu_id),
        "--prompt-length",
        str(args.prompt_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--baseline-vram-gb",
        str(args.baseline_vram_gb),
        "--cpu-memory-gb",
        str(args.cpu_memory_gb),
        "--decode-ms-per-matrix",
        str(args.decode_ms_per_matrix),
        "--decode-overhead-scale",
        str(args.decode_overhead_scale),
        "--compressed-bitwidth",
        str(args.compressed_bitwidth),
        "--attn-implementation",
        args.attn_implementation,
        "--seed",
        str(args.seed),
        "--num-runs",
        str(args.num_runs),
        "--json-output",
        str(json_output),
    ]
    if bandwidth_gbps is not None:
        command.extend(["--bandwidth-gbps", str(bandwidth_gbps)])
    if args.native_bandwidth_gbps is not None:
        command.extend(["--native-bandwidth-gbps", str(args.native_bandwidth_gbps)])
    if args.include_non_layer_params_in_decode:
        command.append("--include-non-layer-params-in-decode")
    if args.skip_warmup:
        command.append("--skip-warmup")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def run_child_process(
    script_path: Path,
    args: argparse.Namespace,
    mode: str,
    bandwidth_gbps: Optional[float] = None,
) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix=f"{mode}-", suffix=".json", delete=False) as tmp:
        output_path = Path(tmp.name)

    command = make_child_command(
        script_path=script_path,
        args=args,
        mode=mode,
        json_output=output_path,
        bandwidth_gbps=bandwidth_gbps,
    )
    env = os.environ.copy()
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"Child process failed: {' '.join(command)}")

    with output_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    output_path.unlink(missing_ok=True)
    return data


def build_compare_payload(
    args: argparse.Namespace,
    fullvram_result: Dict[str, Any],
    baseline_results: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_results = list(baseline_results)
    analytical = fullvram_result["analytical_decode"]
    ours_ttft_ms = fullvram_result["ttft_ms"] + analytical["per_forward_decode_overhead_ms"]
    ours_tpot_ms = (
        None
        if fullvram_result["tpot_ms"] is None
        else fullvram_result["tpot_ms"] + analytical["per_forward_decode_overhead_ms"]
    )
    ours_ttft_std_ms = fullvram_result.get("ttft_std_ms")
    ours_tpot_std_ms = fullvram_result.get("tpot_std_ms")

    comparisons = []
    for baseline in baseline_results:
        comparisons.append(
            {
                "bandwidth_gbps": baseline["bandwidth_gbps"],
                "baseline_ttft_ms": baseline["ttft_ms"],
                "baseline_ttft_std_ms": baseline.get("ttft_std_ms"),
                "baseline_tpot_ms": baseline["tpot_ms"],
                "baseline_tpot_std_ms": baseline.get("tpot_std_ms"),
                "ours_predicted_ttft_ms": ours_ttft_ms,
                "ours_predicted_ttft_std_ms": ours_ttft_std_ms,
                "ours_predicted_tpot_ms": ours_tpot_ms,
                "ours_predicted_tpot_std_ms": ours_tpot_std_ms,
                "baseline_injected_sleep_ms": baseline["injected_sleep_ms"],
                "baseline_hook_call_count": baseline["hook_call_count"],
            }
        )

    return {
        "mode": "compare",
        "model_id": args.model_id,
        "dtype": args.dtype,
        "gpu_id": args.gpu_id,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "num_runs": args.num_runs,
        "compressed_bitwidth_assumption": args.compressed_bitwidth,
        "fullvram": fullvram_result,
        "baseline_runs": baseline_results,
        "ours_prediction": {
            "ttft_ms": ours_ttft_ms,
            "ttft_std_ms": ours_ttft_std_ms,
            "tpot_ms": ours_tpot_ms,
            "tpot_std_ms": ours_tpot_std_ms,
            "analytical_decode": analytical,
        },
        "comparisons": comparisons,
    }


def print_baseline_summary(result: Dict[str, Any]) -> None:
    num_runs = int(result.get("num_runs", 1))
    print("=" * 80)
    print("Baseline: 16-bit Offloading")
    print(
        f"Prompt={result['prompt_length']} tok | NewTokens={result['max_new_tokens']} | "
        f"BW={result['bandwidth_gbps']:.2f} GB/s | Runs={num_runs}"
    )
    print(
        f"TTFT={format_stat(result['ttft_ms'], result.get('ttft_std_ms'), num_runs, unit='ms')} | "
        f"TPOT={format_stat(result['tpot_ms'], result.get('tpot_std_ms'), num_runs, unit='ms')}"
        if result["tpot_ms"] is not None else
        f"TTFT={format_stat(result['ttft_ms'], result.get('ttft_std_ms'), num_runs, unit='ms')} | TPOT=N/A"
    )
    print(
        f"OffloadedParams={result['offloaded_parameter_gb']:.3f} GB | "
        f"InjectedSleep={format_stat(result['injected_sleep_ms'], result.get('injected_sleep_std_ms'), num_runs, unit='ms')} | "
        f"HookCalls={format_stat(result['hook_call_count'], result.get('hook_call_count_std'), num_runs, precision=2)}"
    )


def print_fullvram_summary(result: Dict[str, Any], compressed_bitwidth: float) -> None:
    analytical = result["analytical_decode"]
    num_runs = int(result.get("num_runs", 1))
    print("=" * 80)
    print("Ours Base Measurement: Full-VRAM 16-bit")
    print(
        f"Prompt={result['prompt_length']} tok | NewTokens={result['max_new_tokens']} | "
        f"CompressedAssumption={compressed_bitwidth:.2f}-bit | Runs={num_runs}"
    )
    print(
        f"TTFT={format_stat(result['ttft_ms'], result.get('ttft_std_ms'), num_runs, unit='ms')} | "
        f"TPOT={format_stat(result['tpot_ms'], result.get('tpot_std_ms'), num_runs, unit='ms')}"
        if result["tpot_ms"] is not None else
        f"TTFT={format_stat(result['ttft_ms'], result.get('ttft_std_ms'), num_runs, unit='ms')} | TPOT=N/A"
    )
    print(
        f"AnalyticalDecodeOverhead={analytical['per_forward_decode_overhead_ms']:.3f} ms "
        f"per forward step"
    )
    print(
        f"CountedDecodeParams={analytical['counted_decode_params'] / 1e9:.3f}B params "
        f"({analytical['decode_param_scope']}) | "
        f"DecoderLayers={analytical['num_hidden_layers']}"
    )


def print_compare_table(payload: Dict[str, Any]) -> None:
    ours = payload["ours_prediction"]
    analytical = ours["analytical_decode"]
    num_runs = int(payload.get("num_runs", 1))
    print("=" * 100)
    print("Final Comparison")
    print(
        f"Model={payload['model_id']} | DType={payload['dtype']} | "
        f"Prompt={payload['prompt_length']} tok | NewTokens={payload['max_new_tokens']} | "
        f"Runs={num_runs}"
    )
    print(
        f"Ours assumption: {payload['compressed_bitwidth_assumption']:.2f}-bit weights fully fit in VRAM, "
        f"decode overhead={analytical['per_forward_decode_overhead_ms']:.3f} ms/step, "
        f"scope={analytical['decode_param_scope']}"
    )
    if num_runs > 1:
        print("TTFT/TPOT entries below are run averages.")
    print("-" * 100)
    print(
        f"{'BW(GB/s)':>10} | {'Baseline TTFT(ms)':>18} | {'Baseline TPOT(ms)':>18} | "
        f"{'Ours TTFT(ms)':>14} | {'Ours TPOT(ms)':>14}"
    )
    print("-" * 100)
    for row in payload["comparisons"]:
        baseline_tpot = "N/A" if row["baseline_tpot_ms"] is None else f"{row['baseline_tpot_ms']:.3f}"
        ours_tpot = "N/A" if row["ours_predicted_tpot_ms"] is None else f"{row['ours_predicted_tpot_ms']:.3f}"
        print(
            f"{row['bandwidth_gbps']:>10.2f} | "
            f"{row['baseline_ttft_ms']:>18.3f} | "
            f"{baseline_tpot:>18} | "
            f"{row['ours_predicted_ttft_ms']:>14.3f} | "
            f"{ours_tpot:>14}"
        )
    print("-" * 100)


def maybe_write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 2:
        raise ValueError("--max-new-tokens must be >= 2 to measure TPOT.")
    if args.prompt_length < 1:
        raise ValueError("--prompt-length must be >= 1.")
    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1.")
    if args.baseline_vram_gb <= 0:
        raise ValueError("--baseline-vram-gb must be > 0.")
    if any(bw <= 0 for bw in args.bandwidth_gbps):
        raise ValueError("All --bandwidth-gbps values must be > 0.")
    if args.decode_ms_per_matrix <= 0:
        raise ValueError("--decode-ms-per-matrix must be > 0.")
    if args.decode_overhead_scale <= 0:
        raise ValueError("--decode-overhead-scale must be > 0.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.mode == "baseline-run":
        result = run_baseline(args, bandwidth_gbps=args.bandwidth_gbps[0])
        maybe_write_json(args.json_output, result)
        return

    if args.mode == "fullvram-run":
        result = run_fullvram(args)
        maybe_write_json(args.json_output, result)
        return

    script_path = Path(__file__).resolve()
    fullvram_result = run_child_process(
        script_path=script_path,
        args=args,
        mode="fullvram-run",
    )
    baseline_results = [
        run_child_process(
            script_path=script_path,
            args=args,
            mode="baseline-run",
            bandwidth_gbps=bandwidth,
        )
        for bandwidth in args.bandwidth_gbps
    ]
    payload = build_compare_payload(
        args=args,
        fullvram_result=fullvram_result,
        baseline_results=baseline_results,
    )
    print_compare_table(payload)
    maybe_write_json(args.json_output, payload)


if __name__ == "__main__":
    main()
