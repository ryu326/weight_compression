import argparse
import gc
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


DEFAULT_MODEL_PATHS = [
    # "/workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B",
    # "/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B",
    '/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-3.2-1B'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark one-token latency of a single Llama 3 8B decoder block."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit path to the local Meta-Llama-3-8B checkpoint.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Decoder layer index to benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the benchmark.",
    )
    parser.add_argument(
        "--past-seq-len",
        type=int,
        default=2048,
        help="Past KV cache length to emulate autoregressive decoding.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=20,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=100,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, for example cuda:0 or cpu. Defaults to cuda:0 when available.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype used when loading the checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to generate synthetic hidden states and KV cache tensors.",
    )
    return parser.parse_args()


def resolve_model_path(explicit_path):
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        return str(path)

    for candidate in DEFAULT_MODEL_PATHS:
        path = Path(candidate)
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find Meta-Llama-3-8B. Checked: " + ", ".join(DEFAULT_MODEL_PATHS)
    )


def resolve_device(device_arg):
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_torch_dtype(dtype_arg):
    if dtype_arg == "auto":
        return "auto"
    return getattr(torch, dtype_arg)


def activate_cuda_device(device):
    if device.type == "cuda":
        torch.cuda.set_device(device)


def compute_percentile(values, percentile):
    values_tensor = torch.tensor(values, dtype=torch.float64)
    return torch.quantile(values_tensor, percentile / 100.0).item()


def load_decoder_layer(model_path, layer_idx, device, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    num_layers = len(model.model.layers)
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"layer_idx must be in [0, {num_layers - 1}], got {layer_idx}")

    layer = model.model.layers[layer_idx].eval().to(device)
    rotary_emb = model.model.rotary_emb.to(device)
    config = model.config
    dtype = next(layer.parameters()).dtype

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return layer, rotary_emb, config, dtype


def build_inputs(config, rotary_emb, dtype, device, batch_size, past_seq_len):
    hidden_states = torch.randn(
        batch_size,
        1,
        config.hidden_size,
        device=device,
        dtype=dtype,
    )
    position_ids = torch.full(
        (batch_size, 1),
        past_seq_len,
        device=device,
        dtype=torch.long,
    )
    cache_position = torch.tensor([past_seq_len], device=device, dtype=torch.long)
    position_embeddings = rotary_emb(hidden_states, position_ids)

    head_dim = config.hidden_size // config.num_attention_heads
    base_key_states = torch.randn(
        batch_size,
        config.num_key_value_heads,
        past_seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )
    base_value_states = torch.randn(
        batch_size,
        config.num_key_value_heads,
        past_seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )

    return {
        "hidden_states": hidden_states,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "position_embeddings": position_embeddings,
        "base_key_states": base_key_states,
        "base_value_states": base_value_states,
    }


def build_cache(layer_idx, base_key_states, base_value_states):
    cache = DynamicCache()
    cache.update(base_key_states, base_value_states, layer_idx)
    return cache


def measure_latency_ms(layer, layer_idx, config, benchmark_inputs, warmup_iters, benchmark_iters, device):
    activate_cuda_device(device)

    hidden_states = benchmark_inputs["hidden_states"]
    position_ids = benchmark_inputs["position_ids"]
    cache_position = benchmark_inputs["cache_position"]
    position_embeddings = benchmark_inputs["position_embeddings"]
    base_key_states = benchmark_inputs["base_key_states"]
    base_value_states = benchmark_inputs["base_value_states"]

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    times_ms = []

    with torch.inference_mode():
        for _ in range(warmup_iters):
            cache = build_cache(layer_idx, base_key_states, base_value_states)
            _ = layer(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for _ in range(benchmark_iters):
            cache = build_cache(layer_idx, base_key_states, base_value_states)

            if device.type == "cuda":
                starter.record()
                outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=cache,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                ender.record()
                torch.cuda.synchronize(device)
                times_ms.append(starter.elapsed_time(ender))
            else:
                start_time = time.perf_counter()
                outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=cache,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                end_time = time.perf_counter()
                times_ms.append((end_time - start_time) * 1000.0)

    output_hidden_shape = tuple(outputs[0].shape)
    output_cache_len = cache.get_seq_length()

    return times_ms, output_hidden_shape, output_cache_len


def summarize(times_ms):
    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.pstdev(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "p50_ms": compute_percentile(times_ms, 50),
        "p95_ms": compute_percentile(times_ms, 95),
    }


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    device = resolve_device(args.device)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)

    activate_cuda_device(device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    layer, rotary_emb, config, dtype = load_decoder_layer(
        model_path=model_path,
        layer_idx=args.layer_idx,
        device=device,
        torch_dtype=torch_dtype,
    )
    benchmark_inputs = build_inputs(
        config=config,
        rotary_emb=rotary_emb,
        dtype=dtype,
        device=device,
        batch_size=args.batch_size,
        past_seq_len=args.past_seq_len,
    )
    times_ms, output_hidden_shape, output_cache_len = measure_latency_ms(
        layer=layer,
        layer_idx=args.layer_idx,
        config=config,
        benchmark_inputs=benchmark_inputs,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        device=device,
    )
    stats = summarize(times_ms)

    print("=" * 80)
    print("Llama 3 8B Single Decoder Block Latency Benchmark")
    print("=" * 80)
    print(f"model_path         : {model_path}")
    print(f"device             : {device}")
    print(f"layer_idx          : {args.layer_idx}")
    print(f"batch_size         : {args.batch_size}")
    print(f"past_seq_len       : {args.past_seq_len}")
    print("query_seq_len      : 1")
    print(f"dtype              : {dtype}")
    print(f"load_torch_dtype   : {args.torch_dtype}")
    print(f"attention_impl     : {layer.self_attn.__class__.__name__}")
    print(f"warmup_iters       : {args.warmup_iters}")
    print(f"benchmark_iters    : {args.benchmark_iters}")
    print(f"output_hidden_shape: {output_hidden_shape}")
    print(f"output_cache_len   : {output_cache_len}")
    print("-" * 80)
    print(f"mean latency       : {stats['mean_ms']:.4f} ms")
    print(f"std latency        : {stats['std_ms']:.4f} ms")
    print(f"p50 latency        : {stats['p50_ms']:.4f} ms")
    print(f"p95 latency        : {stats['p95_ms']:.4f} ms")
    print(f"min latency        : {stats['min_ms']:.4f} ms")
    print(f"max latency        : {stats['max_ms']:.4f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
