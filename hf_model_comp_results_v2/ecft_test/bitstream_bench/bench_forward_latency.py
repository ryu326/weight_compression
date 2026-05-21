#!/usr/bin/env python3
"""Latency: forward_from_bitstream (dietgpu decompress+decode+matmul) vs nn.Linear."""
import time
import torch
import torch.nn as nn
from lib.linear.ec_linear_parametric import EntropyConstrainedLinear

torch.manual_seed(0)
DEVICE = torch.device("cuda:0")  # via CUDA_VISIBLE_DEVICES


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


shapes = [
    ("q_proj",     4096, 4096),
    ("gate_proj",  4096, 14336),
    ("down_proj", 14336, 4096),
]
batch_seqs = [(1, 128), (1, 512), (1, 2048)]

print(f"{'shape':<11}{'bxs':<10}{'ECbit (ms)':<14}{'nn.Linear (ms)':<16}{'EC/Lin':<8}{'bpp':<8}")
print("-" * 68)
for name, in_f, out_f in shapes:
    layer = EntropyConstrainedLinear(
        in_f, out_f, bias=False, decoder_type="identity", rht_seed=0,
        entropy_bottleneck_kwargs={"num_gaussian": 3, "num_laplacian": 3},
    ).float().to(DEVICE)
    layer.qs = torch.tensor(1.05)
    with torch.no_grad():
        w = torch.randn(out_f, in_f, device=DEVICE) * 0.02
        layer.initialize_from_weight(w)
    pack = layer.compress_latent(force_update=True)
    bpp = pack["num_bits"] / (in_f * out_f)

    # Pre-compute for fast inference (buffer reuse, qs fused into right_diag)
    layer.prepare_for_inference(pack)

    ref = nn.Linear(in_f, out_f, bias=False).to(DEVICE).to(torch.float16)

    for B, S in batch_seqs:
        x16 = torch.randn(B, S, in_f, device=DEVICE, dtype=torch.float16)
        x32 = x16.float()

        def ec_fn():
            with torch.no_grad():
                return layer.forward_from_bitstream(
                    x32, pack["strings"], shape=pack["shape"],
                    qs=layer.qs, meta=pack["meta"],
                )

        def lin_fn():
            with torch.no_grad():
                return ref(x16)

        ec_ms = bench(ec_fn)
        lin_ms = bench(lin_fn)
        print(f"{name:<11}{f'{B}x{S}':<10}{ec_ms:<14.2f}{lin_ms:<16.2f}"
              f"{ec_ms/lin_ms:<8.1f}{bpp:<8.3f}")
    del layer, ref
    torch.cuda.empty_cache()
