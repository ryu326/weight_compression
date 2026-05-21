#!/usr/bin/env python3
"""Latency benchmark: ec_linear.forward_from_bitstream_nvcomp (GPU decode +
matmul) vs nn.Linear (fp16 reference).

All tensors are pre-placed on the GPU; the timed regions contain no host↔device
transfers. The encoded nvcomp.Array is cached on-device via
`prepare_for_inference_nvcomp`, and the decode + matmul inside the forward call
both run on CUDA streams only — the only CPU involvement is the Python function
call itself."""
import time
import torch
import torch.nn as nn
from lib.linear.ec_linear import EntropyConstrainedLinear

torch.manual_seed(0)
DEVICE = torch.device("cuda:0")


def bench(fn, iters=200, warmup=30):
    # Warmup (JIT/caches/prefetch)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Take the median over several trials to suppress sporadic GPU contention
    trials = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        trials.append((time.time() - t0) / iters * 1000)  # ms
    trials.sort()
    return trials[len(trials) // 2]  # median


shapes = [
    ("q_proj",     4096, 4096),
    ("gate_proj",  4096, 14336),
    ("down_proj", 14336, 4096),
]
batch_seqs = [(1, 128), (1, 512), (1, 2048)]
algos = ["ANS", "LZ4", "Zstd", "Gdeflate", "Deflate"]


def build_layer(in_f, out_f):
    layer = EntropyConstrainedLinear(
        in_f, out_f, bias=False, decoder_type="identity", rht_seed=0,
    ).float().to(DEVICE)
    layer.qs = torch.tensor(1.05)
    with torch.no_grad():
        w = torch.randn(out_f, in_f, device=DEVICE) * 0.02
        layer.initialize_from_weight(w)
    # Brief training so the EB distribution is non-trivial (affects compress
    # ratio; doesn't affect the latency numbers since prep is offline).
    layer.train(); layer.quantize_mode = "noise"
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
    for _ in range(30):
        opt.zero_grad()
        q, _ = layer.quantize_latent()
        loss = (q - layer.latent).pow(2).mean() + 0.1 * layer._last_rate_loss
        loss.backward(); opt.step()
    layer.eval()
    layer.update_entropy_model(force=True, update_quantiles=True)
    return layer


header = (
    f"{'shape':<11}{'bxs':<10}{'algo':<10}"
    f"{'EC16':<10}{'EC16_nd':<9}{'Δdiag':<8}"
    f"{'Lin16':<10}"
    f"{'EC32':<10}{'EC32_nd':<9}"
    f"{'Lin32':<10}{'bpp':<8}"
)
print(header)
print("-" * len(header))

for name, in_f, out_f in shapes:
    layer = build_layer(in_f, out_f)

    # nn.Linear references — fp16 and fp32, both fully on GPU.
    ref_fp16 = nn.Linear(in_f, out_f, bias=False).to(DEVICE).to(torch.float16)
    ref_fp32 = nn.Linear(in_f, out_f, bias=False).to(DEVICE).to(torch.float32)

    # Pre-allocate inputs on GPU — NOT moved during benchmarking.
    xs = {(B, S): (torch.randn(B, S, in_f, device=DEVICE, dtype=torch.float16),
                    torch.randn(B, S, in_f, device=DEVICE, dtype=torch.float32))
          for (B, S) in batch_seqs}

    for algo in algos:
        # Compress once (offline — not timed). The encoded nvcomp.Array stays
        # on-device; prepare caches the codec + qs-fused buffers.
        pack = layer.compress_latent_nvcomp(algorithm=algo, force_update=True)
        bpp = (pack["num_bytes"] * 8) / (in_f * out_f) if pack["num_bytes"] > 0 else float("nan")
        layer.prepare_for_inference_nvcomp(pack)

        for (B, S), (x16, x32) in xs.items():
            # EC uses input.dtype for compute — fp16 input → fp16 matmul
            def ec16_fn():
                with torch.no_grad():
                    return layer.forward_from_bitstream_nvcomp(x16)

            def ec16_nodiag_fn():
                with torch.no_grad():
                    return layer.forward_from_bitstream_nvcomp(x16, skip_diag=True)

            def ec32_fn():
                with torch.no_grad():
                    return layer.forward_from_bitstream_nvcomp(x32)

            def ec32_nodiag_fn():
                with torch.no_grad():
                    return layer.forward_from_bitstream_nvcomp(x32, skip_diag=True)

            def lin16_fn():
                with torch.no_grad():
                    return ref_fp16(x16)

            def lin32_fn():
                with torch.no_grad():
                    return ref_fp32(x32)

            ec16_ms    = bench(ec16_fn)
            ec16_nd_ms = bench(ec16_nodiag_fn)
            ec32_ms    = bench(ec32_fn)
            ec32_nd_ms = bench(ec32_nodiag_fn)
            l16_ms     = bench(lin16_fn)
            l32_ms     = bench(lin32_fn)
            diag_cost  = ec16_ms - ec16_nd_ms  # both diag multiplies combined
            print(
                f"{name:<11}{f'{B}x{S}':<10}{algo:<10}"
                f"{ec16_ms:<10.3f}{ec16_nd_ms:<9.3f}{diag_cost:<8.3f}"
                f"{l16_ms:<10.3f}"
                f"{ec32_ms:<10.3f}{ec32_nd_ms:<9.3f}"
                f"{l32_ms:<10.3f}{bpp:<8.3f}"
            )

    del layer, ref_fp16, ref_fp32
    torch.cuda.empty_cache()
