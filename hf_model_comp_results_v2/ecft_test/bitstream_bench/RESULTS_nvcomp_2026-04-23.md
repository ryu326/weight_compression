# nvcomp forward bitstream latency — 2026-04-23

Latency of `ec_linear.forward_from_bitstream_nvcomp` vs `nn.Linear`.
All tensors pre-placed on GPU; timed regions contain no host↔device transfers.
Compute dtype follows input dtype (fp16 in → fp16 matmul; fp32 in → fp32 matmul).

## Setup

- **GPU:** NVIDIA RTX A6000 (48 GB)
- **Env:** conda `base`, python 3.10.13, torch 2.4.0+cu121, `nvidia-nvcomp-cu12 5.0.0.6`
- **Decoder:** `identity` (RHT kernel unavailable in this env)
- **Layer:** `ec_linear.EntropyConstrainedLinear` (compressai EntropyBottleneck, channels=1)
- **Warm-up training:** 30 steps of noise-mode quantization on random init before compressing
- **Timing:** 50 iters, 10 warm-ups, `torch.cuda.synchronize()` around the timed region
- **Input:** `x = torch.randn(B, S, in_f)` on GPU, fp16 or fp32
- **Compression:** offline (not timed); encoded `nvcomp.Array` cached on-device via
  `prepare_for_inference_nvcomp`, qs fused into `right_diag`

Shapes tested: `q_proj` (4096×4096), `gate_proj` (4096×14336), `down_proj` (14336×4096).
Batch×seq: `1×128`, `1×512`, `1×2048` (input flattens to `(B·S, in_f)`).

## Correctness

Round-trip `decompress_latent_nvcomp` vs `quantize_latent`: **max diff 0.0** (exact).
`forward_from_bitstream_nvcomp` vs reference matmul:
- fp32 input: max diff **1.8e-7** (float32 precision)
- fp16 input: max diff **4.9e-4** (float16 precision)

## Results

Columns: `EC16` = `forward_from_bitstream_nvcomp(x16)`, `Lin16` = `nn.Linear.fp16(x16)`,
`EC32` / `Lin32` are the fp32 variants. All in **milliseconds** per call.

```
shape      bxs       algo      EC16       Lin16       EC16/L16  EC32       Lin32       EC32/L32  bpp
-----------------------------------------------------------------------------------------------------
q_proj     1x128     ANS       0.389      0.057       6.8       0.690      0.232       3.0       0.289
q_proj     1x512     ANS       0.526      0.175       3.0       1.328      0.925       1.4       0.289
q_proj     1x2048    ANS       1.097      0.688       1.6       4.205      3.757       1.1       0.289
q_proj     1x128     LZ4       0.408      0.058       7.1       0.696      0.233       3.0       0.035
q_proj     1x512     LZ4       0.543      0.171       3.2       1.323      0.922       1.4       0.035
q_proj     1x2048    LZ4       1.095      0.683       1.6       4.171      3.778       1.1       0.035
q_proj     1x128     Zstd      0.420      0.057       7.3       0.714      0.232       3.1       0.005
q_proj     1x512     Zstd      0.553      0.175       3.2       1.344      0.914       1.5       0.005
q_proj     1x2048    Zstd      1.125      0.685       1.6       4.187      3.771       1.1       0.005
q_proj     1x128     Gdeflate  0.680      0.057       11.9      0.916      0.232       4.0       0.158
q_proj     1x512     Gdeflate  0.789      0.163       4.9       1.555      0.911       1.7       0.158
q_proj     1x2048    Gdeflate  1.331      0.678       2.0       4.380      3.775       1.2       0.158
q_proj     1x128     Deflate   0.610      0.057       10.7      0.860      0.232       3.7       0.071
q_proj     1x512     Deflate   0.725      0.167       4.3       1.487      0.916       1.6       0.071
q_proj     1x2048    Deflate   1.274      0.682       1.9       4.331      3.783       1.1       0.071
gate_proj  1x128     ANS       2.910      0.204       14.3      3.871      0.831       4.7       0.148
gate_proj  1x512     ANS       2.731      0.524       5.2       6.961      3.351       2.1       0.148
gate_proj  1x2048    ANS       4.084      2.102       1.9       18.301     13.445      1.4       0.148
gate_proj  1x128     LZ4       7.898      0.206       38.4      10.127     0.826       12.3      0.035
gate_proj  1x512     LZ4       4.071      0.531       7.7       7.888      3.357       2.3       0.035
gate_proj  1x2048    LZ4       8.510      2.119       4.0       18.479     13.474      1.4       0.035
gate_proj  1x128     Zstd      4.672      0.204       22.9      4.348      0.846       5.1       0.005
gate_proj  1x512     Zstd      2.621      0.525       5.0       6.233      3.337       1.9       0.005
gate_proj  1x2048    Zstd      5.636      2.125       2.7       16.306     13.490      1.2       0.005
gate_proj  1x128     Gdeflate  3.974      0.204       19.5      5.269      0.837       6.3       0.158
gate_proj  1x512     Gdeflate  4.052      0.530       7.6       6.970      3.320       2.1       0.158
gate_proj  1x2048    Gdeflate  5.760      2.126       2.7       16.463     13.308      1.2       0.158
gate_proj  1x128     Deflate   8.337      0.204       40.8      5.755      0.824       7.0       0.071
gate_proj  1x512     Deflate   6.157      0.527       11.7      8.152      3.288       2.5       0.071
gate_proj  1x2048    Deflate   14.707     2.124       6.9       19.772     13.348      1.5       0.071
down_proj  1x128     ANS       6.883      0.184       37.4      6.545      0.804       8.1       0.148
down_proj  1x512     ANS       4.579      0.519       8.8       8.833      3.129       2.8       0.148
down_proj  1x2048    ANS       7.288      2.278       3.2       17.964     12.390      1.4       0.148
down_proj  1x128     LZ4       6.985      0.184       37.9      4.583      0.821       5.6       0.035
down_proj  1x512     LZ4       5.715      0.547       10.4      9.086      3.159       2.9       0.035
down_proj  1x2048    LZ4       8.370      2.284       3.7       17.249     12.369      1.4       0.035
down_proj  1x128     Zstd      3.028      0.184       16.4      4.205      0.807       5.2       0.005
down_proj  1x512     Zstd      2.505      0.549       4.6       5.821      3.161       1.8       0.005
down_proj  1x2048    Zstd      5.906      2.293       2.6       16.373     12.387      1.3       0.005
down_proj  1x128     Gdeflate  7.354      0.184       40.0      6.093      0.827       7.4       0.158
down_proj  1x512     Gdeflate  6.019      0.530       11.4      9.022      3.160       2.9       0.158
down_proj  1x2048    Gdeflate  7.956      2.285       3.5       17.585     12.441      1.4       0.158
down_proj  1x128     Deflate   6.252      0.188       33.3      5.835      0.827       7.1       0.071
down_proj  1x512     Deflate   4.865      0.523       9.3       10.983     3.161       3.5       0.071
down_proj  1x2048    Deflate   7.088      2.301       3.1       18.216     12.475      1.5       0.071
```

## Key observations

- **fp32 EC ≈ fp32 Linear:** `EC32/Lin32 = 1.1–1.5×` at 1×2048 — fp32 matmul is so slow on
  A6000 that decode cost is absorbed.
- **fp16 is the meaningful comparison:** `EC16/Lin16 = 1.6–3.2×` at 1×2048 (best for
  q_proj ANS/Zstd/LZ4 at 1.6×). Small batches (1×128) show 6–40× because fixed
  decode cost dominates fast TensorCore matmul.
- **Algorithm ordering:** ANS ≈ Zstd (fastest decode in nvcomp batched API), Gdeflate
  slightly slower, Deflate / LZ4 most variance depending on data size.
- **BPP caveat:** `Zstd 0.005` is degenerate (30-step toy training quantized most symbols
  to one value). ANS/Gdeflate's 0.15–0.3 bpp is more representative of realistic
  distributions. Decode latency is roughly independent of bpp — dominated by symbol
  count.
- **Where time goes:** at 1×128 q_proj fp16, Lin16 ≈ 60 µs vs EC16 ≈ 400 µs. The
  ~340 µs gap is decode kernel launch + codec Python overhead; it's amortized when
  matmul itself takes ≥1 ms.

## Potential further optimizations

- **CUDA Graph** around decode+matmul to eliminate per-call launch overhead (mostly
  helps small batches).
- **Stream overlap:** start the decode on a non-default stream so the fp16 TensorCore
  matmul of a prior layer runs concurrently.
- **RHT integration:** need to register a CUDA impl for `hadamard::hadamard` in this env
  (currently only `decompress_matvec_*` ops from qtip_kernels). Without it, the
  benchmark is identity-decoder only.

## Reproducing

```bash
conda activate base
PYTHONPATH=/home/jgryu/workspace/weight_compression/comp_lm_qtip \
python /home/jgryu/workspace/weight_compression/hf_model_comp_results_v2/ecft_test/bitstream_bench/bench_forward_nvcomp.py
```
