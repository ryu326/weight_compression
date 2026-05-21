# NWC_v2 codec training — RD-curve experiment summary

## Configuration

- **Dataset**: gaussian (synthetic N(0, 1)) AND llama8b (per-row std normalized)
- **Codec**: encoder + compressai EntropyBottleneck(channels=M=16) + decoder, both ends operate on 16-dim chunks (input/M=16, dim_encoder=512)
- **λ values**: 32, 64, 128, 256, 1024, 8192
- **Optimizer**: Adam(lr=1e-4) main + Adam(lr=1e-3) aux for `.quantiles`
- **Training**: iter=20000, batch_size=512
- **Early stop**: `patience=3` on `val/loss` for the asymmetric (extra) sweep; disabled for the symmetric (main) sweep

7 transform labels (encoder/decoder combinations):

| Label | Encoder | Decoder | Notes |
|---|---|---|---|
| `rht` | RHR (sign+Hadamard, 16-dim) | RHR | Different rht_seed on each side → reconstruction broken |
| `affine` | per-element a·x+b | per-element a·x+b | |
| `linear` | nn.Linear(16→16) | nn.Linear(16→16) | |
| `resblock(n=1)` | 1 resblock | 1 resblock | from extra sweep config B |
| `resblock(n=2)` | 2 resblocks | 2 resblocks | main sweep symmetric |
| `resblock(n=2)/resblock(n=1)` | 2 resblocks | 1 resblock | extra config C — asymmetric |
| `resblock(n=2)/linear` | 2 resblocks | nn.Linear(16→16) | extra config A — asymmetric |

Total: **84 runs** (7 labels × 2 datasets × 6 λ).

## Files

- Plot: `./checkpoint/rd_curves_combined.png`
- CSV: `./checkpoint/rd_results_combined.csv`
- Run dirs: `./checkpoint/rd_sweep/<run_name>/best.pth.tar` (main, 48 runs) and `./checkpoint/rd_sweep_extra/<run_name>/best.pth.tar` (extra, 36 runs)

## Headline results (gaussian, λ=8192 — high-fidelity end of the RD curve)

| Label | bpp | MSE/std² | iter at best |
|---|---|---|---|
| `rht` | 2.32 | 6.2e-2 | 20000 |
| `affine` | 3.77 | 7.7e-3 | 20000 |
| `linear` | 4.77 | 2.0e-3 | 20000 |
| `resblock(n=1)` | 6.26 | 3.1e-4 | 18000 |
| `resblock(n=2)` (sym) | 5.98 | 5.4e-4 | 20000 |
| `resblock(n=2)/resblock(n=1)` | 6.04 | 4.0e-4 | 20000 |
| **`resblock(n=2)/linear`** | **6.78** | **1.2e-4** | **20000** |

Llama8B numbers within ±5 % — patterns identical.

## Key findings

1. **Linear transforms (rht/affine/linear) are RD-degenerate at low λ.** All λ ∈ {32, 64} converge to nearly the same operating point — adding rate penalty doesn't budge the system. Lambda only affects the weighted loss numerically; the actual (bpp, mse) point is unchanged.

2. **rht transform is broken.** Encoder and decoder use different ±1 sign init (rht_seed 0 vs 1), so the encoder/decoder pair starts far from being inverses. Reconstruction stays at MSE ≈ 0.06 across all λ. Fix: use same rht_seed for both, or constrain decoder = encoder^-1.

3. **resblock spans the RD curve.** With ≥1 resblock the codec can trade rate for distortion meaningfully. The full λ sweep shows a clean 1/n-shaped RD curve.

4. **Asymmetric encoder > symmetric.** `resblock(n=2)/linear` consistently wins at high λ (low MSE), beating `resblock(n=2)`-symmetric by 4× MSE at the same iter budget.
   - Encoder needs nonlinear capacity to discover compressible features.
   - Decoder just needs to project quantized integer codes back to weight space — linear suffices.
   - Fewer total parameters → less overfitting to quantization noise.
   - Matches well-known pattern in image-compression literature ("strong encoder, weak decoder").

5. **Resblock depth: n=1 ≈ n=2.** Adding a second resblock barely helps; encoder capacity is not the bottleneck once you have ≥1 nonlinear block.

6. **gaussian ≈ llama8b.** After per-row std normalization, Llama-3-8B weight rows behave like white Gaussian to within a few %. RD curves overlap.

## Caveats / known issues

- **Early stopping fired prematurely at low λ** in the extra sweep (patience=3 on `val/loss`). At low λ, `loss = λ·mse + bpp ≈ bpp`, and bpp converges within ~2K iters → ES triggers despite MSE still falling. Best-iter for many low-λ runs is 2000 instead of the planned 20000. Fixes for future runs:
  - Use `--early_stop_metric mse` instead of `loss`
  - Set `--early_stop_min_iter 10000` (warmup grace period)
  - Or disable early stop entirely for fair comparison

- **The main sweep didn't use early stop**, so its low-λ points are fully trained (iter=20000) and on a slightly better RD point than the extra-sweep counterparts.

- **rht result is uninformative** (broken arch). Should be re-run with matched encoder/decoder rht_seed.

## Reproducing

```bash
cd /home/jgryu/workspace/weight_compression/NWC_v2

# main sweep (4 symmetric transforms × 2 datasets × 6 lambdas = 48 runs, ~10–13 hours on 4× A6000)
GPU_IDS="4 5 6 7" bash scripts/sweep_rd.sh

# extra sweep (3 asymmetric configs × 2 datasets × 6 lambdas = 36 runs, ~6–9 hours)
GPU_IDS="4 5 6 7" bash scripts/sweep_rd_extra.sh

# combined plot
python scripts/plot_rd_combined.py \
    --save_roots ./checkpoint/rd_sweep ./checkpoint/rd_sweep_extra \
    --out ./checkpoint/rd_curves_combined.png \
    --csv ./checkpoint/rd_results_combined.csv
```
