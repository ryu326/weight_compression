# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for weight compression/quantization of large language models and vision models. The primary active project is **comp_lm_qtip** — an enhanced version of QTIP (Quantization with Trellises and Incoherence Processing, NeurIPS 2024 Spotlight). The repo also contains several related quantization baselines and experiments.

## Key Dependencies

```
pip install -r requirements.txt
```
Core: torch 2.4.0, transformers 4.46.3, accelerate 0.34.2, compressai 1.2.6, lm_eval 0.4.4, wandb

CUDA kernels (required for fast inference):
```bash
cd comp_lm_qtip/qtip-kernels && python setup.py install
# or for original QTIP:
cd qtip/qtip-kernels && python setup.py install
```

fast-hadamard-transform may need to be built from source if pip install fails.

## Architecture

### Primary Project: `comp_lm_qtip/`

Enhanced QTIP with support for LLMs (Llama 2/3, Gemma3, Mixtral, QWen, GPTOss) and vision models (CLIP, DINO, SigLIP).

**Quantization pipeline** (3-stage):
1. **Hessian computation**: `quantize_llama/input_hessian_llama.py` — computes per-layer Hessian matrices for incoherence processing
2. **Quantize + fine-tune**: `quantize_llama/quantize_finetune_llama.py` — main entry point; runs LDLQ quantization with optional end-to-end fine-tuning
3. **HF conversion**: `quantize_llama/hfize_llama.py` (or `hfize_moe.py`, `hfize_clip.py`) — converts quantized weights to Hugging Face format

**Library structure** (`comp_lm_qtip/lib/`):
- `codebook/bitshift.py` — `BitshiftLinear`: trellis-coded quantization inference (HYB, 3INST, 1MAD, LUT decode modes)
- `algo/finetune.py` — Main quantization orchestrator (based on QuIP#). Handles per-layer LDLQ + optional fine-tuning
- `algo/nwc_refactory.py` — Neural weight compression algorithm variant
- `algo/ecsq.py` — Entropy-constrained scalar quantization
- `linear/comp_linear.py`, `comp_linear2.py`, `comp_linear3.py` — Compressed linear layer implementations with codebook lookups
- `utils/data_utils.py` — Calibration data loading (WikiText-2, C4)
- `utils/matmul_had.py` — Hadamard transform operations

**Key QTIP parameters** (in quantize_finetune_llama.py):
- `--L`, `--K`, `--V`: Lattice dimension, vector dimension, codebook size (as in the paper)
- `--tlut_bits`: Tunable LUT bits (Q for HYB code; 0 for 3INST/1MAD)
- `--decode_mode`: `quantlut_sym` (HYB), `3inst`, `1mad`, or `lut`
- `--td_x`, `--td_y`: Trellis tile dimensions (default 16x16)
- `--ft_epochs`, `--ft_lr`, `--ft_bs`: Fine-tuning hyperparameters

**Evaluation**: `comp_lm_qtip/eval/` — perplexity (`eval_ppl.py`), interactive generation (`interactive_gen.py`), vision model eval (`eval_siglip_imagenet.py`)

### Running Experiments

Shell scripts in `comp_lm_qtip/scripts/` drive experiments:
- `comp_lm.sh`, `comp_lm1.sh` — Standard LLM quantization
- `comp_lm_parallel.sh` — Parallel execution across configs
- `comp_lm_moe.sh` — Mixture of Experts models
- `comp_clip.sh` — CLIP model compression
- `comp_lm_ecsq.sh` — Entropy-constrained experiments

Scripts in `qtip/` for original QTIP: `run_qtip.sh`, `run_qtip_parallel.sh`, `run_qtip_e2e.sh`

Typical environment setup in scripts:
```bash
CUDA_VISIBLE_DEVICES=7
HF_HOME=/home/jgryu/.cache/huggingface
WANDB_SILENT=true
```

### Other Projects

- `qtip/` — Original QTIP implementation with pre-quantized HF models and fast CUDA kernels
- `comp_cnn/` — CNN (ResNet18/50) compression with NWC, lambda-sweep experiments
- `NWC/` — Neural Weight Compression training pipelines (lattice transform coding, VQ-VAE, entropy bottleneck)
- `e2e_latency/` — End-to-end latency benchmarking (PCIe bandwidth emulation, offloading vs full-VRAM)
- `quip-sharp/` — QuIP# baseline implementation
- `notebooks/` — Jupyter notebooks for result analysis, BPP vs accuracy plots, publication figures
- `hf_model_comp_results/` — Stored quantization results (perplexity, accuracy) across methods

## Conventions

- Model checkpoints and Hessians are saved to paths like `hf_model_comp/qtip/<model>/<config>/`
- Wandb is used for experiment tracking (project names vary by experiment)
- Scripts use bash associative arrays to map model names to HF paths
- The codebase extends QuIP# patterns — `finetune.py` is the central algorithm file
