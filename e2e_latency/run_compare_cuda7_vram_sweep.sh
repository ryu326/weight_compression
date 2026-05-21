#!/usr/bin/env bash

set -euo pipefail

cd /home/jgryu/workspace/weight_compression/e2e_latency

# sudo nvidia-smi -i 7 -pm 1
# sudo nvidia-smi -i 0 -pl 150

mkdir -p ./results

for PROMPT_LENGTH in 128 512 1024; do
  for BASELINE_VRAM_GB in 12; do
    echo "============================================================"
    echo "Running compare with --prompt-length ${PROMPT_LENGTH} --baseline-vram-gb ${BASELINE_VRAM_GB}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=7 python benchmark_llama_offload.py \
      --mode compare \
      --model-id ../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B \
      --dtype float16 \
      --prompt-length "${PROMPT_LENGTH}" \
      --max-new-tokens 16 \
      --bandwidth-gbps 1000 \
      --baseline-vram-gb "${BASELINE_VRAM_GB}" \
      --baseline-runtime-reserve-gb 2.0 \
      --decode-ms-per-matrix 1.25 \
      --decode-overhead-scale 1.0 \
      --json-output "./results/llama8b_compare_prompt${PROMPT_LENGTH}_${BASELINE_VRAM_GB}GB_150W_10run.json" \
      --num-runs 10
  done
done

# sudo nvidia-smi -i 0 -pl 300
