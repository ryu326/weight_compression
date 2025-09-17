#!/bin/bash

# --- 설정 변수 ---
# 환경 변수 설정
export HF_HOME=/workspace/hf_cache/huggingface_nwc
export CUDA_VISIBLE_DEVICES=1

# 고정할 인자 값 설정
CHECKPOINT_PATH="/workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/quant45/comp_model.pt"
MODE="group"

# {2..8}은 2 3 4 5 6 7 8 과 동일합니다.
for bit in 4 5
do
    echo "=================================================="
    echo "Running evaluation for bit=${bit}, mode=${MODE}"
    echo "=================================================="

    uv run python scripts/eval_quantized_lora.py \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --bit "$bit" \
        --mode "$MODE" \
        --group 128 \
        --full_eval
done
# nohup sh scripts/eval_quantized_lora.sh > ./logs/quant_eval.log 2>&1 &