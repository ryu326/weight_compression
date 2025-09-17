#!/bin/bash

# --- 설정 변수 ---
# 환경 변수 설정
export HF_HOME=/workspace/hf_cache/huggingface_nwc
export CUDA_VISIBLE_DEVICES=2

SAVE_PATH="/workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/group128"
# SAVE_PATH="/workspace/Weight_compression/text-to-lora/train_outputs/qunat_lora/quant_group/group128"
MODE="group"

uv run python scripts/eval_quantized_lora.py \
    --save_dir "$SAVE_PATH" \
    --mode "$MODE" \
    --group 128 \
    --full_eval
    # > ./logs/quant_group128.log 2>&1
    # --full_eval

# 고정할 인자 값 설정
# CHECKPOINT_PATH="/workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/quant23/comp_model.pt"
# CHECKPOINT_PATH="/workspace/Weight_compression/text-to-lora/train_outputs/qunat_lora/quant2/comp_model.pt"
# MODE="group"

# # {2..8}은 2 3 4 5 6 7 8 과 동일합니다.
# for bit in 2 3 3 4 5 6 7 8
# do
#     echo "=================================================="
#     echo "Running evaluation for bit=${bit}, mode=${MODE}"
#     echo "=================================================="

#     uv run python scripts/eval_quantized_lora.py \
#         --checkpoint_path "$CHECKPOINT_PATH" \
#         --bit "$bit" \
#         --mode "$MODE" \
#         --group 128 > ./logs/quant_group128_${bit}bit.log 2>&1
#         # --full_eval
# done

# nohup sh scripts/eval_quantized_lora.sh > ./logs/quant_eval.log 2>&1 &
# nohup sh scripts/eval_quantized_lora1.sh > ./logs/quant_eval1.log 2>&1 &
# nohup sh scripts/eval_quantized_lora2.sh > ./logs/quant_eval2.log 2>&1 &
# nohup sh scripts/eval_quantized_lora3.sh > ./logs/quant_eval3.log 2>&1 &