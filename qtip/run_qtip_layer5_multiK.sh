#!/bin/bash
# QTIP layer 5 only — K=2,3,5 in parallel on separate GPUs.
# K=2 already done; run K=3 and K=5.

set -u
set -o pipefail

MODEL_KEY="llama3_8b"
BASE_MODEL="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"
FT_EPOCHS=5

CKPT="../hf_model_comp/qtip/ckpt"
LOG="./log"
RES="../hf_model_comp_results/qtip"

export WANDB_SILENT=true
export TRANSFORMERS_NO_TORCHVISION=1
export HF_HOME=/home/jgryu/.cache/huggingface

NLAYERS=32
TAGS=(v q k o up gate down)
TARGET=5
skip_list=""
for ((i=0; i<NLAYERS; i++)); do
    [ "$i" -eq "$TARGET" ] && continue
    for tag in "${TAGS[@]}"; do
        [ -z "$skip_list" ] && skip_list="${i}_${tag}" || skip_list="${skip_list},${i}_${tag}"
    done
done

K_list=(3 5)
gpu_list=(6 7)

pids=()
for idx in "${!K_list[@]}"; do
    K="${K_list[$idx]}"
    GPU="${gpu_list[$idx]}"
    EXP_TAG="ft1_layer5_only"
    NAME="${MODEL_KEY}/${EXP_TAG}/${K}bit"
    SAVE_PATH="${CKPT}/${NAME}"
    LOG_FILE="${LOG}/${NAME}.log"
    mkdir -p "$SAVE_PATH" "$(dirname "$LOG_FILE")"

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"
        echo "### [K=${K} | layer ${TARGET} | GPU ${GPU}] ###" | tee "$LOG_FILE"
        python -m quantize_llama.quantize_finetune_llama \
            --save_path "$SAVE_PATH" \
            --codebook bitshift \
            --base_model "$BASE_MODEL" \
            --in_hess_path "$HESS" \
            --scale_override 0.9 \
            --ft_epochs "$FT_EPOCHS" \
            --td_x 16 --td_y 16 --L 16 --K "$K" --V 2 \
            --decode_mode quantlut_sym --tlut_bits 9 \
            --skip_list "$skip_list" \
            2>&1 | tee -a "$LOG_FILE"
    ) &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "All K values done."
