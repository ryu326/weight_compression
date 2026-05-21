#!/bin/bash
# Quantize ONLY decoder layer 10 with QTIP K=2, ft_epochs=5 (ft1 setup).
# Skip list excludes every (layer, sublayer) except layer 10's 7 sublayers.

set -u
set -o pipefail

##########################################################################
##                           CONFIG                                     ##
##########################################################################
MODEL_KEY="llama3_8b"
BASE_MODEL="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

K=2
FT_EPOCHS=5
EXP_TAG="ft1_layer10_only"

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"
RES="../hf_model_comp_results/qtip"

GPU_ID=2  # single GPU

mkdir -p "$CKPT" "$HF" "$LOG" "$RES"

##########################################################################
##                        BUILD SKIP LIST                               ##
##    llama3-8b has 32 decoder blocks. Skip all except block 10.        ##
##########################################################################
NLAYERS=32
TAGS=(v q k o up gate down)
TARGET=10
skip_list=""
for ((i=0; i<NLAYERS; i++)); do
    if [ "$i" -eq "$TARGET" ]; then
        continue
    fi
    for tag in "${TAGS[@]}"; do
        if [ -z "$skip_list" ]; then
            skip_list="${i}_${tag}"
        else
            skip_list="${skip_list},${i}_${tag}"
        fi
    done
done

##########################################################################
##                               RUN                                    ##
##########################################################################
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_SILENT=true
export TRANSFORMERS_NO_TORCHVISION=1
export HF_HOME=/home/jgryu/.cache/huggingface

NAME="${MODEL_KEY}/${EXP_TAG}/${K}bit"
SAVE_PATH="${CKPT}/${NAME}"
HF_PATH="${HF}/${NAME}"
LOG_FILE="${LOG}/${NAME}.log"

mkdir -p "$SAVE_PATH" "$(dirname "$LOG_FILE")"

echo "### [Stage: Quantize | K=${K} | layer ${TARGET} only] ###" | tee "$LOG_FILE"
echo "skip_list length: $(echo "$skip_list" | tr ',' '\n' | wc -l) entries" | tee -a "$LOG_FILE"
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

echo "### [Stage: Hfize | K=${K}] ###" | tee -a "$LOG_FILE"
python -m quantize_llama.hfize_llama \
    --quantized_path "$SAVE_PATH" \
    --hf_output_path "$HF_PATH" \
    --base_model "$BASE_MODEL" 2>&1 | tee -a "$LOG_FILE"

echo "### [Stage: Eval PPL | K=${K}] ###" | tee -a "$LOG_FILE"
python -m eval.eval_ppl \
    --hf_path "$HF_PATH" \
    --output_path "${RES}/${NAME}" \
    --seqlen 2048 2>&1 | tee -a "$LOG_FILE"

if [ "$HF_PATH" != "$HF" ]; then
    echo "Cleaning up ${HF_PATH}"
    rm -rf "$HF_PATH"
fi

echo "### DONE — results in ${RES}/${NAME} ###"
