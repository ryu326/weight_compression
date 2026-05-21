#!/bin/bash
##########################################################################
##  Run QTIP on Qwen3-4B (using already-computed Hessian).
##  Uses hfize_llama_hf.py for HF model export.
##  After hfize, paroquant-style reasoning eval (AIME-2024/2025, GPQA-Diamond)
##  is done by run_qtip_qwen3_4b_eval.sh.
##########################################################################
set -u

# qwen3 conda env (transformers 4.51+) — required for Qwen3 model loading.
PYTHON_BIN="/opt/conda/envs/qwen3/bin/python"
TORCHRUN_BIN="/opt/conda/envs/qwen3/bin/torchrun"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: qwen3 conda env not found at $PYTHON_BIN" >&2
    exit 1
fi

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"
RES="../hf_model_comp_results_v2/qtip"

mkdir -p "$CKPT" "$HF" "$LOG" "$RES"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_SILENT=true
export TRANSFORMERS_NO_TORCHVISION=1
export HF_HOME=/home/jgryu/.cache/huggingface
unset PYTHONPATH

base_model="Qwen/Qwen3-4B"
HESS="../Wparam_dataset/quip_hess/qwen3_4b_1024"
model_key="qwen3_4b"
exp_type="ft1"             # ft1 (ft_epochs=5) — paper-quality; switch to noft for fast smoke test
ft_epochs=0
if [ "$exp_type" == "ft1" ]; then
    ft_epochs=5
fi

# QTIP K (bits) values to sweep — full range used in QTIP paper.
K_VALUES=(2 3 4 5 6)

for K in "${K_VALUES[@]}"; do
    NAME="${model_key}/${exp_type}/${K}bit"
    SAVE_PATH="$CKPT/$NAME"
    HF_PATH="$HF/$NAME"
    LOG_FILE="${LOG}/${NAME}.log"
    mkdir -p "$SAVE_PATH" "$(dirname "$LOG_FILE")"

    echo "============================================================" | tee "$LOG_FILE"
    echo " QTIP | model=$model_key | K=$K bit | exp=$exp_type" | tee -a "$LOG_FILE"
    echo " base_model = $base_model" | tee -a "$LOG_FILE"
    echo " hess_path  = $HESS"        | tee -a "$LOG_FILE"
    echo " save_path  = $SAVE_PATH"   | tee -a "$LOG_FILE"
    echo " hf_path    = $HF_PATH"     | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    # 1) Quantize + (optional) fine-tune
    if [ ! -f "$SAVE_PATH/config.pt" ]; then
        echo "### [Stage: Quantize | K=$K] ###" | tee -a "$LOG_FILE"
        "$PYTHON_BIN" -m quantize_llama.quantize_finetune_llama \
            --save_path "$SAVE_PATH" \
            --codebook bitshift \
            --base_model "$base_model" \
            --in_hess_path "$HESS" \
            --scale_override 0.9 \
            --ft_epochs $ft_epochs \
            --td_x 16 --td_y 16 --L 16 --K $K --V 2 \
            --decode_mode quantlut_sym --tlut_bits 9 \
            2>&1 | tee -a "$LOG_FILE"
    else
        echo "### [Skip Quantize | $SAVE_PATH/config.pt exists] ###" | tee -a "$LOG_FILE"
    fi

    # 2) Hfize via hfize_llama_hf (per user request)
    if [ ! -d "$HF_PATH" ] || [ ! -f "$HF_PATH/config.json" ]; then
        echo "### [Stage: Hfize_HF | K=$K] ###" | tee -a "$LOG_FILE"
        "$PYTHON_BIN" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$SAVE_PATH" \
            --hf_output_path "$HF_PATH" \
            --base_model "$base_model" \
            2>&1 | tee -a "$LOG_FILE"
    else
        echo "### [Skip Hfize | $HF_PATH/config.json exists] ###" | tee -a "$LOG_FILE"
    fi
done

echo "QTIP Qwen3-4B compression+hfize done. Run run_qtip_qwen3_4b_eval.sh next."
