#!/bin/bash
##########################################################################
##  QTIP Qwen3-4B end-to-end per K (interleaved):
##    for K in {2,3,4,5,6}:
##      Quantize -> Hfize_HF -> paroquant eval (AIME-24/25, GPQA-Diamond)
##      -> cleanup HF dir -> Slack notify (per K)
##  ckpt retained, HF dir deleted after eval.
##########################################################################
set -u

# Compress env (transformers 4.51 for Qwen3)
COMP_PY="/opt/conda/envs/qwen3/bin/python"
# Eval env (lighteval + vllm)
EVAL_PY="/opt/conda/envs/paroquant-eval/bin/python"
PARO_ROOT="/home/jgryu/workspace/weight_compression/paroquant"
NOTIFY_PY="/home/jgryu/workspace/weight_compression/comp_lm_qtip/scripts/_paroquant_notify.py"

if [ ! -x "$COMP_PY" ]; then echo "missing $COMP_PY" >&2; exit 1; fi
if [ ! -x "$EVAL_PY" ]; then echo "missing $EVAL_PY" >&2; exit 1; fi

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

# HF token for gated GPQA-Diamond
if [ -f "$HF_HOME/token" ]; then
    HF_TOKEN_VAL="$(cat "$HF_HOME/token")"
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
fi

base_model="Qwen/Qwen3-4B"
HESS="../Wparam_dataset/quip_hess/qwen3_4b_1024"
model_key="qwen3_4b"
exp_type="ft1"
ft_epochs=0
if [ "$exp_type" == "ft1" ]; then ft_epochs=5; fi

K_VALUES=(2 3 4 5 6)
DATASETS=(AIME-2024 AIME-2025 GPQA-Diamond)
SEED=42

for K in "${K_VALUES[@]}"; do
    NAME="${model_key}/${exp_type}/${K}bit"
    SAVE_PATH="$CKPT/$NAME"
    HF_PATH="$HF/$NAME"
    LOG_FILE="${LOG}/${NAME}.log"
    OUT_DIR="${RES}/${NAME}_paroquant_reasoning"
    mkdir -p "$SAVE_PATH" "$(dirname "$LOG_FILE")" "$OUT_DIR"

    echo "============================================================" | tee "$LOG_FILE"
    echo " QTIP cycle | model=$model_key | K=$K bit | exp=$exp_type"   | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    # 1) Quantize + (optional) FT
    if [ ! -f "$SAVE_PATH/config.pt" ]; then
        echo "### [K=$K] Quantize ###" | tee -a "$LOG_FILE"
        "$COMP_PY" -m quantize_llama.quantize_finetune_llama \
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
        echo "### [K=$K] Quantize SKIP (config.pt exists) ###" | tee -a "$LOG_FILE"
    fi

    # 2) Hfize_HF
    if [ ! -d "$HF_PATH" ] || [ ! -f "$HF_PATH/config.json" ]; then
        echo "### [K=$K] Hfize_HF ###" | tee -a "$LOG_FILE"
        "$COMP_PY" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$SAVE_PATH" \
            --hf_output_path "$HF_PATH" \
            --base_model "$base_model" \
            2>&1 | tee -a "$LOG_FILE"
    fi

    if [ ! -f "$HF_PATH/config.json" ]; then
        echo "WARN: hfize failed for K=$K (skip eval+cleanup)" | tee -a "$LOG_FILE"
        continue
    fi

    # 3) Eval (AIME-24, AIME-25, GPQA-Diamond)
    for ds in "${DATASETS[@]}"; do
        echo "### [K=$K] Eval: $ds ###" | tee -a "$LOG_FILE"
        (cd "$PARO_ROOT" && "$EVAL_PY" \
            -m experiments.tasks.reasoning.lighteval_custom.inference \
            --model "$HF_PATH" \
            --dataset "$ds" \
            --seed "$SEED" \
            --output_dir "$OUT_DIR") 2>&1 | tee -a "$LOG_FILE"
    done

    # 4) Cleanup HF dir; ckpt is retained.
    if [ -d "$HF_PATH" ] && [ "$HF_PATH" != "$HF" ]; then
        echo "### [K=$K] Cleanup: rm -rf $HF_PATH (ckpt at $SAVE_PATH retained) ###" | tee -a "$LOG_FILE"
        rm -rf "$HF_PATH"
    fi

    # 5) Slack notify for this K
    python3 "$NOTIFY_PY" "QTIP Qwen3-4B ${exp_type} K=${K}bit eval 완료" "$OUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"
done

echo "QTIP Qwen3-4B ${exp_type} pipeline complete for K=${K_VALUES[*]}"
