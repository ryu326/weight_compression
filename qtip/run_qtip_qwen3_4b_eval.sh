#!/bin/bash
##########################################################################
##  Paroquant-style reasoning eval (AIME-2024/2025, GPQA-Diamond) on
##  QTIP-quantized Qwen3-4B HF dirs produced by run_qtip_qwen3_4b.sh.
##  Cleans up HF dir after eval (ckpt retained).
##########################################################################
set -u

PARO_ROOT="/home/jgryu/workspace/weight_compression/paroquant"
PYTHON_BIN="/opt/conda/envs/paroquant-eval/bin/python"
NOTIFY_PY="/home/jgryu/workspace/weight_compression/comp_lm_qtip/scripts/_paroquant_notify.py"

export HF_HOME=/home/jgryu/.cache/huggingface
if [ -f "$HF_HOME/token" ]; then
    HF_TOKEN_VAL="$(cat "$HF_HOME/token")"
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH

CKPT_ROOT="../hf_model_comp/qtip/ckpt"
HF_ROOT="../hf_model_comp/qtip/hf"
RES_ROOT="../hf_model_comp_results_v2/qtip"
LOG="./log"

model_key="qwen3_4b"
exp_type="ft1"
K_VALUES=(2 3 4 5 6)
DATASETS=(AIME-2024 AIME-2025 GPQA-Diamond)
SEED=42

mkdir -p "$LOG"

for K in "${K_VALUES[@]}"; do
    NAME="${model_key}/${exp_type}/${K}bit"
    hf_path="${HF_ROOT}/${NAME}"
    out_dir="${RES_ROOT}/${NAME}_paroquant_reasoning"
    log_path="${LOG}/${NAME}_paroquant_reasoning.log"
    mkdir -p "$(dirname "$log_path")" "$out_dir"

    if [ ! -d "$hf_path" ]; then
        echo "WARN: HF dir missing for K=${K} bit: $hf_path (skip)" | tee -a "$log_path"
        continue
    fi

    echo "============================================================" | tee "$log_path"
    echo " QTIP eval | model=$model_key | K=$K bit | exp=$exp_type"     | tee -a "$log_path"
    echo " HF path    : $hf_path"                                       | tee -a "$log_path"
    echo " Output dir : $out_dir"                                       | tee -a "$log_path"
    echo " Datasets   : ${DATASETS[*]}"                                 | tee -a "$log_path"
    echo "============================================================" | tee -a "$log_path"

    for ds in "${DATASETS[@]}"; do
        echo "---- Dataset: $ds ----" | tee -a "$log_path"
        (cd "$PARO_ROOT" && "$PYTHON_BIN" \
            -m experiments.tasks.reasoning.lighteval_custom.inference \
            --model "$hf_path" \
            --dataset "$ds" \
            --seed "$SEED" \
            --output_dir "$out_dir") 2>&1 | tee -a "$log_path"
    done

    # Cleanup HF dir to save disk; ckpt kept.
    if [ -d "$hf_path" ] && [ "$hf_path" != "$HF_ROOT" ]; then
        echo "---- Cleanup: rm -rf $hf_path ----" | tee -a "$log_path"
        rm -rf "$hf_path"
    fi
done

# Slack notification with per-K accuracies
python3 "$NOTIFY_PY" "QTIP Qwen3-4B ${exp_type} (K=${K_VALUES[*]}) reasoning eval 완료" \
    "${RES_ROOT}/${model_key}/${exp_type}" \
    2>&1 | tee -a "$LOG/${model_key}_qtip_eval_summary.log"
