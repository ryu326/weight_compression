#!/bin/bash
# QTIP Qwen3-8B MMLU-Pro 2k — GPUs 4,5,6,7 only
set -u

ROOT="/home/jgryu/workspace/weight_compression/comp_lm_qtip"
QTIP_ROOT="/home/jgryu/workspace/weight_compression/qtip"
PARO_ROOT="/home/jgryu/workspace/weight_compression/paroquant"
EVAL_PY="/opt/conda/envs/paroquant-eval/bin/python"
HFIZE_PY="/opt/conda/envs/qwen3/bin/python"
NOTIFY_PY="$ROOT/scripts/_paroquant_notify.py"
LOG_FILE="$ROOT/log/mmlu_pro_2k_8b.log"

export HF_HOME=/home/jgryu/.cache/huggingface
if [ -f "$HF_HOME/token" ]; then
    HF_TOKEN_VAL="$(cat "$HF_HOME/token")"
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
fi
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset PYTHONPATH

DATASET="MMLU-PRO"
MAX_SAMPLES=2000
SEED=42

run_eval() {
    local hf_path="$1" out_dir="$2" label="$3"
    local out_file="${out_dir}/${DATASET}.jsonl"
    mkdir -p "$out_dir"
    if [ -f "$out_file" ]; then
        echo "[$label] $out_file exists — skip" | tee -a "$LOG_FILE"
        return 0
    fi
    echo "[$label] eval ${DATASET} (max_samples=${MAX_SAMPLES}, seed=${SEED})" | tee -a "$LOG_FILE"
    (cd "$PARO_ROOT" && "$EVAL_PY" \
        -m experiments.tasks.reasoning.lighteval_custom.inference \
        --model "$hf_path" \
        --dataset "$DATASET" \
        --seed "$SEED" \
        --max_samples "$MAX_SAMPLES" \
        --output_dir "$out_dir") 2>&1 | tee -a "$LOG_FILE"
    python3 "$NOTIFY_PY" "[$label] MMLU-Pro 2k 완료" "$out_dir" 2>&1 || true
}

QTIP_CKPT_8B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_8b/ft1"
QTIP_HF_8B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_8b/ft1"
QTIP_RES_8B="$QTIP_ROOT/../hf_model_comp_results_v2/qtip/qwen3_8b/ft1"

echo "========== QTIP 8B MMLU-Pro 2k (GPU 4,5,6,7) ==========" | tee -a "$LOG_FILE"

cd "$QTIP_ROOT"
for K in 2 3 4 5 6; do
    ckpt_dir="${QTIP_CKPT_8B}/${K}bit"
    hf_dir="${QTIP_HF_8B}/${K}bit"
    out_dir="${QTIP_RES_8B}/${K}bit_paroquant_reasoning"
    [ -d "$ckpt_dir" ] || { echo "[QTIP 8B K=${K}] ckpt missing — skip" | tee -a "$LOG_FILE"; continue; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[QTIP 8B K=${K}] hfize_hf" | tee -a "$LOG_FILE"
        "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-8B 2>&1 | tee -a "$LOG_FILE"
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[QTIP 8B K=${K}] hfize fail — skip" | tee -a "$LOG_FILE"; continue; }
    run_eval "$hf_dir" "$out_dir" "QTIP 8B K=${K}"
    rm -rf "$hf_dir"
done

python3 "$NOTIFY_PY" "Qwen3-8B QTIP MMLU-Pro 2k 모두 완료" "$QTIP_RES_8B" 2>&1 || true
echo "==================== QTIP 8B MMLU-Pro 2k complete ===================="
