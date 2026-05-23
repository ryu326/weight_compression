#!/bin/bash
##########################################################################
##  MMLU-Pro full (12K) — QTIP K=4
##  Order: Qwen3-8B K=4 → Qwen3-4B K=4
##  Output: hf_model_comp_results_v2/qtip/qwen3_{8b,4b}/ft1/4bit_mmlu12k/
##########################################################################
set -u

ROOT="/home/jgryu/workspace/weight_compression/comp_lm_qtip"
QTIP_ROOT="/home/jgryu/workspace/weight_compression/qtip"
PARO_ROOT="/home/jgryu/workspace/weight_compression/paroquant"
EVAL_PY="/opt/conda/envs/paroquant-eval/bin/python"
HFIZE_PY="/opt/conda/envs/qwen3/bin/python"
NOTIFY_PY="$ROOT/scripts/_paroquant_notify.py"
RESULTS_V2="/home/jgryu/workspace/weight_compression/hf_model_comp_results_v2"

export HF_HOME=/home/jgryu/.cache/huggingface
if [ -f "$HF_HOME/token" ]; then
    HF_TOKEN_VAL="$(cat "$HF_HOME/token")"
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH

LOG_FILE="$ROOT/log/mmlu_pro_full_qtip_k4.log"
mkdir -p "$(dirname "$LOG_FILE")"

run_eval() {
    local hf_path="$1" out_dir="$2" label="$3"
    local out_file="${out_dir}/MMLU-PRO.jsonl"
    mkdir -p "$out_dir"
    if [ -f "$out_file" ]; then
        echo "[$label] $out_file exists — skip" | tee -a "$LOG_FILE"
        return 0
    fi
    echo "[$label] MMLU-PRO full 12K" | tee -a "$LOG_FILE"
    (cd "$PARO_ROOT" && "$EVAL_PY" \
        -m experiments.tasks.reasoning.lighteval_custom.inference \
        --model "$hf_path" \
        --dataset "MMLU-PRO" \
        --seed 42 \
        --output_dir "$out_dir") 2>&1 | tee -a "$LOG_FILE"
    python3 "$NOTIFY_PY" "[$label] MMLU-Pro 12K 완료" "$out_dir" 2>&1 || true
}

echo "================================================================" | tee -a "$LOG_FILE"
echo " MMLU-Pro full 12K — QTIP K=4  $(date)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

##########################################################################
## Qwen3-8B K=4
##########################################################################
CKPT_8B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_8b/ft1/4bit"
HF_8B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_8b/ft1/4bit"
OUT_8B="$RESULTS_V2/qtip/qwen3_8b/ft1/4bit_mmlu12k"

if [ ! -d "$CKPT_8B" ]; then
    echo "[QTIP 8B K=4] ckpt missing — abort" | tee -a "$LOG_FILE"
    exit 1
fi
if [ ! -f "$HF_8B/config.json" ]; then
    echo "[QTIP 8B K=4] hfize_hf" | tee -a "$LOG_FILE"
    (cd "$QTIP_ROOT" && "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
        --quantized_path "$CKPT_8B" \
        --hf_output_path "$HF_8B" \
        --base_model Qwen/Qwen3-8B) 2>&1 | tee -a "$LOG_FILE"
fi
run_eval "$HF_8B" "$OUT_8B" "QTIP 8B K=4"
rm -rf "$HF_8B"

##########################################################################
## Qwen3-4B K=4
##########################################################################
CKPT_4B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_4b/ft1/4bit"
HF_4B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_4b/ft1/4bit"
OUT_4B="$RESULTS_V2/qtip/qwen3_4b/ft1/4bit_mmlu12k"

if [ ! -d "$CKPT_4B" ]; then
    echo "[QTIP 4B K=4] ckpt missing — abort" | tee -a "$LOG_FILE"
    exit 1
fi
if [ ! -f "$HF_4B/config.json" ]; then
    echo "[QTIP 4B K=4] hfize_hf" | tee -a "$LOG_FILE"
    (cd "$QTIP_ROOT" && "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
        --quantized_path "$CKPT_4B" \
        --hf_output_path "$HF_4B" \
        --base_model Qwen/Qwen3-4B) 2>&1 | tee -a "$LOG_FILE"
fi
run_eval "$HF_4B" "$OUT_4B" "QTIP 4B K=4"
rm -rf "$HF_4B"

python3 "$NOTIFY_PY" "QTIP K=4 MMLU-Pro 12K 모두 완료" "$OUT_8B" "$OUT_4B" 2>&1 || true
echo "================================================================" | tee -a "$LOG_FILE"
echo " Done: $(date)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
