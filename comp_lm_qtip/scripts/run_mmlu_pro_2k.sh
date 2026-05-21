#!/bin/bash
##########################################################################
##  MMLU-Pro (max_samples=2000, seed=42) evaluation pipeline.
##  Order: Qwen3-4B (11 configs) → Qwen3-8B (12 configs).
##
##  Same samples across configs: enforced by
##    --max_samples 2000 + --seed 42 (lighteval slices the test split
##    in fixed order, so all configs see identical first-2000 prompts).
##
##  Results saved alongside existing AIME/GPQA jsonl in `*_paroquant_reasoning/`
##  as `MMLU-Pro.jsonl`. inference.py auto-skips when output file exists.
##########################################################################
set -u

ROOT="/home/jgryu/workspace/weight_compression/comp_lm_qtip"
QTIP_ROOT="/home/jgryu/workspace/weight_compression/qtip"
PARO_ROOT="/home/jgryu/workspace/weight_compression/paroquant"
EVAL_PY="/opt/conda/envs/paroquant-eval/bin/python"
HFIZE_PY="/opt/conda/envs/qwen3/bin/python"
NOTIFY_PY="$ROOT/scripts/_paroquant_notify.py"

export HF_HOME=/home/jgryu/.cache/huggingface
if [ -f "$HF_HOME/token" ]; then
    HF_TOKEN_VAL="$(cat "$HF_HOME/token")"
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH

DATASET="MMLU-PRO"     # paroquant inference.py dataset name
MAX_SAMPLES=2000
SEED=42

run_eval() {
    # $1 = hf_model_path  $2 = out_dir  $3 = log_label
    local hf_path="$1"
    local out_dir="$2"
    local label="$3"
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

##########################################################################
## Block A: Qwen3-4B
##########################################################################
LOG_FILE="$ROOT/log/mmlu_pro_2k_4b.log"
mkdir -p "$(dirname "$LOG_FILE")"

# 1) Qwen3-4B FP16 base (no hfize needed — use HF model id directly)
BASE_OUT="$ROOT/../hf_model_comp_results_v2/Qwen--Qwen3-4B/baseline_paroquant_reasoning"
run_eval "Qwen/Qwen3-4B" "$BASE_OUT" "Qwen3-4B base"

# 2) Qwen3-4B Ours (5 lambdas)
OURS_CKPT_4B="$ROOT/../hf_model_comp/comp_qtip/ckpt/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
OURS_HF_4B="$ROOT/../hf_model_comp/comp_qtip/hf/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
OURS_RES_4B="$ROOT/../hf_model_comp_results_v2/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"

cd "$ROOT"
for lmbda in 50 100 300 1000 10000; do
    ckpt_dir="${OURS_CKPT_4B}/lmbda${lmbda}"
    hf_dir="${OURS_HF_4B}/lmbda${lmbda}"
    out_dir="${OURS_RES_4B}/lmbda${lmbda}_paroquant_reasoning"
    [ -d "$ckpt_dir" ] || { echo "[Ours 4B λ=${lmbda}] ckpt missing — skip"; continue; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[Ours 4B λ=${lmbda}] hfize" | tee -a "$LOG_FILE"
        "$HFIZE_PY" -m quantize_llama.hfize_llama \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-4B 2>&1 | tee -a "$LOG_FILE"
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[Ours 4B λ=${lmbda}] hfize fail — skip"; continue; }
    run_eval "$hf_dir" "$out_dir" "Ours 4B λ=${lmbda}"
    rm -rf "$hf_dir"
done

# 3) Qwen3-4B QTIP (5 Ks)
QTIP_CKPT_4B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_4b/ft1"
QTIP_HF_4B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_4b/ft1"
QTIP_RES_4B="$QTIP_ROOT/../hf_model_comp_results_v2/qtip/qwen3_4b/ft1"

cd "$QTIP_ROOT"
for K in 2 3 4 5 6; do
    ckpt_dir="${QTIP_CKPT_4B}/${K}bit"
    hf_dir="${QTIP_HF_4B}/${K}bit"
    out_dir="${QTIP_RES_4B}/${K}bit_paroquant_reasoning"
    [ -d "$ckpt_dir" ] || { echo "[QTIP 4B K=${K}] ckpt missing — skip"; continue; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[QTIP 4B K=${K}] hfize_hf" | tee -a "$LOG_FILE"
        "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-4B 2>&1 | tee -a "$LOG_FILE"
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[QTIP 4B K=${K}] hfize fail — skip"; continue; }
    run_eval "$hf_dir" "$out_dir" "QTIP 4B K=${K}"
    rm -rf "$hf_dir"
done

python3 "$NOTIFY_PY" "Qwen3-4B MMLU-Pro 2k 모두 완료" "$OURS_RES_4B" "$QTIP_RES_4B" "$BASE_OUT" 2>&1 || true

##########################################################################
## Block B: Qwen3-8B
##########################################################################
LOG_FILE="$ROOT/log/mmlu_pro_2k_8b.log"
mkdir -p "$(dirname "$LOG_FILE")"

# 1) Qwen3-8B FP16 base
BASE_OUT8="$ROOT/../hf_model_comp_results_v2/Qwen--Qwen3-8B/baseline_paroquant_reasoning"
run_eval "Qwen/Qwen3-8B" "$BASE_OUT8" "Qwen3-8B base"

# 2) Qwen3-8B Ours (6 lambdas)
OURS_CKPT_8B="$ROOT/../hf_model_comp/comp_qtip/ckpt/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"
OURS_HF_8B="$ROOT/../hf_model_comp/comp_qtip/hf/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"
OURS_RES_8B="$ROOT/../hf_model_comp_results_v2/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"

cd "$ROOT"
for lmbda in 30 50 100 300 1000 10000; do
    ckpt_dir="${OURS_CKPT_8B}/lmbda${lmbda}"
    hf_dir="${OURS_HF_8B}/lmbda${lmbda}"
    out_dir="${OURS_RES_8B}/lmbda${lmbda}_paroquant_reasoning"
    [ -d "$ckpt_dir" ] || { echo "[Ours 8B λ=${lmbda}] ckpt missing — skip"; continue; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[Ours 8B λ=${lmbda}] hfize" | tee -a "$LOG_FILE"
        "$HFIZE_PY" -m quantize_llama.hfize_llama \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-8B 2>&1 | tee -a "$LOG_FILE"
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[Ours 8B λ=${lmbda}] hfize fail — skip"; continue; }
    run_eval "$hf_dir" "$out_dir" "Ours 8B λ=${lmbda}"
    rm -rf "$hf_dir"
done

# 3) Qwen3-8B QTIP (5 Ks)
QTIP_CKPT_8B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_8b/ft1"
QTIP_HF_8B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_8b/ft1"
QTIP_RES_8B="$QTIP_ROOT/../hf_model_comp_results_v2/qtip/qwen3_8b/ft1"

cd "$QTIP_ROOT"
for K in 2 3 4 5 6; do
    ckpt_dir="${QTIP_CKPT_8B}/${K}bit"
    hf_dir="${QTIP_HF_8B}/${K}bit"
    out_dir="${QTIP_RES_8B}/${K}bit_paroquant_reasoning"
    [ -d "$ckpt_dir" ] || { echo "[QTIP 8B K=${K}] ckpt missing — skip"; continue; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[QTIP 8B K=${K}] hfize_hf" | tee -a "$LOG_FILE"
        "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-8B 2>&1 | tee -a "$LOG_FILE"
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[QTIP 8B K=${K}] hfize fail — skip"; continue; }
    run_eval "$hf_dir" "$out_dir" "QTIP 8B K=${K}"
    rm -rf "$hf_dir"
done

python3 "$NOTIFY_PY" "Qwen3-8B MMLU-Pro 2k 모두 완료" "$OURS_RES_8B" "$QTIP_RES_8B" "$BASE_OUT8" 2>&1 || true

echo "==================== MMLU-Pro 2k pipeline complete ===================="
