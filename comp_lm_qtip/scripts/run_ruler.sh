#!/bin/bash
##########################################################################
##  RULER evaluation — Qwen3-4B and Qwen3-8B
##  13 tasks: niah_single/multikey/multivalue/multiquery, vt, cwe, fwe, qa_1/2
##  Context lengths: 4K, 8K, 16K, 32K
##  Results: hf_model_comp_results_v2/<model>/<config>_longctx/ruler.json
##  Pattern: hfize → eval → rm hf_dir (ckpts never deleted)
##########################################################################
set -u

ROOT="/home/jgryu/workspace/weight_compression/comp_lm_qtip"
QTIP_ROOT="/home/jgryu/workspace/weight_compression/qtip"
RESULTS_V2="/home/jgryu/workspace/weight_compression/hf_model_comp_results_v2"
EVAL_PY="/opt/conda/envs/paroquant-eval/bin/python"
HFIZE_PY="/opt/conda/envs/qwen3/bin/python"
NOTIFY_PY="$ROOT/scripts/_paroquant_notify.py"

export HF_HOME=/home/jgryu/.cache/huggingface
if [ -f "$HF_HOME/token" ]; then
    HF_TOKEN_VAL="$(cat "$HF_HOME/token")"
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
fi
GPUS="${1:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPUS"
unset PYTHONPATH

run_ruler() {
    # $1=hf_path  $2=out_base  $3=label  $4=log_file
    local hf_path="$1" out_base="$2" label="$3" log="$4"
    local result_file="${out_base}/ruler.json"
    mkdir -p "$out_base"
    if [ -f "$result_file" ]; then
        echo "[$label] ruler.json exists — skip" | tee -a "$log"
        return 0
    fi
    echo "[$label] RULER (gpus=${GPUS})" | tee -a "$log"
    (cd "$ROOT" && "$EVAL_PY" eval/ruler.py \
        --model "$hf_path" \
        --output-dir "$out_base" \
        --gpus "$GPUS") 2>&1 | tee -a "$log"
    python3 "$NOTIFY_PY" "[$label] RULER 완료" "$out_base" 2>&1 || true
}

##########################################################################
## Block A: Qwen3-4B
##########################################################################
LOG_4B="$ROOT/log/ruler_4b.log"
mkdir -p "$(dirname "$LOG_4B")"

# 1) baseline
run_ruler "Qwen/Qwen3-4B" \
    "$RESULTS_V2/Qwen--Qwen3-4B/baseline_longctx" \
    "Qwen3-4B base" "$LOG_4B"

# 2) Ours 4B
OURS_CKPT_4B="$ROOT/../hf_model_comp/comp_qtip/ckpt/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
OURS_HF_4B="$ROOT/../hf_model_comp/comp_qtip/hf/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
OURS_RES_4B="$RESULTS_V2/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"

cd "$ROOT"
for lmbda in 10000 1000 300 100 50; do
    ckpt_dir="${OURS_CKPT_4B}/lmbda${lmbda}"
    hf_dir="${OURS_HF_4B}/lmbda${lmbda}"
    out_base="${OURS_RES_4B}/lmbda${lmbda}_longctx"
    [ -d "$ckpt_dir" ] || { echo "[Ours 4B λ=${lmbda}] ckpt missing — skip" | tee -a "$LOG_4B"; continue; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[Ours 4B λ=${lmbda}] hfize" | tee -a "$LOG_4B"
        "$HFIZE_PY" -m quantize_llama.hfize_llama \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-4B 2>&1 | tee -a "$LOG_4B"
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[Ours 4B λ=${lmbda}] hfize fail — skip" | tee -a "$LOG_4B"; continue; }
    run_ruler "$hf_dir" "$out_base" "Ours 4B λ=${lmbda}" "$LOG_4B"
    rm -rf "$hf_dir"
done

# 3) QTIP 4B
QTIP_CKPT_4B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_4b/ft1"
QTIP_HF_4B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_4b/ft1"
QTIP_RES_4B="$RESULTS_V2/qtip/qwen3_4b/ft1"

cd "$QTIP_ROOT"
for K in 6 5 4 3 2; do
    ckpt_dir="${QTIP_CKPT_4B}/${K}bit"
    hf_dir="${QTIP_HF_4B}/${K}bit"
    out_base="${QTIP_RES_4B}/${K}bit_longctx"
    [ -d "$ckpt_dir" ] || { echo "[QTIP 4B K=${K}] ckpt missing — skip" | tee -a "$LOG_4B"; continue; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[QTIP 4B K=${K}] hfize_hf" | tee -a "$LOG_4B"
        "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-4B 2>&1 | tee -a "$LOG_4B"
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[QTIP 4B K=${K}] hfize fail — skip" | tee -a "$LOG_4B"; continue; }
    run_ruler "$hf_dir" "$out_base" "QTIP 4B K=${K}" "$LOG_4B"
    rm -rf "$hf_dir"
done

python3 "$NOTIFY_PY" "Qwen3-4B RULER 모두 완료" "$OURS_RES_4B" "$QTIP_RES_4B" 2>&1 || true

##########################################################################
## Block B: Qwen3-8B
##########################################################################
LOG_8B="$ROOT/log/ruler_8b.log"
mkdir -p "$(dirname "$LOG_8B")"

# 1) baseline
run_ruler "Qwen/Qwen3-8B" \
    "$RESULTS_V2/Qwen--Qwen3-8B/baseline_longctx" \
    "Qwen3-8B base" "$LOG_8B"

# 2) Ours 8B
OURS_CKPT_8B="$ROOT/../hf_model_comp/comp_qtip/ckpt/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"
OURS_HF_8B="$ROOT/../hf_model_comp/comp_qtip/hf/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"
OURS_RES_8B="$RESULTS_V2/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"

cd "$ROOT"
for lmbda in 10000 1000 300 100 50 30; do
    ckpt_dir="${OURS_CKPT_8B}/lmbda${lmbda}"
    hf_dir="${OURS_HF_8B}/lmbda${lmbda}"
    out_base="${OURS_RES_8B}/lmbda${lmbda}_longctx"
    [ -d "$ckpt_dir" ] || { echo "[Ours 8B λ=${lmbda}] ckpt missing — skip" | tee -a "$LOG_8B"; continue; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[Ours 8B λ=${lmbda}] hfize" | tee -a "$LOG_8B"
        "$HFIZE_PY" -m quantize_llama.hfize_llama \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-8B 2>&1 | tee -a "$LOG_8B"
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[Ours 8B λ=${lmbda}] hfize fail — skip" | tee -a "$LOG_8B"; continue; }
    run_ruler "$hf_dir" "$out_base" "Ours 8B λ=${lmbda}" "$LOG_8B"
    rm -rf "$hf_dir"
done

# 3) QTIP 8B
QTIP_CKPT_8B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_8b/ft1"
QTIP_HF_8B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_8b/ft1"
QTIP_RES_8B="$RESULTS_V2/qtip/qwen3_8b/ft1"

cd "$QTIP_ROOT"
for K in 6 5 4 3 2; do
    ckpt_dir="${QTIP_CKPT_8B}/${K}bit"
    hf_dir="${QTIP_HF_8B}/${K}bit"
    out_base="${QTIP_RES_8B}/${K}bit_longctx"
    [ -d "$ckpt_dir" ] || { echo "[QTIP 8B K=${K}] ckpt missing — skip" | tee -a "$LOG_8B"; continue; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[QTIP 8B K=${K}] hfize_hf" | tee -a "$LOG_8B"
        "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-8B 2>&1 | tee -a "$LOG_8B"
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[QTIP 8B K=${K}] hfize fail — skip" | tee -a "$LOG_8B"; continue; }
    run_ruler "$hf_dir" "$out_base" "QTIP 8B K=${K}" "$LOG_8B"
    rm -rf "$hf_dir"
done

python3 "$NOTIFY_PY" "Qwen3-8B RULER 모두 완료" "$OURS_RES_8B" "$QTIP_RES_8B" 2>&1 || true
echo "==================== RULER complete ===================="
