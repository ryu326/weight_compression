#!/bin/bash
##########################################################################
##  Needle-in-a-Haystack evaluation — Qwen3-4B and Qwen3-8B
##  Context lengths: 1K 2K 4K 8K 16K 32K, Depths: 10%..100%
##  Results: hf_model_comp_results_v2/<model>/<config>_longctx/niah.json
##########################################################################
set -u

ROOT="/home/jgryu/workspace/weight_compression/comp_lm_qtip"
QTIP_ROOT="/home/jgryu/workspace/weight_compression/qtip"
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

GPUS="0,1,2,3,4,5,6,7"
LOG_FILE="$ROOT/log/niah.log"
mkdir -p "$(dirname "$LOG_FILE")"

run_niah() {
    local hf_path="$1" out_dir="$2" label="$3"
    local result="${out_dir}/niah.json"
    mkdir -p "$out_dir"
    if [ -f "$result" ]; then
        echo "[$label] niah.json exists — skip" | tee -a "$LOG_FILE"
        return 0
    fi
    echo "[$label] NIAH" | tee -a "$LOG_FILE"
    "$EVAL_PY" "$ROOT/eval/niah.py" \
        --model "$hf_path" \
        --output-dir "$out_dir" \
        --gpus "$GPUS" \
        2>&1 | tee -a "$LOG_FILE"
    python3 "$NOTIFY_PY" "[$label] NIAH 완료" "$out_dir" 2>&1 || true
}

##########################################################################
## Block A: Qwen3-4B
##########################################################################
echo "========== Qwen3-4B NIAH ==========" | tee -a "$LOG_FILE"

run_niah "Qwen/Qwen3-4B" \
    "$RESULTS_V2/Qwen--Qwen3-4B/baseline_longctx" \
    "Qwen3-4B base"

OURS_CKPT_4B="$ROOT/../hf_model_comp/comp_qtip/ckpt/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
OURS_HF_4B="$ROOT/../hf_model_comp/comp_qtip/hf/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
OURS_RES_4B="$RESULTS_V2/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"

cd "$ROOT"
for lmbda in 10000 1000 300 100 50; do
    ckpt_dir="${OURS_CKPT_4B}/lmbda${lmbda}"
    hf_dir="${OURS_HF_4B}/lmbda${lmbda}"
    out_dir="${OURS_RES_4B}/lmbda${lmbda}_longctx"
    [ -d "$ckpt_dir" ] || { echo "[Ours 4B λ=${lmbda}] ckpt missing — skip" | tee -a "$LOG_FILE"; continue; }
    if [ ! -f "$hf_dir/config.json" ]; then
        "$HFIZE_PY" -m quantize_llama.hfize_llama \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-4B 2>&1 | tee -a "$LOG_FILE"
    fi
    [ -f "$hf_dir/config.json" ] || continue
    run_niah "$hf_dir" "$out_dir" "Ours 4B λ=${lmbda}"
    rm -rf "$hf_dir"
done

QTIP_CKPT_4B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_4b/ft1"
QTIP_HF_4B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_4b/ft1"
QTIP_RES_4B="$RESULTS_V2/qtip/qwen3_4b/ft1"

cd "$QTIP_ROOT"
for K in 6 5 4 3 2; do
    ckpt_dir="${QTIP_CKPT_4B}/${K}bit"
    hf_dir="${QTIP_HF_4B}/${K}bit"
    out_dir="${QTIP_RES_4B}/${K}bit_longctx"
    [ -d "$ckpt_dir" ] || continue
    if [ ! -f "$hf_dir/config.json" ]; then
        "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-4B 2>&1 | tee -a "$LOG_FILE"
    fi
    [ -f "$hf_dir/config.json" ] || continue
    run_niah "$hf_dir" "$out_dir" "QTIP 4B K=${K}"
    rm -rf "$hf_dir"
done

python3 "$NOTIFY_PY" "Qwen3-4B NIAH 모두 완료" 2>&1 || true

##########################################################################
## Block B: Qwen3-8B
##########################################################################
echo "========== Qwen3-8B NIAH ==========" | tee -a "$LOG_FILE"

run_niah "Qwen/Qwen3-8B" \
    "$RESULTS_V2/Qwen--Qwen3-8B/baseline_longctx" \
    "Qwen3-8B base"

OURS_CKPT_8B="$ROOT/../hf_model_comp/comp_qtip/ckpt/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"
OURS_HF_8B="$ROOT/../hf_model_comp/comp_qtip/hf/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"
OURS_RES_8B="$RESULTS_V2/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"

cd "$ROOT"
for lmbda in 10000 1000 300 100 50 30; do
    ckpt_dir="${OURS_CKPT_8B}/lmbda${lmbda}"
    hf_dir="${OURS_HF_8B}/lmbda${lmbda}"
    out_dir="${OURS_RES_8B}/lmbda${lmbda}_longctx"
    [ -d "$ckpt_dir" ] || continue
    if [ ! -f "$hf_dir/config.json" ]; then
        "$HFIZE_PY" -m quantize_llama.hfize_llama \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-8B 2>&1 | tee -a "$LOG_FILE"
    fi
    [ -f "$hf_dir/config.json" ] || continue
    run_niah "$hf_dir" "$out_dir" "Ours 8B λ=${lmbda}"
    rm -rf "$hf_dir"
done

QTIP_CKPT_8B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_8b/ft1"
QTIP_HF_8B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_8b/ft1"
QTIP_RES_8B="$RESULTS_V2/qtip/qwen3_8b/ft1"

cd "$QTIP_ROOT"
for K in 6 5 4 3 2; do
    ckpt_dir="${QTIP_CKPT_8B}/${K}bit"
    hf_dir="${QTIP_HF_8B}/${K}bit"
    out_dir="${QTIP_RES_8B}/${K}bit_longctx"
    [ -d "$ckpt_dir" ] || continue
    if [ ! -f "$hf_dir/config.json" ]; then
        "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-8B 2>&1 | tee -a "$LOG_FILE"
    fi
    [ -f "$hf_dir/config.json" ] || continue
    run_niah "$hf_dir" "$out_dir" "QTIP 8B K=${K}"
    rm -rf "$hf_dir"
done

python3 "$NOTIFY_PY" "Qwen3-8B NIAH 모두 완료" 2>&1 || true
echo "==================== NIAH complete ===================="
