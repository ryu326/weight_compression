#!/bin/bash
##########################################################################
##  LongBench V1 evaluation — Qwen3-8B then Qwen3-4B
##  Ours (λ) and QTIP (K) interleaved in reverse order:
##    baseline → (λ10000,K6) → (λ1000,K5) → (λ300,K4) → (λ100,K3) → (λ50,K2)
##    8B only: → λ30
##  All configs use GPU 0-7 (tensor_parallel_size=8)
##  Results: hf_model_comp_results_v2/<model>/<config>_longbenchV1/
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
LOG_FILE="$ROOT/log/longbench.log"
mkdir -p "$(dirname "$LOG_FILE")"

run_longbench() {
    # $1=hf_path  $2=out_dir  $3=label
    local hf_path="$1" out_dir="$2" label="$3"
    local result="${out_dir}/longbenchV1.json"
    mkdir -p "$out_dir"
    if [ -f "$result" ]; then
        echo "[$label] longbenchV1.json exists — skip" | tee -a "$LOG_FILE"
        return 0
    fi
    echo "[$label] LongBench V1" | tee -a "$LOG_FILE"
    "$EVAL_PY" "$ROOT/eval/longbench.py" \
        --model "$hf_path" \
        --output-dir "$out_dir" \
        --gpus "$GPUS" \
        2>&1 | tee -a "$LOG_FILE"
    python3 "$NOTIFY_PY" "[$label] LongBench 완료" "$out_dir" 2>&1 || true
}

run_ours() {
    # $1=ckpt_base $2=hf_base $3=res_base $4=lmbda $5=base_model $6=label_prefix
    local ckpt_dir="${1}/lmbda${4}"
    local hf_dir="${2}/lmbda${4}"
    local out_dir="${3}/lmbda${4}_longbenchV1"
    local label="${6} λ=${4}"
    [ -d "$ckpt_dir" ] || { echo "[$label] ckpt missing — skip" | tee -a "$LOG_FILE"; return 0; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[$label] hfize" | tee -a "$LOG_FILE"
        (cd "$ROOT" && "$HFIZE_PY" -m quantize_llama.hfize_llama \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model "$5" 2>&1 | tee -a "$LOG_FILE")
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[$label] hfize fail — skip" | tee -a "$LOG_FILE"; return 0; }
    run_longbench "$hf_dir" "$out_dir" "$label"
    rm -rf "$hf_dir"
}

run_qtip() {
    # $1=ckpt_base $2=hf_base $3=res_base $4=K $5=base_model $6=label_prefix
    local ckpt_dir="${1}/${4}bit"
    local hf_dir="${2}/${4}bit"
    local out_dir="${3}/${4}bit_longbenchV1"
    local label="${6} K=${4}"
    [ -d "$ckpt_dir" ] || { echo "[$label] ckpt missing — skip" | tee -a "$LOG_FILE"; return 0; }
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[$label] hfize_hf" | tee -a "$LOG_FILE"
        (cd "$QTIP_ROOT" && "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model "$5" 2>&1 | tee -a "$LOG_FILE")
    fi
    [ -f "$hf_dir/config.json" ] || { echo "[$label] hfize fail — skip" | tee -a "$LOG_FILE"; return 0; }
    run_longbench "$hf_dir" "$out_dir" "$label"
    rm -rf "$hf_dir"
}

##########################################################################
## Block A: Qwen3-8B  (8 GPUs)
##########################################################################
echo "========== Qwen3-8B LongBench ==========" | tee -a "$LOG_FILE"

OURS_CKPT_8B="$ROOT/../hf_model_comp/comp_qtip/ckpt/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"
OURS_HF_8B="$ROOT/../hf_model_comp/comp_qtip/hf/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"
OURS_RES_8B="$RESULTS_V2/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"
QTIP_CKPT_8B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_8b/ft1"
QTIP_HF_8B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_8b/ft1"
QTIP_RES_8B="$RESULTS_V2/qtip/qwen3_8b/ft1"

run_longbench "Qwen/Qwen3-8B" "$RESULTS_V2/Qwen--Qwen3-8B/baseline_longbenchV1" "Qwen3-8B base"

# Interleaved: λ10000+K6, λ1000+K5, λ300+K4, λ100+K3, λ50+K2, then λ30
lmbdas=(10000 1000 300 100 50)
Ks=(6 5 4 3 2)
for i in "${!lmbdas[@]}"; do
    run_ours  "$OURS_CKPT_8B" "$OURS_HF_8B" "$OURS_RES_8B" "${lmbdas[$i]}" "Qwen/Qwen3-8B" "Ours 8B"
    run_qtip  "$QTIP_CKPT_8B" "$QTIP_HF_8B" "$QTIP_RES_8B" "${Ks[$i]}"     "Qwen/Qwen3-8B" "QTIP 8B"
done
run_ours "$OURS_CKPT_8B" "$OURS_HF_8B" "$OURS_RES_8B" 30 "Qwen/Qwen3-8B" "Ours 8B"

python3 "$NOTIFY_PY" "Qwen3-8B LongBench 모두 완료" "$OURS_RES_8B" "$QTIP_RES_8B" 2>&1 || true

##########################################################################
## Block B: Qwen3-4B  (8 GPUs)
##########################################################################
echo "========== Qwen3-4B LongBench ==========" | tee -a "$LOG_FILE"

OURS_CKPT_4B="$ROOT/../hf_model_comp/comp_qtip/ckpt/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
OURS_HF_4B="$ROOT/../hf_model_comp/comp_qtip/hf/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
OURS_RES_4B="$RESULTS_V2/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
QTIP_CKPT_4B="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_4b/ft1"
QTIP_HF_4B="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_4b/ft1"
QTIP_RES_4B="$RESULTS_V2/qtip/qwen3_4b/ft1"

run_longbench "Qwen/Qwen3-4B" "$RESULTS_V2/Qwen--Qwen3-4B/baseline_longbenchV1" "Qwen3-4B base"

# Interleaved: λ10000+K6, λ1000+K5, λ300+K4, λ100+K3, λ50+K2
for i in "${!lmbdas[@]}"; do
    run_ours  "$OURS_CKPT_4B" "$OURS_HF_4B" "$OURS_RES_4B" "${lmbdas[$i]}" "Qwen/Qwen3-4B" "Ours 4B"
    run_qtip  "$QTIP_CKPT_4B" "$QTIP_HF_4B" "$QTIP_RES_4B" "${Ks[$i]}"     "Qwen/Qwen3-4B" "QTIP 4B"
done

python3 "$NOTIFY_PY" "Qwen3-4B LongBench 모두 완료" "$OURS_RES_4B" "$QTIP_RES_4B" 2>&1 || true
echo "==================== LongBench complete ===================="
