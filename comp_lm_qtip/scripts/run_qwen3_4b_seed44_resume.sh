#!/bin/bash
##########################################################################
##  Qwen3-4B 추가 seed=44 평가:
##    - Ours (comp_lm_qtip) 5 lambda × {AIME-24, AIME-25, GPQA-Diamond}
##    - QTIP 5 K × {AIME-24, AIME-25, GPQA-Diamond}
##  ckpt 는 보존, HF dir 은 eval 후 삭제.
##  결과 저장 경로 끝에 `_seed44` 접미사 — seed=42 결과 (덮어쓰지 않음).
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
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset PYTHONPATH

DATASETS=(AIME-2024 AIME-2025 GPQA-Diamond)
SEED=44

##########################################################################
## Block 1: Ours (comp_lm_qtip) 5 lambda
##########################################################################
LMBDA_VALUES=(1000 10000)
CKPT_OURS_ROOT="$ROOT/../hf_model_comp/comp_qtip/ckpt/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
HF_OURS_ROOT="$ROOT/../hf_model_comp/comp_qtip/hf/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"
RES_OURS_ROOT="$ROOT/../hf_model_comp_results_v2/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft"

cd "$ROOT"

for lmbda in "${LMBDA_VALUES[@]}"; do
    ckpt_dir="${CKPT_OURS_ROOT}/lmbda${lmbda}"
    hf_dir="${HF_OURS_ROOT}/lmbda${lmbda}"
    out_dir="${RES_OURS_ROOT}/lmbda${lmbda}_paroquant_reasoning_seed${SEED}"
    log_file="$ROOT/log/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft/lmbda${lmbda}_seed${SEED}.log"
    mkdir -p "$(dirname "$log_file")" "$out_dir"

    if [ ! -d "$ckpt_dir" ]; then
        echo "[Ours λ=${lmbda}] ckpt missing, skip" | tee -a "$log_file"
        continue
    fi

    echo "============================================================" | tee "$log_file"
    echo " Ours λ=${lmbda} | seed=${SEED}" | tee -a "$log_file"
    echo "============================================================" | tee -a "$log_file"

    # 1) hfize
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[Ours λ=${lmbda}] hfize" | tee -a "$log_file"
        "$HFIZE_PY" -m quantize_llama.hfize_llama \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-4B 2>&1 | tee -a "$log_file"
    fi
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[Ours λ=${lmbda}] hfize failed (skip)" | tee -a "$log_file"
        continue
    fi

    # 2) eval all datasets with seed=44
    for ds in "${DATASETS[@]}"; do
        echo "[Ours λ=${lmbda}] eval $ds seed=${SEED}" | tee -a "$log_file"
        (cd "$PARO_ROOT" && "$EVAL_PY" \
            -m experiments.tasks.reasoning.lighteval_custom.inference \
            --model "$hf_dir" \
            --dataset "$ds" \
            --seed "$SEED" \
            --output_dir "$out_dir") 2>&1 | tee -a "$log_file"
    done

    # 3) cleanup HF dir
    echo "[Ours λ=${lmbda}] cleanup $hf_dir" | tee -a "$log_file"
    rm -rf "$hf_dir"

    # 4) slack per-λ
    python3 "$NOTIFY_PY" "Qwen3-4B Ours λ=${lmbda} seed=${SEED} 완료" "$out_dir" \
        2>&1 | tee -a "$log_file"
done

##########################################################################
## Block 2: QTIP 5 K
##########################################################################
K_VALUES=(2 3 4 5 6)
CKPT_QTIP_ROOT="$QTIP_ROOT/../hf_model_comp/qtip/ckpt/qwen3_4b/ft1"
HF_QTIP_ROOT="$QTIP_ROOT/../hf_model_comp/qtip/hf/qwen3_4b/ft1"
RES_QTIP_ROOT="$QTIP_ROOT/../hf_model_comp_results_v2/qtip/qwen3_4b/ft1"

cd "$QTIP_ROOT"

for K in "${K_VALUES[@]}"; do
    ckpt_dir="${CKPT_QTIP_ROOT}/${K}bit"
    hf_dir="${HF_QTIP_ROOT}/${K}bit"
    out_dir="${RES_QTIP_ROOT}/${K}bit_paroquant_reasoning_seed${SEED}"
    log_file="$QTIP_ROOT/log/qwen3_4b/ft1/${K}bit_seed${SEED}.log"
    mkdir -p "$(dirname "$log_file")" "$out_dir"

    if [ ! -d "$ckpt_dir" ]; then
        echo "[QTIP K=${K}] ckpt missing, skip" | tee -a "$log_file"
        continue
    fi

    echo "============================================================" | tee "$log_file"
    echo " QTIP K=${K}bit | seed=${SEED}" | tee -a "$log_file"
    echo "============================================================" | tee -a "$log_file"

    # 1) hfize (hfize_llama_hf for QTIP)
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[QTIP K=${K}] hfize_hf" | tee -a "$log_file"
        "$HFIZE_PY" -m quantize_llama.hfize_llama_hf \
            --quantized_path "$ckpt_dir" \
            --hf_output_path "$hf_dir" \
            --base_model Qwen/Qwen3-4B 2>&1 | tee -a "$log_file"
    fi
    if [ ! -f "$hf_dir/config.json" ]; then
        echo "[QTIP K=${K}] hfize failed (skip)" | tee -a "$log_file"
        continue
    fi

    # 2) eval all datasets with seed=44
    for ds in "${DATASETS[@]}"; do
        echo "[QTIP K=${K}] eval $ds seed=${SEED}" | tee -a "$log_file"
        (cd "$PARO_ROOT" && "$EVAL_PY" \
            -m experiments.tasks.reasoning.lighteval_custom.inference \
            --model "$hf_dir" \
            --dataset "$ds" \
            --seed "$SEED" \
            --output_dir "$out_dir") 2>&1 | tee -a "$log_file"
    done

    # 3) cleanup HF dir
    echo "[QTIP K=${K}] cleanup $hf_dir" | tee -a "$log_file"
    rm -rf "$hf_dir"

    # 4) slack per-K
    python3 "$NOTIFY_PY" "QTIP Qwen3-4B K=${K}bit seed=${SEED} 완료" "$out_dir" \
        2>&1 | tee -a "$log_file"
done

# Final slack summary
python3 "$NOTIFY_PY" "Qwen3-4B seed=${SEED} (Ours+QTIP) 모두 완료" \
    "$RES_OURS_ROOT" "$RES_QTIP_ROOT" 2>&1 || true

echo "==================== seed=${SEED} pipeline complete ===================="
