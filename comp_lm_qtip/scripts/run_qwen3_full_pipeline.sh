#!/bin/bash
# ##########################################################################
# ##  Master Qwen3 evaluation chain.
# ##  Stage 1: Qwen3-4B FP16 baseline reasoning eval (AIME-24/25, GPQA-Diamond)
# ##  Stage 2: Qwen3-4B compressed (5 lambdas) GPQA-Diamond eval (AIME pre-done)
# ##  Stage 3: Qwen3-8B compression sweep + AIME-24/25 + GPQA-Diamond eval
# ##  Slack notification after each stage with task accuracies.
# ##########################################################################
set -u

ROOT="/home/jgryu/workspace/weight_compression/comp_lm_qtip"
PARO_ROOT="/home/jgryu/workspace/weight_compression/paroquant"
EVAL_PY="/opt/conda/envs/paroquant-eval/bin/python"
NOTIFY_PY="$ROOT/scripts/_paroquant_notify.py"

export HF_HOME=/home/jgryu/.cache/huggingface
if [ -f "$HF_HOME/token" ]; then
    HF_TOKEN_VAL="$(cat "$HF_HOME/token")"
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH

RES_ROOT="../hf_model_comp_results_v2"
LOG="./log"
DATASETS=(AIME-2024 AIME-2025 GPQA-Diamond)
SEED=42

mkdir -p "$LOG"

cd "$ROOT"

##########################################################################
## Stage 1: Qwen3-4B FP16 baseline
##########################################################################
echo "==================== Stage 1: Qwen3-4B FP16 baseline ===================="
BASELINE_OUT="${RES_ROOT}/Qwen--Qwen3-4B/baseline_paroquant_reasoning"
BASELINE_LOG="${LOG}/Qwen--Qwen3-4B/baseline_paroquant_reasoning.log"
mkdir -p "$BASELINE_OUT" "$(dirname "$BASELINE_LOG")"

for ds in "${DATASETS[@]}"; do
    echo "---- Stage 1 | Dataset: $ds ----" | tee -a "$BASELINE_LOG"
    (cd "$PARO_ROOT" && "$EVAL_PY" \
        -m experiments.tasks.reasoning.lighteval_custom.inference \
        --model "Qwen/Qwen3-4B" \
        --dataset "$ds" \
        --seed "$SEED" \
        --output_dir "$BASELINE_OUT") 2>&1 | tee -a "$BASELINE_LOG"
done

python3 "$NOTIFY_PY" "Stage 1: Qwen3-4B FP16 baseline 완료" "$BASELINE_OUT" \
    2>&1 | tee -a "$BASELINE_LOG"

##########################################################################
## Stage 2: Qwen3-4B compressed — GPQA-Diamond (AIME results already exist)
##########################################################################
echo "==================== Stage 2: Qwen3-4B compressed reasoning eval ===================="
# eval_reasoning_paroquant.sh handles hfize → eval (AIME auto-skips) → cleanup.
bash scripts/eval_reasoning_paroquant.sh

python3 "$NOTIFY_PY" "Stage 2: Qwen3-4B 압축 (5 λ) reasoning 완료" \
    "${RES_ROOT}/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft" \
    2>&1 | tee -a "$LOG/Qwen--Qwen3-4B/ql_ldlq128_rnorm_ft/stage2_summary.log"

##########################################################################
## Stage 2.5: QTIP baseline on Qwen3-4B (K=2,3,4) + reasoning eval
##########################################################################
echo "==================== Stage 2.5: QTIP Qwen3-4B (K=2,3,4) ===================="
QTIP_ROOT="/home/jgryu/workspace/weight_compression/qtip"
( cd "$QTIP_ROOT" && bash run_qtip_qwen3_4b.sh )
( cd "$QTIP_ROOT" && bash run_qtip_qwen3_4b_eval.sh )

##########################################################################
## Stage 3: Qwen3-8B compression sweep + reasoning eval
##########################################################################
echo "==================== Stage 3a: Qwen3-8B compression sweep ===================="
bash scripts/comp_lm_reasoning_8b.sh

echo "==================== Stage 3b: Qwen3-8B reasoning eval ===================="
# Switch eval_reasoning_paroquant.sh to evaluate Qwen3-8B by uncommenting the
# 8B model_names entry. (Idempotent; re-running won't change a 4B-only file.)
sed -i 's|^    # "Qwen--Qwen3-8B"|    "Qwen--Qwen3-8B"|' scripts/eval_reasoning_paroquant.sh

# Limit Stage 3b to 8B only; comment out 4B entry (already evaluated in Stage 2).
sed -i 's|^    "Qwen--Qwen3-4B"$|    # "Qwen--Qwen3-4B"  # done in Stage 2|' scripts/eval_reasoning_paroquant.sh

bash scripts/eval_reasoning_paroquant.sh

# Restore script: re-enable 4B + comment out 8B for tidiness.
sed -i 's|^    # "Qwen--Qwen3-4B"  # done in Stage 2|    "Qwen--Qwen3-4B"|' scripts/eval_reasoning_paroquant.sh
sed -i 's|^    "Qwen--Qwen3-8B"$|    # "Qwen--Qwen3-8B"  # 8B compression not yet done; enable after compress sweep|' scripts/eval_reasoning_paroquant.sh

python3 "$NOTIFY_PY" "Stage 3: Qwen3-8B 압축 (6 λ) reasoning 완료" \
    "${RES_ROOT}/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft" \
    2>&1 | tee -a "$LOG/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft/stage3_summary.log"

echo "==================== All 3 stages complete ===================="
