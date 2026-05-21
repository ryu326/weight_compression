#!/bin/bash
##########################################################################
## Qwen3-8B FP16 baseline reasoning eval (AIME-2024, AIME-2025, GPQA-Diamond)
## Output: hf_model_comp_results_v2/Qwen--Qwen3-8B/baseline_paroquant_reasoning/
##########################################################################
set -u

PARO_ROOT="/home/jgryu/workspace/weight_compression/paroquant"
EVAL_PY="/opt/conda/envs/paroquant-eval/bin/python"
NOTIFY_PY="/home/jgryu/workspace/weight_compression/comp_lm_qtip/scripts/_paroquant_notify.py"

export HF_HOME=/home/jgryu/.cache/huggingface
if [ -f "$HF_HOME/token" ]; then
    HF_TOKEN_VAL="$(cat "$HF_HOME/token")"
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH

BASELINE_OUT="/home/jgryu/workspace/weight_compression/hf_model_comp_results_v2/Qwen--Qwen3-8B/baseline_paroquant_reasoning"
LOG_DIR="/home/jgryu/workspace/weight_compression/comp_lm_qtip/log/Qwen--Qwen3-8B"
mkdir -p "$BASELINE_OUT" "$LOG_DIR"
BASELINE_LOG="${LOG_DIR}/baseline_paroquant_reasoning.log"

DATASETS=(AIME-2024 AIME-2025 GPQA-Diamond)
SEED=42

for ds in "${DATASETS[@]}"; do
    echo "---- Qwen3-8B FP16 | Dataset: $ds ----" | tee -a "$BASELINE_LOG"
    (cd "$PARO_ROOT" && "$EVAL_PY" \
        -m experiments.tasks.reasoning.lighteval_custom.inference \
        --model "Qwen/Qwen3-8B" \
        --dataset "$ds" \
        --seed "$SEED" \
        --output_dir "$BASELINE_OUT") 2>&1 | tee -a "$BASELINE_LOG"
done

python3 "$NOTIFY_PY" "Qwen3-8B FP16 baseline reasoning 완료" "$BASELINE_OUT" \
    2>&1 | tee -a "$BASELINE_LOG"
