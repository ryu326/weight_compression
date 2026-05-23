#!/bin/bash
##########################################################################
##  MMLU-Pro balanced evaluation (category-stratified sampling)
##
##  Replaces the biased first-2000 approach (run_mmlu_pro_2k.sh) which
##  only covered business + law. This script samples N_PER_CAT examples
##  per category (default 143 → ~2002 total, all 14 categories).
##
##  Usage:
##    bash run_mmlu_pro_balanced.sh <hf_model_path> <output_dir> [gpus]
##
##  Example (Llama3.1-8B-Instruct, single GPU):
##    bash run_mmlu_pro_balanced.sh \
##        meta-llama/Meta-Llama-3.1-8B-Instruct \
##        /results/llama3.1-8b/baseline_reasoning \
##        0,1,2,3
##
##  Output: <output_dir>/MMLU-PRO.jsonl
##  Format: JSON array compatible with plot_bpp_reasoning.ipynb loader.
##########################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY_PATH="$SCRIPT_DIR/../eval/eval_mmlu_pro.py"

EVAL_PY="${PAROQUANT_EVAL_PYTHON:-/opt/conda/envs/paroquant-eval/bin/python}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <hf_model_path> <output_dir> [gpus]"
    echo "  hf_model_path : HF model id or local path"
    echo "  output_dir    : directory to save MMLU-PRO.jsonl"
    echo "  gpus          : comma-separated GPU ids (default: 0,1,2,3,4,5,6,7)"
    exit 1
fi

MODEL="$1"
OUT_DIR="$2"
GPUS="${3:-0,1,2,3,4,5,6,7}"
N_PER_CAT="${MMLU_N_PER_CAT:-143}"   # 143 × 14 categories = 2002 samples
SEED="${MMLU_SEED:-42}"

export HF_HOME="${HF_HOME:-/home/jgryu/.cache/huggingface}"
if [ -f "$HF_HOME/token" ]; then
    export HF_TOKEN="$(cat "$HF_HOME/token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
unset PYTHONPATH

echo "================================================================"
echo " MMLU-Pro balanced eval"
echo " Model    : $MODEL"
echo " Output   : $OUT_DIR"
echo " GPUs     : $GPUS"
echo " N/cat    : $N_PER_CAT  (~$((N_PER_CAT * 14)) total)"
echo " Seed     : $SEED"
echo "================================================================"

"$EVAL_PY" "$EVAL_PY_PATH" \
    --model    "$MODEL" \
    --output-dir "$OUT_DIR" \
    --gpus     "$GPUS" \
    --n-per-cat "$N_PER_CAT" \
    --seed     "$SEED"

echo "Done → $OUT_DIR/MMLU-PRO.jsonl"
