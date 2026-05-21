#!/bin/bash
# ##########################################################################
# ##  ParoQuant-style reasoning eval (lighteval + vLLM)
# ##  Datasets: AIME-2024 (30), AIME-2025 (30), GPQA-Diamond (198), MMLU-PRO (12k)
# ##  Iterates over HF dirs produced by comp_lm_reasoning.sh.
# ##########################################################################
set -u

PARO_ROOT="/home/jgryu/workspace/weight_compression/paroquant"
PYTHON_BIN="/opt/conda/envs/paroquant-eval/bin/python"
HFIZE_PYTHON_BIN="/opt/conda/envs/qwen3/bin/python"   # transformers 4.51 for Qwen3 hfize

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: paroquant-eval env not found at $PYTHON_BIN" >&2
    echo "Create it with:" >&2
    echo "  conda create -n paroquant-eval python=3.11 -y && \\" >&2
    echo "  $PYTHON_BIN -m pip install -r $PARO_ROOT/experiments/tasks/reasoning/requirements.txt" >&2
    exit 1
fi
if [ ! -x "$HFIZE_PYTHON_BIN" ]; then
    echo "ERROR: qwen3 conda env not found at $HFIZE_PYTHON_BIN (needed for hfize)" >&2
    exit 1
fi

# Per-model HF base id used by hfize (Qwen3-4B, Qwen3-8B, ...). Matches model_names below.
declare -A BASE_MODEL_OF
BASE_MODEL_OF["Qwen--Qwen3-4B"]="Qwen/Qwen3-4B"
BASE_MODEL_OF["Qwen--Qwen3-8B"]="Qwen/Qwen3-8B"

export HF_HOME=/home/jgryu/.cache/huggingface
# Online mode + HF token (from ~/.cache/huggingface/token) so all four
# datasets (incl. gated Idavidrein/gpqa Diamond) can be downloaded/cached.
if [ -f "$HF_HOME/token" ]; then
    HF_TOKEN_FILE="$HF_HOME/token"
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
# All 8 GPUs available (compression paused). vLLM tensor_parallel_size =
# #visible GPUs (set in inference.py).
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH

##########################################################################
##                 PIPELINE OUTPUT LAYOUT (matches comp_lm_reasoning.sh)
##########################################################################
CKPT_ROOT="../hf_model_comp/comp_qtip/ckpt"
HF_ROOT="../hf_model_comp/comp_qtip/hf"
RES_ROOT="../hf_model_comp_results_v2"
LOG="./log"

##########################################################################
##                          EVAL CONFIGURATION
##########################################################################
model_names=(
    "Qwen--Qwen3-4B"
    "Qwen--Qwen3-8B"  # 8B compression not yet done; enable after compress sweep
)
exp_name="ql_ldlq128_rnorm_ft"
lmbda_values=(100 300 1000 10000 50 30)

# ParoQuant Table 2 datasets — MMLU-PRO dropped per user (too slow given
# Qwen3 reasoning trace lengths). GPQA-Diamond enabled now that HF token
# is configured.
DATASETS=(AIME-2024 AIME-2025 GPQA-Diamond)
SEED=42

# Per-dataset --max_samples override. Empty means no limit. AIME-24/25 are
# 30 samples each; GPQA-Diamond is 198 samples — all run fully.
declare -A MAX_SAMPLES_OF

mkdir -p "$LOG"

##########################################################################
##                                MAIN LOOP
##########################################################################
for model_name in "${model_names[@]}"; do
    base_model="${BASE_MODEL_OF[$model_name]:-}"
    if [ -z "$base_model" ]; then
        echo "ERROR: no BASE_MODEL_OF entry for $model_name. Add to the assoc array." >&2
        exit 1
    fi
    for lmbda in "${lmbda_values[@]}"; do
        SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}
        ckpt_path="${CKPT_ROOT}/${SAVE_NAME}"
        hf_path="${HF_ROOT}/${SAVE_NAME}"
        out_dir="${RES_ROOT}/${SAVE_NAME}_paroquant_reasoning"
        log_path="${LOG}/${SAVE_NAME}_paroquant_reasoning.log"
        mkdir -p "$(dirname "$log_path")" "$out_dir"

        # Step 1: hfize from ckpt if HF dir is missing.
        if [ ! -d "$hf_path" ]; then
            if [ ! -d "$ckpt_path" ]; then
                echo "WARN: ckpt missing: $ckpt_path (skip $model_name lmbda=$lmbda)" | tee -a "$log_path"
                continue
            fi
            echo "---- hfize from ckpt -> $hf_path ----" | tee -a "$log_path"
            HF_HOME=/home/jgryu/.cache/huggingface "$HFIZE_PYTHON_BIN" \
                -m quantize_llama.hfize_llama \
                --quantized_path "$ckpt_path" \
                --hf_output_path "$hf_path" \
                --base_model "$base_model" 2>&1 | tee -a "$log_path"
            if [ ! -d "$hf_path" ]; then
                echo "WARN: hfize failed for $SAVE_NAME (skip)" | tee -a "$log_path"
                continue
            fi
        fi

        echo "================================================================" | tee -a "$log_path"
        echo " Reasoning eval | model=${model_name} | lmbda=${lmbda}" | tee -a "$log_path"
        echo " HF path        : $hf_path" | tee -a "$log_path"
        echo " Output dir     : $out_dir" | tee -a "$log_path"
        echo " Datasets       : ${DATASETS[*]}" | tee -a "$log_path"
        echo "================================================================" | tee -a "$log_path"

        # Step 2: run all reasoning datasets against the HF dir.
        for ds in "${DATASETS[@]}"; do
            echo "---- Dataset: $ds ----" | tee -a "$log_path"
            ms_arg=()
            ms_val="${MAX_SAMPLES_OF[$ds]:-}"
            if [ -n "$ms_val" ]; then
                ms_arg=(--max_samples "$ms_val")
            fi
            (cd "$PARO_ROOT" && "$PYTHON_BIN" \
                -m experiments.tasks.reasoning.lighteval_custom.inference \
                --model "$hf_path" \
                --dataset "$ds" \
                --seed "$SEED" \
                --output_dir "$out_dir" \
                "${ms_arg[@]}") 2>&1 | tee -a "$log_path"
        done

        # Step 3: cleanup HF dir to save disk; ckpt is preserved.
        if [ -d "$hf_path" ] && [ "$hf_path" != "$HF_ROOT" ]; then
            echo "---- Cleanup: rm -rf $hf_path (ckpt at $ckpt_path retained) ----" | tee -a "$log_path"
            rm -rf "$hf_path"
        fi
    done
done

echo "All paroquant reasoning evaluations complete." | tee -a "$LOG/paroquant_eval_summary.log"
