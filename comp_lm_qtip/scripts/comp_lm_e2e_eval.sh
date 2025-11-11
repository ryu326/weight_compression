#!/bin/bash

comp_model_bases=(
    "dumy"
)
quantize_flags=(
    "dumy"
)
experiment_names=(
    "ql_ldlq128_rnorm_ft"
)

model_names=(
    "meta-llama--Meta-Llama-3-8B"
)
hess_paths=(
    "../Wparam_dataset/quip_hess/llama3_8b_6144"
)

CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"
RES="../hf_model_comp_results"

mkdir -p "$CKPT"
mkdir -p "$HF"
mkdir -p "$LOG"
mkdir -p "$RES"

export HF_HOME=/home/jgryu/.cache/huggingface

lmbda_values=(50 100 300 1000 10000)
gpus=(3 4 5 6 7)

run_lambda() {
    local model_name="$1"
    local HESS="$2"
    local lm_model_path="$3"
    local exp_name="$4"
    local comp_model_base="$5"
    local current_quantize_flags="$6"
    local lmbda="$7"
    local gpu="$8"

    local SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}
    local E2EOUT_NAME=${model_name}/${exp_name}_e2e/lmbda${lmbda}

    mkdir -p "$(dirname "$LOG/$E2EOUT_NAME.log")"

    echo "======================================================================" 
    echo "[GPU $gpu] MODEL: $model_name | EXP: $exp_name | LAMBDA: $lmbda"
    echo "======================================================================"

    CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_zeroshot_hf \
        --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
        --batch_size 4 \
        --hf_path $CKPT/${E2EOUT_NAME} \
        --output_path $RES/${E2EOUT_NAME}_common_mmlu \
        > "$LOG/$E2EOUT_NAME.log"

    echo "[GPU $gpu] DONE: MODEL=$model_name | EXP=$exp_name | LAMBDA=$lmbda"
}

for j in "${!model_names[@]}"; do
    model_name="${model_names[$j]}"
    HESS="${hess_paths[$j]}"
    lm_model_path="../Wparam_dataset/hf_model/$model_name"

    echo "------------------------------------------------------------------------"
    echo "MODEL: $model_name"
    echo "HESS : $HESS"
    echo "------------------------------------------------------------------------"

    for i in "${!experiment_names[@]}"; do
        exp_name="${experiment_names[$i]}"
        comp_model_base="${comp_model_bases[$i]}"
        current_quantize_flags="${quantize_flags[$i]}"

        echo "========================================================================"
        echo "STARTING EXPERIMENT SET: $exp_name (MODEL: $model_name)"
        echo "========================================================================"

        pids=()
        for idx in "${!lmbda_values[@]}"; do
            lmbda="${lmbda_values[$idx]}"
            gpu="${gpus[$idx]}"

            run_lambda "$model_name" "$HESS" "$lm_model_path" \
                       "$exp_name" "$comp_model_base" "$current_quantize_flags" \
                       "$lmbda" "$gpu" &
            pids+=($!)
        done

        for pid in "${pids[@]}"; do
            wait "$pid"
        done

        echo "========================================================================"
        echo "FINISHED EXPERIMENT SET: $exp_name (MODEL: $model_name)"
        echo "========================================================================"
    done
done
