#!/bin/bash
# OOM test: try different ft_bs x ctx_size combinations
# Only runs quantization (no eval) to check memory

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/home/jgryu/.cache/huggingface

CKPT="../hf_model_comp/comp_qtip/ckpt"
RES="../hf_model_comp_results_v2"
model_name="meta-llama--Meta-Llama-3-8B"
hess_path="../Wparam_dataset/quip_hess/llama3_8b_6144"
lm_model_path="../Wparam_dataset/hf_model/${model_name}"

# Combinations to test: (ft_bs, ctx_size)
combos=(
    "1 8"
    "1 4"
    "1 16"
    "2 8"
    "2 16"
    "4 8"
    "4 16"
)

for combo in "${combos[@]}"; do
    read -r bs ctx <<< "$combo"
    save_name="${model_name}/oom_test/bs${bs}_ctx${ctx}"
    echo ""
    echo "========================================================================"
    echo "TESTING: ft_bs=${bs}, ctx_size=${ctx}"
    echo "========================================================================"

    timeout 300 python -m quantize_llama.quantize_finetune_llama \
        --save_path "${CKPT}/${save_name}" \
        --base_model "${lm_model_path}" \
        --ecft_lmbda 1 \
        --in_hess_path "${hess_path}" \
        --devset_size 384 \
        --ft_valid_size 128 \
        --batch_size 8 \
        --ecft_decoder --ft_epochs 5 --ft_bs ${bs} --ctx_size ${ctx} --ec_linear \
        --row_normalize --scaleHinv --ecft_epochs 0 --ecft_aux_warmup_step 500 \
        --R_target 2 --ec_decoder_type rht --ecft_adaptive_lambda --ecft_lambda_lr 0.1 \
        --res_path "${RES}/${save_name}" \
        2>&1

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo ">>> SUCCESS: ft_bs=${bs}, ctx_size=${ctx} completed without OOM"
        # Clean up
        rm -rf "${CKPT}/${save_name}"
    elif [ $exit_code -eq 124 ]; then
        echo ">>> TIMEOUT (still running after 5min, likely OK): ft_bs=${bs}, ctx_size=${ctx}"
        rm -rf "${CKPT}/${save_name}"
    else
        echo ">>> FAILED (exit code ${exit_code}): ft_bs=${bs}, ctx_size=${ctx}"
        rm -rf "${CKPT}/${save_name}"
    fi

    # Clear GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    sleep 5
done
