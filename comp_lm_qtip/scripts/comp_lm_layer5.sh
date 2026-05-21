#!/bin/bash
# Layer 5 only — NWC quantization with skip_list, GPU-parallel across lambdas
set -u
set -o pipefail

##########################################################################
##                       EXPERIMENT CONFIGURATION                       ##
##########################################################################
comp_model_bases=(
    "/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
    "/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
)
quantize_flags=(
    "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 0"
    "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 5"
)
experiment_names=(
    'layer5_ql_ldlq128_rnorm'
    'layer5_ql_ldlq128_rnorm_ft'
)

##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
    "meta-llama--Meta-Llama-3-8B"
)
hess_paths=(
    "../Wparam_dataset/quip_hess/llama3_8b_6144"
)

##########################################################################
##                     LAYER 5 SKIP LIST + GPU SETUP                    ##
##########################################################################
skip_spec="auto:upto=31,except=5_q|5_k|5_v|5_o|5_up|5_gate|5_down"
gpu_ids=(0 1 2 3)
ngpu=${#gpu_ids[@]}

############################################
##              SCRIPT SETUP              ##
############################################
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"
RES="../hf_model_comp_results_v2"

mkdir -p "$CKPT" "$HF" "$LOG" "$RES"
export HF_HOME=/home/jgryu/.cache/huggingface

lmbda_values=(30 50 100 300 1000 10000)

##########################################################################
##                        MAIN EXECUTION LOOP                           ##
##########################################################################
for j in "${!model_names[@]}"; do
    model_name="${model_names[$j]}"
    HESS="${hess_paths[$j]}"
    lm_model_path="../Wparam_dataset/hf_model/$model_name"

    echo "------------------------------------------------------------------------"
    echo "            MODEL: $model_name (layer 5 only)"
    echo "------------------------------------------------------------------------"

    for i in "${!experiment_names[@]}"; do
        exp_name="${experiment_names[$i]}"
        comp_model_base="${comp_model_bases[$i]}"
        current_quantize_flags="${quantize_flags[$i]}"

        echo "========================================================================"
        echo "            EXPERIMENT: $exp_name"
        echo "========================================================================"

        # Round-robin: each GPU gets a queue of lambdas
        pids=()
        for worker_idx in "${!gpu_ids[@]}"; do
            gpu_id="${gpu_ids[$worker_idx]}"
            (
                lmbda_idx=0
                for lmbda in "${lmbda_values[@]}"; do
                    if (( lmbda_idx % ngpu != worker_idx )); then
                        lmbda_idx=$((lmbda_idx + 1))
                        continue
                    fi
                    lmbda_idx=$((lmbda_idx + 1))

                    SAVE_NAME="${model_name}/${exp_name}/lmbda${lmbda}"
                    comp_model="${comp_model_base}/lmbda${lmbda}_*/best_loss*.pth.tar"
                    log_path="${LOG}/${SAVE_NAME}.log"
                    mkdir -p "$(dirname "$log_path")"

                    echo "[GPU ${gpu_id}] lmbda=${lmbda} exp=${exp_name}" | tee "$log_path"

                    CUDA_VISIBLE_DEVICES="${gpu_id}" python -m quantize_llama.quantize_finetune_llama \
                        --save_path "$CKPT/$SAVE_NAME" \
                        --base_model "$lm_model_path" \
                        --comp_model_path $comp_model \
                        --in_hess_path "$HESS" \
                        --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                        --skip_list "$skip_spec" \
                        ${current_quantize_flags} \
                        2>&1 | tee -a "$log_path"
                done
            ) &
            pids+=($!)
        done

        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        echo "Experiment $exp_name done."
    done
done

echo "All experiments done."