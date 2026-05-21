#!/bin/bash
# Layer 5 ECFT: R_target=4.2, decoder=rht, entropy_model=compressai, lr=1e-3, lambda sweep

set -u
set -o pipefail

model_name="meta-llama--Meta-Llama-3-8B"
hess_path="../Wparam_dataset/quip_hess/llama3_8b_6144"
lm_model_path="../Wparam_dataset/hf_model/${model_name}"

skip_spec="auto:upto=31,except=5_q|5_k|5_v|5_o|5_up|5_gate|5_down"

base_flags="--ec_linear --row_normalize --scaleHinv \
    --ecft_epochs 0 --ecft_aux_warmup_step 500 \
    --R_target 4.2 --ec_decoder_type rht --ecft_mode noise \
    --ecft_entropy_model compressai \
    --ec_entropy_chunk_rows 2048 --ft_grad_ckpt --ft_epochs 10 --ecft_decoder \
    --skip_list ${skip_spec}"

ecft_lmbda_list=(0 0.01 0.1 0.3 1 3 10 30)
gpu_ids=(0 1 2 3 4 5 6 7)

ROOT="/home/jgryu/workspace/weight_compression/comp_lm_qtip"
CKPT="../hf_model_comp/comp_qtip/ckpt"
LOG="./log"
RES="../hf_model_comp_results_v2"
exp_group="ecft_layer5_rht_compressai_r42_sweep"

mkdir -p "$CKPT" "$LOG" "$RES"
export HF_HOME=/home/jgryu/.cache/huggingface

pids=()
for idx in "${!ecft_lmbda_list[@]}"; do
    ld="${ecft_lmbda_list[$idx]}"
    gpu_id="${gpu_ids[$idx]}"
    ld_tag="${ld//./p}"
    exp_name="${exp_group}/lmbda${ld_tag}"
    save_name="${model_name}/${exp_name}"
    run_save_path="${CKPT}/${save_name}"
    log_path="${LOG}/${save_name}.log"
    res_path="${RES}/${save_name}"
    mkdir -p "$(dirname "$log_path")" "$res_path"

    (
        echo "================================================================" | tee "$log_path"
        echo "[LAYER5 RHT+COMPRESSAI R=4.2] lmbda=${ld} ft_lr=1e-3 GPU=${gpu_id}" | tee -a "$log_path"
        echo "================================================================" | tee -a "$log_path"

        CUDA_VISIBLE_DEVICES="${gpu_id}" python -m quantize_llama.quantize_finetune_llama \
            --save_path "${run_save_path}" \
            --base_model "${lm_model_path}" \
            --in_hess_path "${hess_path}" \
            --devset_size 384 --ft_valid_size 128 --batch_size 8 \
            --res_path "${res_path}" \
            --ecft_lmbda "${ld}" \
            --ft_lr 1e-3 \
            --use_wandb --wandb_project "ecft_layer5_rht_compressai" --wandb_group "${exp_group}" \
            ${base_flags} \
            2>&1 | tee -a "$log_path"
    ) &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "Done. Results under ${RES}/${model_name}/${exp_group}/"
