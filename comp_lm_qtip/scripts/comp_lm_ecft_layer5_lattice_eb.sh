#!/bin/bash
# Layer 5 ECFT (lattice_eb / OLVQ): R_target=4.2, lr=0.001, lattice_dim=16 (fixed)
# Nested-loop full Cartesian sweep over (lmbda × lambda_ortho × B_init).
# Concurrency is limited to #gpu_ids; each run is pinned to one GPU and we
# wait for a GPU to free up before launching the next run.

set -u
set -o pipefail

model_name="meta-llama--Meta-Llama-3-8B"
hess_path="../Wparam_dataset/quip_hess/llama3_8b_6144"
lm_model_path="../Wparam_dataset/hf_model/${model_name}"

# skip_spec="auto:upto=31,except=5_q|5_k|5_v|5_o|5_up|5_gate|5_down"
skip_spec="auto:upto=31,except=5_q"

base_flags="--ec_linear --row_normalize --scaleHinv \
    --ecft_epochs 500 --ecft_aux_warmup_step 500 \
    --R_target 4.2 --ec_decoder_type rht --ecft_mode noise \
    --ecft_entropy_model lattice_eb --ecft_lattice_dim 16 \
    --skip_list ${skip_spec}
    --verbose "

    # --ec_entropy_chunk_rows 2048 --ft_grad_ckpt --ft_epochs 5 --ecft_decoder \


# Sweep dimensions (Cartesian product = all combinations)
ecft_lmbda_list=(0.03 0.1 0.3 1 2 3)
lambda_ortho_list=(0 0.01 0.1)
B_init_list=(identity uniform)

# GPU pool — total concurrent runs = #gpu_ids
gpu_ids=(0 1 2 3)

ROOT="/home/jgryu/workspace/weight_compression/comp_lm_qtip"
CKPT="../hf_model_comp/comp_qtip/ckpt"
LOG="./log"
RES="../hf_model_comp_results_v2/ecft_test"
exp_group="ecft_layer5_r4.2_lattice_eb"

mkdir -p "$CKPT" "$LOG" "$RES"
export HF_HOME=/home/jgryu/.cache/huggingface

total=$(( ${#ecft_lmbda_list[@]} * ${#lambda_ortho_list[@]} * ${#B_init_list[@]} ))
echo "Launching ${total} runs across ${#gpu_ids[@]} GPU(s): ${gpu_ids[*]}"

# Per-GPU PID tracking — block until a specific GPU is free, then return it.
declare -A gpu_pid
for g in "${gpu_ids[@]}"; do gpu_pid[$g]=""; done

acquire_gpu() {
    while :; do
        for g in "${gpu_ids[@]}"; do
            local pid="${gpu_pid[$g]}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$g"
                return
            fi
        done
        sleep 5
    done
}

run_idx=0
for ld in "${ecft_lmbda_list[@]}"; do
  for lo in "${lambda_ortho_list[@]}"; do
    for binit in "${B_init_list[@]}"; do
        run_idx=$((run_idx + 1))
        ld_tag="${ld//./p}"
        lo_tag="${lo//./p}"
        exp_name="${exp_group}/lmbda${ld_tag}_ortho${lo_tag}_${binit}"
        save_name="${model_name}/${exp_name}"
        run_save_path="${CKPT}/${save_name}"
        log_path="${LOG}/${save_name}.log"
        res_path="${RES}/${save_name}"
        mkdir -p "$(dirname "$log_path")" "$res_path"

        gpu_id="$(acquire_gpu)"

        (
            echo "================================================================" | tee "$log_path"
            echo "[LAYER5 R=4.2 lattice_eb n=16] run ${run_idx}/${total} lmbda=${ld} ortho=${lo} Binit=${binit} ft_lr=1e-3 GPU=${gpu_id}" | tee -a "$log_path"
            echo "================================================================" | tee -a "$log_path"

            CUDA_VISIBLE_DEVICES="${gpu_id}" python -m quantize_llama.quantize_finetune_llama \
                --save_path "${run_save_path}" \
                --base_model "${lm_model_path}" \
                --in_hess_path "${hess_path}" \
                --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                --res_path "${res_path}" \
                --ecft_lmbda "${ld}" \
                --ecft_lambda_ortho "${lo}" \
                --ecft_B_init "${binit}" \
                --ft_lr 1e-3 \
                --use_wandb --wandb_project "ecft_layer5_r3_lattice_eb" --wandb_group "${exp_group}" \
                ${base_flags} \
                2>&1 | tee -a "$log_path"
        ) &
        gpu_pid[$gpu_id]=$!
    done
  done
done

wait
echo "Done. ${total} runs. Results under ${RES}/${model_name}/${exp_group}/"
