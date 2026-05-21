#!/bin/bash
# RD-curve sweep with EntropyBottleneck channels=1 (--shared_eb).
# Same 4 transforms as sweep_rd_normnone.sh, but ONLY gaussian, ITER=20K, ES=loss.
# Total: 4 × 1 × 4 = 16 runs.
# Output: ./checkpoint/rd_sweep_shared_eb/<run_name>/best.pth.tar

set -u
set -o pipefail
cd "$(dirname "$0")/.."

GPU_IDS="${GPU_IDS:-0 1 2 3}"
LLAMA_DIRECTION="${LLAMA_DIRECTION:-row}"
ITER="${ITER:-20000}"
BATCH="${BATCH:-512}"
INPUT_SIZE="${INPUT_SIZE:-16}"
M="${M:-16}"
DIM_ENCODER="${DIM_ENCODER:-512}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-loss}"
EARLY_STOP_MIN_ITER="${EARLY_STOP_MIN_ITER:-5000}"
WANDB_PROJECT="${WANDB_PROJECT:-nwc_v2_rd_shared_eb}"
WANDB_MODE="${WANDB_MODE:-online}"

config_names=(
    "encR2_decLinear"
    "rht_rht"
    "linear_linear"
    "encR2_decR2"
)
encoder_tfs=(  resblock  rht  linear  resblock )
decoder_tfs=(  linear    rht  linear  resblock )
encoder_ns=(   2         0    0       2        )
decoder_ns=(   0         0    0       2        )

datasets=(gaussian llama8b)
lambdas=(32 128 1024 8192)
read -r -a gpu_ids <<< "$GPU_IDS"

SAVE_ROOT="./checkpoint/rd_sweep_shared_eb"
mkdir -p "$SAVE_ROOT"
LOG_DIR="./log/rd_sweep_shared_eb"
mkdir -p "$LOG_DIR"

declare -A gpu_pid
for g in "${gpu_ids[@]}"; do gpu_pid[$g]=""; done

acquire_gpu() {
    while :; do
        for g in "${gpu_ids[@]}"; do
            local pid="${gpu_pid[$g]}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$g"; return
            fi
        done
        sleep 5
    done
}

total=$(( ${#config_names[@]} * ${#datasets[@]} * ${#lambdas[@]} ))
echo "Launching ${total} shared-EB runs across ${#gpu_ids[@]} GPU(s): ${gpu_ids[*]}"
run_idx=0

for cfg_idx in "${!config_names[@]}"; do
    cfg_name="${config_names[$cfg_idx]}"
    enc_tf="${encoder_tfs[$cfg_idx]}"
    dec_tf="${decoder_tfs[$cfg_idx]}"
    enc_n="${encoder_ns[$cfg_idx]}"
    dec_n="${decoder_ns[$cfg_idx]}"

    for dataset in "${datasets[@]}"; do
        for lmbda in "${lambdas[@]}"; do
            run_idx=$((run_idx + 1))
            run_name="${dataset}_${cfg_name}_compressai_M${M}_lmbda${lmbda}_sharedEB"
            ckpt_dir="${SAVE_ROOT}/${run_name}"
            best_pt="${ckpt_dir}/best.pth.tar"
            log_file="${LOG_DIR}/${run_name}.log"

            if [[ -f "$best_pt" ]]; then
                echo "[skip ${run_idx}/${total}] already done: ${run_name}"
                continue
            fi

            gpu_id="$(acquire_gpu)"

            n_flags=""
            if [[ "$enc_tf" == "resblock" ]]; then
                n_flags="--encoder_n_resblock ${enc_n}"
            fi
            if [[ "$dec_tf" == "resblock" ]]; then
                n_flags="${n_flags} --decoder_n_resblock ${dec_n}"
            fi

            extra_flags=""
            if [[ "$dataset" == "llama8b" ]]; then
                extra_flags="--normalize none --direction ${LLAMA_DIRECTION}"
            fi

            EVAL_EVERY=$(( ITER / 10 ))
            [[ $EVAL_EVERY -lt 1 ]] && EVAL_EVERY=1
            LOG_EVERY=$(( ITER / 20 ))
            [[ $LOG_EVERY -lt 1 ]] && LOG_EVERY=1

            (
                echo "================================================================" | tee "$log_file"
                echo "[RUN ${run_idx}/${total}] dataset=${dataset} cfg=${cfg_name} (enc=${enc_tf}/n${enc_n}, dec=${dec_tf}/n${dec_n}) lmbda=${lmbda} shared_eb=YES GPU=${gpu_id}" | tee -a "$log_file"
                echo "================================================================" | tee -a "$log_file"

                CUDA_VISIBLE_DEVICES=${gpu_id} /opt/conda/bin/python train.py \
                    --dataset "${dataset}" \
                    --encoder_transform "${enc_tf}" --decoder_transform "${dec_tf}" \
                    --entropy_model compressai --shared_eb \
                    --input_size "${INPUT_SIZE}" --M "${M}" \
                    --dim_encoder "${DIM_ENCODER}" \
                    ${n_flags} \
                    --lmbda "${lmbda}" --iter "${ITER}" --batch_size "${BATCH}" \
                    --eval_every "${EVAL_EVERY}" --log_every "${LOG_EVERY}" --num_workers 2 \
                    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
                    --early_stop_metric "${EARLY_STOP_METRIC}" \
                    --early_stop_min_iter "${EARLY_STOP_MIN_ITER}" \
                    --save_dir "${SAVE_ROOT}" --run_name "${run_name}" \
                    --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}" \
                    ${extra_flags} \
                    2>&1 | tee -a "$log_file"
            ) &
            gpu_pid[$gpu_id]=$!
        done
    done
done

wait
echo "Done.  Results under ${SAVE_ROOT}/<run_name>/best.pth.tar"
