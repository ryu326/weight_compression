#!/bin/bash
# Re-run only the rht_rht config with TIED encoder/decoder (decoder = encoder.inverse).
# 8 runs = 2 datasets × 4 lambdas.  All other settings match sweep_rd_normnone.sh.
#
# Output dir: ./checkpoint/rd_sweep_normnone/<run_name>/best.pth.tar  (same root)

set -u
set -o pipefail
cd "$(dirname "$0")/.."

GPU_IDS="${GPU_IDS:-4 6}"
ITER="${ITER:-20000}"
BATCH="${BATCH:-512}"
INPUT_SIZE="${INPUT_SIZE:-16}"
M="${M:-16}"
DIM_ENCODER="${DIM_ENCODER:-512}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-loss}"
EARLY_STOP_MIN_ITER="${EARLY_STOP_MIN_ITER:-5000}"
LLAMA_DIRECTION="${LLAMA_DIRECTION:-row}"
WANDB_PROJECT="${WANDB_PROJECT:-nwc_v2_rd_normnone}"
WANDB_MODE="${WANDB_MODE:-online}"

datasets=(gaussian llama8b)
lambdas=(32 128 1024 8192)
read -r -a gpu_ids <<< "$GPU_IDS"

SAVE_ROOT="./checkpoint/rd_sweep_normnone"
mkdir -p "$SAVE_ROOT"
LOG_DIR="./log/rd_sweep_normnone"
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

total=$(( ${#datasets[@]} * ${#lambdas[@]} ))
echo "Launching ${total} rht_rht (tied) runs across ${#gpu_ids[@]} GPU(s): ${gpu_ids[*]}"
run_idx=0

for dataset in "${datasets[@]}"; do
    for lmbda in "${lambdas[@]}"; do
        run_idx=$((run_idx + 1))
        run_name="${dataset}_rht_rht_compressai_M${M}_lmbda${lmbda}"
        ckpt_dir="${SAVE_ROOT}/${run_name}"
        best_pt="${ckpt_dir}/best.pth.tar"
        log_file="${LOG_DIR}/${run_name}.log"

        if [[ -f "$best_pt" ]]; then
            echo "[skip ${run_idx}/${total}] already done: ${run_name}"
            continue
        fi

        gpu_id="$(acquire_gpu)"

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
            echo "[RUN ${run_idx}/${total}] dataset=${dataset} cfg=rht_rht (tied) lmbda=${lmbda} batch=${BATCH} GPU=${gpu_id}" | tee -a "$log_file"
            echo "================================================================" | tee -a "$log_file"

            CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
                --dataset "${dataset}" \
                --encoder_transform rht --decoder_transform rht \
                --entropy_model compressai \
                --input_size "${INPUT_SIZE}" --M "${M}" \
                --dim_encoder "${DIM_ENCODER}" \
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

wait
echo "Done.  rht_rht (tied) results under ${SAVE_ROOT}/<run_name>/best.pth.tar"
