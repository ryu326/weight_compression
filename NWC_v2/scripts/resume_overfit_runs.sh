#!/bin/bash
# Resume 6 runs whose best.pth.tar got stuck at iter=2000 because
# best.pth.tar previously tracked val/mse but val/mse degraded after iter 2000
# while val/loss was still improving.  Now train.py tracks val[early_stop_metric]
# (here = 'loss'), so resuming with --early_stop_metric loss + a fresh best
# init will let best.pth.tar re-anchor at the loss-optimal iter.

set -u
set -o pipefail
cd "$(dirname "$0")/.."

GPU_IDS="${GPU_IDS:-4 5 6 7}"
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

# Each entry: dataset|enc|dec|enc_n|dec_n|lmbda|run_name
runs=(
  "gaussian|resblock|resblock|2|2|32|gaussian_encR2_decR2_compressai_M16_lmbda32"
  "gaussian|resblock|resblock|2|2|128|gaussian_encR2_decR2_compressai_M16_lmbda128"
  "gaussian|resblock|linear|2|0|32|gaussian_encR2_decLinear_compressai_M16_lmbda32"
  "gaussian|resblock|linear|2|0|128|gaussian_encR2_decLinear_compressai_M16_lmbda128"
  "llama8b|resblock|linear|2|0|32|llama8b_encR2_decLinear_compressai_M16_lmbda32"
  "llama8b|resblock|linear|2|0|128|llama8b_encR2_decLinear_compressai_M16_lmbda128"
)

read -r -a gpu_ids <<< "$GPU_IDS"
SAVE_ROOT="./checkpoint/rd_sweep_normnone"
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

total=${#runs[@]}
echo "Resuming ${total} runs from ckpt_iter8000.pth.tar across ${#gpu_ids[@]} GPU(s): ${gpu_ids[*]}"
run_idx=0

for entry in "${runs[@]}"; do
    IFS='|' read -r dataset enc dec enc_n dec_n lmbda run_name <<< "$entry"
    run_idx=$((run_idx + 1))
    ckpt_dir="${SAVE_ROOT}/${run_name}"
    resume_ckpt="${ckpt_dir}/ckpt_iter8000.pth.tar"
    log_file="${LOG_DIR}/${run_name}.log"

    if [[ ! -f "$resume_ckpt" ]]; then
        echo "[skip ${run_idx}/${total}] no ckpt_iter8000: ${run_name}"
        continue
    fi

    gpu_id="$(acquire_gpu)"

    extra_flags=""
    if [[ "$dataset" == "llama8b" ]]; then
        extra_flags="--normalize none --direction ${LLAMA_DIRECTION}"
    fi
    n_flags="--encoder_n_resblock ${enc_n}"
    if [[ "$dec" == "resblock" ]]; then
        n_flags="${n_flags} --decoder_n_resblock ${dec_n}"
    fi

    EVAL_EVERY=$(( ITER / 10 ))
    [[ $EVAL_EVERY -lt 1 ]] && EVAL_EVERY=1
    LOG_EVERY=$(( ITER / 20 ))
    [[ $LOG_EVERY -lt 1 ]] && LOG_EVERY=1

    (
        echo "================================================================" | tee -a "$log_file"
        echo "[RESUME ${run_idx}/${total}] ${run_name} from iter 8000 with metric=${EARLY_STOP_METRIC} GPU=${gpu_id}" | tee -a "$log_file"
        echo "================================================================" | tee -a "$log_file"

        CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
            --dataset "${dataset}" \
            --encoder_transform "${enc}" --decoder_transform "${dec}" \
            --entropy_model compressai \
            --input_size "${INPUT_SIZE}" --M "${M}" \
            --dim_encoder "${DIM_ENCODER}" \
            ${n_flags} \
            --lmbda "${lmbda}" --iter "${ITER}" --batch_size "${BATCH}" \
            --eval_every "${EVAL_EVERY}" --log_every "${LOG_EVERY}" --num_workers 2 \
            --early_stop_patience "${EARLY_STOP_PATIENCE}" \
            --early_stop_metric "${EARLY_STOP_METRIC}" \
            --early_stop_min_iter "${EARLY_STOP_MIN_ITER}" \
            --checkpoint "${resume_ckpt}" \
            --save_dir "${SAVE_ROOT}" --run_name "${run_name}" \
            --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}" \
            ${extra_flags} \
            2>&1 | tee -a "$log_file"
    ) &
    gpu_pid[$gpu_id]=$!
done

wait
echo "Done.  Resumed ${total} runs."
