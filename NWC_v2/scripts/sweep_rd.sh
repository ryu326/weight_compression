#!/bin/bash
# RD-curve sweep:
#   datasets   = {gaussian, llama8b}
#   transforms = {identity, rht, resblock}   # encoder = decoder
#   lambdas    = {32, 64, 128, 256, 1024, 8192}
#   entropy    = compressai (fixed)
# Total: 2 × 3 × 6 = 36 runs.
#
# Each run is pinned to one GPU; we acquire from a pool so total
# concurrency = #gpu_ids.

set -u
set -o pipefail
cd "$(dirname "$0")/.."

# Defaults (override via env)
GPU_IDS="${GPU_IDS:-4 5 6 7}"
ITER="${ITER:-20000}"
BATCH="${BATCH:-512}"
INPUT_SIZE="${INPUT_SIZE:-16}"
M="${M:-16}"
N_RESBLOCK="${N_RESBLOCK:-2}"
DIM_ENCODER="${DIM_ENCODER:-512}"
LLAMA_NORMALIZE="${LLAMA_NORMALIZE:-row}"
LLAMA_DIRECTION="${LLAMA_DIRECTION:-row}"
WANDB_PROJECT="${WANDB_PROJECT:-nwc_v2_rd_sweep}"
WANDB_MODE="${WANDB_MODE:-online}"

datasets=(gaussian llama8b)
transforms=(resblock affine rht linear)   # resblock first
lambdas=(32 64 128 256 1024 8192)
read -r -a gpu_ids <<< "$GPU_IDS"

SAVE_ROOT="./checkpoint/rd_sweep"
mkdir -p "$SAVE_ROOT"
LOG_DIR="./log/rd_sweep"
mkdir -p "$LOG_DIR"

# Per-GPU PID tracking — block until a specific GPU is free.
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

total=$(( ${#datasets[@]} * ${#transforms[@]} * ${#lambdas[@]} ))
echo "Launching ${total} runs across ${#gpu_ids[@]} GPU(s): ${gpu_ids[*]}"
run_idx=0

for dataset in "${datasets[@]}"; do
    for tf in "${transforms[@]}"; do
        for lmbda in "${lambdas[@]}"; do
            run_idx=$((run_idx + 1))
            run_name="${dataset}_${tf}-${tf}_compressai_M${M}_lmbda${lmbda}"
            save_dir="${SAVE_ROOT}"
            log_file="${LOG_DIR}/${run_name}.log"
            ckpt_dir="${SAVE_ROOT}/${run_name}"
            best_pt="${ckpt_dir}/best.pth.tar"

            if [[ -f "$best_pt" ]]; then
                echo "[skip ${run_idx}/${total}] already done: ${run_name}"
                continue
            fi

            gpu_id="$(acquire_gpu)"

            # Llama-only flags (gaussian ignores them)
            extra_flags=""
            if [[ "$dataset" == "llama8b" ]]; then
                extra_flags="--normalize ${LLAMA_NORMALIZE} --direction ${LLAMA_DIRECTION}"
            fi

            # Eval cadence: 4 evals over the run, plus a final one at iter=ITER.
            EVAL_EVERY=$(( ITER / 10 ))
            [[ $EVAL_EVERY -lt 1 ]] && EVAL_EVERY=1
            LOG_EVERY=$(( ITER / 20 ))
            [[ $LOG_EVERY -lt 1 ]] && LOG_EVERY=1

            (
                echo "================================================================" | tee "$log_file"
                echo "[RUN ${run_idx}/${total}] dataset=${dataset} tf=${tf} lmbda=${lmbda} batch=${BATCH} GPU=${gpu_id}" | tee -a "$log_file"
                echo "================================================================" | tee -a "$log_file"

                CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
                    --dataset "${dataset}" \
                    --encoder_transform "${tf}" --decoder_transform "${tf}" \
                    --entropy_model compressai \
                    --input_size "${INPUT_SIZE}" --M "${M}" \
                    --n_resblock "${N_RESBLOCK}" --dim_encoder "${DIM_ENCODER}" \
                    --lmbda "${lmbda}" --iter "${ITER}" --batch_size "${BATCH}" \
                    --eval_every "${EVAL_EVERY}" --log_every "${LOG_EVERY}" --num_workers 2 \
                    --save_dir "${save_dir}" --run_name "${run_name}" \
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
echo "Plot:  python scripts/plot_rd.py --save_root ${SAVE_ROOT} --out rd_curves.png"
