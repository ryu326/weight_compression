#!/bin/bash

set -u
set -o pipefail

ROOT="/home/jgryu/workspace/weight_compression/comp_lm_qtip"
cd "$ROOT"

##########################################################################
##                               TARGET                                 ##
##########################################################################
model_name="meta-llama--Meta-Llama-3-8B"
hess_path="../Wparam_dataset/quip_hess/llama3_8b_6144"
lm_model_path="../Wparam_dataset/hf_model/${model_name}"

target_keys=("5_q")
target_rates=("2" "3" "4" "5")
# target_rates=("4.2")
decoder_types=("identity" "rht")
ecft_epochs_list=("500")
ecft_lmbda_list=("0" "0.01" "0.1")
gpu_ids=(0 1 2 3)

##########################################################################
##                           RUNTIME OPTIONS                            ##
##########################################################################
ecft_mode=noise
# ecft_entropy_model=parametric
ecft_entropy_model=lattice_eb
ecft_num_gaussian=3
ecft_num_laplacian=3
# lattice_eb only (ignored by compressai/parametric)
ecft_lattice_dim=16
ecft_lambda_ortho=0
ecft_B_init=identity
aux_warmup_step=500
devset_size=8
ft_valid_size=2
batch_size=2

##########################################################################
##                                PATHS                                 ##
##########################################################################
CKPT="../hf_model_comp/comp_qtip/ckpt"
LOG="./log"
RES="../hf_model_comp_results_v2"
exp_tag="${ecft_mode}_${ecft_entropy_model}"
PLOTS_DIR="${RES}/ecft_ablation_subset_${exp_tag}_plots"
RECORDS_CSV="${RES}/ecft_ablation_subset_${exp_tag}_records.csv"
PLOT_LOCK_FILE="${RES}/ecft_ablation_subset_${exp_tag}_plots.lock"
target_order_csv="$(IFS=,; echo "${target_keys[*]}")"

mkdir -p "$CKPT" "$LOG" "$RES" "$PLOTS_DIR"
export HF_HOME=/home/jgryu/.cache/huggingface

if [ "${#gpu_ids[@]}" -ne 8 ]; then
    echo "ERROR: gpu_ids should have 4 entries" >&2
    exit 1
fi

task_file="$(mktemp)"
run_file="$(mktemp)"
cleanup() {
    rm -f "$task_file" "$run_file"
}
trap cleanup EXIT

export RETRY_RECORDS_CSV="$RECORDS_CSV"
export RETRY_CKPT_ROOT="$CKPT"
export RETRY_MODEL_NAME="$model_name"
export RETRY_KEYS="$(IFS=,; echo "${target_keys[*]}")"
export RETRY_RATES="$(IFS=,; echo "${target_rates[*]}")"
export RETRY_DECS="$(IFS=,; echo "${decoder_types[*]}")"
export RETRY_EPOCHS="$(IFS=,; echo "${ecft_epochs_list[*]}")"
export RETRY_LAMBDAS="$(IFS=,; echo "${ecft_lmbda_list[*]}")"
export RETRY_ECFT_MODE="$ecft_mode"
export RETRY_EXP_TAG="$exp_tag"

python - <<'PY' > "$task_file"
import csv
import itertools
import os

records_path = os.environ["RETRY_RECORDS_CSV"]
ckpt_root = os.environ["RETRY_CKPT_ROOT"]
model_name = os.environ["RETRY_MODEL_NAME"]
keys = [x for x in os.environ["RETRY_KEYS"].split(",") if x]
rates = [x for x in os.environ["RETRY_RATES"].split(",") if x]
decs = [x for x in os.environ["RETRY_DECS"].split(",") if x]
epochs = [x for x in os.environ["RETRY_EPOCHS"].split(",") if x]
lambdas = [x for x in os.environ["RETRY_LAMBDAS"].split(",") if x]
ecft_mode = os.environ["RETRY_ECFT_MODE"]
exp_tag = os.environ["RETRY_EXP_TAG"]

records = set()
if os.path.exists(records_path):
    with open(records_path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            records.add(
                (
                    str(r["target_key"]),
                    str(r["target_rate"]),
                    str(r["decoder_type"]),
                    str(r["ecft_epochs"]),
                    str(r["ecft_lmbda"]),
                )
            )

for key, rate, dec, ep, lmbda in itertools.product(keys, rates, decs, epochs, lambdas):
    combo = (key, rate, dec, ep, lmbda)
    if combo in records:
        continue

    lmbda_tag = lmbda.replace(".", "p")
    run_rel = f"{model_name}/ecft_ablation_subset/{key}/rt{rate}_{dec}_ep{ep}_lmbda{lmbda_tag}_{exp_tag}"
    pt_path = os.path.join(ckpt_root, run_rel, f"{key}.pt")
    action = "UPDATE" if os.path.exists(pt_path) else "RUN"
    print("\t".join([action, key, rate, dec, ep, lmbda, run_rel, pt_path]))
PY

total_tasks=$(wc -l < "$task_file")
update_tasks=$(awk -F'\t' '$1=="UPDATE"{c++} END{print c+0}' "$task_file")
run_tasks=$(awk -F'\t' '$1=="RUN"{c++} END{print c+0}' "$task_file")

echo "Total missing combos: ${total_tasks}"
echo "Backfill-only (pt exists): ${update_tasks}"
echo "Need rerun: ${run_tasks}"

if [ "$total_tasks" -eq 0 ]; then
    echo "Nothing missing. Exit."
    exit 0
fi

# 1) Backfill records/plots for pt files that already exist.
if [ "$update_tasks" -gt 0 ]; then
    while IFS=$'\t' read -r action target_key target_rate decoder_type ecft_epoch ecft_lmbda run_rel pt_path; do
        [ "$action" = "UPDATE" ] || continue
        log_path="${LOG}/${run_rel}_retry.log"
        mkdir -p "$(dirname "$log_path")"
        echo "[BACKFILL] ${target_key} rt=${target_rate} dec=${decoder_type} ep=${ecft_epoch} lmbda=${ecft_lmbda}" | tee -a "$log_path"
        if command -v flock >/dev/null 2>&1; then
            (
                flock -x 200
                python "${ROOT}/scripts/update_ecft_ablation_plots.py" \
                    --records_csv "${RECORDS_CSV}" \
                    --plots_dir "${PLOTS_DIR}" \
                    --target_order "${target_order_csv}" \
                    --pt_path "${pt_path}" \
                    --target_key "${target_key}" \
                    --target_rate "${target_rate}" \
                    --decoder_type "${decoder_type}" \
                    --ecft_epochs "${ecft_epoch}" \
                    --ecft_lmbda "${ecft_lmbda}" \
                    --ecft_mode "${ecft_mode}" \
                    --ecft_entropy_model "${ecft_entropy_model}" \
                    >> "${log_path}" 2>&1
            ) 200>"${PLOT_LOCK_FILE}"
        else
            python "${ROOT}/scripts/update_ecft_ablation_plots.py" \
                --records_csv "${RECORDS_CSV}" \
                --plots_dir "${PLOTS_DIR}" \
                --target_order "${target_order_csv}" \
                --pt_path "${pt_path}" \
                --target_key "${target_key}" \
                --target_rate "${target_rate}" \
                --decoder_type "${decoder_type}" \
                --ecft_epochs "${ecft_epoch}" \
                --ecft_lmbda "${ecft_lmbda}" \
                >> "${log_path}" 2>&1
        fi
    done < "$task_file"
fi

awk -F'\t' '$1=="RUN"{print}' "$task_file" > "$run_file"

if [ "$run_tasks" -eq 0 ]; then
    echo "All missing combos were backfilled from existing pt files. Exit."
    exit 0
fi

ngpu="${#gpu_ids[@]}"
pids=()

for worker_idx in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[$worker_idx]}"
    (
        line_idx=0
        while IFS=$'\t' read -r action target_key target_rate decoder_type ecft_epoch ecft_lmbda run_rel pt_path; do
            if (( line_idx % ngpu != worker_idx )); then
                line_idx=$((line_idx + 1))
                continue
            fi

            run_save_path="${CKPT}/${run_rel}"
            log_path="${LOG}/${run_rel}_retry.log"
            mkdir -p "$(dirname "$log_path")"
            skip_spec="auto:upto=31,except=${target_key}"

            current_quantize_flags="--ec_linear \
                    --row_normalize --scaleHinv --ecft_epochs ${ecft_epoch} --ecft_aux_warmup_step ${aux_warmup_step} \
                    --R_target ${target_rate} --ec_decoder_type ${decoder_type} --ecft_lmbda ${ecft_lmbda} \
                    --ecft_mode ${ecft_mode} \
                    --ecft_entropy_model ${ecft_entropy_model} \
                    --ecft_num_gaussian ${ecft_num_gaussian} --ecft_num_laplacian ${ecft_num_laplacian} \
                    --ecft_lattice_dim ${ecft_lattice_dim} --ecft_lambda_ortho ${ecft_lambda_ortho} --ecft_B_init ${ecft_B_init}"

            echo "========================================================================" | tee -a "$log_path"
            echo "[RUN] TARGET=${target_key} | R=${target_rate} dec=${decoder_type} ep=${ecft_epoch} lmbda=${ecft_lmbda} mode=${ecft_mode} em=${ecft_entropy_model} | GPU=${gpu_id}" | tee -a "$log_path"
            echo "========================================================================" | tee -a "$log_path"

            CUDA_VISIBLE_DEVICES="${gpu_id}" python -m quantize_llama.quantize_finetune_llama \
                --save_path "${run_save_path}" \
                --base_model "${lm_model_path}" \
                --in_hess_path "${hess_path}" \
                --devset_size "${devset_size}" \
                --ft_valid_size "${ft_valid_size}" \
                --batch_size "${batch_size}" \
                --skip_list "${skip_spec}" \
                --res_path "${RES}/${run_rel}" \
                ${current_quantize_flags} \
                2>&1 | tee -a "$log_path"

            run_exit_code=${PIPESTATUS[0]}
            if [ "$run_exit_code" -eq 0 ] && [ -f "$pt_path" ]; then
                if command -v flock >/dev/null 2>&1; then
                    (
                        flock -x 200
                        python "${ROOT}/scripts/update_ecft_ablation_plots.py" \
                            --records_csv "${RECORDS_CSV}" \
                            --plots_dir "${PLOTS_DIR}" \
                            --target_order "${target_order_csv}" \
                            --pt_path "${pt_path}" \
                            --target_key "${target_key}" \
                            --target_rate "${target_rate}" \
                            --decoder_type "${decoder_type}" \
                            --ecft_epochs "${ecft_epoch}" \
                            --ecft_lmbda "${ecft_lmbda}" \
                            --ecft_mode "${ecft_mode}" \
                            --ecft_entropy_model "${ecft_entropy_model}" \
                            >> "${log_path}" 2>&1
                    ) 200>"${PLOT_LOCK_FILE}"
                else
                    python "${ROOT}/scripts/update_ecft_ablation_plots.py" \
                        --records_csv "${RECORDS_CSV}" \
                        --plots_dir "${PLOTS_DIR}" \
                        --target_order "${target_order_csv}" \
                        --pt_path "${pt_path}" \
                        --target_key "${target_key}" \
                        --target_rate "${target_rate}" \
                        --decoder_type "${decoder_type}" \
                        --ecft_epochs "${ecft_epoch}" \
                        --ecft_lmbda "${ecft_lmbda}" \
                        --ecft_mode "${ecft_mode}" \
                        --ecft_entropy_model "${ecft_entropy_model}" \
                        >> "${log_path}" 2>&1
                fi
            else
                echo "WARN: run failed or result missing: ${pt_path}" | tee -a "$log_path"
            fi

            line_idx=$((line_idx + 1))
        done < "$run_file"
    ) &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "Retry completed."
