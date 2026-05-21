#!/bin/bash
# SigLIP-B/16 rnorm seed1/2/3 Phase 1 (Compress only) — GPU 0~3

export HF_HOME=/home/jgryu/.cache/huggingface
PYTHON_BIN="/opt/conda/bin/python"

CKPT="../hf_model_comp/comp_qtip/ckpt"
LOG="./log"
mkdir -p $CKPT $LOG

HESS_BASE="/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess"
NWC_BASE="/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/MultiSeed_rdloss_ql_size16_encdim512_M16_Q4_nRB4R0_m0_batch_size2048_total_iter200000_lr0.0001_seed4.0"

GPU_IDS=(0 1 2 3)
N_GPU=4
JOBS_PER_GPU=4
MAX_COMP_JOBS=$((N_GPU * JOBS_PER_GPU))  # 16

model_name="google--siglip-base-patch16-224"
hess_path="${HESS_BASE}/siglip-base-patch16-224_512"
lm_model_path="/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/google--siglip-base-patch16-224"
quant_module="quantize_llama.quantize_finetune_clip"
lmbda_values=(30 50 100 300 1000 10000)
quant_flags="--direction col --ql --Q 4 --row_normalize --comp_batch_size 64 --ldlq --ft_epochs 0"

experiment_names=("rnorm_ldlq64_seed1" "rnorm_ldlq64_seed2" "rnorm_ldlq64_seed3")
declare -A exp_nwc_seed
exp_nwc_seed["rnorm_ldlq64_seed1"]="seed1"
exp_nwc_seed["rnorm_ldlq64_seed2"]="seed2"
exp_nwc_seed["rnorm_ldlq64_seed3"]="seed3"

get_comp_gpu() { echo ${GPU_IDS[$((( $1 % MAX_COMP_JOBS ) / JOBS_PER_GPU))]}; }

echo "════════════════════════════════════════"
echo "SigLIP Phase 1 — Compress only (GPU 0~3)"
echo "════════════════════════════════════════"

comp_job_idx=0
for exp_name in "${experiment_names[@]}"; do
    nwc_seed="${exp_nwc_seed[$exp_name]}"
    comp_model_base="${NWC_BASE}/${nwc_seed}"
    for lmbda in "${lmbda_values[@]}"; do
        gpu_id=$(get_comp_gpu $comp_job_idx)
        SAVE_NAME="${model_name}/${exp_name}/lmbda${lmbda}"
        comp_model="${comp_model_base}/lmbda${lmbda}_*/best_loss*.pth.tar"
        LOG_FILE="${LOG}/${SAVE_NAME}.log"
        mkdir -p "$(dirname "$LOG_FILE")"
        echo "  >> [GPU $gpu_id] $exp_name lmbda=$lmbda"
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id
            $PYTHON_BIN -m $quant_module \
                --save_path ${CKPT}/${SAVE_NAME} \
                --base_model $lm_model_path \
                --comp_model_path $comp_model \
                --in_hess_path $hess_path \
                $quant_flags \
                > $LOG_FILE 2>&1
            echo "  [COMP DONE] $SAVE_NAME" >> $LOG_FILE
        ) &
        ((comp_job_idx++))
        if (( comp_job_idx % MAX_COMP_JOBS == 0 )); then
            echo "  Waiting for comp batch ($comp_job_idx jobs)..."
            wait
        fi
    done
done
wait
echo "[Phase 1] SigLIP compress all done."
