#!/bin/bash
# Vision encoder compression — phased approach
#
# Phase 1 (Compress):   4 jobs/GPU × 4 GPUs = 16 simultaneous  (comp barely uses GPU)
# Phase 2 (Hfize+Eval): N_GPU jobs at a time (one per GPU)
#
# Models: CLIP-ViT-L/16, SigLIP-B/16, DINOv2-L
# Experiments: rnorm_seed1/2/3 only (3 seeds × 6 lambda = 18 jobs per model)

export HF_HOME=/home/jgryu/.cache/huggingface
PYTHON_BIN="/opt/conda/bin/python"

CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
RES="../hf_model_comp_results_v2"
LOG="./log"
mkdir -p $CKPT $HF $LOG

HF_MODEL_BASE="/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model"
HESS_BASE="/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess"
IMAGENET_PATH="/data/ILSVRC2012"

NWC_BASE="/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/MultiSeed_rdloss_ql_size16_encdim512_M16_Q4_nRB4R0_m0_batch_size2048_total_iter200000_lr0.0001_seed4.0"

GPU_IDS=(4 5 6 7)
N_GPU=${#GPU_IDS[@]}
JOBS_PER_GPU=4                          # comp phase: 4 jobs per GPU simultaneously
MAX_COMP_JOBS=$((N_GPU * JOBS_PER_GPU)) # = 16

lmbda_values=(30 50 100 300 1000 10000)

# ─────────────────────────────────────────────────────────────────────────────
# Model configs: "model_name|hess_name|model_id|quant_mod|hfize_mod|eval_mod|lm_model_path"
# ─────────────────────────────────────────────────────────────────────────────
model_configs=(
    "openai--clip-vit-large-patch14|clip-vit-large-patch14_512|openai/clip-vit-large-patch14|quantize_llama.quantize_finetune_clip|quantize_llama.hfize_clip|eval.eval_clip_imagenet|${HF_MODEL_BASE}/openai--clip-vit-large-patch14"
    "google--siglip-base-patch16-224|siglip-base-patch16-224_512|google/siglip-base-patch16-224|quantize_llama.quantize_finetune_clip|quantize_llama.hfize_clip|eval.eval_siglip_imagenet|/home/jgryu/.cache/huggingface/hub/models--google--siglip-base-patch16-224//home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/google--siglip-base-patch16-224""
    "facebook--dinov2-large-imagenet1k-1-layer|dinov2-large-imagenet1k-1-layer_cc1024||quantize_llama.quantize_finetune_dino|quantize_llama.hfize_dino|eval.eval_dino|${HF_MODEL_BASE}/facebook--dinov2-large-imagenet1k-1-layer"
)

# ─────────────────────────────────────────────────────────────────────────────
# Experiments: rnorm seed1/2/3 only
# ─────────────────────────────────────────────────────────────────────────────
experiment_names=("rnorm_ldlq64_seed1" "rnorm_ldlq64_seed2" "rnorm_ldlq64_seed3")

declare -A exp_nwc_seed
exp_nwc_seed["rnorm_ldlq64_seed1"]="seed1"
exp_nwc_seed["rnorm_ldlq64_seed2"]="seed2"
exp_nwc_seed["rnorm_ldlq64_seed3"]="seed3"

declare -A exp_quant_flags
exp_quant_flags["rnorm_ldlq64_seed1"]="--direction col --ql --Q 4 --row_normalize --comp_batch_size 64 --ldlq --ft_epochs 0"
exp_quant_flags["rnorm_ldlq64_seed2"]="--direction col --ql --Q 4 --row_normalize --comp_batch_size 64 --ldlq --ft_epochs 0"
exp_quant_flags["rnorm_ldlq64_seed3"]="--direction col --ql --Q 4 --row_normalize --comp_batch_size 64 --ldlq --ft_epochs 0"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: GPU index for comp phase (round-robin, 4 jobs per GPU)
# job_idx 0-3 → GPU4, 4-7 → GPU5, 8-11 → GPU6, 12-15 → GPU7
# ─────────────────────────────────────────────────────────────────────────────
get_comp_gpu() {
    local idx=$1
    echo ${GPU_IDS[$((( idx % MAX_COMP_JOBS ) / JOBS_PER_GPU))]}
}

# ─────────────────────────────────────────────────────────────────────────────
# Main loop: per model, Phase1 then Phase2
# ─────────────────────────────────────────────────────────────────────────────
PYTHON_BIN="/opt/conda/bin/python"

for model_cfg in "${model_configs[@]}"; do
    IFS='|' read -r model_name hess_name model_id quant_module hfize_module eval_module lm_model_path <<< "$model_cfg"
    hess_path="${HESS_BASE}/${hess_name}"

    echo ""
    echo "════════════════════════════════════════"
    echo "Model: $model_name"
    echo "════════════════════════════════════════"

    # ── Phase 1: Compress all (16 simultaneous) ───────────────────────────
    echo "[Phase 1] Compress — $(( ${#experiment_names[@]} * ${#lmbda_values[@]} )) jobs, ${MAX_COMP_JOBS} simultaneous"

    comp_job_idx=0
    for exp_name in "${experiment_names[@]}"; do
        nwc_seed="${exp_nwc_seed[$exp_name]}"
        comp_model_base="${NWC_BASE}/${nwc_seed}"
        quant_flags="${exp_quant_flags[$exp_name]}"

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
                echo "  Waiting for comp batch (${comp_job_idx} jobs)..."
                wait
            fi
        done
    done
    wait
    echo "[Phase 1] All compress done for $model_name"

    # ── Phase 2: Hfize + Eval (one per GPU at a time) ─────────────────────
    echo "[Phase 2] Hfize+Eval — one per GPU (${N_GPU} parallel)"

    eval_job_idx=0
    for exp_name in "${experiment_names[@]}"; do
        for lmbda in "${lmbda_values[@]}"; do
            gpu_id=${GPU_IDS[$((eval_job_idx % N_GPU))]}
            SAVE_NAME="${model_name}/${exp_name}/lmbda${lmbda}"
            LOG_FILE="${LOG}/${SAVE_NAME}.log"

            echo "  >> [GPU $gpu_id] hfize+eval $exp_name lmbda=$lmbda"
            (
                export CUDA_VISIBLE_DEVICES=$gpu_id

                echo ">> hfize lmbda=${lmbda}" >> $LOG_FILE
                $PYTHON_BIN -m $hfize_module \
                    --quantized_path ${CKPT}/${SAVE_NAME} \
                    --base_model $lm_model_path \
                    --hf_output_path ${HF}/${SAVE_NAME} \
                    >> $LOG_FILE 2>&1

                echo ">> eval lmbda=${lmbda}" >> $LOG_FILE
                if [ "$eval_module" = "eval.eval_dino" ]; then
                    $PYTHON_BIN -m $eval_module \
                        --hf_path ${HF}/${SAVE_NAME} \
                        --output_path ${RES}/${SAVE_NAME} \
                        --imagenet_path $IMAGENET_PATH \
                        >> $LOG_FILE 2>&1
                else
                    $PYTHON_BIN -m $eval_module \
                        --hf_path ${HF}/${SAVE_NAME} \
                        --output_path ${RES}/${SAVE_NAME} \
                        --model_id $model_id \
                        >> $LOG_FILE 2>&1
                fi

                echo ">> cleanup" >> $LOG_FILE
                rm -rf "${HF}/${SAVE_NAME}"
            ) &

            ((eval_job_idx++))
            # N_GPU개마다 wait (one per GPU at a time)
            if (( eval_job_idx % N_GPU == 0 )); then
                wait
            fi
        done
    done
    wait
    echo "[Phase 2] All hfize+eval done for $model_name"

done

echo ""
echo "All models done."
