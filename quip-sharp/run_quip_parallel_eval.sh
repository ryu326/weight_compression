#!/bin/bash
MODELS_TO_RUN=(
    # "llama3.2_1b_inst"
    # "llama3.2_3b_inst"
    "llama3_8b"
    # "llama2_7b"
    # "llama2_13b"
    # "llama3.2_3b"
)

CKPT="../hf_model_comp/quip-sharp/ckpt"
HF="../hf_model_comp/quip-sharp/hf"
LOG="./log"
RES="../hf_model_comp_results/quip-sharp"

K_VALUES=(2) 
GPUS_TO_USE=(1) 

MAX_PARALLEL_JOBS=${#GPUS_TO_USE[@]}

export WANDB_SILENT=true
export HF_HOME=/home/jgryu/.cache/huggingface

declare -A MODEL_PATHS
MODEL_PATHS=(
    ["llama2_7b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"
    ["llama2_13b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"
    ["llama3_8b"]="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"
    ["vicuna_7b"]="../Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5"
    ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
    ["llama3.2_3b_inst"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B-Instruct"
    ["llama3.2_1b_inst"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-1B-Instruct"
)
declare -A HESS_PATHS
HESS_PATHS=(
    ["llama2_7b"]="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
    ["llama2_13b"]="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
    ["llama3_8b"]="../Wparam_dataset/quip_hess/llama3_8b_6144"
    ["vicuna_7b"]="../Wparam_dataset/quip_hess/lmsys--vicuna-7b-v1.5_256"
    ["llama3.2_3b"]="../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256"
    ["llama3.2_3b_inst"]="../Wparam_dataset/quip_hess/Llama-3.2-3B-Instruct-Hessians"
    ["llama3.2_1b_inst"]="../Wparam_dataset/quip_hess/Llama-3.2-1B-Instruct-Hessians"
)

mkdir -p $CKPT $HF $LOG $RES

run_eval_for_k() {
    local K=$1
    local GPU=$2
    local model_key=$3 # model_key도 인자로 받도록 수정

    export CUDA_VISIBLE_DEVICES=$GPU

    NAME="${model_key}/ft1_e2e_after/${K}bit"
    CKPT_PATH="${CKPT}/${NAME}"
    HF_PATH="${HF}/${NAME}"
    LOG_FILE="${LOG}/evals/${NAME}.log"

    mkdir -p $CKPT_PATH
    mkdir -p $(dirname "$LOG_FILE")

    echo "[GPU $GPU / K=$K] [Stage: Convert to HF format]" | tee $LOG_FILE
    python -m quantize_llama.hfize_llama \
        --quantized_path ${CKPT_PATH} \
        --hf_output_path $HF_PATH 2>&1 | tee -a $LOG_FILE

    echo "[GPU $GPU / K=$K] ### [Stage: Eval PPL] ###" | tee -a $LOG_FILE
    python -m eval.eval_ppl \
        --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME} \
        --no_use_cuda_graph \
        --seqlen 2048  2>&1 | tee -a $LOG_FILE

    echo "[GPU $GPU / K=$K] ### [Stage: Eval Zero-shot] ###" | tee -a $LOG_FILE
    python -m eval.eval_zeroshot_ \
        --tasks arc_challenge,arc_easy,boolq,piqa,winogrande,hellaswag,mmlu \
        --batch_size 2  --hf_path ${HF_PATH} \
        --output_path ${RES}/${NAME}_common_mmlu 2>&1 | tee -a $LOG_FILE

    if [ "$HF_PATH" != "$HF" ]; then
        echo "[GPU $GPU / K=$K] Cleaning up temporary files for $NAME" | tee -a $LOG_FILE
        rm -rf "$HF_PATH"
    fi
}

for model_key in "${MODELS_TO_RUN[@]}"; do
    base_model=${MODEL_PATHS[$model_key]}
    HESS=${HESS_PATHS[$model_key]}

    echo "============================================================"
    echo "           STARTING DYNAMIC QUEUE FOR MODEL: [$model_key]"
    echo "           (Max ${MAX_PARALLEL_JOBS} jobs on GPUs: ${GPUS_TO_USE[*]})"
    echo "============================================================"

    gpu_index=0 # 사용할 GPU 인덱스 카운터
    
    for K in ${K_VALUES[@]}; do
        current_jobs=$(jobs -p | wc -l)

        if [ $current_jobs -ge $MAX_PARALLEL_JOBS ]; then
            echo "[Queue Manager] Waiting for a GPU to become free..."
            wait -n 
        fi

        GPU=${GPUS_TO_USE[$gpu_index]}
        
        echo "[Queue Manager] Assigning K=$K to GPU $GPU"
        
        run_eval_for_k $K $GPU $model_key &
        gpu_index=$(( (gpu_index + 1) % MAX_PARALLEL_JOBS ))
        
        sleep 1 
    done


    echo "[Queue Manager] All K-values dispatched. Waiting for remaining jobs of $model_key to finish..."
    wait

    echo "============================================================"
    echo "           ALL K-EVALS FINISHED FOR MODEL: [$model_key]"
    echo "============================================================"

done # model_key 루프 종료

echo "############################################################"
echo "##               All experiments finished.                ##"
echo "############################################################"