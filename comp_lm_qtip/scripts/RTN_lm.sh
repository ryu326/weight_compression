#!/bin/bash

############################################
##                  SCRIPT SETUP                  ##
############################################
HF="../hf_model_comp/RTN"
LOG="./log"
RES="../hf_model_comp_results/RTN"

mkdir -p "$HF"
mkdir -p "$LOG"
mkdir -p "$RES"

export WANDB_SILENT=true

#-------------------------------------------------------------------------
#   실험 실행 함수 정의
#   인자: 1) 모델 이름, 2) GPU ID
#-------------------------------------------------------------------------
run_experiment() {
    local model_name=$1
    local gpu_id=$2
    local lm_model_path="../Wparam_dataset/hf_model/$model_name"

    # 이 함수에서 실행되는 모든 프로세스에 지정된 GPU 할당
    export CUDA_VISIBLE_DEVICES=$gpu_id

    echo "========================================================================"
    echo "          STARTING: ${model_name} on GPU ${gpu_id}"
    echo "========================================================================"

    for b in 3 4 5 6 8; do
        
        local SAVE_NAME="${model_name}/W${b}g128"
        
        echo "### [GPU ${gpu_id}] Running compression | ${SAVE_NAME} ###"
        mkdir -p "$(dirname "$LOG/$SAVE_NAME.log")"
        
        local pretrain_path="$HF/$SAVE_NAME"

        python -m quantize_llama.RTN_quantization \
            --model_path "$lm_model_path" \
            --output_path "$pretrain_path" \
            --num_bits "$b" \
            2>&1 | tee "$LOG/$SAVE_NAME.log"
        
        echo "### [GPU ${gpu_id}] Running PPL evaluation | ${SAVE_NAME} ###"
        python -m eval.eval_ppl_hf \
            --hf_path "$pretrain_path" \
            --seqlen 2048 \
            --output_path "$RES/$SAVE_NAME" \
            --dataset wikitext2,c4 \
            --no_use_cuda_graph 2>&1 | tee -a "$LOG/$SAVE_NAME.log"

        echo "### [GPU ${gpu_id}] Running benchmark evaluation | ${SAVE_NAME} ###"
        python -m eval.eval_zeroshot_hf \
            --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
            --batch_size 1 \
            --hf_path "$pretrain_path" \
            --output_path "$RES/$SAVE_NAME"

        if [ "$pretrain_path" != "$HF" ]; then
            echo "Cleaning up temporary files for $SAVE_NAME on GPU $gpu_id"
            rm -rf "$pretrain_path"
        fi
    done

    echo "========================================================================"
    echo "          FINISHED: ${model_name} on GPU ${gpu_id}"
    echo "========================================================================"
}

##########################################################################
##                      MAIN EXECUTION                                  ##
##    각 모델을 백그라운드(&)에서 동시에 실행                             ##
##########################################################################

echo "Starting experiments in parallel..."

# Llama-2-7b-hf 모델을 GPU 0에서 백그라운드로 실행
run_experiment "meta-llama--Llama-2-7b-hf" 0 &

# Llama-2-13b-hf 모델을 GPU 1에서 백그라운드로 실행
run_experiment "meta-llama--Llama-2-13b-hf" 1 &

# 'wait' 명령어는 모든 백그라운드 작업이 끝날 때까지 스크립트 종료를 대기
echo "Waiting for all background jobs to complete..."
wait

echo "========================================================================"
echo "                      All experiments finished."
echo "========================================================================"