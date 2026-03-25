#!/bin/bash
# PYTHON_BIN="/opt/conda/bin/python"
# unset PYTHONPATH
# export PATH="/opt/conda/bin:$PATH"  # PATH의 맨 앞에 base 경로 강제 삽입
# echo "Running with explicit python: $PYTHON_BIN"


# ##########################################################################
# ##                       EXPERIMENT CONFIGURATION                       ##
# ##########################################################################
quantize_flags=(
    "--ecsq --row_normalize --scaleHinv"
    # "--ecsq"
)
experiment_names=(
    'ecsq_rnorm_scaleHinv'
    # 'ecsq'
)
##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
    "meta-llama--Meta-Llama-3-8B"
    # "meta-llama--Llama-2-7b-hf"
)
hess_paths=(
    "../Wparam_dataset/quip_hess/llama3_8b_6144"
    # "../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
)
############################################
##              SCRIPT SETUP              ##
############################################
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"
RES="../hf_model_comp_results_v2"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES

# 사용할 GPU 목록 설정 (여기서 정의한 GPU들을 돌아가며 사용합니다)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}

export HF_HOME=/home/jgryu/.cache/huggingface

# 모든 실험에 공통으로 적용될 Lambda 값
# lmbda_values=(30 50 100 300 1000 10000)
R_targets=(2 2.3 2.5 3 3.5 4 4.5 5 5.5 6 6.5)

##########################################################################
##                        MAIN EXECUTION LOOP                           ##
##########################################################################

for j in "${!model_names[@]}"; do
    model_name="${model_names[$j]}"
    HESS="${hess_paths[$j]}"
    lm_model_path="../Wparam_dataset/hf_model/$model_name"

    echo "------------------------------------------------------------------------"
    echo "            MODEL: $model_name"
    echo "------------------------------------------------------------------------"
    echo "Using Hessian path: $HESS"
    echo "------------------------------------------------------------------------"

    for i in "${!experiment_names[@]}"; do
        exp_name="${experiment_names[$i]}"
        current_quantize_flags="${quantize_flags[$i]}"

        echo "========================================================================"
        echo "            STARTING EXPERIMENT SET: $exp_name"
        echo "========================================================================"
        
        # GPU 할당을 위한 카운터 초기화
        counter=0
        
        for R in "${R_targets[@]}"; do
            # 현재 작업에 할당할 GPU ID 계산
            gpu_idx=$((counter % NUM_GPUS))
            CURRENT_GPU=${GPU_LIST[$gpu_idx]}
            
            (
                echo ">>> [GPU $CURRENT_GPU] Starting R_target=${R}..."
                
                export CUDA_VISIBLE_DEVICES=$CURRENT_GPU
                
                SAVE_NAME=${model_name}/${exp_name}/lmbda${R}

                mkdir -p $(dirname "$LOG/$SAVE_NAME.log")

                echo "################## Running compression | R_target=${R} | GPU: $CURRENT_GPU ##################" | tee $LOG/$SAVE_NAME.log
                # taskset은 CPU 코어 할당입니다. 병렬 실행 시 충돌을 막기 위해 제거하거나 범위를 나누는 것이 좋으나, 
                # OS 스케줄링에 맡기기 위해 여기서는 그대로 두되 필요시 조정하십시오.
                # taskset -c 0-63 \
                python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
                    --base_model $lm_model_path \
                    --R_target $R \
                    --in_hess_path $HESS \
                    --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                    ${current_quantize_flags} \
                    >> "$LOG/$SAVE_NAME.log" 2>&1

                echo "################## Running hfize | R_target=${R} | GPU: $CURRENT_GPU ##################" | tee $LOG/${SAVE_NAME}_eval.log
                python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                        --hf_output_path $HF/${SAVE_NAME} \
                        --base_model $lm_model_path \
                        >> "$LOG/${SAVE_NAME}_eval.log" 2>&1

                echo "################## Running PPL evaluation | R_target=${R} | GPU: $CURRENT_GPU ##################"  | tee -a $LOG/${SAVE_NAME}_eval.log
                python -m eval.eval_ppl_hf \
                    --hf_path $HF/${SAVE_NAME} \
                    --seqlen 2048 \
                    --output_path ${RES}/${SAVE_NAME} \
                    --datasets wikitext2,c4 \
                    --no_use_cuda_graph \
                    >> "$LOG/${SAVE_NAME}_eval.log" 2>&1                

                echo "################## Running benchmark evaluation | R_target=${R} | GPU: $CURRENT_GPU ##################"  | tee -a $LOG/${SAVE_NAME}_eval.log
                python -m eval.eval_zeroshot_hf \
                    --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
                    --batch_size 2 \
                    --hf_path $HF/$SAVE_NAME \
                    --output_path $RES/${SAVE_NAME}_common_mmlu \
                    >> "$LOG/${SAVE_NAME}_eval.log" 2>&1

                if [ "$HF/$SAVE_NAME" != "$HF" ]; then
                    echo "Cleaning up temporary files for $SAVE_NAME"
                    rm -rf "$HF/$SAVE_NAME"
                fi                
                # ft_epochs 값 추출
                if [[ "$current_quantize_flags" =~ --ft_epochs[[:space:]]+([0-9]+) ]]; then
                    ft_epochs=${BASH_REMATCH[1]}
                else
                    ft_epochs=0
                fi                
                # ft_epochs가 0보다 크면 CKPT 디렉토리를 삭제하지 않음
                if [ "$ft_epochs" -le 0 ]; then
                    echo "Cleaning up checkpoint files for $SAVE_NAME (ft_epochs=$ft_epochs)"
                    rm -rf "$CKPT/$SAVE_NAME"
                else
                    echo "Keeping checkpoint files for $SAVE_NAME (ft_epochs=$ft_epochs > 0)"
                fi
                
                echo "<<< [GPU $CURRENT_GPU] Finished R_target=${R}"
            ) & 

            # Stagger job launches to reduce startup contention.
            sleep 10
            
            # 카운터 증가
            counter=$((counter + 1))

            # GPU 개수만큼 작업이 실행되었으면 모두 끝날 때까지 대기 (Batch Wait)
            # 예: GPU가 4개라면, 4개의 작업이 던져진 후 wait가 걸림
            if [ $((counter % NUM_GPUS)) -eq 0 ]; then
                echo "Waiting for current batch of $NUM_GPUS jobs to finish..."
                wait
            fi
        done
        
        # 남은 작업이 있다면 대기
        wait
    done
done
