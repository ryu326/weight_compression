#!/bin/bash
# ##########################################################################
# ##                       EXPERIMENT CONFIGURATION                       ##
# ##########################################################################
comp_model_bases=(
    "/home/jgryu/workspace/weight_compression/NWC/checkpoint2/n_rb1"
    "/home/jgryu/workspace/weight_compression/NWC/checkpoint2/n_rb2"
)
quantize_flags=(
    "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128"
    "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128"
)
experiment_names=(
    "ql_ldlq128_rnorm_nres1"
    "ql_ldlq128_rnorm_nres2"
)
##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
    "meta-llama--Meta-Llama-3-8B"
)
hess_paths=(
    "../Wparam_dataset/quip_hess/llama3_8b_6144"
)
############################################
##              SCRIPT SETUP              ##
############################################
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"
RES="../hf_model_comp_results"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES

# 사용할 GPU 목록 설정 (여기서 정의한 GPU들을 돌아가며 사용합니다)
export CUDA_VISIBLE_DEVICES=3,4,5,6,7
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}

export HF_HOME=/home/jgryu/.cache/huggingface

# 모든 실험에 공통으로 적용될 Lambda 값
lmbda_values=(30.0 50.0 100.0 300.0 10000.0)

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
        comp_model_base="${comp_model_bases[$i]}"
        current_quantize_flags="${quantize_flags[$i]}"

        echo "========================================================================"
        echo "            STARTING EXPERIMENT SET: $exp_name"
        echo "========================================================================"
        
        # GPU 할당을 위한 카운터 초기화
        counter=0
        
        for lmbda in "${lmbda_values[@]}"; do
            # 현재 작업에 할당할 GPU ID 계산
            gpu_idx=$((counter % NUM_GPUS))
            CURRENT_GPU=${GPU_LIST[$gpu_idx]}
            
            # 백그라운드 병렬 실행 시작 (괄호로 묶어 서브쉘에서 실행)
            (
                echo ">>> [GPU $CURRENT_GPU] Starting lmbda=${lmbda}..."
                
                # 서브쉘 내부에서 해당 프로세스만의 CUDA_VISIBLE_DEVICES 설정
                export CUDA_VISIBLE_DEVICES=$CURRENT_GPU
                
                SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}
                comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
                
                # 로그 디렉토리 생성 (동시 접근 충돌 방지 위해 -p 사용)
                mkdir -p $(dirname "$LOG/$SAVE_NAME.log")

                echo "################## Running compression | lmbda=${lmbda} | GPU: $CURRENT_GPU ##################"
                # taskset은 CPU 코어 할당입니다. 병렬 실행 시 충돌을 막기 위해 제거하거나 범위를 나누는 것이 좋으나, 
                # OS 스케줄링에 맡기기 위해 여기서는 그대로 두되 필요시 조정하십시오.
                taskset -c 0-63 \
                python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
                    --base_model $lm_model_path \
                    --comp_model_path $comp_model \
                    --in_hess_path $HESS \
                    --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                    ${current_quantize_flags} \
                    > "$LOG/$SAVE_NAME.log" 2>&1

                echo "################## Running hfize | lmbda=${lmbda} | GPU: $CURRENT_GPU ##################"
                python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                        --hf_output_path $HF/${SAVE_NAME} \
                        >> "$LOG/$SAVE_NAME.log" 2>&1

                echo "################## Running PPL evaluation | lmbda=${lmbda} | GPU: $CURRENT_GPU ##################"
                python -m eval.eval_ppl_hf \
                    --hf_path $HF/${SAVE_NAME} \
                    --seqlen 2048 \
                    --output_path ${RES}/${SAVE_NAME} \
                    --datasets wikitext2,c4 \
                    --no_use_cuda_graph >> "$LOG/$SAVE_NAME.log" 2>&1

                if [ "$HF/$SAVE_NAME" != "$HF" ]; then
                    echo "Cleaning up temporary files for $SAVE_NAME"
                    rm -rf "$HF/$SAVE_NAME"
                    # rm -rf "$CKPT/$SAVE_NAME"
                fi
                
                echo "<<< [GPU $CURRENT_GPU] Finished lmbda=${lmbda}"
            ) & 
            
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