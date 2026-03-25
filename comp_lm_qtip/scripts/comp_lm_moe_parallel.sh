#!/bin/bash
# ##########################################################################
# ##                       EXPERIMENT CONFIGURATION                       ##
# ##########################################################################
# PYTHON_BIN="/opt/conda/bin/python"
# unset PYTHONPATH
# export PATH="/opt/conda/bin:$PATH"  # PATH의 맨 앞에 base 경로 강제 삽입
# echo "Running with explicit python: $PYTHON_BIN"

comp_model_bases=(
    "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
)
quantize_flags=(
    "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 64 --ft_epochs 0"
    # "--direction col --ql --Q 4 --col_normalize --ldlq --comp_batch_size 64 --ft_epochs 5"
)
experiment_names=(
    "ql_ldlq64_rnorm"
    # "ql_ldlq64_rnorm_ft"
)
##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
    "openai/gpt-oss-20b"
)
hess_paths=(
    "/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess/gpt-oss-20b_1024"
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
export CUDA_VISIBLE_DEVICES=0
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}

export HF_HOME=/home/jgryu/.cache/huggingface

# 모든 실험에 공통으로 적용될 Lambda 값
lmbda_values=(30 50 100 300 1000 10000)

##########################################################################
##                        MAIN EXECUTION LOOP                           ##
##########################################################################

for j in "${!model_names[@]}"; do
    model_name="${model_names[$j]}"
    HESS="${hess_paths[$j]}"
    # lm_model_path="../Wparam_dataset/hf_model/$model_name"
    lm_model_path="$model_name"

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
                # taskset -c 0-63 \
                python -m quantize_llama.quantize_finetune_moe --save_path $CKPT/$SAVE_NAME \
                    --base_model $lm_model_path \
                    --comp_model_path $comp_model \
                    --in_hess_path $HESS \
                    --devset_size 384 --ft_valid_size 128 \
                    --batch_size 4 --ft_bs 1 \
                    ${current_quantize_flags} \
                    > $LOG/$SAVE_NAME.log 2>&1

                echo "################## Running hfize | lmbda=${lmbda} | GPU: $CURRENT_GPU ##################"
                python -m quantize_llama.hfize_moe_hf --quantized_path $CKPT/${SAVE_NAME} \
                        --hf_output_path $HF/${SAVE_NAME} \
                        --base_model $lm_model_path \
                        > "$LOG/${SAVE_NAME}_eval.log" 2>&1

                # echo "################## Running PPL evaluation | lmbda=${lmbda} | GPU: $CURRENT_GPU ##################"
                # python -m eval.eval_ppl_hf \
                #     --hf_path $HF/${SAVE_NAME} \
                #     --seqlen 2048 \
                #     --output_path ${RES}/${SAVE_NAME} \
                #     --datasets wikitext2,c4 \
                #     --gptoss_replace_version standard \
                #     --no_use_cuda_graph >> "$LOG/${SAVE_NAME}_eval.log" 2>&1                

                # echo "################## Running benchmark evaluation | lmbda=${lmbda} | GPU: $CURRENT_GPU ##################"
                # python -m eval.eval_zeroshot_hf \
                #     --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
                #     --hf_path $HF/$SAVE_NAME \
                #     --output_path $RES/${SAVE_NAME}_common_mmlu \
                #     --gptoss_replace_version standard \
                #     >> "$LOG/${SAVE_NAME}_eval.log" 2>&1 

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