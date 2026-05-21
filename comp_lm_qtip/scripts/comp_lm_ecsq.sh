#!/bin/bash
# PYTHON_BIN="/opt/conda/bin/python"
# unset PYTHONPATH
# export PATH="/opt/conda/bin:$PATH"  # PATH의 맨 앞에 base 경로 강제 삽입
# echo "Running with explicit python: $PYTHON_BIN"

quantize_flags=(
    # "--ecsq --normalization_search --scaleH"
    # "--ecsq --row_normalize --scaleH"
    # "--ecsq --row_normalize"
    # "--ec_linear --row_normalize --scaleHinv --ecft_epochs 1000"
    "--ec_linear --row_normalize --scaleHinv --ecft_epochs 0 --ecft_aux_warmup_step 500"
    # "--ec_linear --row_normalize --scaleHinv --ecft_epochs 200 --ecft_aux_warmup_step 100"
)
experiment_names=(
    'eclinear2_rnorm_scaleHiv_ep0_warmup500_target3'    
    # 'eclinear2_rnorm_scaleHiv_ep200_warmup100'    
    # 'eclinear2_rnorm_scaleHiv_ep1000'    
    # 'ecsq_normalization_search_scaleH'
    # 'ecsq_rnorm_scaleH'
    # 'ecsq_rnorm'
)
##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
    "meta-llama--Meta-Llama-3-8B"
    # "meta-llama--Llama-2-7b-hf"
    # "meta-llama--Llama-3.2-3B"
    # "meta-llama--Llama-2-13b-hf"
    # "meta-llama--Llama-2-70b-hf_"
)
hess_paths=(
    "../Wparam_dataset/quip_hess/llama3_8b_6144"
    # "../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
    # "../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256"
    # "../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
    # "../Wparam_dataset/quip_hess/llama2_70b_relaxml_git/Hessians-Llama-2-70b-6144"
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HF_HOME=/workspace/hf_cache/huggingface_nwc
export HF_HOME=/home/jgryu/.cache/huggingface

# export TRANSFORMERS_CACHE=$HF_HOME
# export HF_DATASETS_CACHE=$HF_HOME/datasets
# export HF_METRICS_CACHE=$HF_HOME/metrics

# 모든 실험에 공통으로 적용될 Lambda 값
R_targets=3
ecft_lmbda=(10 50 100)
##########################################################################
##                        MAIN EXECUTION LOOP                           ##
##########################################################################
# 2. 중간 루프: 설정된 모든 모델을 순회
for j in "${!model_names[@]}"; do
    model_name="${model_names[$j]}"
    HESS="${hess_paths[$j]}"
    lm_model_path="../Wparam_dataset/hf_model/$model_name"

    echo "------------------------------------------------------------------------"
    echo "            MODEL: $model_name"
    echo "------------------------------------------------------------------------"
    echo "Using quantize flags: $current_quantize_flags"
    echo "Using Hessian path: $HESS"
    echo "------------------------------------------------------------------------"

    # 1. 외부 루프: 설정된 모든 실험을 순회
    for i in "${!experiment_names[@]}"; do
        # 현재 실험에 대한 변수 설정
        exp_name="${experiment_names[$i]}"
        current_quantize_flags="${quantize_flags[$i]}"

        echo "========================================================================"
        echo "            STARTING EXPERIMENT SET: $exp_name"
        echo "========================================================================"
        
        # 3. 내부 루프: 설정된 모든 Lambda 값을 순회
        for ld in "${ecft_lmbda[@]}"; do
            SAVE_NAME=${model_name}/${exp_name}/lmbda${R}

            echo "################## Running compression | R_target=${R} | Exp: ${exp_name} | Model: ${model_name} ##################" | tee $LOG/$SAVE_NAME.log
            mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
            
            # taskset -c 0-63 \
            python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
                --base_model $lm_model_path \
                --ecft_lmbda $ld \
                --in_hess_path $HESS \
                --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                --res_path ${RES}/${SAVE_NAME} \
                --R_target $R_targets \
                ${current_quantize_flags} \
                2>&1 | tee -a $LOG/$SAVE_NAME.log
                

                # --R_target $R \
                # --perlayer_ft_lmbda $lmbda \
                # --devset_size 48 --ft_valid_size 16 --batch_size 1 \
                # --devset_size 384 --ft_valid_size 128 --batch_size 8 \
            
            echo "################## Running hfize | R_target=${R} | Exp: ${exp_name} | Model: ${model_name} ##################" | tee $LOG/${SAVE_NAME}_eval.log
            python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                    --hf_output_path $HF/${SAVE_NAME} \
                    --base_model $lm_model_path \
                    2>&1 | tee -a $LOG/${SAVE_NAME}_eval.log

            echo "################## Running PPL evaluation | R_target=${R} | Exp: ${exp_name} | Model: ${model_name} ##################"  | tee -a $LOG/${SAVE_NAME}_eval.log
            echo "Running evaluation for directory: $HF/$SAVE_NAME"
            python -m eval.eval_ppl_hf \
                --hf_path $HF/${SAVE_NAME} \
                --seqlen 2048 \
                --output_path ${RES}/${SAVE_NAME} \
                --datasets wikitext2,c4 \
                --no_use_cuda_graph \
                2>&1 | tee -a $LOG/${SAVE_NAME}_eval.log

                # --datasets wikitext2,c4 \

            # echo "################## Running benchmark evaluation | R_target=${R} | Exp: ${exp_name} | Model: ${model_name} ##################"  | tee -a $LOG/${SAVE_NAME}_eval.log
            # python -m eval.eval_zeroshot_hf \
            #     --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
            #     --batch_size 8 \
            #     --hf_path $HF/$SAVE_NAME \
            #     --output_path $RES/${SAVE_NAME}_common_mmlu \
            #     2>&1 | tee -a $LOG/${SAVE_NAME}_eval.log

                # --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
                # --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag,mmlu \
                # --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \


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
        done
    done
done


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m eval.eval_zeroshot_hf \
#     --tasks mmlu \
#     --batch_size 8  \
#     --hf_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B \
#     --output_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B_mmlu

# python -m eval.eval_zeroshot_hf \
#     --tasks mmlu \
#     --batch_size 8  \
#     --hf_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf \
#     --output_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf_mmlu