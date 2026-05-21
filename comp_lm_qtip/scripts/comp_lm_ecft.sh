#!/bin/bash
# PYTHON_BIN="/opt/conda/bin/python"
# unset PYTHONPATH
# export PATH="/opt/conda/bin:$PATH"  # PATH의 맨 앞에 base 경로 강제 삽입
# echo "Running with explicit python: $PYTHON_BIN"

quantize_flags=(
    "--ec_linear --row_normalize --scaleHinv --ecft_epochs 0 --ecft_aux_warmup_step 500 \
    --R_target 2 --ec_decoder_type identity --ecft_lmbda 0.01 --ecft_mode noise \
    --ecft_entropy_model parametric --ecft_num_gaussian 3 --ecft_num_laplacian 3 \
    --ec_entropy_chunk_rows 2024 --ft_grad_ckpt --ft_epochs 5 --ecft_decoder"
)
experiment_names=(
    'ecft_rnorm_scaleHiv_dec/iden_noise_parametric33_r2_wu500_ld0p01'    
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

ecft_lmbda=(10)
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
            SAVE_NAME=${model_name}/${exp_name}/lmbda${ld}

            echo "################## Running compression | Exp: ${exp_name} | Model: ${model_name} ##################" | tee $LOG/$SAVE_NAME.log
            mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
            
            # taskset -c 0-63 \
            python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
                --base_model $lm_model_path \
                --ecft_lmbda $ld \
                --in_hess_path $HESS \
                --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                --res_path ${RES}/${SAVE_NAME} \
                ${current_quantize_flags} \
                2>&1 | tee -a $LOG/$SAVE_NAME.log
            
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


            # echo "################## Running benchmark evaluation | R_target=${R} | Exp: ${exp_name} | Model: ${model_name} ##################"  | tee -a $LOG/${SAVE_NAME}_eval.log
            # python -m eval.eval_zeroshot_hf \
            #     --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
            #     --batch_size 8 \
            #     --hf_path $HF/$SAVE_NAME \
            #     --output_path $RES/${SAVE_NAME}_common_mmlu \
            #     2>&1 | tee -a $LOG/${SAVE_NAME}_eval.log

 

            if [ $eval_exit_code -eq 0 ]; then
                echo "Benchmark finished successfully. Starting cleanup..."

                if [ "$HF/$SAVE_NAME" != "$HF" ]; then
                    echo "Cleaning up temporary HF files: $HF/$SAVE_NAME"
                    rm -rf "$HF/$SAVE_NAME"
                    rm -rf "$CKPT/$SAVE_NAME"
                fi
            else
                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                echo "CRITICAL: Benchmark failed with exit code $eval_exit_code."
                echo "Skipping cleanup to allow manual inspection: $SAVE_NAME"
                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                rm -rf "$HF/$SAVE_NAME"
            fi
        done
    done
done
