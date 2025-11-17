#!/bin/bash
# ##########################################################################
# ##                       EXPERIMENT CONFIGURATION                       ##
# ##########################################################################
comp_model_bases=(
    # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
    "dumy"
)
quantize_flags=(
    # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --use_codes"
    "dumy"
)
experiment_names=(
    # "ql_ldlq128_rnorm_70b_targetbit"
    # "ql_ldlq128_rnorm_double_check"
    "ql_ldlq128_rnorm_ft"
)
##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
    # "meta-llama--Meta-Llama-3-8B"
    "meta-llama--Llama-2-7b-hf"
    # "meta-llama--Llama-3.2-3B"
    # "meta-llama--Llama-2-13b-hf"
    # "meta-llama--Llama-2-70b-hf_"
)
hess_paths=(
    # "../Wparam_dataset/quip_hess/llama3_8b_6144"
    "../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
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
RES="../hf_model_comp_results"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# export HF_HOME=/workspace/hf_cache/huggingface_nwc
export HF_HOME=/home/jgryu/.cache/huggingface

# export TRANSFORMERS_CACHE=$HF_HOME
# export HF_DATASETS_CACHE=$HF_HOME/datasets
# export HF_METRICS_CACHE=$HF_HOME/metrics

# 모든 실험에 공통으로 적용될 Lambda 값
lmbda_values=(30 50 100 300 1000 10000)
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
    echo "Using comp_model_base: $comp_model_base"
    echo "Using quantize flags: $current_quantize_flags"
    echo "Using Hessian path: $HESS"
    echo "------------------------------------------------------------------------"


    # 1. 외부 루프: 설정된 모든 실험을 순회
    for i in "${!experiment_names[@]}"; do
        # 현재 실험에 대한 변수 설정
        exp_name="${experiment_names[$i]}"
        comp_model_base="${comp_model_bases[$i]}"
        current_quantize_flags="${quantize_flags[$i]}"

        echo "========================================================================"
        echo "            STARTING EXPERIMENT SET: $exp_name"
        echo "========================================================================"
        
        # 3. 내부 루프: 설정된 모든 Lambda 값을 순회
        for lmbda in "${lmbda_values[@]}"; do
            SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}
            # E2EOUT_NAME=${model_name}/${exp_name}_e2e_ft_rnorm/lmbda${lmbda}
            E2EOUT_NAME=${model_name}/${exp_name}_e2e/lmbda${lmbda}
            mkdir -p $(dirname "$LOG/$E2EOUT_NAME.log")

            echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                    --hf_output_path $HF/${SAVE_NAME} \
                    2>&1 | tee $LOG/${E2EOUT_NAME}.log
                    # --skip_list 1_down \
                    # --sep_rnorm \

            # echo "################## Running PPL evaluation before | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            # python -m eval.eval_ppl_hf \
            #     --hf_path $HF/${SAVE_NAME} \
            #     --seqlen 2048 \
            #     --output_path ${RES}/${E2EOUT_NAME}_before \
            #     --datasets wikitext2 \
            #     --sep_rnorm \
            #     --no_use_cuda_graph 2>&1 | tee -a $LOG/$E2EOUT_NAME.log

            echo "################## End-to-End Finetuning | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m quantize_llama.finetune_e2e_llama --base_model $lm_model_path \
                --hf_path $HF/${SAVE_NAME} --devset_size 640 --ft_valid_size 128 \
                --ft_epochs 4 --ft_update_freq 4 --ft_bs 2 --ctx_size 4096 \
                --hf_output_path $CKPT/${E2EOUT_NAME} 2>&1 | tee -a $LOG/${E2EOUT_NAME}.log 

            echo "################## Running PPL evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m eval.eval_ppl_hf \
                --hf_path $CKPT/${E2EOUT_NAME} \
                --seqlen 2048 \
                --output_path ${RES}/${E2EOUT_NAME} \
                --datasets wikitext2,c4 \
                --no_use_cuda_graph 2>&1 | tee -a $LOG/$E2EOUT_NAME.log
                # --sep_rnorm \
                # --datasets wikitext2,c4 \

            echo "################## Running benchmark evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m eval.eval_zeroshot_hf \
                --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
                --batch_size 16 \
                --hf_path $CKPT/${E2EOUT_NAME} \
                --output_path $RES/${E2EOUT_NAME}_common_mmlu \
                2>&1 | tee -a $LOG/$E2EOUT_NAME.log
                # --sep_rnorm \

                # --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
                # --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag,mmlu \
                # --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
                # --tasks arc_challenge,arc_easy,winogrande,boolq,hellaswag \


            if [ "$HF/$SAVE_NAME" != "$HF" ]; then
                echo "Cleaning up temporary files for $SAVE_NAME"
                rm -rf "$HF/$SAVE_NAME"
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