#!/bin/bash
# ##########################################################################
# ##                       EXPERIMENT CONFIGURATION                       ##
# ##########################################################################
comp_model_bases=(
    "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
)
quantize_flags=(
    "--direction col --ql --Q 4 --col_normalize --ldlq --comp_batch_size 128 --ft_epochs 5"
)
experiment_names=(
    "ql_ldlq128_rnorm_ft"
)
##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
    "Qwen3-30B-A3B"
)
hess_paths=(
    # "/home/jgryu/workspace/weight_compression/Wparam_dataset/quip_hess/Mixtral-8x7B-v0.1_256"
    "../Wparam_dataset/quip_hess/Qwen3-30B-A3B"
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
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export HF_HOME=/workspace/hf_cache/huggingface_nwc
# export HF_HOME=/home/jgryu/.cache/huggingface
export HF_HOME=/workspace/Weight_compression/hf_cache/

# export TRANSFORMERS_CACHE=$HF_HOME
# export HF_DATASETS_CACHE=$HF_HOME/datasets
# export HF_METRICS_CACHE=$HF_HOME/metrics

# 모든 실험에 공통으로 적용될 Lambda 값
lmbda_values=(10000)
##########################################################################
##                        MAIN EXECUTION LOOP                           ##
##########################################################################
# 2. 중간 루프: 설정된 모든 모델을 순회
for j in "${!model_names[@]}"; do
    model_name="${model_names[$j]}"
    HESS="${hess_paths[$j]}"
    # lm_model_path="../Wparam_dataset/hf_model/$model_name"
    # lm_model_path="$model_name"
    lm_model_path="/workspace/Weight_compression/Wparam_dataset/hf_model/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"

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

            echo "################## Running compression | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
            mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
            
            # taskset -c 0-63 \
            # python -m quantize_llama.quantize_finetune_moe --save_path $CKPT/$SAVE_NAME \
            #     --base_model $lm_model_path \
            #     --comp_model_path $comp_model \
            #     --in_hess_path $HESS \
            #     --devset_size 384 --ft_valid_size 128 --batch_size 8 \
            #     ${current_quantize_flags} \
            #     2>&1 | tee $LOG/$SAVE_NAME.log

                # --devset_size 48 --ft_valid_size 16 --batch_size 1 \
                # --devset_size 384 --ft_valid_size 128 --batch_size 8 \

            
            echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m quantize_llama.hfize_moe --quantized_path $CKPT/${SAVE_NAME} \
                    --hf_output_path $HF/${SAVE_NAME} \
                    2>&1 | tee -a $LOG/${SAVE_NAME}_eval.log

                    # --skip_list 1_down \
            # SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}_skip1down


            # echo "################## Running PPL evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            # echo "Running evaluation for directory: $HF/$SAVE_NAME"
            # python -m eval.eval_ppl_hf \
            #     --hf_path $HF/${SAVE_NAME} \
            #     --seqlen 2048 \
            #     --output_path ${RES}/${SAVE_NAME} \
            #     --datasets wikitext2,c4 \
            #     --no_use_cuda_graph 2>&1 | tee -a $LOG/${SAVE_NAME}_eval.log

                # --datasets wikitext2,c4 \

            echo "################## Running benchmark evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m eval.eval_zeroshot_hf \
                --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
                --batch_size 16 \
                --hf_path $HF/$SAVE_NAME \
                --output_path $RES/${SAVE_NAME}_common_mmlu \
                2>&1 | tee -a $LOG/$SAVE_NAME.log

                # --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
                # --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag,mmlu \
                # --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \


            if [ "$HF/$SAVE_NAME" != "$HF" ]; then
                echo "Cleaning up temporary files for $SAVE_NAME"
                rm -rf "$HF/$SAVE_NAME"
                # rm -rf "$CKPT/$SAVE_NAME"
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