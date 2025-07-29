#!/bin/bash
# ##########################################################################
# ##                       EXPERIMENT CONFIGURATION                       ##
# ##########################################################################
comp_model_bases=(
    "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
    # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M32__rdloss_ql_size16_encdim512_M32_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100" # 예시
    # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__scaleHinv_sig0.0001_std_rnormed_lidx_row_1024.pt/rdloss_ql_size128_encdim1024_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
    # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_lidx_row_1024.pt/rdloss_ql_size128_encdim1024_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
    # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_lidx_row_1024.pt/rdloss_ql_size128_encdim2048_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
    # "../NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/rdloss_size128_encdim1024_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
    # "../NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_uniform31.6_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/rdloss_size128_encdim1024_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
    # "../NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/rdloss_size16_encdim512_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
)
quantize_flags=(
    # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 5"
    "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 5 --ft_rnorm"
    # "--direction row --scaleHinv --row_normalize --ldlq --comp_batch_size 128"
    # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128"
    # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond --scale_cond_test 0.01" 
    # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond" 
    # "--direction row --scaleH --row_normalize --comp_batch_size 2048 --scale_cond" 
)
experiment_names=(
    # "ql_ldlq128_rnorm_ft"
    "ql_ldlq128_rnorm_ft_ft"
    # "scaleHinv_std_ldlq128/size128_encdim1024_M256"
    # "scaleH_std_ldlq128/size128_encdim1024_M256"
    # "scaleH_std_ldlq128/size128_encdim2048_M256"
    # "scaleH_std_ldlq128_scale_cond(col)_uniform31.6/test0.01_size128_encdim2048_M256"
    # "scaleH_std_ldlq128_scale_cond(col)_uniform10/size128_encdim2048_M256"
    # "test2_ql_rnorm"
    # "scaleH_std_ldlq128_scale_cond(col)/size16_encdim512_M256"

)
##########################################################################
##                           MODEL CONFIGURATION                        ##
##########################################################################
model_names=(
    # "meta-llama--Meta-Llama-3-8B"
    # "meta-llama--Llama-2-7b-hf"
    # "meta-llama--Llama-2-13b-hf"
    "meta-llama--Llama-2-70b-hf_"
)
hess_paths=(
    # "../Wparam_dataset/quip_hess/llama3_8b_6144"
    # "../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
    # "../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
    "../Wparam_dataset/quip_hess/llama2_70b_relaxml_git/Hessians-Llama-2-70b-6144"
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7
export WANDB_SILENT=true

# 모든 실험에 공통으로 적용될 Lambda 값
lmbda_values=(100)
##########################################################################
##                        MAIN EXECUTION LOOP                           ##
##########################################################################
# 1. 외부 루프: 설정된 모든 실험을 순회
for i in "${!experiment_names[@]}"; do
    # 현재 실험에 대한 변수 설정
    exp_name="${experiment_names[$i]}"
    comp_model_base="${comp_model_bases[$i]}"
    current_quantize_flags="${quantize_flags[$i]}"

    echo "========================================================================"
    echo "            STARTING EXPERIMENT SET: $exp_name"
    echo "========================================================================"

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
        
        # 3. 내부 루프: 설정된 모든 Lambda 값을 순회
        for lmbda in "${lmbda_values[@]}"; do
            SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}

            echo "################## Running compression | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
            mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
            
            taskset -c 0-31 \
            python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
                --base_model $lm_model_path \
                --comp_model_path $comp_model \
                --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 --batch_size 8 \
                ${current_quantize_flags} \
                2>&1 | tee $LOG/$SAVE_NAME.log

            echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
                    --hf_output_path $HF/${SAVE_NAME} 2>&1 | tee -a $LOG/$SAVE_NAME.log

            # echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            # python -m quantize_llama.hfize_llama_rnorm --quantized_path $CKPT/${SAVE_NAME} \
            #         --hf_output_path $HF/${SAVE_NAME} 2>&1 | tee -a $LOG/$SAVE_NAME.log

            echo "################## Running PPL evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            echo "Running evaluation for directory: $HF/$SAVE_NAME"
            python -m eval.eval_ppl_hf \
                --hf_path $HF/$SAVE_NAME \
                --seqlen 4096 \
                --output_path $RES/$SAVE_NAME \
                --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log

            echo "################## Running benchmark evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
            python -m eval.eval_zeroshot_hf \
                --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \
                --batch_size 1  \
                --hf_path $HF/$SAVE_NAME \
                --output_path $RES/$SAVE_NAME

                # --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
                # --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \

            if [ "$HF/$SAVE_NAME" != "$HF" ]; then
                echo "Cleaning up temporary files for $SAVE_NAME"
                rm -rf "$HF/$SAVE_NAME"
                # rm -rf "$CKPT/$SAVE_NAME"
            fi
        done
    done
done

# #!/bin/bash
# # ##########################################################################
# # ##                       EXPERIMENT CONFIGURATION                       ##
# # ##########################################################################
# comp_model_bases=(
#     "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
#     # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M32__rdloss_ql_size16_encdim512_M32_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100" # 예시
#     # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__scaleHinv_sig0.0001_std_rnormed_lidx_row_1024.pt/rdloss_ql_size128_encdim1024_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
#     # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_lidx_row_1024.pt/rdloss_ql_size128_encdim1024_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
#     # "../NWC/checkpoint/nwc_ql/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_lidx_row_1024.pt/rdloss_ql_size128_encdim2048_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
#     # "../NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/rdloss_size128_encdim1024_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
#     # "../NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_uniform31.6_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/rdloss_size128_encdim1024_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
#     # "../NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/rdloss_size16_encdim512_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
# )
# quantize_flags=(
#     # "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 5"
#     "--direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 5 --ft_rnorm"
#     # "--direction row --scaleHinv --row_normalize --ldlq --comp_batch_size 128"
#     # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128"
#     # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond --scale_cond_test 0.01" 
#     # "--direction row --scaleH --row_normalize --ldlq --comp_batch_size 128 --scale_cond" 
#     # "--direction row --scaleH --row_normalize --comp_batch_size 2048 --scale_cond" 
# )
# experiment_names=(
#     # "ql_ldlq128_rnorm_ft"
#     "ql_ldlq128_rnorm_ft_ft"
#     # "scaleHinv_std_ldlq128/size128_encdim1024_M256"
#     # "scaleH_std_ldlq128/size128_encdim1024_M256"
#     # "scaleH_std_ldlq128/size128_encdim2048_M256"
#     # "scaleH_std_ldlq128_scale_cond(col)_uniform31.6/test0.01_size128_encdim2048_M256"
#     # "scaleH_std_ldlq128_scale_cond(col)_uniform10/size128_encdim2048_M256"
#     # "test2_ql_rnorm"
#     # "scaleH_std_ldlq128_scale_cond(col)/size16_encdim512_M256"

# )
# ##########################################################################
# ##                           MODEL CONFIGURATION                        ##
# ##########################################################################
# model_names=(
#     # "meta-llama--Meta-Llama-3-8B"
#     # "meta-llama--Llama-2-7b-hf"
#     "meta-llama--Llama-2-13b-hf"
#     # "meta-llama--Llama-2-70b-hf_"
# )
# hess_paths=(
#     # "../Wparam_dataset/quip_hess/llama3_8b_6144"
#     # "../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
#     "../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
#     # "../Wparam_dataset/quip_hess/llama2_70b_relaxml_git/Hessians-Llama-2-70b-6144"
# )
# ############################################
# ##              SCRIPT SETUP              ##
# ############################################
# CKPT="../hf_model_comp/comp_qtip/ckpt"
# HF="../hf_model_comp/comp_qtip/hf"
# LOG="./log"
# RES="../hf_model_comp_results"

# mkdir -p $CKPT
# mkdir -p $HF
# mkdir -p $LOG
# mkdir -p $RES
# export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7
# export WANDB_SILENT=true

# # 모든 실험에 공통으로 적용될 Lambda 값
# lmbda_values=(30 50 100 300 1000)
# # lmbda_values=(30 50 100 300)
# ##########################################################################
# ##                        MAIN EXECUTION LOOP                           ##
# ##########################################################################
# # 1. 외부 루프: 설정된 모든 실험을 순회
# for i in "${!experiment_names[@]}"; do
#     # 현재 실험에 대한 변수 설정
#     exp_name="${experiment_names[$i]}"
#     comp_model_base="${comp_model_bases[$i]}"
#     current_quantize_flags="${quantize_flags[$i]}"

#     echo "========================================================================"
#     echo "            STARTING EXPERIMENT SET: $exp_name"
#     echo "========================================================================"

#     # 2. 중간 루프: 설정된 모든 모델을 순회
#     for j in "${!model_names[@]}"; do
#         model_name="${model_names[$j]}"
#         HESS="${hess_paths[$j]}"
#         lm_model_path="../Wparam_dataset/hf_model/$model_name"

#         echo "------------------------------------------------------------------------"
#         echo "            MODEL: $model_name"
#         echo "------------------------------------------------------------------------"
#         echo "Using comp_model_base: $comp_model_base"
#         echo "Using quantize flags: $current_quantize_flags"
#         echo "Using Hessian path: $HESS"
#         echo "------------------------------------------------------------------------"
        
#         # 3. 내부 루프: 설정된 모든 Lambda 값을 순회
#         for lmbda in "${lmbda_values[@]}"; do
#             SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}

#             echo "################## Running compression | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
#             comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
#             mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
            
#             taskset -c 0-31 \
#             python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
#                 --base_model $lm_model_path \
#                 --comp_model_path $comp_model \
#                 --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 --batch_size 8 \
#                 ${current_quantize_flags} \
#                 2>&1 | tee $LOG/$SAVE_NAME.log

#             echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
#             python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
#                     --hf_output_path $HF/${SAVE_NAME} 2>&1 | tee -a $LOG/$SAVE_NAME.log

#             # echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
#             # python -m quantize_llama.hfize_llama_rnorm --quantized_path $CKPT/${SAVE_NAME} \
#             #         --hf_output_path $HF/${SAVE_NAME} 2>&1 | tee -a $LOG/$SAVE_NAME.log

#             echo "################## Running PPL evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
#             echo "Running evaluation for directory: $HF/$SAVE_NAME"
#             python -m eval.eval_ppl_hf \
#                 --hf_path $HF/$SAVE_NAME \
#                 --seqlen 2048 \
#                 --output_path $RES/$SAVE_NAME \
#                 --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log

#             echo "################## Running benchmark evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
#             python -m eval.eval_zeroshot_hf \
#                 --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#                 --batch_size 4  \
#                 --hf_path $HF/$SAVE_NAME \
#                 --output_path $RES/$SAVE_NAME

#                 # --tasks arc_challenge,arc_easy,piqa,winogrande,hellaswag \

#             if [ "$HF/$SAVE_NAME" != "$HF" ]; then
#                 echo "Cleaning up temporary files for $SAVE_NAME"
#                 rm -rf "$HF/$SAVE_NAME"
#                 rm -rf "$CKPT/$SAVE_NAME"
#             fi
#         done
#     done
# done