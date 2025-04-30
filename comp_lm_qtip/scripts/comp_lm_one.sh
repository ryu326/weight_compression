model_name="meta-llama--Meta-Llama-3-8B"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

# model_name="meta-llama--Llama-3.2-3B"
# HESS="../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256"

# ql="../Wparam_dataset/hessian/$model_name/quip_hess_n6144_top3_qlevel3.pt"
# ql="../Wparam_dataset/hessian/$model_name/pileval_n_samples128_seqlen512_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt"
# ql='../Wparam_dataset/hessian/meta-llama--Llama-2-7b-hf/quip_hess_n6144_all_layers_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt'
############################################

lm_model_path="../Wparam_dataset/hf_model/$model_name"

CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG

export CUDA_VISIBLE_DEVICES=1
export WANDB_SILENT=true

lmbda_values=(1000)
for lmbda in "${lmbda_values[@]}"; do
    echo "################## Running compression lmbda=${lmbda} ##################"
    
    ## ========= Change this =========
    # SAVE_NAME=noft_ql/${model_name}_channelwise_scale_ql_tuned/lmbda${lmbda}
    SAVE_NAME=${model_name}/ql_M32_lnormed_lnorm_trained/lmbda${lmbda}
    # SAVE_NAME=${model_name}/ql/lmbda${lmbda}
    ## ========= Change this =========

    # comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
    # comp_model=../NWC/checkpoint/nwc_ql_ste/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/lmbda${lmbda}_*/best_loss*.pth.tar
    # comp_model=../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M32/lmbda${lmbda}_*/best_loss*.pth.tar
    # comp_model=../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_layerwise_normed.pt/lmbda${lmbda}_*/best_loss*.pth.tar
    # comp_model=../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_colwise_normed.pt/lmbda${lmbda}_*/best_loss*.pth.tar
    mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
    
    # taskset -c 0-31 \
    # python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
    #     --base_model $lm_model_path \
    #     --comp_model_path $comp_model \
    #     --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 --batch_size 8 \
    #     --channelwise_scale \
    #     --ql \
    #     --ft_epochs 0 \
    #     2>&1 | tee $LOG/$SAVE_NAME.log

        # --incoh_mode had  --rescale_WH_2  --sigma_reg 1e-4 --use_train_scale \
        # --ldlq --comp_batch_size 1 \
        # --ft_comp_model2 --ft_comp_lmbda $lmbda --ft_comp_steps 400 --direction row \
        # --layerwise_scale \
        # --ql_tuned \
        # --ql \

    echo "################## Running hfize lmbda=${lmbda} ##################"
    python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
            --hf_output_path $HF/$SAVE_NAME 2>&1 | tee -a $LOG/$SAVE_NAME.log

    # echo "################## Running PPL evaluation lmbda=${lmbda} ##################"
    # pretrain_path=$HF/$SAVE_NAME
    # log_path=$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt
    # log_dir=$(dirname "$log_path")
    # mkdir -p "$log_dir"
    # echo "Running evaluation for directory: $pretrain_path"
    # python -m eval.eval_ppl_hf \
    #     --hf_path $pretrain_path \
    #     --seqlen 2048 \
    #     --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log

    # echo "################## Running benchmark evaluation lmbda=${lmbda} ##################"
    # pretrain_path=$HF/$SAVE_NAME
    # output_path=$(echo "$pretrain_path" | sed 's|model_reconstructed|model_eval|')_harness_results
    # lm_eval --model hf \
    #     --model_args "pretrained=$pretrain_path,parallelize=True" \
    #     --tasks arc_easy,arc_challenge,winogrande,piqa,boolq \
    #     --batch_size 1 \
    #     --output_path $output_path \
    #     --trust_remote_code \
    #     2>&1 | tee -a $LOG/$SAVE_NAME.log

    rm -r $pretrain_path

done