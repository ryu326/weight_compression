# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_layerwise_normed.pt"
# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_rnormed_row_1024.pt/rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
# comp_model_base="../NWC/checkpoint/nwc_ql_pe/block_seq_ql_random_pos_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_rnormed_lidx_row_1024.pt/rdloss_ql_size128_encdim1024_M256_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
comp_model_base="../NWC/checkpoint/nwc_qmap2/block_seq_qmap_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_qmap2_size16_encdim512_M17_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
comp_model_base="/workspace/Weight_compression/NWC/checkpoint/nwc_qmap2/block_seq_qmap_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_qmap2_size16_encdim512_M17_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100"
model_name="meta-llama--Llama-2-7b-hf"
HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"

# model_name="meta-llama--Meta-Llama-3-8B"
# HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

# model_name="meta-llama--Llama-2-13b-hf"
# HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"

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
RES="../hf_model_comp_results"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG
mkdir -p $RES
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_SILENT=true

lmbda_values=(300)
for lmbda in "${lmbda_values[@]}"; do
    echo "################## Running compression lmbda=${lmbda} ##################"
    ## ========= Change this =========
    # SAVE_NAME=${model_name}/scaleH_pe/size128_encdim1024_M256_ql1/lmbda${lmbda}
    SAVE_NAME=${model_name}/ql_qmap22/hessian_ql/lmbda${lmbda}
    # ## ========= Change this =========

    comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
    # comp_model=$(ls -t $comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar | head -n 1)
    mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
    
    taskset -c 0-31 \
    python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
        --base_model $lm_model_path \
        --comp_model_path $comp_model \
        --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 --batch_size 8 \
        --ft_epochs 0 \
        --qmap_hessian_ql \
        2>&1 | tee $LOG/$SAVE_NAME.log

        # --direction row --scaleH \


    echo "################## Running hfize lmbda=${lmbda} ##################"
    python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
            --hf_output_path $HF/$SAVE_NAME 2>&1 | tee -a $LOG/$SAVE_NAME.log

    echo "################## Running PPL evaluation lmbda=${lmbda} ##################"
    pretrain_path=$HF/$SAVE_NAME
    mkdir -p "$log_dir"
    echo "Running evaluation for directory: $pretrain_path"
    python -m eval.eval_ppl_hf \
        --hf_path $pretrain_path \
        --seqlen 2048 \
        --output_path $RES/$SAVE_NAME \
        --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log

    if [ "$pretrain_path" != "$HF" ]; then
        rm -r "$pretrain_path"
        rm -r "$CKPT/$SAVE_NAME"
    fi

done

