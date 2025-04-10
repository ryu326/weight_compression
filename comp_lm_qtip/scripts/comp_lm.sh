# comp_model_base="../NWC/checkpoint/nwc/block_seq_row_16"
# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random__llama-3-8b-hf/block_seq_ql_random_col_16"
# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_4096_RHT.pt"
# comp_model_base="../NWC/checkpoint/nwc_tr_with_hyp/block_seq_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"
# comp_model_base="../NWC/checkpoint/nwc_tr/block_seq_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"
# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"
# comp_model_base="../NWC/checkpoint/nwc/block_seq_scaler_meta-llama--Meta-Llama-3-8B__scaled3_RHT_sig1e-06_col_1024.pt"
comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"

model_name="meta-llama--Llama-2-7b-hf"
HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"

# model_name="meta-llama--Meta-Llama-3-8B"
# HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

lm_model_path="../Wparam_dataset/hf_model/$model_name"
# ql="../Wparam_dataset/hessian/$model_name/quip_hess_n6144_top3_qlevel3.pt"
# ql="../Wparam_dataset/hessian/$model_name/pileval_n_samples128_seqlen512_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt"
# ql='../Wparam_dataset/hessian/meta-llama--Llama-2-7b-hf/quip_hess_n6144_all_layers_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt'

CKPT="./ckpt/ft_ql_ldlq"
HF="./hf/ft_ql_ldlq"
LOG="./log/ft_ql_ldlq"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG

lmbda_values=(30 10)
for lmbda in "${lmbda_values[@]}"; do
    echo "Running with lmbda=${lmbda}"

    comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
    SAVE_NAME=${model_name}/lmbda${lmbda}
    mkdir -p $LOG/${model_name}

    CUDA_VISIBLE_DEVICES=0,1,2 taskset -c 0-31 \
    python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
        --base_model $lm_model_path \
        --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 \
        --batch_size 8 \
        --ft_epochs 5 \
        --ldlq \
        --ql \
        --comp_batch_size 1 \
        --comp_model_path $comp_model 2>&1 | tee $LOG/$SAVE_NAME.log
    
        # --incoh_mode had  --rescale_WH_2  --sigma_reg 1e-4 --use_train_scale \

    python -m quantize_llama.hfize_llama --quantized_path $CKPT/$SAVE_NAME \
            --hf_output_path $HF/$SAVE_NAME 2>&1 | tee -a $LOG/$SAVE_NAME.log 

    pretrain_path=$HF/$SAVE_NAME
    log_path=$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt
    log_dir=$(dirname "$log_path")
    mkdir -p "$log_dir"
    echo "Running evaluation for directory: $pretrain_path"
    export CUDA_VISIBLE_DEVICES=3
    python eval_ppl.py \
        --hf_path $pretrain_path \
        --seqlen 2048 \
        --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log &

    rm -r $pretrain_path

done