CKPT="./ckpt/ft"
HF="./hf/ft"
LOG="./log/ft"

mkdir $CKPT
mkdir $HF
mkdir $LOG

# comp_model_base="../NWC/checkpoint/nwc/block_seq_row_16"
comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random__llama-3-8b-hf/block_seq_ql_random_col_16"
# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_4096_RHT.pt"
# comp_model_base="../NWC/checkpoint/nwc/block_seq_scaler_meta-llama--Meta-Llama-3-8B__col_4096_RHT.pt"

# model_name="meta-llama--Meta-Llama-3-8B"
model_name="meta-llama--Llama-2-7b-hf"
# ql="../Wparam_dataset/hessian/$model_name/quip_hess_n6144_top3_qlevel3.pt"
# ql="../Wparam_dataset/hessian/$model_name/pileval_n_samples128_seqlen512_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt"
HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
# HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"
lm_model_path="../Wparam_dataset/hf_model/$model_name"

lmbda_values=(50 100 200 300)
for lmbda in "${lmbda_values[@]}"; do
    echo "Running with lmbda=${lmbda}"
    comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
    CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-31 \
    python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$model_name/lmbda${lmbda}_ql_had2 \
        --base_model $lm_model_path \
        --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 \
        --batch_size 8 \
        --ft_epochs 0 \
        --incoh_mode had \
        --ql \
        --comp_model_path $comp_model 2>&1 | tee $LOG/${model_name}_lmbda${lmbda}_ql_had2.log
    
    python -m quantize_llama.hfize_llama --quantized_path $CKPT/$model_name/lmbda${lmbda}_ql_had2 \
            --hf_output_path $HF/$model_name/lmbda${lmbda}_ql_had2 2>&1 | tee $LOG/${model_name}_lmbda${lmbda}_ql_had2_hfize.log 

    pretrain_path=$HF/$model_name/lmbda${lmbda}_ql_had2
    log_path=$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt
    log_dir=$(dirname "$log_path")
    mkdir -p "$log_dir"
    echo "Running evaluation for directory: $pretrain_path"
    export CUDA_VISIBLE_DEVICES=3 
    python eval_ppl.py \
        --hf_path $pretrain_path \
        --seqlen 2048 \
        --no_use_cuda_graph 2>&1 | tee -a "$log_path"
done