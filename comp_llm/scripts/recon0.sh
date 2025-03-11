export CUDA_VISIBLE_DEVICES=7

model_names=(
    # "meta-llama--Meta-Llama-3-8B"
    "meta-llama--Llama-2-7b-hf"
    # "meta-llama--Llama-2-13b-hf"
)

save_path="./model_lm_reconstructed/diag_scale"
comp_model_base="../VQVAE_v2/checkpoint/nwc/block_seq_scalar_mean_meta-llama--Llama-2-7b-hf__scaled_sig0.001_row_4096.pt"
# comp_model_base="../VQVAE_v2/checkpoint/nwc_ql/block_seq_ql_random__llama-3-8b-hf/block_seq_ql_random_col_16"
batch_size=1024

for model_name in "${model_names[@]}"; do
    lm_model_path="/home/jgryu/Weight_compression/Wparam_dataset/hf_model/$model_name"
    ql="/home/jgryu/Weight_compression/Wparam_dataset/hessian/$model_name/quip_hess_n6144_top3_qlevel3.pt"
    # ql="/home/jgryu/Weight_compression/Wparam_dataset/hessian/$model_name/pileval_n_samples128_seqlen512_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt"
    quip_hess="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"


    lmbda_values=(50 100 200 1000)
    for lmbda in "${lmbda_values[@]}"; do
        echo "Running with lmbda=${lmbda}"
        # comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
        # taskset -c 0-31 python compress_lm_nwc.py \
        #     --lm_model_path "$lm_model_path" \
        #     --comp_model_path $comp_model \
        #     --direction row \
        #     --save_path "$save_path" \
        #     --batch_size "$batch_size" \
        #     --diag_scale \
        #     --quip_hess "$quip_hess"
        #     # --ql "$ql"
        #     # --diag_scale \
        #     # > ../logs/recon${lmbda}.log 2>&1 &


        base_dir=$save_path/$model_name/block_seq_scalar_mean_meta-llama--Llama-2-7b-hf__scaled_sig0.001_row_4096.pt/lmbda${lmbda}_*/best_loss*.pth.tar
        pretrain_paths=($(find $base_dir -type f -name "tokenizer.json" -exec dirname {} \; | sort -u))
        for pretrain_path in "${pretrain_paths[@]}"
        do
            log_path=$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt
            log_dir=$(dirname "$log_path")
            mkdir -p "$log_dir"
            echo "Running evaluation for directory: $pretrain_path"
            python eval_ppl.py \
                --hf_path $pretrain_path \
                --seqlen 2048 \
                --no_use_cuda_graph | tee -a "$log_path"
        done
    done
done
