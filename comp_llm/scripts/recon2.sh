export CUDA_VISIBLE_DEVICES=6

model_names=(
    # "meta-llama--Meta-Llama-3-8B"
    "meta-llama--Llama-2-7b-hf"
    # "meta-llama--Llama-2-13b-hf"
)

save_path="/home/jgryu/Weight_compression/comp_llm/model_lm_reconstructed/ldlq"

for model_name in "${model_names[@]}"; do
    lm_model_path="/home/jgryu/Weight_compression/Wparam_dataset/hf_model/$model_name"
    ql="/home/jgryu/Weight_compression/Wparam_dataset/hessian/$model_name/pileval_n_samples128_seqlen512_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt"
    quip_hess="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
    batch_size=1

    lmbda_values=(200 300 1000 10000 100000)
    # lmbda_values=(100000)
    for lmbda in "${lmbda_values[@]}"; do
        echo "Running with lmbda=${lmbda}"
        comp_model=../VQVAE_v2/checkpoint/nwc_ql/block_seq_ql_random__llama-3-8b-hf/block_seq_ql_random_col_16/lmbda${lmbda}_*/best_loss*.pth.tar
        taskset -c 0-31 python compress_lm_nwc.py \
            --lm_model_path "$lm_model_path" \
            --comp_model_path $comp_model \
            --direction col \
            --save_path "$save_path" \
            --batch_size "$batch_size" \
            --ldlq \
            --quip_hess "$quip_hess" \
            --ql "$ql"

            # > ../logs/recon${lmbda}.log 2>&1 &  # 로그 저장을 원하면 주석 해제
        base_dir=$save_path/$model_name/block_seq_ql_random_col_16/lmbda${lmbda}_*/best_loss*.pth.tar
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


export CUDA_VISIBLE_DEVICES=6

model_names=(
    # "meta-llama--Meta-Llama-3-8B"
    # "meta-llama--Llama-2-7b-hf"
    "meta-llama--Llama-2-13b-hf"
)

save_path="/home/jgryu/Weight_compression/comp_llm/model_lm_reconstructed/ldlq"

for model_name in "${model_names[@]}"; do
    lm_model_path="/home/jgryu/Weight_compression/Wparam_dataset/hf_model/$model_name"
    ql="/home/jgryu/Weight_compression/Wparam_dataset/hessian/$model_name/pileval_n_samples128_seqlen512_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt"
    quip_hess="../Wparam_dataset/quip_hess/Hessians-Llama-2-13b-6144"
    batch_size=1

    lmbda_values=(50 100 200 300 1000 10000 100000)
    # lmbda_values=(100000)
    for lmbda in "${lmbda_values[@]}"; do
        echo "Running with lmbda=${lmbda}"
        comp_model=../VQVAE_v2/checkpoint/nwc_ql/block_seq_ql_random__llama-3-8b-hf/block_seq_ql_random_col_16/lmbda${lmbda}_*/best_loss*.pth.tar
        taskset -c 0-31 python compress_lm_nwc.py \
            --lm_model_path "$lm_model_path" \
            --comp_model_path $comp_model \
            --direction col \
            --save_path "$save_path" \
            --batch_size "$batch_size" \
            --ldlq \
            --quip_hess "$quip_hess" \
            --ql "$ql"

            # > ../logs/recon${lmbda}.log 2>&1 &  # 로그 저장을 원하면 주석 해제
        base_dir=$save_path/$model_name/block_seq_ql_random_col_16/lmbda${lmbda}_*/best_loss*.pth.tar
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

# base_dirs=(
#     '/home/jgryu/Weight_compression/model_lm_reconstructed/test'
# )
# for base_dir in "${base_dirs[@]}"
# do
#     pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u))
#     # pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u | grep 'ROWCOL'))
#     for pretrain_path in "${pretrain_paths[@]}"
#     do
#         log_path=$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt

#         log_dir=$(dirname "$log_path")
#         mkdir -p "$log_dir"

#         echo "Running evaluation for directory: $pretrain_path"

#         python eval_ppl.py \
#             --hf_path $pretrain_path \
#             --seqlen 2048 \
#             --no_use_cuda_graph | tee -a "$log_path"
#     done
# done

# --comp_model_path ../checkpoint/nwc_ql/block_seq_ql_random_meta-llama--Me*/lmbda${lmbda}_rdloss_ql_size16_encdim512_M16_Q8*/best_loss*.pth.tar \
# --comp_model_path ../../VQVAE/checkpoint/nwc_ql/block_seq_ql_random_col_8/lmbda${lmbda}_*/best_loss*.pth.tar \