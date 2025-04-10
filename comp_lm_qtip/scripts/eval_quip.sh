base_dirs=(
    # '/home/jgryu/Weight_compression/model_lm_reconstructed/awq/meta-llama--Llama-2-7b-hf'
    '/home/jgryu/Weight_compression/Wparam_dataset/hf_model'
  )

export CUDA_VISIBLE_DEVICES=3

# base_dir 배열을 반복
for base_dir in "${base_dirs[@]}"
do
    # pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u))
    pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u | grep 'relaxml'))
    for pretrain_path in "${pretrain_paths[@]}"
    do
        log_path=$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt

        log_dir=$(dirname "$log_path")
        mkdir -p "$log_dir"

        echo "Running evaluation for directory: $pretrain_path"

        python -m eval.eval_ppl \
            --hf_path $pretrain_path \
            --seqlen 2048 2>&1 | tee -a "$log_path"
    done
done