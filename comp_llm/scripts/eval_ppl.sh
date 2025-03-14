export CUDA_VISIBLE_DEVICES=1

base_dir="/workspace/Weight_compression/comp_lm_quip-sharp/hf/meta-llama--Llama-2-7b-hf"
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
