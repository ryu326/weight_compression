
pretrain_paths=(
    "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf"
    "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf"
    "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-8x8-hf"
    "ISTA-DASLab/Llama-2-13b-AQLM-2Bit-1x16-hf"
    "ISTA-DASLab/Llama-2-13b-AQLM-2Bit-2x8-hf"
    "ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16"
)

for pretrain_path in "${pretrain_paths[@]}"
do
    log_path=$(echo "./$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt
    log_dir=$(dirname "$log_path")
    mkdir -p "$log_dir"
    echo "Running evaluation for directory: $pretrain_path"
    CUDA_VISIBLE_DEVICES=2 python eval_ppl_aqlm.py \
        --hf_path $pretrain_path \
        --seqlen 2048 \
        --no_use_cuda_graph | tee -a "./$log_path"
done