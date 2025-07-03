base_dirs=(
    # '/home/jgryu/Weight_compression/model_lm_reconstructed/awq/meta-llama--Llama-2-7b-hf'
    # '/workspace/Weight_compression/Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5'
    # '/workspace/Weight_compression/Wparam_dataset/hf_model'
    '/workspace/Weight_compression/hf_model_comp/awq/lmsys--vicuna-7b-v1.5'
  )

export CUDA_VISIBLE_DEVICES=3

# base_dir 배열을 반복
for base_dir in "${base_dirs[@]}"
do
    pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u))
    # pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u | grep 'relaxml'))
    # pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u | grep 'vicuna-7b-v1.5'))
    for pretrain_path in "${pretrain_paths[@]}"
    do
        echo "################## Running PPL evaluation ${pretrain_path} ##################"
        echo "Running evaluation for directory: $pretrain_path"
        python -m eval.eval_ppl_hf \
            --hf_path $pretrain_path \
            --seqlen 2048 \
            --output_path $pretrain_path \
            --no_use_cuda_graph
            # --dataset ptb \

        echo "################## Running benchmark evaluation ${pretrain_path} ##################"
        python -m eval.eval_zeroshot_hf \
            --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
            --batch_size 8  \
            --hf_path $pretrain_path \
            --output_path $pretrain_path
    done
done