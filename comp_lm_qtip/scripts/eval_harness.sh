#### v2
base_dirs=(
    '../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf'
)
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=2

# base_dir 배열을 반복
for base_dir in "${base_dirs[@]}"
do
    pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u))

    # 경로 배열을 반복
    for pretrain_path in "${pretrain_paths[@]}"
    do
        log_path=$(echo "$pretrain_path" | sed 's|model_reconstructed|model_eval|')_harness_results

        echo "Running evaluation for directory: $pretrain_path"

        # 평가 실행 및 로그 저장
        lm_eval --model hf \
            --model_args pretrained=$pretrain_path,parallelize=True \
            --tasks arc_easy,arc_challenge,winogrande,piqa \
            --batch_size 1 \
            --output_path "$log_path"
    done
done

echo "Evaluation completed for all directories and tasks."
# ,arc_challenge,winogrande,boolq,hellaswag

# CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
#     --model_args pretrained=/home/jgryu/Weight_compression/model_cache/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920,parallelize=True \
#     --tasks arc_easy,arc_challenge,winogrande,boolq,hellaswag \
#     --batch_size 1 \
#     --output_path /home/jgryu/Weight_compression/model_eval/original_harness_result.txt

#### v1
# base_dirs=(
#     '../model_cache/models--meta-llama--Meta-Llama-3-8B/snapshots'
#     '../model_reconstructed_v0/awq'
#     '../model_reconstructed/nwc_ql/block_seq_ql_random_col_16'
#     '../model_reconstructed/nwc/block_seq_row_16'
# )
# tasks=("arc_easy" "arc_challenge" "winogrande" "boolq" "hellaswag")

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# # base_dir 배열을 반복
# for base_dir in "${base_dirs[@]}"
# do
#     pretrain_paths=($(find "$base_dir" -type f -name "config.json" -exec dirname {} \; | sort -u))

#     # 경로 배열을 반복
#     for pretrain_path in "${pretrain_paths[@]}"
#     do
#         log_path=$(echo "$pretrain_path" | sed 's|model_reconstructed|model_eval|')_harness_result.txt

#         log_dir=$(dirname "$log_path")
#         mkdir -p "$log_dir"

#         for task in "${tasks[@]}"
#         do
#             echo "Running evaluation for directory: $pretrain_path on task: $task"

#             # 평가 실행 및 로그 저장
#             lm_eval --model hf \
#                 --model_args pretrained=$pretrain_path,parallelize=True \
#                 --tasks $task \
#                 --batch_size 4 | tee -a "$log_path"
#         done
#     done
# done

# echo "Evaluation completed for all directories and tasks."
