#!/bin/bash

base_dirs=(
    # '/home/jgryu/Weight_compression/model_lm_reconstructed/awq/meta-llama--Llama-2-7b-hf'
    # '/workspace/Weight_compression/Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5'
    # '/workspace/Weight_compression/Wparam_dataset/hf_model'
    # '/workspace/Weight_compression/hf_model_comp/awq/lmsys--vicuna-7b-v1.5'
    '/workspace/Weight_compression/hf_model_comp/gptq/meta-llama--Meta-Llama-3-8B'
    '/workspace/Weight_compression/hf_model_comp/gptq/meta-llama--Llama-2-7b-hf'
    '/workspace/Weight_compression/hf_model_comp/gptq/meta-llama--Llama-2-13b-hf'
)

# 사용할 GPU 목록을 배열로 정의
gpus=(0 1 2 3)
num_gpus=${#gpus[@]}
job_count=0

# base_dir 배열을 반복
for base_dir in "${base_dirs[@]}"
do
    # pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u))
    pretrain_paths=($(find "$base_dir" -type f -name "config.json" -exec dirname {} \; | sort -u))
    # pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u | grep 'relaxml'))
    # pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u | grep 'vicuna-7b-v1.5'))
    
    for pretrain_path in "${pretrain_paths[@]}"
    do
        # 현재 작업에 할당할 GPU 결정
        gpu_id=${gpus[$((job_count % num_gpus))]}

        output_path=${pretrain_path/hf_model_comp/hf_model_comp_result}
        
        #
        # 백그라운드에서 평가 스크립트 실행
        #
        (
            echo "################## [GPU ${gpu_id}] Starting PPL evaluation for ${pretrain_path} ##################"
            
            # 해당 프로세스에 특정 GPU를 할당하여 실행
            CUDA_VISIBLE_DEVICES=$gpu_id python -m eval.eval_ppl_hf \
                --hf_path "$pretrain_path" \
                --seqlen 2048 \
                --output_path "$output_path" \
                --no_use_cuda_graph
            
            echo "################## [GPU ${gpu_id}] Finished PPL evaluation for ${pretrain_path} ##################"
        ) & # '&'를 붙여 백그라운드로 실행

        job_count=$((job_count + 1))
        # GPU 수만큼 작업이 실행되었다면 모든 작업이 끝날 때까지 대기
        if [[ $((job_count % num_gpus)) -eq 0 ]]; then
            echo "--- All GPUs are busy. Waiting for the current batch of jobs to finish... ---"
            wait
            echo "--- Batch finished. Proceeding with the next batch. ---"
        fi
    done
done

# 루프가 끝난 후 남아있는 모든 백그라운드 작업이 완료될 때까지 대기
echo "--- Waiting for the last remaining jobs to finish... ---"
wait

echo "--- All evaluations are complete. ---"

# base_dirs=(
#     # '/home/jgryu/Weight_compression/model_lm_reconstructed/awq/meta-llama--Llama-2-7b-hf'
#     # '/workspace/Weight_compression/Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5'
#     # '/workspace/Weight_compression/Wparam_dataset/hf_model'
#     # '/workspace/Weight_compression/hf_model_comp/awq/lmsys--vicuna-7b-v1.5',
#     '/workspace/Weight_compression/hf_model_comp/gptq/meta-llama--Meta-Llama-3-8B',
#     '/workspace/Weight_compression/hf_model_comp/gptq/meta-llama--Llama-2-7b-hf0',
#     '/workspace/Weight_compression/hf_model_comp/gptq/meta-llama--Llama-2-13b-hf',

#   )

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# # base_dir 배열을 반복
# for base_dir in "${base_dirs[@]}"
# do
#     pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u))
#     # pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u | grep 'relaxml'))
#     # pretrain_paths=($(find "$base_dir" -type f -name "tokenizer.json" -exec dirname {} \; | sort -u | grep 'vicuna-7b-v1.5'))
#     for pretrain_path in "${pretrain_paths[@]}"
#     do
#         output_path=${pretrain_path/hf_model_comp/hf_model_comp_result}

#         echo "################## Running PPL evaluation ${pretrain_path} ##################"
#         echo "Running evaluation for directory: $pretrain_path"
#         python -m eval.eval_ppl_hf \
#             --hf_path $pretrain_path \
#             --seqlen 2048 \
#             --output_path $output_path \
#             --no_use_cuda_graph
#             # --dataset ptb \

#         # echo "################## Running benchmark evaluation ${pretrain_path} ##################"
#         # python -m eval.eval_zeroshot_hf \
#         #     --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#         #     --batch_size 8  \
#         #     --hf_path $pretrain_path \
#         #     --output_path $output_path
#     done
# done