export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/jgryu/.cache/huggingface


# lm_model_path=mistralai/Mixtral-8x7B-v0.1
# output_path=/home/jgryu/workspace/weight_compression/hf_model_comp_results/mistralai/base_model

# lm_model_path=openai/gpt-oss-20b
# output_path=/home/jgryu/workspace/weight_compression/hf_model_comp_results_v2/gpt-oss-20b/base_model

lm_model_path=/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf
output_path=/home/jgryu/workspace/weight_compression/hf_model_comp_results_v2/meta-llama--Llama-2-7b-hf/base_model_seqlen4096

python -m eval.eval_ppl_hf \
    --hf_path $lm_model_path \
    --seqlen 4096 \
    --output_path $output_path \
    --datasets wikitext2,c4 \
    --no_use_cuda_graph

    # --gptoss_replace_version v1

# python -m eval.eval_zeroshot_hf \
#     --tasks arc_challenge,arc_easy,boolq,piqa,winogrande,hellaswag,mmlu \
#     --batch_size 1 \
#     --hf_path $lm_model_path \
#     --output_path $output_path \
#     --gptoss_replace_version v1

    # --output_path /home/jgryu/workspace/weight_compression/hf_model_comp_results/meta-llama--Meta-Llama-3-8B/base_model_svd_auto 


# python -m eval.eval_zeroshot_hf \
#     --tasks arc_challenge,piqa,winogrande,hellaswag \
#     --batch_size 4 \
#     --hf_path /home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf \
#     --output_path /home/jgryu/workspace/weight_compression/hf_model_comp_results/meta-llama--Llama-2-7b-hf/base_model_qat_bf16