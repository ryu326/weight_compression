export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/jgryu/.cache/huggingface
# python -m eval.eval_zeroshot_hf \
#     --tasks arc_challenge,piqa,winogrande,hellaswag \
#     --batch_size 4 \
#     --hf_path /home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf \
#     --output_path /home/jgryu/workspace/weight_compression/hf_model_comp_results/meta-llama--Llama-2-7b-hf/base_model_qat_bf16

lm_model_path=mistralai/Mixtral-8x7B-v0.1
output_path=/home/jgryu/workspace/weight_compression/hf_model_comp_results/mistralai/base_model
python -m eval.eval_ppl_hf \
    --hf_path $lm_model_path \
    --seqlen 2048 \
    --output_path $output_path \
    --datasets wikitext2,c4 \
    --no_use_cuda_graph

# python -m eval.eval_zeroshot_hf \
#     --tasks openbookqa,mathqa \
#     --batch_size 4 \
#     --hf_path $lm_model_path \
#     --output_path $output_path

    # --output_path /home/jgryu/workspace/weight_compression/hf_model_comp_results/meta-llama--Meta-Llama-3-8B/base_model_svd_auto 