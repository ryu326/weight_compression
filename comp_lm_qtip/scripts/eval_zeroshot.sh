export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/jgryu/.cache/huggingface
# python -m eval.eval_zeroshot_hf \
#     --tasks arc_challenge,piqa,winogrande,hellaswag \
#     --batch_size 4 \
#     --hf_path /home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf \
#     --output_path /home/jgryu/workspace/weight_compression/hf_model_comp_results/meta-llama--Llama-2-7b-hf/base_model_qat_bf16

python -m eval.eval_zeroshot_hf \
    --tasks openbookqa,mathqa \
    --batch_size 4 \
    --hf_path /home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B \
    --output_path /home/jgryu/workspace/weight_compression/hf_model_comp_results/meta-llama--Meta-Llama-3-8B/base_model_svd_auto 