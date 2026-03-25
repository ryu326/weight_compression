# torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_llama \
#     --batch_size 8 --devset_size 256 \
#     --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B \
#     --save_path ../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256

# torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_llama \
#     --batch_size 8 --devset_size 256 \
#     --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5 \
#     --save_path ../Wparam_dataset/quip_hess/lmsys--vicuna-7b-v1.5_256

# python -m quantize_llama.input_hessian_llama \
#     --batch_size 8 --devset_size 256 \
#     --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B \
#     --save_path ../Wparam_dataset/quip_hess/meta-llama--Llama-3.2-3B-256

export HF_HOME=/home/jgryu/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 -m quantize_llama.input_hessian_gptoss \
    --batch_size 2 --devset_size 128 \
    --large_batch_size 128 \
    --base_model openai/gpt-oss-20b \
    --save_path ../Wparam_dataset/quip_hess/gpt-oss-20b_1024 \
    --gptoss_replace_version v1
