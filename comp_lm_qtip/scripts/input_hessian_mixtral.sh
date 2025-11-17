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
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_mixtral \
    --batch_size 8 --devset_size 256 \
    --base_model mistralai/Mixtral-8x7B-v0.1 \
    --save_path ../Wparam_dataset/quip_hess/mistralai/Mixtral-8x7B-v0.1_256
