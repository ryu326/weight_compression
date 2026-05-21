# export HF_HOME=/home/jgryu/.cache/huggingface
# export HF_HOME=/workspace/hf_cache/huggingface_nwc
export HF_HOME=/workspace/Weight_compression/hf_cache/
# export HF_HOME=/root/.cache/huggingface
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_qwenmoe \
    --batch_size 8 --devset_size 256 \
    --base_model /workspace/Weight_compression/Wparam_dataset/hf_model/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
    --save_path ../Wparam_dataset/quip_hess/mistralai/Qwen3-30B-A3B
