## generate hessian for llama3_8B model
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 -m quantize_llama.input_hessian_llama \
    --batch_size 8 --devset_size 6144 \
    --base_model ../dataset/hf_model/meta-llama--Meta-Llama-3-8B \
    --save_path "./hess/llama3_8b_6144"