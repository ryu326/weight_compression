export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/hf_cache/huggingface_nwc
LAMBDAS=(10000 100000)
for lmbda in "${LAMBDAS[@]}"
do
    uv run python scripts/eval_compnet_checkpoint.py \
        --checkpoint_path /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld${lmbda}_*/comp_model.pt \
        --full_eval > ./logs/${lmbda}.log
done


export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/workspace/hf_cache/huggingface_nwc
LAMBDAS=(50 100 1000)
for lmbda in "${LAMBDAS[@]}"
do
    uv run python scripts/eval_compnet_checkpoint.py \
        --checkpoint_path /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v4_ld${lmbda}_*/comp_model.pt \
        --full_eval > ./logs/${lmbda}.log
done


export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/workspace/hf_cache/huggingface_nwc
paths=(
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v5_ld30.0_20250911-174318_SgEOjip7/comp_model.pt
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v5_ld100.0_20250910-093246_Xp3EQZJ3/comp_model.pt
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v5_ld100.0_20250910-093437_r0vC6NtR/comp_model.pt
    /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/compnet/v5_ld1000.0_20250910-093437_CKnpK2as/comp_model.pt
)
i=1
for path in "${paths[@]}"; do
    log_name="./logs/eval_${i}.log"
    echo "Running $path -> $log_name"
    uv run python scripts/eval_compnet_checkpoint.py \
        --checkpoint_path "$path" \
        --full_eval > "$log_name"
    ((i++))
done