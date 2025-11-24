path="/workspace/Weight_compression/Wparam_dataset/hf_model/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"

export CUDA_VISIBLE_DEVICES=4,5,6,7
# python -m eval.eval_ppl \
#     --hf_path "${path}" \
#     --output_path /workspace/Weight_compression/hf_model_comp_results/qwen30a3b_base \
#     --max_mem_ratio 0.9 \
#     --seqlen 2048

python -m eval.eval_zeroshot \
    --tasks arc_challenge,arc_easy,boolq,piqa,winogrande,hellaswag,mmlu \
    --batch_size 1 \
    --hf_path "${path}" \
    --max_mem_ratio 0.7 \
    --output_path /workspace/Weight_compression/hf_model_comp_results/qwen30a3b_base_common_mmlu
