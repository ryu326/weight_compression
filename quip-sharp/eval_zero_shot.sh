CUDA_VISIBLE_DEVICES=3 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,piqa,winogrande \
    --batch_size 1  --hf_path relaxml/Llama-2-7b-E8P-2Bit \
    --output_path ./eval_result/Llama-2-7b-E8P-2Bit > ./log/Llama-2-7b-E8P-2Bit.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,piqa,winogrande \
    --batch_size 1  --hf_path relaxml/Llama-2-7b-E8PRVQ-3Bit \
    --output_path ./eval_result/Llama-2-7b-E8PRVQ-3Bit > ./log/Llama-2-7b-E8PRVQ-3Bit.log 2>&1 &