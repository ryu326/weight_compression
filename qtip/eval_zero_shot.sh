CUDA_VISIBLE_DEVICES=2 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,piqa,winogrande \
    --batch_size 1  --hf_path ../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf \
    --output_path ./eval_result/meta-llama--Llama-2-7b-hf_ft_result >> ./log/meta-llama--Llama-2-7b-hf_ft_result.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,piqa,winogrande \
    --batch_size 1  --hf_path relaxml/Llama-2-7b-QTIP-3Bit \
    --output_path ./eval_result/Llama-2-7b-QTIP-3Bit_ft_result >> ./log/Llama-2-7b-QTIP-3Bit_ft_result.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,piqa,winogrande \
    --batch_size 1  --hf_path relaxml/Llama-2-7b-QTIP-2Bit \
    --output_path ./eval_result/Llama-2-7b-QTIP-2Bit_ft_result.pt >> ./log/Llama-2-7b-QTIP-2Bit_ft_result.log 2>&1 &