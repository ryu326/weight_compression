MODEL=llama3-8b-my

# evaluate the AWQ quantize model (simulated pseudo quantization)
bits=(2 3 5 6 8 10 12)
for bit in "${bits[@]}"
do
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m awq.entry --model_path ../model_cache/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m awq.entry --model_path ../Wparam_dataset_v0/model_zoo/huggingface/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6 \
    --w_bit $bit --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w${bit}-g128.pt

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m awq.entry --model_path ../Wparam_dataset_v0/model_zoo/huggingface/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6 \
        --tasks wikitext \
        --w_bit $bit --q_group_size 128 \
        --load_awq awq_cache/$MODEL-w${bit}-g128.pt \
        --q_backend fake \
        --dump_fake ../model_cache_reconstructed/awq_/$MODEL-w${bit}-g128-fake-quantized
done



# # run AWQ search (optional; we provided the pre-computed results)
# python -m awq.entry --model_path /dataset/models/llama3/$MODEL \
#     --w_bit 4 --q_group_size 128 \
#     --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
# python -m awq.entry --model_path /dataset/models/llama3/$MODEL \
#     --tasks wikitext \
#     --w_bit 4 --q_group_size 128 \
#     --load_awq awq_cache/$MODEL-w4-g128.pt \
#     --q_backend fake

# # generate real quantized weights (w4)
# python -m awq.entry --model_path /dataset/models/llama3/$MODEL \
#     --w_bit 4 --q_group_size 128 \
#     --load_awq awq_cache/$MODEL-w4-g128.pt \
#     --q_backend real --dump_quant quant_cache/$MODEL-w4-g128-awq.pt

# # load and evaluate the real quantized model (smaller gpu memory usage)
# python -m awq.entry --model_path /dataset/models/llama3/$MODEL \
#     --tasks wikitext \
#     --w_bit 4 --q_group_size 128 \
#     --load_quant quant_cache/$MODEL-w4-g128-awq.pt