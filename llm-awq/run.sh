
# MODEL_0="meta-llama--Llama-2-7b-hf"  # GPU 0에서 실행
MODEL_1="meta-llama--Meta-Llama-3-8B"  # GPU 1에서 실행
# # 사용할 비트 수 설정
BITS=(4 5 6 7 8 9)

# # GPU 0에서 Llama-2-7B 실행
# (
# for bit in "${BITS[@]}"
# do
#     # CUDA_VISIBLE_DEVICES=0 python -m awq.entry \
#     #     --model_path ../Wparam_dataset/hf_model/$MODEL_0 \
#     #     --w_bit $bit --q_group_size 128 \
#     #     --run_awq --dump_awq awq_cache/$MODEL_0/w${bit}-g128.pt

#     CUDA_VISIBLE_DEVICES=2 python -m awq.entry \
#         --model_path ../Wparam_dataset/hf_model/$MODEL_0 \
#         --tasks wikitext \
#         --w_bit $bit --q_group_size 128 \
#         --load_awq awq_cache/$MODEL_0/w${bit}-g128.pt \
#         --q_backend fake \
#         --dump_fake ../reconstructed_hf_model/awq/$MODEL_0/w${bit}-g128-fake-quantized
# done
# )

# GPU 1에서 Meta-Llama-3-8B 실행

for bit in "${BITS[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python -m awq.entry \
        --model_path ../Wparam_dataset/hf_model/$MODEL_1 \
        --w_bit $bit --q_group_size 128 \
        --run_awq --dump_awq awq_cache/$MODEL_1/w${bit}-g128.pt

    CUDA_VISIBLE_DEVICES=1 python -m awq.entry \
        --model_path ../Wparam_dataset/hf_model/$MODEL_1 \
        --tasks wikitext \
        --w_bit $bit --q_group_size 128 \
        --load_awq awq_cache/$MODEL_1/w${bit}-g128.pt \
        --q_backend fake \
        --dump_fake ../model_lm_reconstructed/awq/$MODEL_1/w${bit}-g128-fake-quantized
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait
echo "✅ 모든 작업 완료!"
