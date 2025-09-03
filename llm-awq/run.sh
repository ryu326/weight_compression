
# MODEL_0="meta-llama--Llama-2-13b-hf"  # GPU 0에서 실행
# MODEL_1="meta-llama--Meta-Llama-3-8B"  # GPU 1에서 실행
# MODEL_0="lmsys--vicuna-7b-v1.5"  # GPU 1에서 실행
# MODEL_0="Qwen--Qwen2.5-7B"  # GPU 1에서 실행

#!/bin/bash

MODEL_0="meta-llama--Llama-3.2-1B-Instruct"
MODEL_1="meta-llama--Llama-3.2-3B-Instruct"
BITS=(3 4 5 6 7 8)

LOG_DIR="./logs"
mkdir -p $LOG_DIR/$MODEL_0
mkdir -p $LOG_DIR/$MODEL_1

# GPU 6에서 MODEL_0 처리
for bit in "${BITS[@]}"
do
    (
        CUDA_VISIBLE_DEVICES=6 python -m awq.entry \
            --model_path ../Wparam_dataset/hf_model/$MODEL_0 \
            --w_bit $bit --q_group_size 128 \
            --run_awq --dump_awq ../hf_model_comp/awq_cache/$MODEL_0/w${bit}-g128.pt \
            > $LOG_DIR/$MODEL_0/awq_w${bit}.log 2>&1

        CUDA_VISIBLE_DEVICES=6 python -m awq.entry \
            --model_path ../Wparam_dataset/hf_model/$MODEL_0 \
            --w_bit $bit --q_group_size 128 \
            --load_awq ../hf_model_comp/awq_cache/$MODEL_0/w${bit}-g128.pt \
            --q_backend fake \
            --dump_fake ../hf_model_comp/awq/$MODEL_0/w${bit}bit-g128-fake-quantized \
            >> $LOG_DIR/$MODEL_0/awq_w${bit}.log 2>&1
    ) &
done

# GPU 7에서 MODEL_1 처리
for bit in "${BITS[@]}"
do
    (
        CUDA_VISIBLE_DEVICES=7 python -m awq.entry \
            --model_path ../Wparam_dataset/hf_model/$MODEL_1 \
            --w_bit $bit --q_group_size 128 \
            --run_awq --dump_awq ../hf_model_comp/awq_cache/$MODEL_1/w${bit}-g128.pt \
            > $LOG_DIR/$MODEL_1/awq_w${bit}.log 2>&1

        CUDA_VISIBLE_DEVICES=7 python -m awq.entry \
            --model_path ../Wparam_dataset/hf_model/$MODEL_1 \
            --tasks wikitext \
            --w_bit $bit --q_group_size 128 \
            --load_awq ../hf_model_comp/awq_cache/$MODEL_1/w${bit}-g128.pt \
            --q_backend fake \
            --dump_fake ../model_lm_reconstructed/awq/$MODEL_1/w${bit}-g128-fake-quantized \
            >> $LOG_DIR/$MODEL_1/awq_w${bit}.log 2>&1
    ) &
done

# 모든 백그라운드 작업 대기
wait
echo "✅ 모든 작업 완료!"
