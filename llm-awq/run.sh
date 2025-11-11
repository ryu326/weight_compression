export HF_HOME=/home/jgryu/.cache/huggingface

# MODEL_0="meta-llama--Llama-2-13b-hf"  # GPU 0에서 실행
MODEL_0="meta-llama--Meta-Llama-3-8B"  # GPU 1에서 실행
# MODEL_0="lmsys--vicuna-7b-v1.5"  # GPU 1에서 실행
# MODEL_0="Qwen--Qwen2.5-7B"  # GPU 1에서 실행
# # 사용할 비트 수 설정
# BITS=(2 3)

# #!/bin/bash

# MODEL_0="meta-llama--Llama-3.2-1B-Instruct"
# MODEL_1="meta-llama--Llama-3.2-3B-Instruct"
# BITS=(3 4 5 6 7 8)

# LOG_DIR="./logs"
# mkdir -p $LOG_DIR/$MODEL_0
# mkdir -p $LOG_DIR/$MODEL_1

# # GPU 6에서 MODEL_0 처리
# for bit in "${BITS[@]}"
# do
#     CUDA_VISIBLE_DEVICES=0 python -m awq.entry \
#         --model_path ../Wparam_dataset/hf_model/$MODEL_0 \
#         --w_bit $bit --q_group_size 128 \
#         --run_awq --dump_awq ../hf_model_comp/awq_cache/$MODEL_0/w${bit}-g128.pt

#     CUDA_VISIBLE_DEVICES=0 python -m awq.entry \
#         --model_path ../Wparam_dataset/hf_model/${MODEL_0} \
#         --w_bit $bit --q_group_size 128 \
#         --load_awq ../hf_model_comp/awq_cache/${MODEL_0}/w${bit}-g128.pt \
#         --q_backend fake \
#         --dump_fake ../hf_model_comp/awq/$MODEL_0/w${bit}bit-g128-fake-quantized

# done

# # 모든 백그라운드 작업 대기
# wait
# echo "✅ 모든 작업 완료!"


# generate real quantized weights (w4)
CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path ../Wparam_dataset/hf_model/$MODEL_0 \
    --w_bit 4 --q_group_size 128 \
    --load_awq ../hf_model_comp/awq_cache/$MODEL_0/w4-g128.pt \
    --q_backend real --dump_quant ../complexity_test/8b-w4-g128-awq.pt

