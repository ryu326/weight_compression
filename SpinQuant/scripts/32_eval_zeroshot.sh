export CUDA_VISIBLE_DEVICES=0,1,2,3
MODELS_TO_RUN=(
    # "llama3_8b"
    # "llama2_7b"
    # # "llama3.2_3b"
    # "llama2_13b"
    # # "vicuna_7b"
    # "8B"
    # "7B"
    "13B"
)

declare -A MODEL_PATHS
MODEL_PATHS=(
    ["7B"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"
    ["13B"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"
    ["8B"]="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"
    ["vicuna_7b"]="../Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5"
    ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
)

for model_key in "${MODELS_TO_RUN[@]}"; do
    base_model=${MODEL_PATHS[$model_key]}

    for b in 5 6 7 8; do
        torchrun --nnodes=1 --nproc_per_node=4 --master_port=29502 ptq_eval_zeroshot.py \
        --input_model $base_model \
        --do_train False \
        --do_eval True \
        --per_device_eval_batch_size 1 \
        --model_max_length 2048 \
        --fp16 False \
        --bf16 True \
        --save_safetensors False \
        --w_bits $b \
        --a_bits 16 \
        --w_clip \
        --a_asym \
        --rotate \
        --optimized_rotation_path /workspace/Weight_compression/hf_model_comp/spinquant/output_rotation/${model_key}_w${b}a16/R.bin \
        --output_path ../hf_model_comp_results/spinquant/${model_key}_/${model_key}_w${b}a16 \
        2>&1 | tee ./logs/eval_zeroshot_hs_mmlu_${model_key}_w${b}a16.log
    done
done

    # 안 됨
    # --save_qmodel_path ../hf_model_comp/spinquant/output_qmodel/${model_key}_w${b}a16.pth \

    # --optimized_rotation_path /workspace/Weight_compression/hf_model_comp/spinquant/output_rotation/8b_w${b}a16/R.bin \
    # --optimized_rotation_path /workspace/Weight_compression/hf_model_comp/spinquant/output_rotation/${model_key}_W4A16KV16_lr_1.5_seed_0/R.bin \


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# for b in 7 8; do
#     torchrun --nnodes=1 --nproc_per_node=4 --master_port=29502 ptq_eval_ppl_zeroshot.py \
#     --input_model ../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf \
#     --do_train False \
#     --do_eval True \
#     --per_device_eval_batch_size 1 \
#     --model_max_length 2048 \
#     --fp16 False \
#     --bf16 True \
#     --save_safetensors False \
#     --w_bits $b \
#     --a_bits 16 \
#     --w_clip \
#     --a_asym \
#     --rotate \
#     --optimized_rotation_path /workspace/Weight_compression/hf_model_comp/spinquant/output_rotation/7b_w${b}a16/R.bin \
#     --save_qmodel_path ../hf_model_comp/spinquant/output_qmodel/7B_w${b}a16.pth \
#     2>&1 | tee ./logs/eval_zeroshot_7B_w${b}a16.log
# done


    # --load_qmodel_path ../hf_model_comp/spinquant/output_qmodel/7B_w${b}a16.pth \
    # --save_qmodel_path ../hf_model_comp/spinquant/output_qmodel/13B_w${b}a16.pth \
    # --save_qmodel_path ./output_qmodel/7B_w${b}a16.pth \
    # --export_to_et \
    # --w_groupsize 32 \
    # --input_model ../Wparam_dataset/hf_model/meta-llama--Llama-2-13B-hf \
# meta-llama--Meta-Llama-3-13B