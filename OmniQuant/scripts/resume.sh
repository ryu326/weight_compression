MODELS_TO_RUN=(
    # "llama3_8b"
    "llama2_7b"
    # "llama3.2_3b"
    # "llama2_13b"
    # "llama3.2_1b_inst"
    # "llama3.2_3b_inst"
)

CKPT="/home/jgryu/workspace/weight_compression/hf_model_comp/omniquant"
LOG="./log"
RES="/home/jgryu/workspace/weight_compression/hf_model_comp_results/omniquant"

export CUDA_VISIBLE_DEVICES=1
export WANDB_SILENT=true
export TRANSFORMERS_NO_TORCHVISION=1
export HF_HOME=/home/jgryu/.cache/huggingface

declare -A MODEL_PATHS
MODEL_PATHS=(
    ["llama2_7b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf"
    ["llama2_13b"]="../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf"
    ["llama3_8b"]="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B"
    ["vicuna_7b"]="../Wparam_dataset/hf_model/lmsys--vicuna-7b-v1.5"
    ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
    ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
    ["llama3.2_3b"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B"
    ["llama3.2_3b_inst"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B-Instruct"
    ["llama3.2_1b_inst"]="../Wparam_dataset/hf_model/meta-llama--Llama-3.2-1B-Instruct"
)

mkdir -p $CKPT $HF $LOG $RES

for model_key in "${MODELS_TO_RUN[@]}"; do
    base_model=${MODEL_PATHS[$model_key]}

    for wb in 2 3 4; do
        CUDA_VISIBLE_DEVICES=1 python main.py \
        --model $base_model  \
        --epochs 0 --output_dir ./log/test \
        --wbits ${wb} --abits 16 --lwc \
        --resume ${CKPT}/Llama-2-7b-w${wb}a16.pth \
        --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,mmlu \
        --eval_out_dir ${RES}/${model_key}
        # --eval_ppl 
    done
done

