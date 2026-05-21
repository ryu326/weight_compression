#!/bin/bash
# Re-run hfize + eval for CLIP jobs that failed at eval (CKPT exists)
PYTHON_BIN="/opt/conda/bin/python"
CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
RES="../hf_model_comp_results_v2"
LOG="./log"
export HF_HOME=/home/jgryu/.cache/huggingface
MODEL_PATH="/home/jgryu/workspace/weight_compression/Wparam_dataset/hf_model/openai--clip-vit-large-patch14"
MODEL_ID="openai/clip-vit-large-patch14"

GPU_IDS=(0 1 2 3)
N_GPU=4
job_idx=0

for exp in norm_search_ldlq64_seed1 norm_search_ldlq64_seed2; do
    for lmbda in 30 50 100 300 1000 10000; do
        SAVE_NAME="openai--clip-vit-large-patch14/${exp}/lmbda${lmbda}"
        LOG_FILE="${LOG}/${SAVE_NAME}_rerun_eval.log"
        gpu_id=${GPU_IDS[$((job_idx % N_GPU))]}
        echo ">> [GPU $gpu_id] hfize+eval $exp lmbda=$lmbda"
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id
            $PYTHON_BIN -m quantize_llama.hfize_clip \
                --quantized_path ${CKPT}/${SAVE_NAME} \
                --base_model $MODEL_PATH \
                --hf_output_path ${HF}/${SAVE_NAME} \
                > $LOG_FILE 2>&1
            $PYTHON_BIN -m eval.eval_clip_imagenet \
                --hf_path ${HF}/${SAVE_NAME} \
                --output_path ${RES}/${SAVE_NAME} \
                --model_id $MODEL_ID \
                >> $LOG_FILE 2>&1
            rm -rf "${HF}/${SAVE_NAME}"
        ) &
        ((job_idx++))
        if (( job_idx % N_GPU == 0 )); then wait; fi
    done
done
wait
echo "Re-run eval done."
