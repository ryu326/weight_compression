model_name="google/siglip-base-patch16-224"
HESS="../Wparam_dataset/quip_hess/siglip-base-patch16-224_512"
export HF_HOME=/workspace/hf_cache/huggingface_qtip

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"
RES="../hf_model_comp_results/qtip"


mkdir -p $CKPT
mkdir -p $LOG
mkdir -p $HF

BITS=(2 3 4 5 6)
gpu_ids=(0 1 2 3)
i=0

for idx in "${!BITS[@]}"; do
    K=${BITS[$idx]}
    gpu_id=${gpu_ids[$((i % 4))]}

    SAVE_NAME=${model_name}/${K}bit
    LOG_FILE=${LOG}/${SAVE_NAME}.log
    mkdir -p $(dirname "$LOG_FILE")

    echo ">> Launching full pipeline on GPU $gpu_id: BIT=$K"
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id

        # python -m quantize_llama.quantize_finetune_clip \
        #     --save_path ${CKPT}/${SAVE_NAME} \
        #     --codebook bitshift \
        #     --base_model $model_name \
        #     --in_hess_path $HESS \
        #     --scale_override 0.9 \
        #     --ft_epochs 0 \
        #     --td_x 16 \
        #     --td_y 16 \
        #     --L 16 \
        #     --K $K \
        #     --V 2 \
        #     --decode_mode quantlut_sym \
        #     --tlut_bits 9 \
        #     > $LOG_FILE 2>&1

        python -m quantize_llama.hfize_non_llama \
            --quantized_path ${CKPT}/${SAVE_NAME} \
            --hf_output_path ${HF}/${SAVE_NAME} \
            --base_model $model_name \
            > $LOG_FILE 2>&1

        # python -m quantize_llama.hfize_siglip \
        #     --quantized_path ${CKPT}/${SAVE_NAME} \
        #     --hf_output_path ${HF}/${SAVE_NAME} \
        #     --base_model $model_name \
        #     > $LOG_FILE 2>&1

        python -m eval.eval_siglip_imagenet \
            --hf_path ${HF}/${SAVE_NAME} \
            --output_path $RES/$SAVE_NAME \
            --model_id $model_name \
            >> $LOG_FILE 2>&1

        if [ "$HF/$SAVE_NAME" != "$HF" ]; then
            echo "Cleaning up temporary files for $SAVE_NAME"
            rm -rf "$HF/$SAVE_NAME"
        fi
    ) &

    ((i+=1))
    if (( i % 4 == 0 )); then
        wait
    fi
done

wait
