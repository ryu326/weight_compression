export CUDA_VISIBLE_DEVICES=0,1,2,3

CKPT="../hf_model_comp/qtip/ckpt"
HF="../hf_model_comp/qtip/hf"
LOG="./log"
HESS="../Wparam_dataset/quip_hess/clip-vit-large-patch14_8192"
base_model="../Wparam_dataset/hf_model/openai--clip-vit-large-patch14"

mkdir -p $CKPT
mkdir -p $LOG
mkdir -p $HF

BITS=(6 5 4 3 2)

for idx in "${!BITS[@]}"; do
    K=${BITS[$idx]}
    gpu_id=$idx

    SAVE_PATH="$CKPT/clip-vit-large-patch14_${K}bit"
    LOG_FILE="$LOG/clip-vit-large-patch14_${K}_bit.txt"
    HF_PATH="$HF/clip-vit-large-patch14_${K}bit"

    (
        echo "########## [GPU $gpu_id] Running quantization for K=$K bits ##########"

        # CUDA_VISIBLE_DEVICES=$gpu_id python -m quantize_llama.quantize_finetune_clip \
        #     --save_path $SAVE_PATH \
        #     --codebook bitshift \
        #     --base_model $base_model \
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
        #     2>&1 | tee $LOG_FILE

        echo "########## [GPU $gpu_id] HFizing for K=$K ##########"
        CUDA_VISIBLE_DEVICES=$gpu_id python -m quantize_llama.hfize_clip \
            --quantized_path $SAVE_PATH \
            --hf_output_path $HF_PATH \
            --base_model $base_model \
            2>&1 | tee -a $LOG_FILE

        echo "########## [GPU $gpu_id] Evaluating ImageNet for K=$K ##########"
        CUDA_VISIBLE_DEVICES=$gpu_id python -m eval.eval_clip_imagenet \
            --hf_path $HF_PATH \
            2>&1 | tee -a $LOG_FILE
    ) &
done

wait
