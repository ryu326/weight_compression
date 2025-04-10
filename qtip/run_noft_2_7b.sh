export CUDA_VISIBLE_DEVICES=0,1,2,3

CKPT="./ckpt/noft2"
HF="./hf/noft2"
LOG="./log/noft2"
HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"

mkdir -p $CKPT
mkdir -p $LOG
mkdir -p $HF

for K in 3
do
    SAVE_PATH="$CKPT/2_7b_${K}bit"
    LOG_FILE="$LOG/2_7b_${K}bit.txt"
    HF_PATH="$HF/2_7b_${K}bit"

    echo "Running quantization for K=$K bits"

    python -m quantize_llama.quantize_finetune_llama \
        --save_path $SAVE_PATH \
        --codebook bitshift \
        --base_model ../Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf \
        --in_hess_path $HESS \
        --scale_override 0.9 \
        --ft_epochs 0 \
        --td_x 16 \
        --td_y 16 \
        --L 16 \
        --K $K \
        --V 2 \
        --decode_mode quantlut_sym \
        --tlut_bits 9 2>&1 \
        | tee -a $LOG_FILE 

    # python -m quantize_llama.hfize_llama \
    #     --quantized_path $SAVE_PATH \
    #     --hf_output_path $HF_PATH 2>&1  \
    #     | tee -a $LOG_FILE 

    # python -m quantize_llama.finetune_e2e_llama --base_model ../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B \
    #     --hf_path $HF_PATH --devset_size 640 --ft_valid_size 128 \
    #     --ft_epochs 4 --ft_update_freq 4 --ft_bs 1 --ctx_size 2048 \
    #     --ft_train_lut --hf_output_path ${HF_PATH}_ft2 2>&1 | tee -a $LOG_FILE 

    # python -m eval.eval_ppl \
    #     --hf_path ${HF_PATH}_ft2 \
    #     --seqlen 2048 2>&1 | tee -a ${HF_PATH}_ft2_result 2>&1
done
