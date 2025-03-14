# export CUDA_VISIBLE_DEVICES=0,1,2,3

CKPT="./ckpt"
HF="./hf"
LOG="./log"
HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

mkdir -p $CKPT
mkdir -p $LOG
mkdir -p $HF

for K in 2 3 4
do
    SAVE_PATH="$CKPT/3_8b_${K}bit"
    LOG_FILE="$LOG/3_8b_${K}bit.txt"
    HF_PATH="$HF/3_8b_${K}bit"

    echo "Running quantization for K=$K bits"

    python -m quantize_llama.quantize_finetune_llama \
        --save_path $SAVE_PATH \
        --codebook bitshift \
        --base_model meta-llama/Meta-Llama-3-8B \
        --in_hess_path $HESS \
        --scale_override 0.9 \
        --ft_epochs 5 \
        --td_x 16 \
        --td_y 16 \
        --L 16 \
        --K $K \
        --V 2 \
        --decode_mode quantlut_sym \
        --tlut_bits 9 \
        >> $LOG_FILE 2>&1

    python -m quantize_llama.hfize_llama \
        --quantized_path $SAVE_PATH \
        --hf_output_path $HF_PATH \
        >> $LOG_FILE 2>&1 

    python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Meta-Llama-3-8B \
        --hf_path $HF_PATH --devset_size 640 --ft_valid_size 128 \
        --ft_epochs 4 --ft_update_freq 4 --ft_bs 2 --ctx_size 4096 \
        --ft_train_lut --hf_output_path $HF_PATH >> $LOG_FILE 2>&1

    python -m eval.eval_ppl \
        --hf_path $HF_PATH \
        --seqlen 2048 2>&1 | tee -a $LOG_FILE 2>&1
done
