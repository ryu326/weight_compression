# Example script to quantize Llama 2 7B to 2 bits
# Fill these in with your own paths
export CUDA_VISIBLE_DEVICES=0,2,3

CKPT="./ckpt"
HF="./hf"
LOG="./log"
HESS="/home/jgryu/Weight_compression/quip-sharp/hess/relaxml/Hessians-Llama-2-13b-6144"

# mkdir $CKPT
# mkdir $LOG
# mkdir $HF

python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/2_13b_4bit \
       --codebook bitshift \
       --base_model /home/jgryu/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf \
       --in_hess_path $HESS \
       --scale_override 0.9 \
       --ft_epochs 0 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 4 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9
       >> $LOG/2_13b_4bit 2>&1

python -m quantize_llama.hfize_llama --quantized_path ./ckpt/2_13b_4bit --hf_output_path ./hf/2_13b_4bit >> $LOG/2_13b_4bit 2>&1 

python -m eval.eval_ppl  --hf_path $HF/2_13b_4bit --seqlen 2048 2>&1 | tee -a ./logs/2_13b_4bit.txt 2>&1
