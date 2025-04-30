export CUDA_VISIBLE_DEVICES=0,1,2,3
for b in 5 6 7 8 9; do
    torchrun --nnodes=1 --nproc_per_node=4 ptq_eval_ppl_zeroshot.py \
    --input_model ../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B \
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
    --optimized_rotation_path ./output_rotation/8B_w${b}a16/R.bin \
    --load_qmodel_path ./output_qmodel/8B_w${b}a16.pth \
    2>&1 | tee ./logs/eval_zeroshot_8B_w${b}a16.log
done

    # --save_qmodel_path ./output_qmodel/8B_w${b}a16.pth \
    # --export_to_et \
    # --w_groupsize 32 \