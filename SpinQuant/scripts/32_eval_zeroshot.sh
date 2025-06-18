export CUDA_VISIBLE_DEVICES=0,1,2,3
for b in 4; do
    torchrun --nnodes=1 --nproc_per_node=4 --master_port=29502 ptq_eval_ppl_zeroshot.py \
    --input_model ../Wparam_dataset/hf_model/meta-llama--Llama-2-13b-hf \
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
    --optimized_rotation_path /workspace/Weight_compression/hf_model_comp/spinquant/output_rotation/13B_W4A16KV16_lr_1.5_seed_0/R.bin \
    --save_qmodel_path ../hf_model_comp/spinquant/output_qmodel/13B_w${b}a16.pth \
    2>&1 | tee ./logs/eval_zeroshot_13B_w${b}a16.log
done

    # --load_qmodel_path ../hf_model_comp/spinquant/output_qmodel/7B_w${b}a16.pth \
    # --save_qmodel_path ../hf_model_comp/spinquant/output_qmodel/13B_w${b}a16.pth \
    # --save_qmodel_path ./output_qmodel/7B_w${b}a16.pth \
    # --export_to_et \
    # --w_groupsize 32 \
    # --input_model ../Wparam_dataset/hf_model/meta-llama--Llama-2-13B-hf \
# meta-llama--Meta-Llama-3-13B