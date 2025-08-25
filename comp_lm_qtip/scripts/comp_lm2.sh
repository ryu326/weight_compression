export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m eval.eval_zeroshot_hf \
    --tasks hellaswag,openbookqa,mathqa,sciq,pubmedqa \
    --batch_size 8  \
    --hf_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B \
    --output_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B_2

python -m eval.eval_zeroshot_hf \
    --tasks hellaswag,openbookqa,mathqa,sciq,pubmedqa \
    --batch_size 8  \
    --hf_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B \
    --output_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-3.2-3B_2

python -m eval.eval_zeroshot_hf \
    --tasks hellaswag,openbookqa,mathqa,sciq,pubmedqa \
    --batch_size 8  \
    --hf_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf \
    --output_path /workspace/Weight_compression/Wparam_dataset/hf_model/meta-llama--Llama-2-7b-hf_2


# python -m eval.eval_zeroshot_hf \
#     --tasks hellaswag,openbookqa,mathqa,sciq,pubmedqa \
#     --batch_size 4  \
#     --hf_path $HF/$SAVE_NAME \
#     --output_path ${RES}/${SAVE_NAME}_2