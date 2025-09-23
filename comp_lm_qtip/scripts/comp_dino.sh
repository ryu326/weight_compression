model_name="facebook--dinov2-large-imagenet1k-1-layer"
HESS="../Wparam_dataset/quip_hess/llava-hf--dinov2-large-imagenet1k-1-layer_cc512"
export HF_HOME=/workspace/hf_cache/huggingface_nwc
lm_model_path="../Wparam_dataset/hf_model/$model_name"

comp_model_bases=(
    "../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16"
)

experiment_names=(
    "(llama)ql_rnorm_ldlq128"
)

CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
RES="../hf_model_comp_results"
LOG="./log"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG

lmbda_values=(10 30 50 100 300 1000 10000 100000)
PYTHON_BIN=$(which python)

for i in "${!experiment_names[@]}"; do
    exp_name="${experiment_names[$i]}"
    comp_model_base="${comp_model_bases[$i]}"

    gpu_ids=(0 1 2 3)
    i=0
    
    for lmbda in "${lmbda_values[@]}"; do
        gpu_id=${gpu_ids[$((i % 4))]}
        SAVE_NAME=${model_name}/${exp_name}/lmbda${lmbda}
        comp_model=${comp_model_base}/lmbda${lmbda}_*/best_loss*.pth.tar
        LOG_FILE=${LOG}/${SAVE_NAME}.log
        mkdir -p $(dirname "$LOG_FILE")

        echo ">> Launching full pipeline on GPU $gpu_id: lmbda=$lmbda"
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id

            taskset -c 0-7 $PYTHON_BIN -m quantize_llama.quantize_finetune_clip \
                --save_path ${CKPT}/${SAVE_NAME} \
                --base_model $lm_model_path \
                --comp_model_path $comp_model \
                --in_hess_path $HESS \
                --direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 \
                --ft_epochs 0 \
                > $LOG_FILE 2>&1

            echo ">> hfize lmbda=${lmbda}" >> $LOG_FILE
            $PYTHON_BIN -m quantize_llama.hfize_dino \
                --quantized_path $CKPT/$SAVE_NAME \
                --base_model $lm_model_path \
                --hf_output_path $HF/$SAVE_NAME \
                > $LOG_FILE 2>&1

            echo ">> eval lmbda=${lmbda}" >> $LOG_FILE
            $PYTHON_BIN -m eval.eval_dino \
                --hf_path $HF/$SAVE_NAME \
                --output_path ${RES}/${SAVE_NAME} \
                --model_id $model_name \
                >> $LOG_FILE 2>&1

                # --output_path ${RES}/${SAVE_NAME} \

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
done

# export CUDA_VISIBLE_DEVICES=0
# python -m eval.eval_dino \
#     --hf_path /workspace/Weight_compression/Wparam_dataset/hf_model/facebook--dinov2-large-imagenet1k-1-layer  \
#     --output_path /workspace/Weight_compression/hf_model_comp_results/dino_base