model_name="openai--clip-vit-large-patch14"
HESS="../Wparam_dataset/quip_hess/clip-vit-large-patch14_512"
# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"
# comp_model_base="/workspace/Weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_openai--clip-vit-large-patch14__vision_text_col_256.pt"
comp_model_base="/workspace/Weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_openai--clip-vit-large-patch14__vision_text_col_256.pt/clip_llama8b_col1024_pretrained_rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size4096_total_iter100000_lr0.0001_seed100"
lm_model_path="../Wparam_dataset/hf_model/$model_name"

CKPT="../hf_model_comp/comp_qtip/ckpt"
HF="../hf_model_comp/comp_qtip/hf"
LOG="./log"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG

lmbda_values=(1000 10000 100000)
gpu_ids=(0 1 2 3)
i=0
PYTHON_BIN=$(which python)

for lmbda in "${lmbda_values[@]}"; do
    gpu_id=${gpu_ids[$((i % 4))]}
    SAVE_NAME=clip_ql_llama_clip_ft/${model_name}/lmbda${lmbda}
    comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
    LOG_FILE=$LOG/$SAVE_NAME.log
    mkdir -p $(dirname "$LOG_FILE")

    echo ">> Launching full pipeline on GPU $gpu_id: lmbda=$lmbda"

    (
        export CUDA_VISIBLE_DEVICES=$gpu_id

        # 1. quantize + fine-tune
        taskset -c 0-7 $PYTHON_BIN -m quantize_llama.quantize_finetune_clip \
            --save_path $CKPT/$SAVE_NAME \
            --base_model $lm_model_path \
            --comp_model_path $comp_model \
            --in_hess_path $HESS \
            --ql \
            --ft_epochs 0 \
            >> $LOG_FILE 2>&1

        echo ">> hfize lmbda=${lmbda}" >> $LOG_FILE
        $PYTHON_BIN -m quantize_llama.hfize_clip \
            --quantized_path $CKPT/$SAVE_NAME \
            --base_model $lm_model_path \
            --hf_output_path $HF/$SAVE_NAME \
            >> $LOG_FILE 2>&1

        echo ">> eval lmbda=${lmbda}" >> $LOG_FILE
        $PYTHON_BIN -m eval.eval_clip_imagenet \
            --hf_path $HF/$SAVE_NAME \
            >> $LOG_FILE 2>&1
    ) &

    ((i+=1))
    if (( i % 4 == 0 )); then
        wait
    fi
done

wait


# # comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_openai--clip-vit-large-patch14__vision_text_col_256.pt"
# comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"

# model_name="openai--clip-vit-large-patch14"
# HESS="../Wparam_dataset/quip_hess/clip-vit-large-patch14_8192"

# # ql="../Wparam_dataset/hessian/$model_name/quip_hess_n6144_top3_qlevel3.pt"
# # ql="../Wparam_dataset/hessian/$model_name/pileval_n_samples128_seqlen512_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt"
# # ql='../Wparam_dataset/hessian/meta-llama--Llama-2-7b-hf/quip_hess_n6144_all_layers_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt'
# ############################################

# lm_model_path="../Wparam_dataset/hf_model/$model_name"

# CKPT="../hf_model_comp/comp_qtip/ckpt"
# HF="../hf_model_comp/comp_qtip/hf"
# LOG="./log"

# mkdir -p $CKPT
# mkdir -p $HF
# mkdir -p $LOG

# # lmbda_values=(100 300 1000 3000)
# lmbda_values=(50 100 200 300 1000 10000 100000)

# for lmbda in "${lmbda_values[@]}"; do
#     echo "################## Running compression lmbda=${lmbda} ##################"
    
#     ## ========= Change this =========
#     SAVE_NAME=clip_ql_llama_trained/${model_name}/lmbda${lmbda}
#     ## ========= Change this =========

#     comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
#     mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
    
#     CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-31 \
#     python -m quantize_llama.quantize_finetune_clip --save_path $CKPT/$SAVE_NAME \
#         --base_model $lm_model_path \
#         --comp_model_path $comp_model \
#         --in_hess_path $HESS \
#         --ql \
#         --ft_epochs 0 \
#         2>&1 | tee $LOG/$SAVE_NAME.log

#         # --incoh_mode had  --rescale_WH_2  --sigma_reg 1e-4 --use_train_scale \
#         # --ldlq --comp_batch_size 1 \
#         # --ft_comp_model --ft_comp_lmbda $lmbda --ft_comp_ep 100 --direction row \
#         # --ql \

#     echo "################## Running hfize lmbda=${lmbda} ##################"
#     python -m quantize_llama.hfize_clip --quantized_path $CKPT/${SAVE_NAME} \
#                 --base_model $lm_model_path \
#                 --hf_output_path $HF/$SAVE_NAME 2>&1 | tee -a $LOG/$SAVE_NAME.log 

#     echo "################## Running Imagenet evaluation lmbda=${lmbda} ##################"
#     python -m eval.eval_clip_imagenet \
#         --hf_path $HF/$SAVE_NAME  | tee $LOG_FILE 

#     # rm -r $pretrain_path

# done