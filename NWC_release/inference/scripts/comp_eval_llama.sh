HESS="./hess/llama3_8b_6144"
hf_model_path="../dataset/hf_model/meta-llama--Meta-Llama-3-8B"
comp_model_ckpt_base="../train/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt"

CKPT="./ckpt"
HF="./hf"
RES="./results"
LOG="./log"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG


lmbda_values=(30 50 100 300 1000 10000)

for lmbda in "${lmbda_values[@]}"; do

    SAVE_NAME=llama3_8b/lmbda${lmbda}

    comp_model_ckpt=$comp_model_ckpt_base/lmbda${lmbda}_*/best_loss*.pth.tar
    mkdir -p $(dirname "$LOG/$SAVE_NAME.log")
    
    echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
    python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$SAVE_NAME \
        --base_model $hf_model_path \
        --comp_model_path $comp_model_ckpt \
        --in_hess_path $HESS \
        --devset_size 384 --ft_valid_size 128 --batch_size 8 \
        --direction col --ql --Q 4 --row_normalize --ldlq --comp_batch_size 128 --ft_epochs 5 --ft_rnorm \
    
    echo "################## Running hfize | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
    python -m quantize_llama.hfize_llama --quantized_path $CKPT/${SAVE_NAME} \
            --hf_output_path $HF/${SAVE_NAME} \


    echo "################## Running PPL evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
    echo "Running evaluation for directory: $HF/$SAVE_NAME"
    python -m eval.eval_ppl_hf \
        --hf_path $HF/${SAVE_NAME} \
        --seqlen 2048 \
        --output_path ${RES}/${SAVE_NAME} \
        --datasets wikitext2,c4 \
        --no_use_cuda_graph 2>&1 | tee -a $LOG/$SAVE_NAME.log


    echo "################## Running benchmark evaluation | lmbda=${lmbda} | Exp: ${exp_name} | Model: ${model_name} ##################"
    python -m eval.eval_zeroshot_hf \
        --tasks arc_challenge,arc_easy,piqa,winogrande,boolq,hellaswag,mmlu \
        --batch_size 16 \
        --hf_path $HF/$SAVE_NAME \
        --output_path $RES/${SAVE_NAME} \


    if [ "$HF/$SAVE_NAME" != "$HF" ]; then
        echo "Cleaning up temporary files for $SAVE_NAME"
        rm -rf "$HF/$SAVE_NAME"
        # rm -rf "$CKPT/$SAVE_NAME"
    fi
done