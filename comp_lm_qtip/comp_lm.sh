
model_names=(
    # "meta-llama--Meta-Llama-3-8B"
    "meta-llama--Llama-2-7b-hf"
    # "meta-llama--Llama-2-13b-hf"
)

CKPT="./ckpt"
HF="./hf"
LOG="./log"
HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"

mkdir $CKPT
mkdir $HF
mkdir $LOG

save_path="./model_lm_reconstructed/diag_scale"
comp_model_base="../NWC/checkpoint/nwc/block_seq_row_16"
comp_model_base="../NWC/checkpoint/nwc_ql/block_seq_ql_random__llama-3-8b-hf/block_seq_ql_random_col_16"

for model_name in "${model_names[@]}"; do
    lm_model_path="../Wparam_dataset/hf_model/$model_name"
    # ql="../Wparam_dataset/hessian/$model_name/quip_hess_n6144_top3_qlevel3.pt"
    ql="../Wparam_dataset/hessian/$model_name/pileval_n_samples128_seqlen512_top[ 0.1  1.  10. ]_qlevel[3, 2, 1].pt"
    lmbda_values=(200 300 1000 10000 100000)
    for lmbda in "${lmbda_values[@]}"; do
        echo "Running with lmbda=${lmbda}"
        comp_model=$comp_model_base/lmbda${lmbda}_*/best_loss*.pth.tar
        CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-31 \
        python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$model_name/lmbda${lmbda}_ql_ldlq \
            --base_model $lm_model_path \
            --in_hess_path $HESS --devset_size 384 --ft_valid_size 128 \
            --batch_size 8 \
            --ql "$ql" \
            --ldlq \
            --comp_batch_size 1 \
            --comp_model_path $comp_model 2>&1 | tee $LOG/${model_name}_lmbda${lmbda}_ql_ldlq.log
        
        python -m quantize_llama.hfize_llama --quantized_path $CKPT/$model_name/lmbda${lmbda}_ql_ldlq \
                --hf_output_path $HF/$model_name/lmbda${lmbda}_ql_ldlq 2>&1 | tee $LOG/${model_name}_lmbda${lmbda}_ql_ldlq_hfize.log 

        pretrain_path=$HF/$model_name/lmbda${lmbda}_ql_ldlq
        log_path=$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt
        log_dir=$(dirname "$log_path")
        mkdir -p "$log_dir"
        echo "Running evaluation for directory: $pretrain_path"
        export CUDA_VISIBLE_DEVICES=3 
        python eval_ppl.py \
            --hf_path $pretrain_path \
            --seqlen 2048 \
            --no_use_cuda_graph | tee -a "$log_path" &
    done

    # #!/bin/bash
    # # lmbda 값과 GPU 목록 정의
    # lmbda_values=(100 200 300 1000)
    # gpu_list=(0 1 2 3)

    # # 각 lmbda에 대해 GPU를 할당하여 동시에 실행
    # for i in "${!lmbda_values[@]}"; do
    #     lmbda="${lmbda_values[$i]}"
    #     gpu="${gpu_list[$i]}"

    #     pretrain_path="$HF/$model_name/lmbda${lmbda}_ql"
    #     log_path="$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt"
    #     log_dir=$(dirname "$log_path")
    #     mkdir -p "$log_dir"

    #     echo "Running evaluation for directory: $pretrain_path on GPU ${gpu}"

    #     # nohup으로 백그라운드 실행하고 로그는 지정한 파일에 저장 (표준에러도 함께 기록)
    #     export CUDA_VISIBLE_DEVICES=${gpu}
    #     nohup python eval_ppl.py \
    #         --hf_path "$pretrain_path" \
    #         --seqlen 2048 \
    #         --no_use_cuda_graph >> "$log_path" 2>&1 &
    # done

    # echo "All evaluation processes have been started."


done
