CKPT="./ckpt/noft"
HF="./hf/skip_test"
LOG="./log/skip_test"

mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG

model_name="meta-llama--Llama-2-7b-hf"
HESS="../Wparam_dataset/quip_hess/Hessians-Llama-2-7b-6144"
# HESS="../Wparam_dataset/quip_hess/llama3_8b_6144"

# lmbda_values=(50 100 300 1000 10000 100000)
lmbda_values=(100)
skip_list_values=("1_v" "1_k" "1_q" "1_o" "1_up" "1_gate" "1_down")

for lmbda in "${lmbda_values[@]}"; do
    for skip_list in "${skip_list_values[@]}"; do
        # python -m quantize_llama.hfize_llama --quantized_path $CKPT/$model_name/lmbda${lmbda}_ql \
        #     --skip_list "$skip_list" \
        #     --hf_output_path $HF/$model_name/lmbda${lmbda}_ql_${skip_list} 2>&1 | tee $LOG/${model_name}_lmbda${lmbda}_ql_${skip_list}.log 

        pretrain_path=$HF/$model_name/lmbda${lmbda}_ql_${skip_list}
        log_path=$(echo "$pretrain_path" | sed 's|_reconstructed|_eval|')_quip_result.txt
        log_dir=$(dirname "$log_path")
        mkdir -p "$log_dir"
        echo "Running evaluation for directory: $pretrain_path"
        export CUDA_VISIBLE_DEVICES=4
        python eval_ppl.py \
            --hf_path $pretrain_path \
            --seqlen 2048 \
            --no_use_cuda_graph 2>&1 | tee -a "$log_path"
    done
done
