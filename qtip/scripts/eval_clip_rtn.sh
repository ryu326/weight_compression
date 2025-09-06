gpu_ids=(0 1 2 3)
i=0
PYTHON_BIN=$(which python)

for b in {10..16}; do
    gpu_id=${gpu_ids[$((i % 4))]}
    PATH="../hf_model_comp/RTN/clip_L_14_W${b}g128"

    echo "Launching on GPU $gpu_id: W=$b"
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_BIN -m eval.eval_clip_imagenet \
        --hf_path $PATH &

    ((i+=1))

    # every 4 jobs, wait for all to finish
    if (( i % 4 == 0 )); then
        wait
    fi
done

wait