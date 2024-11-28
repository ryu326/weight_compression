# parallel -j1 python dataset_generation.py --d {} --model_filter llama 7b ::: 128 32 16 2
# parallel -j1 python dataset_generation.py --d {} --model_filter gemma 2b ::: 128 32 16 2

# python dataset_generation.py --d 256 --model_filter google--gemma-2-2b

values="2048 1024 512 256 128 32"

# for value in $values; do
#     python dataset_generation.py --d "$value" --model_filter gemma 2b
# done

for value in $values; do
    python dataset_generation.py --d "$value" --model_filter llama 7b
done

for value in $values; do
    python dataset_generation.py --d "$value" --model_filter gemma 7b
done

for value in $values; do
    python dataset_generation.py --d "$value" --model_filter qwen 0.5b
done

for value in $values; do
    python dataset_generation.py --d "$value" --model_filter qwen 1.5b
done

for value in $values; do
    python dataset_generation.py --d "$value" --model_filter qwen 7b
done