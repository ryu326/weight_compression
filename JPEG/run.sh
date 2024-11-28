# CUDA_VISIBLE_DEVICES=1 taskset -c 32-39 nohup python -u handcraft_codecs.py /home/taco/VisDiff_for_compression/VisDiff/data/examples/set_a /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/set_a jp --bpp 0.13 > log_JPEG_0.1_a.txt 2>&1 &
# # CUDA_VISIBLE_DEVICES=2 taskset -c 32-39 nohup python -u handcraft_codecs.py /home/taco/VisDiff_for_compression/VisDiff/data/examples/set_a /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/set_a jp --bpp 0.6 > log_JPEG_0.6_a.txt 2>&1 &
 
# CUDA_VISIBLE_DEVICES=7 taskset -c 32-39 nohup python -u handcraft_codecs.py /home/taco/VisDiff_for_compression/VisDiff/data/examples/set_b /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/set_b jp --bpp 0.13 > log_JPEG_0.1_b.txt 2>&1 &
# # CUDA_VISIBLE_DEVICES=6 taskset -c 32-39 nohup python -u  handcraft_codecs.py /home/taco/VisDiff_for_compression/VisDiff/data/examples/set_b /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/set_b jp --bpp 0.6 > log_JPEG_0.6_b.txt 2>&1 &
 
# ImageNet Star
# CUDA_VISIBLE_DEVICES=4 taskset -c 32-39 nohup python -u handcraft_codecs.py /data/ImageNet-S/in_the_fog/n02106662 /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/foggy/0.1 jp --bpp 0.13 > log_JPEG_0.1_foggy.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=5 taskset -c 32-39 nohup python -u handcraft_codecs.py /data/ImageNet-S/in_the_fog/n02106662 /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/foggy/0.6 jp --bpp 0.6 > log_JPEG_0.6_foggy.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=6 taskset -c 32-39 nohup python -u handcraft_codecs.py /data/ILSVRC2012/val/n02106662 /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/general/0.1 jp --bpp 0.13 > log_JPEG_0.1_general.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=7 taskset -c 32-39 nohup python -u handcraft_codecs.py /data/ILSVRC2012/val/n02106662 /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/general/0.6 jp --bpp 0.6 > log_JPEG_0.6_general.txt 2>&1 &
 
# CUDA_VISIBLE_DEVICES=7 taskset -c 32-39 python -u handcraft_codecs.py /data/ImageNet-S/Fog /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/fog/0.1 jp --bpp 0.12 > log_JPEG_0.1_fog.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=4 taskset -c 11-18 python -u handcraft_codecs.py /data/ImageNet-S/Fog /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/fog/0.6 jp --bpp 0.5 > log_JPEG_0.6_fog.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=7 taskset -c 32-39 python -u handcraft_codecs.py /home/taco/TACO_development_kit/datasets_for_compression/imagenet_val /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/imagenet_valid/0.1 jp --bpp 0.12 > log_JPEG_0.1_imagenet_val.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=4 taskset -c 11-18 python -u handcraft_codecs.py /home/taco/TACO_development_kit/datasets_for_compression/imagenet_val /home/taco/VisDiff_for_compression/Examples_Compressed/JPEG/imagenet_valid/0.5 jp --bpp 0.5 > log_JPEG_0.5_imagenet_val.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -u handcraft_codecs.py /home/jgryu/Weight_compression/JPEG/wp_image/c=1/ /home/jgryu/Weight_compression/JPEG/wp_image_result/c=1/0.5 jp --bpp 0.5 > log_JPEG_wp_c=1_bpp_0.5.txt 2>&1 &

# c_values=(0.5 0.7 1 1.2 1.5 2)
# bpp_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
# t_values=(1 1.2 1.5 2 2.5 3 3.5 4 5)

# 각 c 값에 대해 명령어 실행

# for c in "${c_values[@]}"; do
#     for bpp in "${bpp_values[@]}"; do
#         CUDA_VISIBLE_DEVICES=0 python -u handcraft_codecs.py \
#             "/home/jgryu/Weight_compression/JPEG/wp_image/size32_32/c=${c}/" \
#             "/home/jgryu/Weight_compression/JPEG/wp_image_result/size32_32/c=${c}/bpp${bpp}" \
#             jp --bpp ${bpp} 2>&1 | tee "./logs/size32_32/log_JPEG_wp_32_32_c=${c}_bpp_${bpp}.txt" 
#     done
# done

# for c in "${c_values[@]}"; do
#     for bpp in "${bpp_values[@]}"; do
#         CUDA_VISIBLE_DEVICES=3 python -u handcraft_codecs.py \
#             "/home/jgryu/Weight_compression/JPEG/wp_image/meta-llama-3-8b_mlp_val_json/256_256/c=${c}/" \
#             "/home/jgryu/Weight_compression/JPEG/wp_image_result/meta-llama-3-8b_mlp_val_json/256_256/c=${c}/bpp${bpp}" \
#             jp --bpp ${bpp} 2>&1 | tee "./logs/mlp_size256_256/log_JPEG_wp_32_32_c=${c}_bpp_${bpp}.txt" 
#     done
# done

dim=(64 256 512 1024)
c_values=(0.5 0.7 1 1.2 1.5 2)
bpp_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
t_values=(1 1.2 1.5 2 2.5 3 3.5 4 5)
for d in "${dim[@]}"; do
    for t in "${t_values[@]}"; do
        mkdir -p "./logs/llama-2-7b_mlp_train_json/${d}_${d}"
        for bpp in "${bpp_values[@]}"; do
            CUDA_VISIBLE_DEVICES=3 python -u handcraft_codecs.py \
                "/home/jgryu/Weight_compression/JPEG/wp_image/llama-2-7b_mlp_train_json/${d}_${d}/t=${t}/" \
                "/home/jgryu/Weight_compression/JPEG/wp_image_result/llama-2-7b_mlp_train_json/${d}_${d}/t=${t}/bpp${bpp}" \
                jp --bpp ${bpp} 2>&1 | tee "./logs/llama-2-7b_mlp_train_json/${d}_${d}/log_t=${t}_bpp${bpp}.txt" 
        done
    done
done

for d in "${dim[@]}"; do
    for t in "${t_values[@]}"; do
        mkdir -p "./logs/llama-2-7b_attn_train_json/${d}_${d}"
        for bpp in "${bpp_values[@]}"; do
            CUDA_VISIBLE_DEVICES=3 python -u handcraft_codecs.py \
                "/home/jgryu/Weight_compression/JPEG/wp_image/llama-2-7b_attn_train_json/${d}_${d}/t=${t}/" \
                "/home/jgryu/Weight_compression/JPEG/wp_image_result/llama-2-7b_attn_train_json/${d}_${d}/t=${t}/bpp${bpp}" \
                jp --bpp ${bpp} 2>&1 | tee "./logs/llama-2-7b_attn_train_json/${d}_${d}/log_t=${t}_bpp${bpp}.txt" 
        done
    done
done