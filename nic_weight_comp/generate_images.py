# taskset -c 0-7 python -u generate_images.py

device='cuda'

import os
import math
import shutil

import torch
import torch.nn.functional as F
from PIL import Image

import torchvision
from tqdm import tqdm

import json
import lpips
import pandas as pd

from models import NIC_Fair

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)
def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


# 압축 성능 평가
def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_alex = loss_fn_alex.to(device)
loss_fn_alex.requires_grad_(False)

def main():
    
    dataset_folder_dict = {
        'imagenet_val_best_class': './imagenet_val_best_class'
        ,'imagenet_val_worst_class': './imagenet_val_worst_class'
        }
    for dataset in dataset_folder_dict:

        image_list = os.listdir(dataset_folder_dict[dataset])
        image_list.sort()

        for lmbda in [0.0007, 0.0025, 0.0035, 0.0067, 0.013]:
            
            save_path = f'./NIC_Fair_compressed_images/{dataset}/{lmbda}'
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            try:
                os.mkdir(save_path)
            except:
                os.makedirs(save_path)

            # 모델 선언 + 체크 포인트 불러오기
            net = NIC_Fair(M=320)        
            net = net.to(device)
            
            
            file_list = os.listdir(f'./checkpoint/exp_NIC_Fair_lambda_{lmbda}_seed_100.0_batch_size_8_radius_denominator_8.0')
            
            for file_name in file_list:
                if ('recent_model_PSNR' in file_name):
                    checkpoint_path = f'./checkpoint/exp_NIC_Fair_lambda_{lmbda}_seed_100.0_batch_size_8_radius_denominator_8.0/{file_name}'
                    break
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            dictory = {}
            for k, v in checkpoint["state_dict"].items():
                dictory[k.replace("module.", "")] = v
            net.load_state_dict(dictory)

            net = net.eval().to(device)
            net.requires_grad_(False)
            net.update()

            stat_csv = {
                'image_name': [],
                'bpp': [],
                'psnr': [],
                'lpips': []
                }

            mean_csv = {
                'bpp':0.0,
                'psnr': 0.0,
                'lpips':0.0
            }
            
            for image_name in tqdm(image_list, desc=f"Model lambda: {lmbda}, Dataset: {dataset}"):

                save_image_name = image_name

                image_path = f'{dataset_folder_dict[dataset]}/{image_name}'
                recon_image_path = f'{save_path}/{save_image_name}'
                
                x = Image.open(image_path).convert('RGB')
                x = torchvision.transforms.ToTensor()(x).unsqueeze(0).to(device) # 원본 이미지

                try:
                    x_paddeimg, padding = pad(x, p = 128)
                    out_enc = net.compress(x_paddeimg.to(device))
                except:
                    x_paddeimg, padding = pad(x, p = 256)
                    out_enc = net.compress(x_paddeimg.to(device))
                
                out_dec = net.decompress(out_enc["strings_high_freq"], out_enc["strings_not_high_freq"], out_enc["shape"], out_enc["topk_indices"])
                
                num_pixels = x.size(0) * x.size(2) * x.size(3)
                
                bpp = sum(len(s[0]) for s in out_enc["strings_high_freq"]) * 8.0 / num_pixels + \
                    sum(len(s[0]) for s in out_enc["strings_not_high_freq"]) * 8.0 / num_pixels
                
                ts_topk_list = torch.Tensor(out_enc["topk_indices"])
                data_num = ts_topk_list.size(0) * ts_topk_list.size(1)
                
                topk_bpp = (data_num / num_pixels) * 32
                bpp += topk_bpp # 정수 하나에 32비트니까

                x_hat = crop(out_dec["x_hat"], padding).clone().detach()


                psnr = compute_psnr(x, x_hat)
                lpips_score = loss_fn_alex(x, x_hat).item()

                mean_csv['bpp'] += bpp
                mean_csv['psnr'] += psnr
                mean_csv['lpips'] += lpips_score

                stat_csv['image_name'].append(save_image_name)
                stat_csv['bpp'].append(bpp)
                stat_csv['psnr'].append(psnr)
                stat_csv['lpips'].append(lpips_score)

                print(f"Image: {save_image_name}, BPP: {bpp}, PSNR: {psnr}, LPIPS: {lpips_score}")

                torchvision.utils.save_image(x_hat, recon_image_path, nrow=1)
            
            data_stat_csv = pd.DataFrame(stat_csv)
            data_stat_csv.to_csv(f'./NIC_Fair_compressed_images/{dataset}/{lmbda}_stat_per_image.csv')

            mean_csv['bpp'] /= len(image_list)
            mean_csv['psnr'] /= len(image_list)
            mean_csv['lpips'] /= len(image_list)

            with open(f'./NIC_Fair_compressed_images/{dataset}/{lmbda}_mean_stat.json', 'w') as f:
                json.dump(mean_csv, f, indent=4)   

            print(f"Compression complete. {lmbda}, Dataset: {dataset}, BPP: {mean_csv['bpp']}, PSNR: {mean_csv['psnr']}, LPIPS: {mean_csv['lpips']}")

if __name__ == "__main__":
    main()