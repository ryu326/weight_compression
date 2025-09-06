import argparse
import csv
import functools
import glob
import itertools
import operator
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager

import matplotlib as mpl
import numpy as np
import scipy.ndimage
from PIL import Image
import torch
from .jpeg import jp_compress, jp_decompress
from .bpg import compress_tensor_with_bpg
from .webp import compress_tensor_with_webp

def compress_linear(W, H, args, device='cpu'):

    save_path = f'{args.save_path}/{args.layer_idx}_{args.layer_name}'
    
    num_pixels = W.numel()
    
    if args.scaleH:
        diagH = torch.diag(H)
        diagH = torch.clamp(diagH, min=1e-8)
        scaleH = diagH.sqrt()
        W = W * scaleH[None, :]
        # W = W.T
    
    W_uint8, scale, zero_point = quantize_to_uint8(
        W, quant_type=args.quant_method, group_size=args.group_sz
    )    
    if args.handcraft_mode == "jp":
        compressed_path, bits = jp_compress(W_uint8, save_path, args.jp_quality)
        What_uint8 = jp_decompress(compressed_path)
    elif args.handcraft_mode == "bpg":
        What_uint8, bits = compress_tensor_with_bpg(W_uint8, args.bpg_quality, save_path)
    elif args.handcraft_mode == "webp":
        What_uint8, bits = compress_tensor_with_webp(W_uint8, args.webp_quality, save_path)
    else:
        raise NotImplementedError("Not implemented compression method: {}".format(args.handcraft_mode))
    
    What = dequantize_from_uint8(
        What_uint8, scale, zero_point, quant_type=args.quant_method, group_size=args.group_sz
    )
    bpp_sum = bits +  scale.numel() * 16 + zero_point.numel() * 16
    
    if args.scaleH:
        # W = W.T
        What = What / scaleH[None, :]
        bpp_sum += scaleH.numel() * 16    
    
    out = {
        'W_hat': What,
        'bpp_loss': 0,
        'bpp': bpp_sum/num_pixels,
        'bpp_loss_sum': 0,
        'bpp_sum': bpp_sum,
        'num_pixels': num_pixels,
        'codes': None,
    }
    
    return out, None, None, None, None, None

def quantize_to_uint8(weight: torch.Tensor, quant_type: str = 'per_tensor', group_size: int = -1):
    device = weight.device
    num_bits = 8
    qmin, qmax = 0, 255

    if quant_type == 'per_tensor':
        real_max, real_min = weight.max(), weight.min()
        scale = (real_max - real_min) / (qmax - qmin)
        if scale == 0:
            scale = torch.tensor(1.0, device=device)
        zero_point = torch.round(qmin - real_min / scale)

        scale = scale.to(torch.float16)
        zero_point = zero_point.to(torch.float16)
        
        quant = torch.round(weight / scale + zero_point)
        quant = torch.clamp(quant, qmin, qmax).to(torch.uint8)
        
        return quant, scale, zero_point

    elif quant_type == 'per_channel':
        real_max = weight.amax(dim=1, keepdim=True)
        real_min = weight.amin(dim=1, keepdim=True)
        scale = (real_max - real_min) / (qmax - qmin)
        scale[scale == 0] = 1.0
        zero_point = torch.round(qmin - real_min / scale)

        scale = scale.to(torch.float16)
        zero_point = zero_point.to(torch.float16)

        quant = torch.round(weight / scale + zero_point)
        quant = torch.clamp(quant, qmin, qmax).to(torch.uint8)
        
        return quant, scale, zero_point

    elif quant_type == 'group':
        if not (group_size > 0 and weight.dim() == 2):
            raise ValueError("Group quantization requires a 2D weight tensor and a positive group_size.")
        
        out_dim, in_dim = weight.shape
        quant = torch.empty_like(weight, dtype=torch.uint8, device=device)
        scales, zero_points = [], []

        for i in range(0, in_dim, group_size):
            group_slice = weight[:, i:i + group_size]
            g_max, g_min = group_slice.max(), group_slice.min()
            s = (g_max - g_min) / (qmax - qmin)
            if s == 0:
                s = torch.tensor(1.0, device=device)
            z = torch.round(qmin - g_min / s)

            s = s.to(torch.float16)
            z = z.to(torch.float16)
            
            q_slice = torch.round(group_slice / s + z)
            quant[:, i:i + group_size] = torch.clamp(q_slice, qmin, qmax).to(torch.uint8)
            
            scales.append(s)
            zero_points.append(z)
            
        return quant, torch.stack(scales), torch.stack(zero_points)
    
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")

def dequantize_from_uint8(
    quant_tensor: torch.Tensor, 
    scale: torch.Tensor, 
    zero_point: torch.Tensor,
    quant_type: str,
    group_size: int = -1
):
    """
    uint8 텐서와 scale, zero_point를 사용하여 Float 텐서로 역양자화합니다.
    """
    quant_tensor_float = quant_tensor.to(torch.float32)

    if quant_type in ['per_tensor', 'per_channel']:
        return (quant_tensor_float - zero_point) * scale

    elif quant_type == 'group':
        if not (group_size > 0 and quant_tensor.dim() == 2):
            raise ValueError("Group dequantization requires a 2D tensor and a positive group_size.")
        
        out_dim, in_dim = quant_tensor.shape
        dequant = torch.empty_like(quant_tensor, dtype=torch.float32, device=quant_tensor.device)
        
        num_groups = scale.numel()
        for i in range(num_groups):
            start_col = i * group_size
            end_col = start_col + group_size
            
            # 각 그룹에 맞는 scale과 zero_point 적용
            s = scale[i]
            z = zero_point[i]
            dequant[:, start_col:end_col] = (quant_tensor_float[:, start_col:end_col] - z) * s
            
        return dequant
        
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")


# def jp_compress(tensor_uint8, output_p, quality, verbose=False):
#     img = Image.fromarray(tensor_uint8.cpu().numpy().astype(np.uint8))
#     out_path = f"{output_p}.jpg"    
    
#     img.save(out_path, quality=quality)
#     # dim = float(img.size[0] * img.size[1])
#     bits = (8 * _jpeg_content_length(out_path))
#     return out_path, bits
    
# def jp_decompress(jpeg_path: str):
#     if not os.path.exists(jpeg_path):
#         raise FileNotFoundError(f"파일을 찾을 수 없습니다: {jpeg_path}")

#     img = Image.open(jpeg_path)
#     decomp_numpy = np.array(img)
#     decomp_tensor = torch.from_numpy(decomp_numpy)
#     return decomp_tensor

# def _jpeg_content_length(p):
#     """
#     Determines the length of the content of the JPEG file stored at `p` in bytes, i.e., size of the file without the
#     header. Note: Note sure if this works for all JPEGs...
#     :param p: path to a JPEG file
#     :return: length of content
#     """
#     with open(p, "rb") as f:
#         last_byte = ""
#         header_end_i = None
#         for i in itertools.count():
#             current_byte = f.read(1)
#             if current_byte == b"":
#                 break
#             # some files somehow contain multiple FF DA sequences, don't know what that means
#             if header_end_i is None and last_byte == b"\xff" and current_byte == b"\xda":
#                 header_end_i = i
#             last_byte = current_byte
#         # at this point, i is equal to the size of the file
#         return i - header_end_i - 2  # minus 2 because all JPEG files end in FF D0