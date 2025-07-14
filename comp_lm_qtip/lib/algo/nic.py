import torch
from math import ceil
import torch.nn.functional as F
import math
from neuralcompression.metrics import (
    MultiscaleStructuralSimilarity,
    calc_psnr,
    pickle_size_of,
    update_patch_fid,
)

def compress_linear(W, comp_model, args, patch_size=512, device='cpu'):
    """
    Compress a 2D weight matrix W using a NIC model by splitting into patches.
    Returns reconstructed weights and bits-per-pixel metrics.
    """
    comp_model = comp_model.to(device)
    if args.nic_model == 'illm':
        comp_model.eval()
        comp_model.update()
        comp_model.update_tensor_devices("compress")

    W = W.to(device)
    quant, scale, zp = normalize(W, quant_type=args.quant_method, group_size=args.group_sz)

    h, w = quant.shape
    num_patches_h = ceil(h / patch_size)
    num_patches_w = ceil(w / patch_size)

    # Prepare output and bit counter
    recon = torch.zeros_like(quant, dtype=torch.float32, device=device)
    total_bits = 0

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Extract patch slice
            h_start = i * patch_size
            w_start = j * patch_size
            h_end = min(h_start + patch_size, h)
            w_end = min(w_start + patch_size, w)
            patch = quant[h_start:h_end, w_start:w_end]

            p = patch.unsqueeze(0).unsqueeze(0).float()  # [1,1,h_p,w_p]
            p_pad, padding = pad(p, patch_size)
            p3 = p_pad.repeat(1, 3, 1, 1)  # [1,3,patch_size,patch_size]

            if args.nic_model == 'tcm':
                out_enc = comp_model.compress(p3)
                out_dec = comp_model.decompress(out_enc["strings"], out_enc["shape"])
                
                rec1 = out_dec["x_hat"][:, 0:1, :, :]

                total_bits += sum(len(s[0]) for s in out_enc["strings"]) * 8.0
            
            elif args.nic_model == 'ftic':
                out = comp_model(p3)
                rec1 = out["x_hat"][:, 0:1, :, :]
                total_bits += sum(
                    (torch.log(likelihoods).sum() / (-math.log(2)))
                    for likelihoods in out["likelihoods"].values()
                ).item()
            elif args.nic_model == 'illm':
                with torch.no_grad():
                    compressed = comp_model.compress(p3, force_cpu=False)
                    decompressed = comp_model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)
                    rec1 = decompressed[:, 0:1, :, :]
                    
                    num_bytes = pickle_size_of(compressed)
                    total_bits += num_bytes * 8

            rec_crop = crop(rec1, padding)  # [1,1,h_p,w_p]        
            recon[h_start:h_end, w_start:w_end] = rec_crop.squeeze(0).squeeze(0)

    # Denormalize
    W_hat = denormalize(recon, scale, zp, quant_type=args.quant_method, group_size=args.group_sz).cpu()

    num_pixels = W.numel()
    bpp_sum = total_bits +  scale.numel() * 16 + zp.numel() * 16
    bpp = bpp_sum / num_pixels
    
    out = {
        'W_hat': W_hat,
        'bpp_loss': 0,
        'bpp': bpp,
        'bpp_loss_sum': 0,
        'bpp_sum': bpp_sum,
        'num_pixels': num_pixels,
        'codes': None,
    }
    
    return out, None, None, None, None, None


# 기존 compress_linear 함수에 업스케일링/다운스케일링 로직 추가
# def compress_linear(W, comp_model, args, patch_size=1024, device='cpu', upscale_factor=4):
#     """
#     W를 업스케일링하여 압축하고, 복원 후 다운스케일링합니다.
#     """
#     comp_model = comp_model.to(device)
#     W = W.to(device)

#     # 1. 원본 W의 크기 저장 및 업스케일링
#     h_orig, w_orig = W.shape
    
#     # row_norms = torch.linalg.norm(W, dim=1)
#     # sorted_row_indices = torch.argsort(row_norms)
#     # W_sorted = W[sorted_row_indices, :]

#     # col_norms = torch.linalg.norm(W_sorted, dim=0)
#     # sorted_col_indices = torch.argsort(col_norms)
#     # W_sorted = W_sorted[:, sorted_col_indices]
    
#     # F.interpolate를 위해 4D 텐서로 변경 [N, C, H, W]
#     W_reshaped = W.unsqueeze(0).unsqueeze(0)
#     W_upscaled = F.interpolate(W_reshaped, scale_factor=upscale_factor, mode='bicubic', align_corners=False)
#     W_upscaled = W_upscaled.squeeze(0).squeeze(0)

#     # 업스케일링된 W를 양자화
#     quant, scale, zp = normalize_to_uint8(W_upscaled, quant_type=args.quant_method, group_size=args.group_sz)

#     h, w = quant.shape
#     num_patches_h = ceil(h / patch_size)
#     num_patches_w = ceil(w / patch_size)

#     recon = torch.zeros_like(quant, dtype=torch.float32, device=device)
#     total_bits = 0

#     # 압축/복원 로직은 동일
#     for i in range(num_patches_h):
#         for j in range(num_patches_w):
#             h_start, w_start = i * patch_size, j * patch_size
#             h_end, w_end = min(h_start + patch_size, h), min(w_start + patch_size, w)
#             patch = quant[h_start:h_end, w_start:w_end]

#             p = patch.unsqueeze(0).unsqueeze(0).float()
#             p_pad, padding = pad(p, patch_size)
#             p3 = p_pad.repeat(1, 3, 1, 1)

#             out_enc = comp_model.compress(p3)
#             out_dec = comp_model.decompress(out_enc["strings"], out_enc["shape"])
            
#             rec1 = out_dec["x_hat"][:, 0:1, :, :]
#             rec_crop = crop(rec1, padding)

#             total_bits += sum(len(s[0]) for s in out_enc["strings"]) * 8.0
#             recon[h_start:h_end, w_start:w_end] = rec_crop.squeeze(0).squeeze(0)

#     # 역양자화
#     W_hat_upscaled = denormalize_from_uint8(recon, scale, zp, quant_type=args.quant_method, group_size=args.group_sz)

#     # 2. 복원된 W_hat을 다시 원본 크기로 다운스케일링
#     W_hat_reshaped = W_hat_upscaled.unsqueeze(0).unsqueeze(0)
#     W_hat_downscaled = F.interpolate(W_hat_reshaped, size=(h_orig, w_orig), mode='bicubic', align_corners=False)
#     W_hat = W_hat_downscaled.squeeze(0).squeeze(0).cpu()

#     # 2. 원래 순서로 되돌리기 위한 역-정렬(unsort) 인덱스 생성
#     # unsort_row_indices = torch.argsort(sorted_row_indices)
#     # unsort_col_indices = torch.argsort(sorted_col_indices)
#     # W_hat = W_hat[:, unsort_col_indices]
#     # W_hat = W_hat[unsort_row_indices, :].cpu()

#     # BPP 계산 시 num_pixels는 원본 W 기준이어야 함
#     num_pixels = W.numel()
#     # bits_for_indices = (h_orig + w_orig) * 16 
#     bits_for_indices = 0
#     bpp_sum = total_bits + scale.numel() * 16 + zp.numel() * 16 + bits_for_indices
#     bpp = bpp_sum / num_pixels
    
#     out = {
#         'W_hat': W_hat,
#         'bpp_loss': 0,
#         'bpp': bpp,
#         'bpp_loss_sum': 0,
#         'bpp_sum': bpp_sum,
#         'num_pixels': num_pixels,
#         'codes': None,
#     }
    
#     return out, None, None, None, None, None

# def compress_linear(W, comp_model, args, patch_size=512, device='cpu'):
#     """
#     Compress a 2D weight matrix W using a NIC model.
#     Each patch is normalized to match ImageNet statistics before compression.
#     """
#     comp_model = comp_model.to(device)
#     W = W.to(device)

#     # ImageNet statistics for normalization (mean and std for R, G, B channels)
#     # These will be the target statistics for each patch before compression.
#     imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
#     imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

#     h, w = W.shape
#     num_patches_h = ceil(h / patch_size)
#     num_patches_w = ceil(w / patch_size)

#     # Prepare output and bit counter
#     recon = torch.zeros_like(W, dtype=torch.float32, device=device)
#     total_bits = 0

#     for i in range(num_patches_h):
#         for j in range(num_patches_w):
#             # Extract patch slice
#             h_start = i * patch_size
#             w_start = j * patch_size
#             h_end = min(h_start + patch_size, h)
#             w_end = min(w_start + patch_size, w)
#             patch = W[h_start:h_end, w_start:w_end]

#             # --- Per-Patch Normalization ---
#             patch_mu = patch.mean()
#             patch_std = patch.std()

#             # Add bits for storing patch_mu and patch_std as 16-bit floats
#             total_bits += 2 * 16

#             # Normalize patch to have zero mean and unit variance
#             # Add a small epsilon to std to prevent division by zero
#             if patch_std > 1e-8:
#                 normalized_patch = (patch - patch_mu) / patch_std
#             else:
#                 normalized_patch = patch - patch_mu # If std is zero, just center it

#             # Prepare the patch for the compression model (pad and repeat channels)
#             p = normalized_patch.unsqueeze(0).unsqueeze(0)  # [1, 1, h_p, w_p]
#             p_pad, padding = pad(p, patch_size)
#             p3 = p_pad.repeat(1, 3, 1, 1)  # [1, 3, patch_size, patch_size]

#             # Scale to ImageNet statistics
#             p3_imagenet = p3 * imagenet_std + imagenet_mean

#             # --- Compression and Decompression ---
#             out_enc = comp_model.compress(p3_imagenet)
#             out_dec = comp_model.decompress(out_enc["strings"], out_enc["shape"])
            
#             # Add compressed bits
#             total_bits += sum(len(s[0]) for s in out_enc["strings"]) * 8.0

#             # --- Per-Patch Denormalization ---
#             rec_hat_imagenet = out_dec["x_hat"] # [1, 3, patch_size, patch_size]

#             # Reverse ImageNet normalization
#             rec_hat_normalized = (rec_hat_imagenet - imagenet_mean) / imagenet_std
            
#             # Average the 3 channels to get a single-channel image
#             rec_hat_single_channel = rec_hat_normalized.mean(dim=1, keepdim=True)
            
#             # Crop to original patch size
#             rec_crop = crop(rec_hat_single_channel, padding)  # [1, 1, h_p, w_p]

#             # Reverse the initial normalization (denormalize)
#             if patch_std > 1e-8:
#                 rec_final_patch = rec_crop * patch_std + patch_mu
#             else:
#                 rec_final_patch = rec_crop + patch_mu

#             recon[h_start:h_end, w_start:w_end] = rec_final_patch.squeeze(0).squeeze(0)

#     W_hat = recon.cpu()
#     num_pixels = W.numel()
#     bpp = total_bits / num_pixels
    
#     out = {
#         'W_hat': W_hat,
#         'bpp': bpp,
#         'bpp_sum': total_bits,
#         'num_pixels': num_pixels,
#         'bpp_loss': 0, # Placeholder, can be filled if needed
#         'bpp_loss_sum': 0, # Placeholder
#         'codes': None, # Placeholder
#     }
#     return out, None, None, None, None, None

# def compress_linear(W, comp_model, args, device='cpu'):
#     """
#     Compress a 2D weight matrix W using a NIC model with decoupled normalization.
#     1. First, normalizes W in blocks of 'norm_patch_size' and stores the factors.
#     2. Then, compresses the normalized W in blocks of 'patch_size'.
#     """
#     comp_model = comp_model.to(device)
#     W = W.to(device)
#     h, w = W.shape

#     # --- 1. Pre-computation: Block-wise Normalization ---

#     # Calculate grid dimensions for normalization patches
#     patch_size = args.nic_patch_size
#     norm_patch_size = args.nic_norm_patch_size
    
#     num_norm_patches_h = ceil(h / norm_patch_size)
#     num_norm_patches_w = ceil(w / norm_patch_size)

#     # Tensors to store the normalized version of W and the normalization factors
#     W_normalized = torch.zeros_like(W)
#     mus = torch.zeros(num_norm_patches_h, num_norm_patches_w, device=device)
#     stds = torch.zeros(num_norm_patches_h, num_norm_patches_w, device=device)

#     for i in range(num_norm_patches_h):
#         for j in range(num_norm_patches_w):
#             h_start = i * norm_patch_size
#             w_start = j * norm_patch_size
#             h_end = min(h_start + norm_patch_size, h)
#             w_end = min(w_start + norm_patch_size, w)
            
#             patch = W[h_start:h_end, w_start:w_end]
            
#             mu = patch.mean()
#             std = patch.std()

#             # Store factors
#             mus[i, j] = mu
#             stds[i, j] = std

#             # Normalize and store in the new tensor
#             if std > 1e-8:
#                 W_normalized[h_start:h_end, w_start:w_end] = (patch - mu) / std
#             else:
#                 W_normalized[h_start:h_end, w_start:w_end] = patch - mu

#     # Calculate bits required to store all normalization factors (as FP16)
#     total_bits = (mus.numel() + stds.numel()) * 16.0

#     # --- 2. Main Loop: Compression and Decompression ---

#     # ImageNet statistics for scaling before compression
#     imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
#     imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

#     # Calculate grid dimensions for compression patches
#     num_comp_patches_h = ceil(h / patch_size)
#     num_comp_patches_w = ceil(w / patch_size)
    
#     recon = torch.zeros_like(W, dtype=torch.float32, device=device)

#     for i in range(num_comp_patches_h):
#         for j in range(num_comp_patches_w):
#             h_start = i * patch_size
#             w_start = j * patch_size
#             h_end = min(h_start + patch_size, h)
#             w_end = min(w_start + patch_size, w)

#             # Extract patch from the pre-normalized matrix
#             normalized_patch = W_normalized[h_start:h_end, w_start:w_end]
            
#             # Prepare for compression model
#             p = normalized_patch.unsqueeze(0).unsqueeze(0)
#             p_pad, padding = pad(p, patch_size)
#             p3 = p_pad.repeat(1, 3, 1, 1)
#             p3_imagenet = p3 * imagenet_std + imagenet_mean

#             # Compress, decompress, and count bits
#             out_enc = comp_model.compress(p3_imagenet)
#             out_dec = comp_model.decompress(out_enc["strings"], out_enc["shape"])
#             total_bits += sum(len(s[0]) for s in out_enc["strings"]) * 8.0

#             # Reverse ImageNet scaling
#             rec_hat_imagenet = out_dec["x_hat"]
#             rec_hat_normalized = (rec_hat_imagenet - imagenet_mean) / imagenet_std
#             rec_hat_single_channel = rec_hat_normalized.mean(dim=1, keepdim=True)
#             rec_crop = crop(rec_hat_single_channel, padding)

#             # --- Denormalization using stored factors ---
#             # Find which normalization blocks this compression patch overlaps with
#             i_norm_start = h_start // norm_patch_size
#             j_norm_start = w_start // norm_patch_size
#             i_norm_end = (h_end - 1) // norm_patch_size
#             j_norm_end = (w_end - 1) // norm_patch_size
            
#             # Get the average mu and std over the patch's area
#             relevant_mus = mus[i_norm_start:i_norm_end+1, j_norm_start:j_norm_end+1]
#             relevant_stds = stds[i_norm_start:i_norm_end+1, j_norm_start:j_norm_end+1]
            
#             avg_mu = relevant_mus.mean()
#             avg_std = relevant_stds.mean()

#             # Denormalize the entire reconstructed patch with the average factors
#             if avg_std > 1e-8:
#                 rec_final_patch = rec_crop * avg_std + avg_mu
#             else:
#                 rec_final_patch = rec_crop + avg_mu

#             recon[h_start:h_end, w_start:w_end] = rec_final_patch.squeeze(0).squeeze(0)

#     W_hat = recon.cpu()
#     num_pixels = W.numel()
#     bpp = total_bits / num_pixels
    
#     out = {
#         'W_hat': W_hat,
#         'bpp': bpp,
#         'bpp_sum': total_bits,
#         'num_pixels': num_pixels,
#         'bpp_loss': 0, 'bpp_loss_sum': 0, 'codes': None
#     }
#     return out, None, None, None, None, None

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


def normalize(weight: torch.Tensor, quant_type: str = 'per_tensor', group_size: int = -1):
    device = weight.device
    qmin, qmax = 0, 1

    if quant_type == 'per_tensor':
        real_max, real_min = weight.max(), weight.min()
        scale = (real_max - real_min) / (qmax - qmin)
        if scale == 0:
            scale = torch.tensor(1.0, device=device)
        zero_point = qmin - real_min / scale

        scale = scale.to(torch.float16)
        zero_point = zero_point.to(torch.float16)
        
        quant = weight / scale + zero_point
        quant = torch.clamp(quant, qmin, qmax)
        
        return quant, scale, zero_point

    elif quant_type == 'per_channel':
        real_max = weight.amax(dim=1, keepdim=True)
        real_min = weight.amin(dim=1, keepdim=True)
        scale = (real_max - real_min) / (qmax - qmin)
        scale[scale == 0] = 1.0
        zero_point = qmin - real_min / scale

        scale = scale.to(torch.float16)
        zero_point = zero_point.to(torch.float16)

        quant = weight / scale + zero_point
        quant = torch.clamp(quant, qmin, qmax)
        
        return quant, scale, zero_point

    elif quant_type == 'group':
        if not (group_size > 0 and weight.dim() == 2):
            raise ValueError("Group quantization requires a 2D weight tensor and a positive group_size.")
        
        out_dim, in_dim = weight.shape
        quant = torch.empty_like(weight, device=device)
        scales, zero_points = [], []

        for i in range(0, in_dim, group_size):
            group_slice = weight[:, i:i + group_size]
            g_max, g_min = group_slice.max(), group_slice.min()
            s = (g_max - g_min) / (qmax - qmin)
            if s == 0:
                s = torch.tensor(1.0, device=device)
            z = qmin - g_min / s

            s = s.to(torch.float16)
            z = z.to(torch.float16)
            
            q_slice = group_slice / s + z
            quant[:, i:i + group_size] = torch.clamp(q_slice, qmin, qmax)
            
            scales.append(s)
            zero_points.append(z)
            
        return quant, torch.stack(scales), torch.stack(zero_points)
    
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")

def denormalize(
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