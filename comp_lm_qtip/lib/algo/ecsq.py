import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import constriction
from lib import utils
import glog
# from nvidia import nvcomp
# import cupy as cp

def binary_search_qs(
    Z,
    target_rate,
    low=0.001,
    high=10.0,
    num_iters=15,
    max_sample_size=50000,
):
    data = Z.flatten()

    best_step = 1.0
    if data.numel() < max_sample_size:
        sample_for_ent = data
    else:
        indices = torch.randint(0, data.numel(), (max_sample_size,), device=data.device)
        sample_for_ent = data[indices]

    for _ in range(num_iters):
        mid = (low + high) / 2
        q_vals_sample = torch.round(sample_for_ent / mid)
        h = calc_entropy_torch(q_vals_sample).item()
        if h > target_rate:
            low = mid
        else:
            high = mid
            best_step = mid

    q_vals_full = torch.round(data / best_step)
    ecsq_recon = q_vals_full * best_step
    ecsq_mse = torch.mean((data - ecsq_recon) ** 2).item()
    theory_rate = calc_entropy_torch(q_vals_full).item()

    return best_step, theory_rate, ecsq_mse

def uniform_ecsq_gpu(W, H, args, device = 'cpu'):
    
    W = W.to(device)
    H = H.to(device)
    (m, n) = W.shape
    Wr, Hr, metadata = utils.standardize_W(W, H, args, device)
    
    if not hasattr(args, "R_target") or args.R_target is None:
        raise ValueError("uniform_ecsq_gpu requires args.R_target")

    data = Wr.flatten()
    best_step, theory_rate, ecsq_mse = binary_search_qs(
        Wr, target_rate=float(args.R_target)
    )
    q_vals_full = torch.round(data / best_step)
    ecsq_recon = q_vals_full * best_step

    rans_rate = run_actual_rans(data.detach().cpu().numpy(), best_step)
    # rans_rate = run_actual_rans_gpu(data, best_step, device) ## rate가 안 맞음 왜?
    
    hatWr = ecsq_recon.reshape(m, n)
    
    total_metadata_bits = utils.calculate_metadata_bpp(metadata, W.shape, args)
    bpp_loss_sum = rans_rate*m*n + total_metadata_bits
    bpp_sum = total_metadata_bits
    
    glog.info(f'Target R: {args.R_target:<10.5f}, ANS rate: {rans_rate:<10.5f}, Theoritical rate: {theory_rate:<10.5f}')
    
    out = {
        'hatWr': hatWr.to(device),
        'bpp_loss': rans_rate,
        'bpp': 0,
        'bpp_loss_sum': bpp_loss_sum,
        'bpp_sum': bpp_sum,
        'bpp_target': theory_rate,
        'mse_normed': ecsq_mse,
        'metadata': metadata,
        'num_pixels': m*n,
        'codes': None,
    }
    return out


def calc_entropy(labels):
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-10))

def calc_entropy_torch(labels):
    unique, counts = torch.unique(labels, return_counts=True)
    probs = counts.float() / labels.numel()
    return -torch.sum(probs * torch.log2(probs + 1e-10))

def uniform_quantization(x, step_size):
    if step_size < 1e-6: return x 
    return np.round(x / step_size) * step_size

def run_actual_rans(data_np, step_size):
    q_indices = np.round(data_np / step_size).astype(np.int32)
    
    min_val = q_indices.min()
    symbols = q_indices - min_val # 0, 1, 2, ... 로 변환
    symbols = symbols.astype(np.int32) # constriction은 int32 요구
    
    unique, counts = np.unique(symbols, return_counts=True)
    
    max_sym = symbols.max()
    freqs = np.zeros(max_sym + 1, dtype=np.float64)
    freqs[unique] = counts
    
    probs = freqs / freqs.sum()
    
    model = constriction.stream.model.Categorical(probs)
    
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(symbols, model)
    
    compressed_data = coder.get_compressed() 
    total_bits = len(compressed_data) * 32 
    
    actual_bpp = total_bits / len(data_np)
    return actual_bpp

def run_actual_rans_gpu(data, step_size, device, chunk_size=262144, bitstream_kind=None, checksum_policy=None):
    if step_size < 1e-6:
        return 0.0

    if not data.is_cuda:
        data = data.to(device)

    q_indices = torch.round(data / step_size).to(torch.int32).contiguous()

    # try:
    #     import cupy as cp
    # except Exception as exc:
    #     raise RuntimeError("cupy is required for nvcomp ANS encoding") from exc
    # try:
    #     from nvidia import nvcomp
    # except Exception as exc:
    #     raise RuntimeError("nvcomp is required for ANS encoding") from exc

    if hasattr(cp, "fromDlpack"):
        q_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(q_indices))
    else:
        q_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(q_indices))

    nvarr_d = nvcomp.as_array(q_cp)

    codec_kwargs = {"algorithm": "ANS", "chunk_size": chunk_size}
    if bitstream_kind is not None:
        codec_kwargs["bitstream_kind"] = bitstream_kind
    else:
       codec_kwargs["bitstream_kind"] = nvcomp.BitstreamKind.NVCOMP_NATIVE
    if checksum_policy is not None:
        codec_kwargs["checksum_policy"] = checksum_policy
    else:
        codec_kwargs["checksum_policy"] = nvcomp.ChecksumPolicy.COMPUTE_AND_VERIFY

    codec = nvcomp.Codec(**codec_kwargs)
    comp_arr = codec.encode(nvarr_d)
    total_bits = comp_arr.buffer_size * 8
    return total_bits / q_indices.numel()
    

# def uniform_ecsq(W, H, args, device = 'cpu'):
    
#     W = W.to(device)
#     H = H.to(device)
#     (m, n) = W.shape
#     Wr, Hr, metadata = utils.standardize_W(W, H, args, device)
    
#     data_np = Wr.flatten().detach().cpu().numpy()
    
#     low, high = 0.001, 10.0
#     best_step = 1.0
#     if len(data_np) < 50000:
#         sample_for_ent = data_np
#     else:
#         indices = np.random.choice(len(data_np), size=50000, replace=False)
#         sample_for_ent = data_np[indices]
    
#     for _ in range(15): 
#         mid = (low + high) / 2
#         q_vals = np.round(sample_for_ent / mid)
#         h = calc_entropy(q_vals)
#         if h > args.R_target: low = mid
#         else: high = mid; best_step = mid
    
#     ecsq_recon = uniform_quantization(data_np, best_step)
#     ecsq_mse = np.mean((data_np - ecsq_recon)**2)
    
#     q_vals_full = np.round(data_np / best_step)
#     theory_rate = calc_entropy(q_vals_full)
    
#     rans_rate = run_actual_rans(data_np, best_step)
    
#     hatWr = torch.from_numpy(ecsq_recon).reshape(m, n)
    
#     total_metadata_bits = utils.calculate_metadata_bpp(metadata, W.shape, args)
#     bpp_loss_sum = rans_rate*m*n + total_metadata_bits
#     bpp_sum = total_metadata_bits
    
#     glog.info(f'Target R: {args.R_target:<10.5f}, ANS rate: {rans_rate:<10.5f}, Theoritical rate: {theory_rate:<10.5f}')
    
#     out = {
#         'hatWr': hatWr.to(device),
#         'bpp_loss': rans_rate,
#         'bpp': 0,
#         'bpp_loss_sum': bpp_loss_sum,
#         'bpp_sum': bpp_sum,
#         'bpp_target': theory_rate,
#         'mse_normed': ecsq_mse,
#         'metadata': metadata,
#         'num_pixels': m*n,
#         'codes': None,
#     }
#     return out
