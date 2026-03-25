import copy
import math
from contextlib import contextmanager
from operator import attrgetter

import glog
import torch
from torch import multiprocessing as mp
from torch import nn
from transformers import AutoModelForCausalLM
import argparse
import numpy as np
import sys
import os
import time ## ryu
import gc

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', 'qtip'))

if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added to path: {project_root}")

try:
    from lib import codebook, utils
    # from lib.linear.quantized_linear_exp import QuantizedLinear
    from lib.linear.quantized_linear import QuantizedLinear
    # from lib.codebook import bitshift_exp
    from lib.codebook import bitshift
    from lib.algo import ldlq   
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    print("Check if the path exists:", os.path.exists(project_root))


def qtip_quantize(quip_params, W, HR, device):    
    rcp = 'col'
    orig_dtype = torch.float32
    dtype_ = torch.float32
    split_for_tp = False
    tp_rank = 8
    scale_override = -1
    
    td_x = quip_params['td_x']
    td_y = quip_params['td_y']
    L = quip_params['L']
    K = quip_params['K']
    V = quip_params['V']
    tlut_bits = quip_params['tlut_bits']
    decode_mode = quip_params['decode_mode']
    
    # cb = bitshift_exp.bitshift_codebook(L=L,
    #                             K=K,
    #                             V=V,
    #                             tlut_bits=tlut_bits,
    #                             decode_mode=decode_mode)
    
    cb = bitshift.bitshift_codebook(L=L,
                                K=K,
                                V=V,
                                tlut_bits=tlut_bits,
                                decode_mode=decode_mode)
    
    has_kernel = utils.has_kernel(decode_mode, L, K, V,
                tlut_bits, td_x, td_y)
    
    cb = cb.to(device).to(orig_dtype)    
    # use_bias = (orig_linear.bias != None) ## ryu
    # bias = orig_linear.bias ## ryu
    # del orig_linear
    (m, n) = W.shape
    SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
    SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)

    # in_hess_path = f'{in_hess_path}/{idx}_{in_hess_name}.pt'
    # # in_hess_path = f'{in_hess_path}/lang_{idx}_{in_hess_name}.pt'
    # H_data = torch.load(in_hess_path, map_location=torch.device('cpu'))
    # HR = utils.flat_to_sym(H_data['flatH'], H_data['n'])
    # if 'mu' in H_data:
    #     mu = H_data['mu']
    #     HR += mu[None, :] * mu[:, None]
    #     del mu
    # del H_data

    # HR = utils.regularize_H(HR, sigma_reg)

    W = W.to(device)
    HR = HR.to(device)
    torch.cuda.synchronize()
    gc.collect()
    gc.disable()  # GC 비활성화

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()

    if split_for_tp:
        pass
        # if rcp == 'col':
        #     # split along output dimension
        #     Wr = utils.matmul_hadUt(
        #         utils.matmul_hadUt((W.T.to(device) * SV).reshape(
        #             n * tp_rank, m // tp_rank)).reshape(
        #                 W.T.shape).T * SU)
        #     HRr = utils.matmul_hadUt(
        #         utils.matmul_hadUt(HR.to(device) * SU).T * SU)

        #     Wscale = Wr.reshape(
        #         tp_rank, m * n // tp_rank).square().mean(
        #             dim=-1).sqrt() / (cb.lut.to(
        #                 torch.float64).square().mean().sqrt().float() *
        #                                 scale_override)
        #     Wr = Wr.reshape(tp_rank,
        #                     m * n // tp_rank) / Wscale.unsqueeze(-1)
        #     Wr = Wr.reshape(m, n)

        # elif rcp == 'row':
        #     # split along input dimension
        #     Wr = utils.matmul_hadUt(
        #         (utils.matmul_hadUt(W.T.to(device) * SV).T * SU).reshape(
        #             m * tp_rank, n // tp_rank)).reshape(W.shape)
        #     HRr = utils.matmul_hadUt(
        #         (utils.matmul_hadUt((HR.to(device) * SU).reshape(
        #             n * tp_rank, n // tp_rank)).reshape(n, n).T *
        #             SU).reshape(n * tp_rank,
        #                         n // tp_rank)).reshape(n, n)
        #     Wscale = Wr.reshape(
        #         m, tp_rank,
        #         n // tp_rank).transpose(0, 1).reshape(
        #             tp_rank, m * n // tp_rank).square().mean(
        #                 dim=-1).sqrt() / (cb.lut.to(
        #                     torch.float64).square().mean().sqrt().float() *
        #                                     scale_override)
        #     Wr = Wr.reshape(m, tp_rank, n // tp_rank).transpose(
        #         0, 1).reshape(tp_rank,
        #                         m * n // tp_rank) / Wscale.unsqueeze(-1)
        #     Wr = Wr.reshape(tp_rank, m,
        #                     n // tp_rank).transpose(0,
        #                                                     1).reshape(m, n)
    else:
        Wr = utils.matmul_hadUt(
            utils.matmul_hadUt(W.T.to(device) * SV).T * SU)
        HRr = utils.matmul_hadUt(
            utils.matmul_hadUt(HR.to(device) * SU).T * SU)
        
        Wscale = Wr.square().mean().sqrt() / (
            cb.lut.to(torch.float64).square().mean().sqrt().float() *
            scale_override)
        Wr /= Wscale

    LRr, _ = utils.block_LDL(HRr, td_y)
    diag = torch.arange(n, device=LRr.device)
    LRr[diag, diag] = 0

    args = argparse.Namespace(**quip_params)
    hatWr, Qidxs = ldlq.LDLQ(Wr, LRr, cb, args, for_kernel=has_kernel)

    Qidxs = Qidxs.cpu()
    packed = cb.pack_trellis(
        Qidxs.reshape(m // td_x, td_x, n // td_y,
                        td_y // V).transpose(1, 2).reshape(
                            -1, td_x * td_y // V))

    if has_kernel:
        packed = packed.view(torch.uint8).view(-1, 2).flip(
            (-1, )).reshape(m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8,
                            K).permute(0, 2, 4, 3, 1, 5).flip(
                                (-1, )).contiguous().flatten().view(
                                    torch.int16).reshape(packed.shape)
    else:
        packed = packed.view(torch.int16)

    end_event.record()
    torch.cuda.synchronize()
    gc.enable() # GC 다시 활성화
    elapsed_time_ms = start_event.elapsed_time(end_event)
    elapsed_time = elapsed_time_ms / 1000.0
   
    glog.info(f"Total encoding Time: {elapsed_time_ms*1000:.4f} ms")

    if rcp == 'col':
        Wr = (Wr.reshape(tp_rank, m * n // tp_rank) *
                Wscale.unsqueeze(-1)).reshape(m, n)
        hatWr = (hatWr.reshape(tp_rank, m * n // tp_rank) *
                    Wscale.unsqueeze(-1)).reshape(m, n)
    elif rcp == 'row':
        Wr = Wr.reshape(m, tp_rank, n // tp_rank).transpose(
            0, 1).reshape(tp_rank, -1) * Wscale.unsqueeze(-1)
        Wr = Wr.reshape(tp_rank, m,
                        n // tp_rank).transpose(0, 1).reshape(m, n)
        hatWr = hatWr.reshape(m, tp_rank,
                                n // tp_rank).transpose(0, 1).reshape(
                                    tp_rank, -1) * Wscale.unsqueeze(-1)
        hatWr = hatWr.reshape(tp_rank, m,
                                n // tp_rank).transpose(0, 1).reshape(
                                    m, n)
    else:
        Wr *= Wscale
        hatWr *= Wscale

    err = torch.trace(
        (Wr - hatWr) @ HRr @ (Wr - hatWr).T) / torch.trace(Wr @ HRr @ Wr.T)
    print(
        f'proxy err {err.item()} tr(WHW.T) {torch.trace(Wr @ HRr @ Wr.T)}'
    )

    save_path = f'./tmp_qtip_ckpt.pt'

    # 0 = no tensor parallelism, 1 = row parallel, 2 = column parallel
    rcp_int = 0
    if split_for_tp:
        rcp_int = 1 if rcp == 'row' else 2

    torch.save(
        {
            'trellis':
            packed.cpu(),
            'SU':
            SU.to(orig_dtype).cpu(),
            'SV':
            SV.to(orig_dtype).cpu(),
            'Wscale':
            Wscale,
            'proxy_err':
            err.item(),
            'tr(WHW.T)': torch.trace(Wr @ HRr @ Wr.T).item(),
            'mse': torch.mean((Wr - hatWr) ** 2).item(),
            'tlut':
            cb.tlut.data.to(orig_dtype).cpu()
            if hasattr(cb, 'tlut') else None,
            'rcp':
            rcp_int,
            'tp_rank':
            tp_rank,
            # 'bias':bias, ## ryu
            'time': elapsed_time_ms
        }, save_path)

    del HRr, Wr, hatWr, LRr, Qidxs, cb
    utils.clean()
    return elapsed_time_ms

def initialize_codebook(quant_layer):
    assert not hasattr(quant_layer, 'built_codebook_class') or not quant_layer.built_codebook_class
    # quant_layer.codebook_class = bitshift_exp.BitshiftLinear(
    #     quant_layer.td_x, quant_layer.td_y, quant_layer.L,
    #     quant_layer.K, quant_layer.V, quant_layer.tlut_bits,
    #     quant_layer.decode_mode, dtype=quant_layer.dtype,
    #     tlut=quant_layer.tlut, has_kernel=quant_layer.has_kernel
    # )
    quant_layer.codebook_class = bitshift.BitshiftLinear(
        quant_layer.td_x, quant_layer.td_y, quant_layer.L,
        quant_layer.K, quant_layer.V, quant_layer.tlut_bits,
        quant_layer.decode_mode, dtype=quant_layer.dtype,
        tlut=quant_layer.tlut, has_kernel=quant_layer.has_kernel
    )
    rcp = quant_layer.rcp.item()
    del quant_layer.rcp
    quant_layer.rcp = rcp
    quant_layer.built_codebook_class = True

def get_What(quip_params, orig_layer_weight, saved_layer_data, layer_name, device):
    """
    양자화된 데이터를 기반으로 복원된 가중치(W_hat)를 계산하여 반환합니다.
    LLaMA 수정: MoE Router용 예외 처리(td_x // 2)를 제거했습니다.
    """
    td_x = quip_params['td_x']
    td_y = quip_params['td_y']
    L = quip_params['L']
    K = quip_params['K']
    V = quip_params['V']
    tlut_bits = quip_params['tlut_bits']
    decode_mode = quip_params['decode_mode']
    
    # 임시 QuantizedLinear 생성
    quant_layer = QuantizedLinear(orig_layer_weight.shape[1],
                    orig_layer_weight.shape[0],
                    td_x, td_y, L, K, V, tlut_bits, decode_mode,
                    dtype=orig_layer_weight.dtype,
                    bias=True)
    
    quant_layer.mode = 'train-fixW'
    quant_layer.to(device) # 계산은 GPU에서 수행
    utils.unpack_quip(quant_layer, saved_layer_data)
    
    quant_layer.has_kernel = utils.has_kernel(decode_mode, L, K, V, tlut_bits, td_x, td_y)
    initialize_codebook(quant_layer)
    
    torch.cuda.synchronize()
    gc.collect()
    gc.disable()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    total_timings = None
    quant_layer.codebook_class.cache_hatW(quant_layer.trellis, quant_layer.had_left,
                                       quant_layer.had_right, quant_layer.K_left,
                                       quant_layer.K_right, len(quant_layer.SV),
                                       len(quant_layer.SU), quant_layer.rcp,
                                       quant_layer.tp_rank)

    end_event.record()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.enable()
    elapsed_time_ms = start_event.elapsed_time(end_event) 
    glog.info(f"{layer_name} Total decoding time: {elapsed_time_ms} ms")
    
    hatW = quant_layer.codebook_class.hatW
    SU = quant_layer.SU
    SV = quant_layer.SV
    scale = quant_layer.codebook_class.scale

    target_dtype = hatW.dtype     
    SV = SV.to(target_dtype)
    SU = SU.to(target_dtype)

    # torch.cuda.synchronize()
    # gc.collect()
    # gc.disable()
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()

    W_reconstructed = torch.diag(SV * scale) @ hatW @ torch.diag(SU)
    
    # end_event.record()
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    # gc.enable()
    # elapsed_time_ms = start_event.elapsed_time(end_event) 
    # glog.info(f"{layer_name} Total decoding time: {elapsed_time_ms} ms")
    
    del quant_layer
    return elapsed_time_ms, total_timings


device = 'cuda:7'
num_iter = 10        # 반복 횟수 (통계용)
warmup_iter = 5      # 워밍업 횟수 (캐싱 등 오버헤드 제거)
std_val = 0.012528750114142895
N = 4096             # Matrix Size

modes = [
    # ('1mad', 1, 0), 
    # ('3inst', 1, 0), 
    ('quantlut_sym', 2, 9), 
    # ('lut', 2, 16)
    ]

K_values = [3]

H = torch.eye(N, device=device) 
W = torch.normal(mean=0.0, std=std_val, size=(N, N), device=device)

import collections

for decode_mode, V, tlut_bits in modes:
    for K in K_values:
        quip_params = {
            'td_x' : 16,
            'td_y' : 16,
            'L' : 16,
            'K' : K,
            'V' : V,
            'tlut_bits' : tlut_bits,
            'decode_mode' : decode_mode
        }
        
        enc_times = []
        dec_times = []
        detailed_stats = collections.defaultdict(list)
        
        encoding_time = qtip_quantize(quip_params, W, H, device)
        saved = torch.load(f'./tmp_qtip_ckpt.pt', map_location='cpu', weights_only=False)

        for i in range(num_iter + warmup_iter):

            decoding_time, timings = get_What(quip_params, W, saved, layer_name=" ", device = device)

            if i >= warmup_iter:
                enc_times.append(encoding_time)
                dec_times.append(decoding_time)
                
                # for k, v in timings.items():
                #     detailed_stats[k].append(v)

        # 통계 계산
        enc_mean = np.mean(enc_times)
        enc_std = np.std(enc_times)
        dec_mean = np.mean(dec_times)
        dec_std = np.std(dec_times)
        
                
        # print(f"{'Mode':<15} | {'K':<3} | {'Enc Mean(ms)':<12} | {'Enc Std':<10} | {'Dec Mean(ms)':<12} | {'Dec Std':<10}")
        # print("-" * 80)        
        # print(f"{decode_mode:<15} | {K:<3} | {enc_mean:<12.4f} | {enc_std:<10.4f} | {dec_mean:<12.4f} | {dec_std:<10.4f}")
        # 2. 출력 (가독성을 위해 블록 형태로 출력)
        print(f"Config: Mode={decode_mode:<12} | K={K:<2} | V={V:<1} | Bits={tlut_bits}")
        print(f"-" * 60)
        print(f"{'Metric':<30} | {'Mean (ms)':<12} | {'Std (ms)':<10}")
        print(f"-" * 60)
        
        # Total Decoding Time
        print(f"{'Total Decoding (End-to-End)':<30} | {dec_mean:<12.4f} | {dec_std:<10.4f}")
        
        # Detailed Breakdown
        # 키를 정렬하여 출력 순서 고정 (예: decode -> scaling -> hadamard)
        # sorted_keys = sorted(detailed_stats.keys()) 
        # for k in sorted_keys:
        #     v_list = detailed_stats[k]
        #     k_mean = np.mean(v_list)
        #     k_std = np.std(v_list)
        #     # 들여쓰기로 하위 항목임 표시
        #     print(f"  - {k:<26} | {k_mean:<12.4f} | {k_std:<10.4f}")
            
        print(f"{'='*100}\n")