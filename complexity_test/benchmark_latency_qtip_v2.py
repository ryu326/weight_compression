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

# RTX A6000 Specs
A6000_PEAK_FP16_TFLOPS = 38.71

def measure_decoding_latency_subtraction(quip_params, orig_layer_weight, saved_layer_data, layer_name, device, num_repeats=100):
    """
    Forward Pass 시간을 측정하고, 이론적인 MatMul 시간(T_math)을 빼서
    Effective Decoding Time을 계산합니다.
    """
    td_x = quip_params['td_x']
    td_y = quip_params['td_y']
    L = quip_params['L']
    K = quip_params['K']
    V = quip_params['V']
    tlut_bits = quip_params['tlut_bits']
    decode_mode = quip_params['decode_mode']
    
    M, N_in = orig_layer_weight.shape
    BATCH_SIZE = 10  # Inference (Token Generation) 가정

    # 1. QuantizedLinear 생성 (Mode='eval'로 설정하여 Fused Kernel 유도)
    quant_layer = QuantizedLinear(N_in, M,
                    td_x, td_y, L, K, V, tlut_bits, decode_mode,
                    dtype=orig_layer_weight.dtype,
                    bias=True,
                    mode='eval') # 중요: eval 모드여야 forward 시 커널을 탐
    
    quant_layer.to(device)
    utils.unpack_quip(quant_layer, saved_layer_data)
    
    # 커널 사용 가능 여부 확인
    quant_layer.has_kernel = utils.has_kernel(decode_mode, L, K, V, tlut_bits, td_x, td_y)
    assert quant_layer.has_kernel == True
    assert quant_layer.mode == 'eval'
    
    # 2. 입력 데이터 생성 (Batch Size = 1)
    x = torch.randn(BATCH_SIZE, N_in, device=device, dtype=torch.float16)

    # 3. Warmup (커널 컴파일 및 초기화)
    # 첫 실행 시 codebook_class가 초기화되므로 반드시 Warmup 필요
    for _ in range(10):
        _ = quant_layer(x)
    torch.cuda.synchronize()
    
    # GC 및 메모리 정리
    gc.collect()
    # torch.cuda.empty_cache() # 잦은 호출은 오버헤드가 될 수 있어 생략하거나 필요시 추가

    # 4. Forward Latency 측정
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_repeats):
        _ = quant_layer(x)
    end_event.record()
    
    torch.cuda.synchronize()
    total_forward_ms = start_event.elapsed_time(end_event) / num_repeats
    
    # 5. 이론적 MatMul 시간 계산 (T_math)
    # FLOPs = 2 * M * N * K
    flops = 2 * M * BATCH_SIZE * N_in
    peak_flops_per_sec = A6000_PEAK_FP16_TFLOPS * 1e12
    
    t_math_sec = flops / peak_flops_per_sec
    t_math_ms = t_math_sec * 1000.0
    
    # 6. Effective Decoding Time 도출
    # Forward Time - Pure Math Time = Decoding + Memory Overhead
    decoding_time_ms = total_forward_ms - t_math_ms
    
    # glog.info(f"{layer_name} Forward: {total_forward_ms:.4f}ms, T_math: {t_math_ms:.6f}ms")
    
    del quant_layer
    return decoding_time_ms, total_forward_ms, t_math_ms


import collections
device = 'cuda:7'
num_iter = 50        # 측정 반복 횟수 늘림 (안정성 확보)
warmup_iter = 5
std_val = 0.012528750114142895
N = 4096             # Matrix Size

modes = [
    # ('1mad', 1, 0), 
    # ('3inst', 1, 0), 
    ('quantlut_sym', 2, 9), 
    # ('lut', 2, 16)
    ]

K_values = [4]

H = torch.eye(N, device=device) 
W = torch.normal(mean=0.0, std=std_val, size=(N, N), device=device)

print(f"{'='*100}")
print(f"Effective Decoding Latency Benchmark (Forward - T_math)")
print(f"GPU: RTX A6000 (Peak FP16: {A6000_PEAK_FP16_TFLOPS} TFLOPS)")
print(f"{'='*100}")

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
        
        # 인코딩 수행 (Checkpoint 생성용)
        # qtip_quantize는 기존 함수 그대로 사용
        encoding_time = qtip_quantize(quip_params, W, H, device)
        print('encoding_time:', encoding_time)
        saved = torch.load(f'./tmp_qtip_ckpt.pt', map_location='cpu', weights_only=False)

        # 측정 수행
        dec_latency_list = []
        forward_latency_list = []
        
        # --- [수정됨] 개별 측정값 출력 헤더 ---
        print(f"\n>> Start Measuring: Mode={decode_mode}, K={K}")
        print(f"{'Iter':<5} | {'Forward (ms)':<15} | {'T_math (ms)':<15} | {'Decoding (ms)':<15}")
        print("-" * 60)
        
        for i in range(10): # 여러 번 측정하여 평균
             dec_ms, fwd_ms, t_math_ms = measure_decoding_latency_subtraction(
                 quip_params, W, saved, layer_name="", device=device, num_repeats=100
             )
             dec_latency_list.append(dec_ms)
             forward_latency_list.append(fwd_ms)
             
             print(f"{i+1:<5} | {fwd_ms:<15.4f} | {t_math_ms:<15.4f} | {dec_ms:<15.4f}")

        dec_mean = np.mean(dec_latency_list)
        dec_std = np.std(dec_latency_list)
        fwd_mean = np.mean(forward_latency_list)
        
        # 결과 출력
        print(f"Config: Mode={decode_mode:<12} | K={K:<2} | V={V:<1} | Bits={tlut_bits}")
        print(f"-" * 60)
        print(f"{'Metric':<30} | {'Mean (ms)':<12} | {'Std (ms)':<10}")
        print(f"-" * 60)
        print(f"{'Total Forward Time':<30} | {fwd_mean:<12.4f} | {np.std(forward_latency_list):<10.4f}")
        print(f"{'Theoretical MatMul (T_math)':<30} | {t_math_ms:<12.4f} | {'0.0':<10}")
        print(f"{'Effective Decoding Time':<30} | {dec_mean:<12.4f} | {dec_std:<10.4f}")
        print(f"{'='*100}\n")
        
        
        dec_median = np.median(dec_latency_list)
        fwd_median = np.median(forward_latency_list)
        
        # 또는 최소값 사용 (Best Latency)
        dec_min = np.min(dec_latency_list)
        fwd_min = np.min(forward_latency_list)

        print(f"-" * 60)
        print(f"Config: Mode={decode_mode:<12} | K={K:<2} | V={V:<1}")
        print(f"-" * 60)
        # Median과 Min을 같이 리포트
        print(f"{'Metric':<30} | {'Median (ms)':<12} | {'Min (ms)':<12}") 
        print(f"-" * 60)
        print(f"{'Total Forward Time':<30} | {fwd_median:<12.4f} | {fwd_min:<12.4f}")
        print(f"{'Effective Decoding Time':<30} | {dec_median:<12.4f} | {dec_min:<12.4f}")
        print(f"{'='*100}\n")