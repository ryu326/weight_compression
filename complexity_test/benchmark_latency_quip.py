import copy
import math
from contextlib import contextmanager
from operator import attrgetter

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

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', 'quip-sharp'))

if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added to path: {project_root}")

try:
    from lib import codebook, utils
    from lib.linear import *
    from lib.algo.quip import *
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    print("Check if the path exists:", os.path.exists(project_root))
    raise


def quip_quantize(W_in, HR, device, codebook_id = 'E8P12RVQ4B'):    

    args_dict = {
        'use_fp64': False,
        'incoh_mode': 'had',
        'save_pfx': "",
        'rescale_WH': False,
        'lora_rank':0,
        'scale_override' : 0.9,
        'no_use_buffered':False,
        'lowmem_ldlq':False,
        'quip_tune_iters' : 10,
        'resid_scale_override' : -1,
        
    }
    args = argparse.Namespace(**args_dict)


    cb = codebook.get_codebook(codebook_id)

    weights = [W_in]
    dtype_ = torch.float32
    shapes = [_.shape for _ in weights]
    scales = [_.to(dtype_).square().mean().sqrt() for _ in weights]

    save_path = f'./tmp_quip_ckpt.pt'
    if os.path.exists(save_path):
        return

    W = torch.vstack([
        weights[i].to(dtype_) / scales[i] for i in range(len(weights))
    ]).to(dtype_)
    H = HR
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    hatW, attr = quantize(H, W, 0, cb, args, device)
    
    end_event.record()
    torch.cuda.synchronize()
    gc.enable() # GC 다시 활성화
    elapsed_time_ms = start_event.elapsed_time(end_event)
    
    if len(scales) == 1:
        # fuse single scale into SV too
        attr['SV'] *= scales[0]
        scales = [1.0]
    attr.update({
        'fused': len(shapes) > 1,
        'shapes': shapes,
        'scales': scales,
    })
    torch.save(attr, save_path)
    utils.show_metrics(hatW, W, H.to(dtype_), save_path)
    utils.clean()

    return elapsed_time_ms

# RTX A6000 Specs
A6000_PEAK_FP16_TFLOPS = 38.71

def measure_decoding_latency_subtraction(orig_layer_weight, device, num_repeats=100, codebook_id = 'E8P12RVQ4B'):
  
    cb = codebook.get_codebook(codebook_id)
    codebook_id = codebook.get_id(codebook_id)
    
    shared_args = (cb.codesz, cb.packsz, cb.pack_out, str(cb.idx_dtype),
                   cb.version)
    shared_kwargs = {
        'rank': 0,
        'rescale_WH': False,
        'resid_scale_override': -1,
        'bias': False,
        'train_mode': False,
        'grad_ckpt': False,
        'train_mode':False
    }
  
    save_path = f'./tmp_quip_ckpt.pt'
    saved_linear = torch.load(save_path,
        map_location=torch.device('cpu'))
    if saved_linear['fused']:
        raise
        quant_linear = FusedQuantizedLinear(
            -1, [_[0] for _ in saved_linear['shapes']],
            saved_linear['shapes'][0][1],
            sum([_[0] for _ in saved_linear['shapes']]), *shared_args,
            **shared_kwargs)
        for i in range(len(saved_linear['scales'])):
            quant_linear.fuse_scales[i].copy_(
                saved_linear['scales'][i])
    else:
        quant_linear = QuantizedLinear(saved_linear['shapes'][0][1],
                                        saved_linear['shapes'][0][0],
                                        *shared_args, **shared_kwargs)
    utils.unpack_quip(quant_linear, saved_linear, codebook_id,
                        cb.codesz)
    quant_linear.SU = nn.Parameter(quant_linear.SU.float(),
                                    requires_grad=True)
    quant_linear.SV = nn.Parameter(quant_linear.SV.float(),
                                        requires_grad=True)
    
    quant_linear.to(device)
    
    M, N_in = orig_layer_weight.shape
    BATCH_SIZE = 10
    x = torch.randn(BATCH_SIZE, N_in, device=device, dtype=torch.float16)

    # 3. Warmup (커널 컴파일 및 초기화)
    # 첫 실행 시 codebook_class가 초기화되므로 반드시 Warmup 필요
    for _ in range(10):
        _ = quant_linear(x)
    torch.cuda.synchronize()
    
    # GC 및 메모리 정리
    gc.collect()
    # torch.cuda.empty_cache() # 잦은 호출은 오버헤드가 될 수 있어 생략하거나 필요시 추가

    # 4. Forward Latency 측정
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_repeats):
        _ = quant_linear(x)
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
    
    del quant_linear
    
    return decoding_time_ms, total_forward_ms, t_math_ms


import collections
device = 'cuda:7'
num_iter = 50        # 측정 반복 횟수 늘림 (안정성 확보)
warmup_iter = 5
std_val = 0.012528750114142895
N = 4096             # Matrix Size

modes = [
    'E8P12RVQ4B'
    ]


H = torch.eye(N, device=device) 
W = torch.normal(mean=0.0, std=std_val, size=(N, N), device=device)

print(f"{'='*100}")
print(f"Effective Decoding Latency Benchmark (Forward - T_math)")
print(f"GPU: RTX A6000 (Peak FP16: {A6000_PEAK_FP16_TFLOPS} TFLOPS)")
print(f"{'='*100}")

for codebook_id in modes:

    encoding_time = quip_quantize(W, H, device, codebook_id)
    print('encoding_time:', encoding_time)
    # 측정 수행
    dec_latency_list = []
    forward_latency_list = []
    
    # --- [수정됨] 개별 측정값 출력 헤더 ---
    print(f"\n>> Start Measuring: Mode={codebook_id}")
    print(f"{'Iter':<5} | {'Forward (ms)':<15} | {'T_math (ms)':<15} | {'Decoding (ms)':<15}")
    print("-" * 60)
    with torch.no_grad():
        
        for i in range(20): # 여러 번 측정하여 평균
            
                dec_ms, fwd_ms, t_math_ms = measure_decoding_latency_subtraction(
                W, device, num_repeats=100, codebook_id = codebook_id
                )
                dec_latency_list.append(dec_ms)
                forward_latency_list.append(fwd_ms)
                
                print(f"{i+1:<5} | {fwd_ms:<15.4f} | {t_math_ms:<15.4f} | {dec_ms:<15.4f}")

        dec_mean = np.mean(dec_latency_list)
        dec_std = np.std(dec_latency_list)
        fwd_mean = np.mean(forward_latency_list)
        
    # 결과 출력
    print(f"Config: Mode={codebook_id:<12}")
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
    print(f"Config: Mode={codebook_id:<12}")
    print(f"-" * 60)
    # Median과 Min을 같이 리포트
    print(f"{'Metric':<30} | {'Median (ms)':<12} | {'Min (ms)':<12}") 
    print(f"-" * 60)
    print(f"{'Total Forward Time':<30} | {fwd_median:<12.4f} | {fwd_min:<12.4f}")
    print(f"{'Effective Decoding Time':<30} | {dec_median:<12.4f} | {dec_min:<12.4f}")
    print(f"{'='*100}\n")