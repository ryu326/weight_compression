import argparse
import os
import time

import glog, json

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from operator import attrgetter

from pathlib import Path
import sys
notebook_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
project_root = notebook_dir.parent
sys.path.append(str(project_root))

from NWC.models import get_model


device = torch.device('cuda:7')

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
        
# comp_model_path = '../NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda50_rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/best_loss_model_loss_3.87239_bpp_4.65884_MSE_0.0162_total_iter_95000.pth.tar'
# comp_model_path = '/home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda10000_rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/best_loss_model_loss_11.43015_bpp_6.30844_MSE_0.00043_total_iter_200000.pth.tar'
# comp_model_path = '/workspace/Weight_compression/NWC/checkpoint/nwc_scale_cond/block_seq_scale_cond_scaler_meta-llama--Meta-Llama-3-8B__scaleH_sig0.0001_std_rnormed_with_col_std_lidx_row_1024.pt/rdloss_size128_encdim1024_M256_Q0_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/lmbda50_/best_loss_model_loss_3.94749_bpp_3.26997_MSE_4.91093_total_iter_192500.pth.tar'
# comp_model_path = '/home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda300_rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/best_loss_model_loss_5.32101_bpp_5.72603_MSE_0.00289_total_iter_95000.pth.tar'  ## 3.95 bits
# comp_model_path = '/home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda75_/best_loss_model_loss_4.18422_bpp_4.90131_MSE_0.01087_total_iter_47500.pth.tar' # 2.96 bit
# comp_model_path = "/home/jgryu/workspace/weight_compression/NWC/checkpoint/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/M16/lmbda30_rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/best_loss_model_loss_3.46368_bpp_4.27494_MSE_0.02685_total_iter_95000.pth.tar" # 2.31 bit
# comp_model_path = "/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/n_rb1/lmbda300_/best_loss_model_loss_5.27489_bpp_5.82522_MSE_0.0027_total_iter_47500.pth.tar" ## nres1 4bit
comp_model_path = "/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/n_rb2/lmbda300_/best_loss_model_loss_5.2877_bpp_5.76578_MSE_0.00275_total_iter_47500.pth.tar"  ## nres2 4bit
# comp_model_path = "/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_ql_size4_encdim64_M4_Q4_R0_m0_batch_size8192_total_iter200000_lr0.0001_seed100/lmbda300_/best_loss_model_loss_5.11241_bpp_5.99794_MSE_0.00262_total_iter_82500.pth.tar"   ## nres2 size 4  4bit

config = os.path.join(os.path.dirname(comp_model_path), 'config.json')
with open(config, 'r', encoding='utf-8') as file:
    config = json.load(file)
config = Config(**config)

shift, scale = None, None
if config.architecture == 'nwc_ql' and not hasattr(config, "Q"):
    config.Q = 4
if not hasattr(config, "no_layernorm"):
    config.no_layernorm = False


comp_model = get_model(config.architecture, config, scale=scale, shift=shift)
comp_model.config = config
ckpt = torch.load(comp_model_path, weights_only=False)
scale, shift  = torch.zeros(1), torch.zeros(1)

comp_model.load_state_dict(ckpt["state_dict"], strict = False)
comp_model.scale = scale.to(device)
comp_model.shift = shift.to(device)

comp_model.to(device)
comp_model.eval()
comp_model.update()

comp_model.update(force=True)

import torch
from compressai.entropy_models import EntropyBottleneck

entropy_bottleneck = comp_model.entropy_bottleneck
cdf_size = entropy_bottleneck._quantized_cdf.numel()
length_size = entropy_bottleneck._cdf_length.numel()
offset_size = entropy_bottleneck._offset.numel()

print(f"메인 코드북(_quantized_cdf) 요소 개수: {cdf_size}")
print(f"CDF 길이 정보(_cdf_length) 요소 개수: {length_size}")
print(f"오프셋 정보(_offset) 요소 개수: {offset_size}")

total_decoding_params = cdf_size + length_size + offset_size
print(f"디코딩용 총 파라미터 개수: {total_decoding_params}")

import torch
from compressai.entropy_models import EntropyBottleneck

entropy_bottleneck = comp_model.entropy_bottleneck

def get_buffer_size_in_bytes(buffer_tensor):
    return buffer_tensor.numel() * buffer_tensor.element_size()

cdf_bytes = get_buffer_size_in_bytes(entropy_bottleneck._quantized_cdf)
length_bytes = get_buffer_size_in_bytes(entropy_bottleneck._cdf_length)
offset_bytes = get_buffer_size_in_bytes(entropy_bottleneck._offset)

print(f"1. _quantized_cdf (확률 테이블): {cdf_bytes} Bytes ({cdf_bytes/1024:.2f} KB)")
print(f"2. _cdf_length    (길이 정보)  : {length_bytes} Bytes")
print(f"3. _offset        (시작점 정보): {offset_bytes} Bytes")

total_bytes = cdf_bytes + length_bytes + offset_bytes
print(f"---")
print(f"총 헤더(코드북) 용량: {total_bytes} Bytes ({total_bytes/1024:.2f} KB)")

print("Total parameters:", sum(p.numel() for p in comp_model.parameters()))
print("Trainable parameters:", sum(p.numel() for p in comp_model.parameters() if p.requires_grad))

import torch
from compressai.entropy_models import EntropyModel, EntropyBottleneck, GaussianConditional

def count_decoding_params(model: torch.nn.Module):
    """
    compressai 기반 모델에서 'decoding(복호화)에 필요한' 파라미터/버퍼 개수만 세는 함수.
    반환값:
      total_numel: 전체 개수
      detail: 모듈별 세부 개수 dict
    """
    total_numel = 0
    detail = {}

    for module_name, module in model.named_modules():
        if not isinstance(module, EntropyModel):
            continue

        module_count = 0

        for buf_name in ["_quantized_cdf", "_cdf_length", "_offset"]:
            buf = getattr(module, buf_name, None)
            if isinstance(buf, torch.Tensor):
                n = buf.numel()
                module_count += n
                total_numel += n

        if isinstance(module, EntropyBottleneck):
            q = getattr(module, "quantiles", None)
            if isinstance(q, torch.Tensor):
                n = q.numel()
                module_count += n
                total_numel += n

        if isinstance(module, GaussianConditional):
            st = getattr(module, "scale_table", None)
            if isinstance(st, torch.Tensor):
                n = st.numel()
                module_count += n
                total_numel += n

        if module_count > 0:
            detail[module_name] = module_count

    return total_numel, detail

def count_decoding_params_by_module(model: torch.nn.Module):
    """
    quality_embedding, g_s, entropy_bottleneck에서 디코딩에 필요한 파라미터 개수를 각각 세는 함수.
    반환값:
      result: 각 모듈별 파라미터 개수 dict
    """
    result = {
        'quality_embedding': 0,
        'g_s': 0,
        'entropy_bottleneck': 0
    }
    
    if hasattr(model, 'quality_embedding'):
        quality_embedding_params = sum(p.numel() for p in model.quality_embedding.parameters())
        result['quality_embedding'] = quality_embedding_params
    
    if hasattr(model, 'g_s'):
        g_s_params = sum(p.numel() for p in model.g_s.parameters())
        result['g_s'] = g_s_params
    
    if hasattr(model, 'entropy_bottleneck'):
        entropy_bottleneck_count = 0
        eb = model.entropy_bottleneck
        
        for buf_name in ["_quantized_cdf", "_cdf_length", "_offset"]:
            buf = getattr(eb, buf_name, None)
            if isinstance(buf, torch.Tensor):
                entropy_bottleneck_count += buf.numel()
        
        q = getattr(eb, "quantiles", None)
        if isinstance(q, torch.Tensor):
            entropy_bottleneck_count += q.numel()
        
        result['entropy_bottleneck'] = entropy_bottleneck_count
    
    return result

print("\n" + "="*60)
print("디코딩에 필요한 파라미터 개수")
print("="*60)

total_entropy_params, entropy_detail = count_decoding_params(comp_model)
print(f"\n[EntropyModel 계열 전체]: {total_entropy_params:,} 개")
if entropy_detail:
    for module_name, count in entropy_detail.items():
        print(f"  - {module_name}: {count:,} 개")

module_params = count_decoding_params_by_module(comp_model)
print(f"\n[모듈별 디코딩 파라미터]:")
print(f"  - quality_embedding: {module_params['quality_embedding']:,} 개")
print(f"  - g_s: {module_params['g_s']:,} 개")
print(f"  - entropy_bottleneck: {module_params['entropy_bottleneck']:,} 개")

total_decoding_params = sum(module_params.values())
print(f"\n[디코딩 총 파라미터]: {total_decoding_params:,} 개")
print("="*60)

import torch
import time
import statistics
from collections import defaultdict

num_iterations = 10
warmup_iter = 5
parallelization_factor = 568

all_timings = defaultdict(list)

accelerated_results = []
residual_results = []

T = torch.zeros(4096, 4096)
T = T.reshape(32, -1, comp_model.input_size).to(device)

with torch.no_grad():
    for i in range(num_iterations + warmup_iter):
        data = {}
        data['weight_block'] = T
        data['q_level'] = torch.zeros(1, T.shape[1], dtype=torch.int).to(device)

        comp_model.to(device)
        out_enc = comp_model.compress(data)

        out_dec, timings = comp_model.fast_decompress_v2(out_enc)
        
        if i >= warmup_iter:
            for step, duration in timings.items():
                all_timings[step].append(duration)

            try:
                current_total_time = timings['total_decompress_ms']
                
                current_transfer_time = sum(
                    value for key, value in timings.items() if 'transfer:' in key
                )
                current_decode_time = timings.get('entropy_dec_loop_decode_call_ms', 0)
                
                new_decode_time = current_decode_time / parallelization_factor
                time_saved_parallel = current_decode_time - new_decode_time
                total_saved = current_transfer_time + time_saved_parallel
                
                simulated_time = current_total_time - total_saved
                accelerated_results.append(simulated_time)

                current_residual = current_total_time - current_transfer_time - current_decode_time
                residual_results.append(current_residual)

            except KeyError:
                print(f"Warning: Iteration {i} missing timing keys.")

print(f"\n--- 각 단계별 평균 소요 시간 (Raw Data, {num_iterations}회 평균) ---")
sorted_avg_timings = []
for step, duration_list in all_timings.items():
    avg = statistics.mean(duration_list)
    sorted_avg_timings.append((step, avg))

for step, avg_duration in sorted_avg_timings:
    print(f"{step:<35}: {avg_duration:.4f} ms")


print("\n" + "="*60)
print(f"🚀 분석 결과 리포트 ({num_iterations}회 수행)")
print("="*60)

if len(accelerated_results) > 0:
    original_mean = statistics.mean(all_timings['total_decompress_ms'])
    
    acc_mean = statistics.mean(accelerated_results)
    acc_stdev = statistics.stdev(accelerated_results) if len(accelerated_results) > 1 else 0.0
    
    res_mean = statistics.mean(residual_results)
    res_stdev = statistics.stdev(residual_results) if len(residual_results) > 1 else 0.0
    
    print(f"1. [원본] 총 시간 평균          : {original_mean:.4f} ms")
    print(f"-"*60)
    
    print(f"2. [가속 예측] (Overhead 제거 + 병렬화):")
    print(f"   ▶ 평균 (Mean)              : {acc_mean:.4f} ms")
    print(f"   ▶ 표준편차 (Std Dev)       : {acc_stdev:.4f} ms")
    print(f"   ▶ 논문 표기용              : {acc_mean:.2f} ± {acc_stdev:.2f} ms")
    print(f"-"*60)

    print(f"3. [기타 부분] (Total - Transfer - Decode):")
    print(f"   * 순수 커널 런타임, 메모리 할당 등 기타 오버헤드")
    print(f"   ▶ 평균 (Mean)              : {res_mean:.4f} ms")
    print(f"   ▶ 표준편차 (Std Dev)       : {res_stdev:.4f} ms")
    print(f"   ▶ 논문 표기용              : {res_mean:.2f} ± {res_stdev:.2f} ms")
    print(f"-"*60)

else:
    print("데이터가 수집되지 않았습니다.")
