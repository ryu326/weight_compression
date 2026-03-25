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
# comp_model_path = "/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_ql_size16_encdim512_M16_Q4_R0_m0_batch_size2048_total_iter200000_lr0.0001_seed100/n_rb2/lmbda300_/best_loss_model_loss_5.2877_bpp_5.76578_MSE_0.00275_total_iter_47500.pth.tar"  ## nres2 4bit
comp_model_path = "/home/jgryu/workspace/weight_compression/NWC/checkpoint2/nwc_ql/block_seq_ql_random_scaler_meta-llama--Meta-Llama-3-8B__col_1024_gaussian_padding.pt/rdloss_ql_size4_encdim64_M4_Q4_R0_m0_batch_size8192_total_iter200000_lr0.0001_seed100/lmbda300_/best_loss_model_loss_5.11241_bpp_5.99794_MSE_0.00262_total_iter_82500.pth.tar"   ## nres2 size 4  4bit

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

comp_model.update(force=True)              # CompressAI: CDF 고정 및 버퍼 등록
# comp_model.entropy_bottleneck._quantized_cdf  # 캐시되어 이후 재계산 안 됨


import torch
import time
import statistics  # 평균, 분산, 표준편차 계산용
from collections import defaultdict

# --- 설정 ---
num_iterations = 10
warmup_iter = 5      # 워밍업 횟수 (캐싱 등 오버헤드 제거)
parallelization_factor = 568 # GPU 병렬화 계수

# 각 스텝별 raw 데이터를 저장 (디버깅/상세 분석용)
all_timings = defaultdict(list)

# 결과 저장용 리스트
accelerated_results = [] # 가속 시나리오 결과
residual_results = []    # 제외할 거 제외한 나머지 시간 결과

T = torch.zeros(4096, 4096)
# T = T.reshape(32, 32768, 16).to(device)
T = T.reshape(32, -1, comp_model.input_size).to(device)

with torch.no_grad():
    for i in range(num_iterations + warmup_iter):
        # --- 데이터 준비 ---
        # T = torch.zeros(256, 256)
        # T = T.reshape(1, -1, 16).to(device)

        
        data = {}
        data['weight_block'] = T
        data['q_level'] = torch.zeros(1, T.shape[1], dtype=torch.int).to(device)

        comp_model.to(device)
        out_enc = comp_model.compress(data)

        # fast_decompress 실행
        out_dec, timings = comp_model.fast_decompress_v2(out_enc)
        
        if i >= warmup_iter:
            # 1. Raw Timing 저장
            for step, duration in timings.items():
                all_timings[step].append(duration)

            # ---------------------------------------------------------
            # 2. 통계 계산 (가속 시나리오 & 나머지 부분 분석)
            # ---------------------------------------------------------
            try:
                # A. 현재 반복의 원본 총 시간
                current_total_time = timings['total_decompress_ms']
                
                # B. 타겟 성분 추출
                # (1) Transfer 관련 시간 합계
                current_transfer_time = sum(
                    value for key, value in timings.items() if 'transfer:' in key
                )
                # (2) Decode Loop 시간
                current_decode_time = timings.get('entropy_dec_loop_decode_call_ms', 0)
                
                # --- [시나리오 1] 가속 예측 시간 계산 ---
                new_decode_time = current_decode_time / parallelization_factor
                time_saved_parallel = current_decode_time - new_decode_time
                total_saved = current_transfer_time + time_saved_parallel # Overhead(Transfer) 제거 + Decode 가속
                
                simulated_time = current_total_time - total_saved
                accelerated_results.append(simulated_time)

                # --- [시나리오 2] 나머지(Residual) 시간 계산 ---
                # Total - (Transfer + Decode Loop)
                current_residual = current_total_time - current_transfer_time - current_decode_time
                residual_results.append(current_residual)

            except KeyError:
                print(f"Warning: Iteration {i} missing timing keys.")

# --- 결과 출력 ---

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
    # 1. 원본 총 시간
    original_mean = statistics.mean(all_timings['total_decompress_ms'])
    
    # 2. 가속 시나리오 통계
    acc_mean = statistics.mean(accelerated_results)
    acc_stdev = statistics.stdev(accelerated_results) if len(accelerated_results) > 1 else 0.0
    
    # 3. 나머지 부분(Residual) 통계
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

# print(accelerated_results)