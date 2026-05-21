# e2e_latency

Llama-3 8B의 16-bit 오프로딩 baseline과, "가중치 압축 + on-the-fly decoding" 방식의 분석적 예측치를 비교하기 위한 PyTorch/Transformers 벤치마크입니다.

## 파일

- `benchmark_llama_offload.py`
  - `compare`: 전체 비교 실행
  - `baseline-run`: 8GB VRAM 강제 제한 + CPU offload + PCIe 지연 주입
  - `fullvram-run`: 16-bit 전체 VRAM 적재 순수 측정
- `run_compare_cuda7.sh`
  - 물리 GPU `7`만 노출시키고 내부에서는 `--gpu-id 0`으로 실행하는 bash launcher

## 실험 가정

1. Baseline
   - `torch.cuda.set_per_process_memory_fraction`으로 단일 GPU 가용 VRAM을 8GB로 제한합니다.
   - `transformers` + `accelerate`의 `device_map="auto"`로 모델 일부를 CPU로 offload합니다.
   - baseline은 offloaded 모듈이 실행 시점에 다시 GPU로 올라올 수 있도록, device map을 짤 때 `--baseline-runtime-reserve-gb`만큼의 GPU headroom을 따로 남겨둡니다.
   - `hf_device_map`에서 CPU에 배치된 모듈을 찾고, 해당 모듈의 `forward_pre_hook`에 `time.sleep()` 기반 지연을 넣어 목표 PCIe 대역폭을 에뮬레이션합니다.

2. Ours
   - 3~4bit 압축으로 전체 모델이 VRAM에 완전히 올라간다고 가정합니다.
   - 실제 측정은 "전체 16-bit 모델을 순수 GPU에 올린 상태"에서 TTFT/TPOT를 구합니다.
   - 여기에 디코딩 오버헤드를 분석적으로 더합니다.
   - 기본 공식:
     - `4096 x 4096 parameters -> 1.07 ms`
   - 스크립트는 실제 로드된 모델에서 decoder layer 파라미터 수만 세고, 이를 기준으로 `per_forward_decode_overhead_ms`를 계산합니다.
   - 즉, embedding과 `lm_head` 파라미터는 기본 계산에서 제외됩니다.

## 중요한 주의점

- `time.sleep()` 기반 PCIe 에뮬레이션은 실제 복사 시간 위에 추가 지연을 넣는 방식입니다.
- 현재 시스템의 실제 host<->GPU 대역폭을 알고 있다면 `--native-bandwidth-gbps`를 넣어서 "목표 대역폭까지의 추가 지연만" 주입할 수 있습니다.
- 멀티 GPU 머신에서는 일반 PC/Edge 에뮬레이션을 위해 단일 GPU만 보이게 두는 편이 안전합니다.
- Meta-Llama-3 모델은 Hugging Face 인증/라이선스 수락이 필요할 수 있습니다.

## 전력 제한 예시

이 부분은 Python 밖에서 설정합니다.

```bash
sudo nvidia-smi -i 0 -pl 150
sudo nvidia-smi -i 0 -pl 100
```

CUDA 7 기준이면 다음처럼 실행하면 됩니다.

```bash
sudo nvidia-smi -i 7 -pl 150
sudo nvidia-smi -i 7 -pl 100
```

## 실행 예시

```bash
cd /home/jgryu/workspace/weight_compression/e2e_latency

CUDA_VISIBLE_DEVICES=0 python benchmark_llama_offload.py \
  --mode compare \
  --model-id meta-llama/Meta-Llama-3-8B \
  --dtype float16 \
  --prompt-length 1024 \
  --max-new-tokens 16 \
  --bandwidth-gbps 4 12 25 \
  --baseline-vram-gb 8 \
  --decode-ms-per-matrix 1.07 \
  --decode-overhead-scale 1.0
```

CUDA 7 기준 launcher는 바로 아래처럼 실행하면 됩니다.

```bash
cd /home/jgryu/workspace/weight_compression/e2e_latency

./run_compare_cuda7.sh
```

환경변수로 기본값을 덮어쓸 수 있습니다.

```bash
cd /home/jgryu/workspace/weight_compression/e2e_latency

PROMPT_LENGTH=2048 \
MAX_NEW_TOKENS=32 \
BANDWIDTHS_STR="4 12 25" \
DECODE_MS_PER_MATRIX=1.25 \
DECODE_OVERHEAD_SCALE=1.10 \
JSON_OUTPUT=./results/llama3_cuda7_custom.json \
./run_compare_cuda7.sh
```

## 출력

`compare` 모드는 다음을 출력합니다.

- baseline TTFT / TPOT
- full-VRAM 16-bit TTFT / TPOT
- analytical decode overhead
- 각 대역폭별 `[Baseline TTFT, Baseline TPOT] vs [Ours predicted TTFT, Ours predicted TPOT]`

필요하면 JSON으로도 저장할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_llama_offload.py \
  --mode compare \
  --bandwidth-gbps 12 \
  --json-output result.json
```

## 주요 인자

- `--prompt-length`: 더미 프롬프트 길이(token 수)
- `--max-new-tokens`: 생성 토큰 수. TPOT는 2번째 생성 토큰부터 평균
- `--bandwidth-gbps`: baseline용 목표 PCIe 대역폭 리스트
- `--native-bandwidth-gbps`: 현재 시스템의 실제 host<->GPU 대역폭을 알고 있을 때 보정용
- `--baseline-vram-gb`: baseline VRAM 제한값
- `--baseline-runtime-reserve-gb`: baseline에서 offloaded 모듈 재적재용으로 비워둘 GPU headroom
- `--decode-ms-per-matrix`: `4096x4096 -> X ms`에서 X를 조절하는 인자
- `--decode-overhead-scale`: 분석적 디코딩 오버헤드를 일괄 배수 조정
- `--include-non-layer-params-in-decode`: embedding/lm_head까지 분석 오버헤드에 포함

## 구현 메모

- TTFT는 첫 토큰이 산출될 때까지의 시간을 직접 측정합니다.
- TPOT는 KV cache를 사용한 1-token decode step들의 평균입니다.
- baseline과 full-VRAM 측정은 서로 다른 subprocess에서 수행합니다.
  - 이유: CUDA allocator 상태와 `set_per_process_memory_fraction`의 영향을 분리하기 위해서입니다.
