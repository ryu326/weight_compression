# DietGPU bitstream forward vs nn.Linear latency

## Setup
- GPU: 1x (CUDA_VISIBLE_DEVICES=6, single visible)
- dtype: fp16 for nn.Linear, fp32 for EC bitstream path
- Layer: `EntropyConstrainedLinear` with parametric MixtureEntropyModel (G3+L3)
- Bitstream: dietgpu rANS (byte-level)
- `forward_from_bitstream` = decompress_latent (dietgpu ANS decode) + decode_latent (identity decoder) + matmul

## Results (ms/iter, 30 iters after 5 warmup)

| shape                    | B×S     | EC-bitstream | nn.Linear | ratio (EC/Lin) | bpp   |
|--------------------------|---------|--------------|-----------|----------------|-------|
| q_proj (4096×4096)       | 1×128   | 1.68         | 0.06      | 28.9x          | 0.266 |
| q_proj                   | 1×512   | 2.31         | 0.15      | 15.4x          | 0.266 |
| q_proj                   | 1×2048  | 4.78         | 0.61      | 7.9x           | 0.266 |
| gate_proj (14336×4096)   | 1×128   | 5.74         | 0.20      | 28.1x          | 0.266 |
| gate_proj                | 1×512   | 7.71         | 0.50      | 15.4x          | 0.266 |
| gate_proj                | 1×2048  | 16.36        | 2.06      | 7.9x           | 0.266 |
| down_proj (4096×14336)   | 1×128   | 5.56         | 0.18      | 30.5x          | 0.266 |
| down_proj                | 1×512   | 7.51         | 0.52      | 14.4x          | 0.266 |
| down_proj                | 1×2048  | 15.90        | 2.26      | 7.0x           | 0.266 |

## 해석

- **EC bitstream 이 7~30x 느림** — 매 forward 마다 decompress 전체 수행하기 때문
- **작은 배치일수록 상대적 부담 ↑** — decompress 고정 비용이 matmul 시간보다 큼
- **실전 사용 시**: decompress 한 번 하고 decoded latent 를 캐시해서 반복 matmul 에 쓰는 게 정석
- bpp 0.266 은 random-initialized latent 가 거의 다 0 (round) 이라 극한적 압축 된 케이스. 실제 학습된 latent 에선 2-5 bpp 범위

## 파일

- `bench_forward_latency.py` — 벤치마크 스크립트
- `latency_table.txt` — 정제된 결과 표
- `raw.log` — 전체 실행 로그 (warning 포함)
