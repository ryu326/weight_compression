# my-kernal (ANS only)

ANS-only fused matvec CUDA extension. QTIP-like fixed-bit paths were removed.

## Layout

- `src/my_ans_inference.cu`: ANS decode + fused Tensor Core matvec kernel
- `src/my_torch_ans.cu`: PyTorch dispatch and shape/prob-bits routing
- `src/wrapper.cpp`: pybind export
- `setup.py`: extension build script

## Build

```bash
cd /home/jgryu/workspace/weight_compression/qtip/my-kernal
python setup.py build_ext --inplace
```

## Python API

```python
import my_kernels

my_kernels.decompress_matvec_ans(
    out,               # [M, 1], float32, cuda
    ans_words,         # [W], int16, cuda (encoded word buffer)
    ans_states,        # [num_streams * 32], int32, cuda (initial states)
    ans_stream_starts, # [num_streams], int32, cuda (word start offsets)
    ans_stream_words,  # [num_streams], int32, cuda (word lengths)
    ans_lookup,        # [2^prob_bits], int32, cuda
    x,                 # [K, 1], float16, cuda
    codebook,          # [512], float16, cuda (256 half2 entries)
    prob_bits=9,       # 9/10/11
)

# Uniform quantization + ANS (16-bit symbol) fused decode+matvec
my_kernels.decompress_matvec_ans_uniform(
    out,               # [M, 1], float32, cuda
    ans_words,         # [W], int16, cuda
    ans_states,        # [num_streams * 32], int32, cuda
    ans_stream_starts, # [num_streams], int32, cuda
    ans_stream_words,  # [num_streams], int32, cuda
    ans_lookup_u16,    # [2^prob_bits], int64, cuda (packed u16 symbol/pdf/s_minus_cdf)
    x,                 # [K, 1], float16, cuda
    step_size,         # float, >0
    symbol_offset,     # int, typically min(q_indices)
    prob_bits=9,       # 9/10/11
)
```

## Encoding helper (2D tensor + step size)

`ans_uniform_encoder.py` provides a host-side encoder that converts a 2D weight tensor into
`decompress_matvec_ans_uniform` inputs.

```python
import torch
import my_kernels
from ans_uniform_encoder import encode_uniform_2d_to_ans

w = ...  # [M, K], float tensor (cuda or cpu)
x = ...  # [K, 1], float16, cuda

enc = encode_uniform_2d_to_ans(
    w,
    step_size=0.02,
    prob_bits=9,   # 9 / 10 / 11
    device="cuda",
)

out = torch.zeros((enc.m, 1), dtype=torch.float32, device="cuda")
my_kernels.decompress_matvec_ans_uniform(
    out,
    enc.ans_words,
    enc.ans_states,
    enc.ans_stream_starts,
    enc.ans_stream_words,
    enc.ans_lookup_u16,
    x,
    enc.step_size,
    enc.symbol_offset,
    enc.prob_bits,
)
```

## ANS format notes

- One ANS stream per `(tile_pair, warp)`, with `tile_pair = 2 x MMA_M rows`.
- Each stream has 32 lane states (`ans_states` flattened).
- Decoder consumes words backward per stream (dietgpu style).
- `ans_lookup` pack format:
  - bits `[7:0]` symbol
  - bits `[19:8]` pdf
  - bits `[31:20]` s_minus_cdf

- `decompress_matvec_ans_uniform` uses `ans_lookup_u16` pack format (`int64`):
  - bits `[15:0]` dequantized weight half bits (`uint16`)
  - bits `[31:16]` pdf (`uint16`)
  - bits `[47:32]` s_minus_cdf (`uint16`)
  - weight values are pre-dequantized at encode time (no runtime `symbol -> float -> half`)
