#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define ANS_MMA_M 16
#define ANS_MMA_N 8
#define ANS_MMA_K 16

#define ANS_BLOCK_COUNT 128
#define ANS_WARP_SIZE 32
#define ANS_BLOCK_SIZE 1024

#define ANS_FULL_MASK 0xFFFFFFFFU
#define ROUND_UP(a, b) (((a) + (b)-1) / (b))

#define gpuErrchk(ans) \
    do { gpuAssert((ans), __FILE__, __LINE__); } while (false)

__host__ static inline void gpuAssert(
    cudaError_t code,
    const char* file,
    int line,
    bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) {
            exit(code);
        }
    }
}

typedef union ditto2 {
    unsigned long long ull;
    uint64_t u64;
    uint2 u32x2;
    float2 f32x2;
    ushort4 u16x4;
    uint32_t u32[2];
    half2 f16x2[2];
    half2* ptr2f16x2;
} ditto2;

typedef union ditto4 {
    uint4 u32x4;
    uint32_t u32[4];
    float4 f32x4;
    half2 f16x2[4];
    uint16_t u16[8];
    float f32[4];
} ditto4;

constexpr uint32_t kANSStateBits = 31;
constexpr uint32_t kANSEncodedBits = 16;
constexpr uint32_t kANSMinState = uint32_t(1) << (kANSStateBits - kANSEncodedBits);

__inline__ __device__ uint32_t ld_x_ans(const uint32_t* p) {
    uint32_t out;
    asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}

__inline__ __device__ void prefetch_ans(uint32_t* a) {
    asm("prefetch.global.L1 [%0];" : : "l"(a));
}

__device__ __forceinline__ uint32_t get_lane_mask_ge_ans(uint32_t lane_id) {
    return 0xFFFFFFFFu << lane_id;
}

template <int ProbBits>
__device__ __forceinline__ uint8_t ans_decode_symbol(
    uint32_t& state,
    const uint16_t*& in,
    const uint32_t* __restrict__ lookup,
    uint32_t lane_id) {
    constexpr uint32_t state_mask = (uint32_t(1) << ProbBits) - 1u;

    uint32_t packed = lookup[state & state_mask];

    uint32_t sym = packed & 0xFFu;
    packed >>= 8;
    uint32_t pdf = packed & 0xFFFu;
    uint32_t s_minus_cdf = packed >> 12;

    state = pdf * (state >> ProbBits) + s_minus_cdf;

    bool read = state < kANSMinState;
    uint32_t vote = __ballot_sync(ANS_FULL_MASK, read);
    uint32_t prefix = __popc(vote & get_lane_mask_ge_ans(lane_id));

    if (read) {
        uint16_t v = in[-static_cast<int>(prefix)];
        state = (state << kANSEncodedBits) + uint32_t(v);
    }

    uint32_t num_read = __popc(vote);
    in -= num_read;

    return static_cast<uint8_t>(sym);
}

template <int ProbBits>
__device__ __forceinline__ uint16_t ans_decode_symbol_u16(
    uint32_t& state,
    const uint16_t*& in,
    const uint64_t* __restrict__ lookup,
    uint32_t lane_id) {
    constexpr uint32_t state_mask = (uint32_t(1) << ProbBits) - 1u;

    // Packed layout per entry (64-bit):
    // bits [15:0]   : dequantized weight half bits (uint16)
    // bits [31:16]  : pdf (uint16)
    // bits [47:32]  : s_minus_cdf (uint16)
    uint64_t packed = lookup[state & state_mask];

    uint32_t weight_half_bits = static_cast<uint32_t>(packed & 0xFFFFu);
    uint32_t pdf = static_cast<uint32_t>((packed >> 16) & 0xFFFFu);
    uint32_t s_minus_cdf = static_cast<uint32_t>((packed >> 32) & 0xFFFFu);

    state = pdf * (state >> ProbBits) + s_minus_cdf;

    bool read = state < kANSMinState;
    uint32_t vote = __ballot_sync(ANS_FULL_MASK, read);
    uint32_t prefix = __popc(vote & get_lane_mask_ge_ans(lane_id));

    if (read) {
        uint16_t v = in[-static_cast<int>(prefix)];
        state = (state << kANSEncodedBits) + uint32_t(v);
    }

    uint32_t num_read = __popc(vote);
    in -= num_read;

    return static_cast<uint16_t>(weight_half_bits);
}

template <int ProbBits>
__device__ inline void decode_weight_fragment_ans(
    const half2* __restrict__ codebook_half2,
    const uint32_t* __restrict__ lookup,
    uint32_t& state,
    const uint16_t*& in,
    uint32_t lane_id,
    ditto4& reg_w) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
        uint8_t sym = ans_decode_symbol<ProbBits>(state, in, lookup, lane_id);
        reg_w.f16x2[j] = codebook_half2[sym];
    }
}

template <int ProbBits>
__device__ inline void decode_weight_fragment_ans_uniform(
    const uint64_t* __restrict__ lookup,
    uint32_t& state,
    const uint16_t*& in,
    uint32_t lane_id,
    ditto4& reg_w) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
        uint16_t w0_bits = ans_decode_symbol_u16<ProbBits>(state, in, lookup, lane_id);
        uint16_t w1_bits = ans_decode_symbol_u16<ProbBits>(state, in, lookup, lane_id);
        reg_w.f16x2[j] = __halves2half2(__ushort_as_half(w0_bits), __ushort_as_half(w1_bits));
    }
}

template <int ProbBits, uint32_t S, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__global__ static void __launch_bounds__(ANS_BLOCK_SIZE, 1) kernel_ans_fused_matvec(
    float* __restrict__ out,
    const uint16_t* __restrict__ ans_words,
    const uint32_t* __restrict__ ans_states,
    const uint32_t* __restrict__ ans_stream_starts,
    const uint32_t* __restrict__ ans_stream_words,
    const half2* __restrict__ x,
    const half2* __restrict__ codebook,
    const uint32_t* __restrict__ ans_lookup) {
    extern __shared__ half2 smem_codebook[];

    const uint32_t thread_id = threadIdx.x;
    const uint32_t lane_id = thread_id % ANS_WARP_SIZE;
    const uint32_t warp_id = thread_id / ANS_WARP_SIZE;
    const uint32_t block_id = blockIdx.x;

    constexpr uint32_t tile_count_m = M / ANS_MMA_M;
    constexpr uint32_t tile_count_k = K / ANS_MMA_K;
    constexpr uint32_t warps_per_block = ANS_BLOCK_SIZE / ANS_WARP_SIZE;

    static_assert(tile_count_m % 2 == 0);
    constexpr uint32_t m_per_block = ROUND_UP(tile_count_m, (2 * ANS_BLOCK_COUNT));
    constexpr uint32_t k_per_block = tile_count_k / (warps_per_block * 4) * 2;
    static_assert((tile_count_k % (warps_per_block * 4)) % 4 == 0);
    constexpr uint32_t extra_warps = (tile_count_k % (warps_per_block * 4)) / 4;
    uint32_t this_warp_k = k_per_block;
    if constexpr (extra_warps > 0) {
        if (warp_id < extra_warps) {
            this_warp_k = k_per_block + 2;
        }
    }

    constexpr uint32_t f16x2_per_x_tile = ANS_MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = ANS_MMA_M;
    constexpr uint32_t codebook_half2 = 1u << (S + V - 1);

    uint32_t tile_id_m = m_per_block * block_id;

    for (uint32_t mi = 0; mi < m_per_block; ++mi) {
        if (tile_id_m * 2 >= tile_count_m) {
            return;
        }

        // One ANS stream per (tile-pair, warp); each lane has its own state.
        const uint32_t stream_idx = tile_id_m * warps_per_block + warp_id;
        uint32_t state = ans_states[stream_idx * ANS_WARP_SIZE + lane_id];
        const uint32_t start = ans_stream_starts[stream_idx];
        const uint32_t words = ans_stream_words[stream_idx];
        const uint16_t* in = ans_words + start + words;

        float4 reg_p[2] = {};

        uint32_t x_idx = warp_id * f16x2_per_x_tile * 4 + lane_id;
        const uint32_t x_idx_step = warps_per_block * f16x2_per_x_tile * 4;

        if (mi == 0) {
            for (uint32_t i = thread_id; i < codebook_half2; i += blockDim.x) {
                smem_codebook[i] = codebook[i];
            }
            __syncthreads();
        }

        __shared__ ditto2 x_buf[2][ANS_BLOCK_SIZE / ANS_WARP_SIZE][4][4];

#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ++ki) {
            if (ki % 2 == 0) {
                __syncwarp();
                x_buf[0][warp_id][lane_id / 8][lane_id % 4].u32[(lane_id % 8) / 4] =
                    ld_x_ans(reinterpret_cast<const uint32_t*>(x) + x_idx);
                __syncwarp();
                x_idx += x_idx_step;
            }

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; ++subki) {
                ditto2 reg_a = {};
                if (lane_id < 4) {
                    reg_a.u32x2 = x_buf[0][warp_id][ki % 2 * 2 + subki][lane_id].u32x2;
                }

#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; ++submi) {
                    ditto4 reg_w = {};
                    decode_weight_fragment_ans<ProbBits>(
                        smem_codebook,
                        ans_lookup,
                        state,
                        in,
                        lane_id,
                        reg_w);

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+f"(reg_p[submi].x), "+f"(reg_p[submi].y), "+f"(reg_p[submi].z), "+f"(reg_p[submi].w)
                        : "r"(reg_w.u32[0]),
                          "r"(reg_w.u32[1]),
                          "r"(reg_w.u32[2]),
                          "r"(reg_w.u32[3]),
                          "r"(reg_a.u32[0]),
                          "r"(reg_a.u32[1]));
                }
            }

            if (ki % 2 == 0) {
                prefetch_ans(reinterpret_cast<uint32_t*>(const_cast<half2*>(x + x_idx + x_idx_step * 4)));
            }
        }

        __shared__ __align__(16 * 8 * 32) float reduce_gather[ANS_BLOCK_SIZE / ANS_WARP_SIZE][2][16];
        if (lane_id % 4 == 0) {
            for (int pi = 0; pi < 2; ++pi) {
                reduce_gather[warp_id][pi][lane_id / 4] = reg_p[pi].x;
                reduce_gather[warp_id][pi][lane_id / 4 + 8] = reg_p[pi].z;
            }
        }
        __syncthreads();

        float reduced = 0.0f;
        if (warp_id < 1) {
            const int pi = lane_id / 16;
            for (int warpi = 0; warpi < ANS_BLOCK_SIZE / ANS_WARP_SIZE; ++warpi) {
                reduced += reduce_gather[warpi][pi][lane_id % 16];
            }

            float* out_tile = out + (tile_id_m * 2) * f32_per_out_tile;
            out_tile[lane_id] = reduced;
        }

        if constexpr (m_per_block > 1) {
            __syncthreads();
        }
        tile_id_m += 1;
    }
}

template <int ProbBits, uint32_t S, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__host__ static void ans_fused_matvec_ptr(
    float* __restrict__ out,
    const uint16_t* __restrict__ ans_words,
    const uint32_t* __restrict__ ans_states,
    const uint32_t* __restrict__ ans_stream_starts,
    const uint32_t* __restrict__ ans_stream_words,
    const half2* __restrict__ x,
    const half2* __restrict__ codebook,
    const uint32_t* __restrict__ ans_lookup,
    CUstream_st* stream) {
    static_assert(S + V == 9, "ANS fused path expects 8-bit symbol -> 256 half2 codebook entries");
    static_assert(M % ANS_MMA_M == 0);
    static_assert(N == 1);
    static_assert(K % ANS_MMA_K == 0);

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    assert(device_prop.warpSize == ANS_WARP_SIZE);

    constexpr uint32_t grid_size = ANS_BLOCK_COUNT;
    constexpr uint32_t block_size = ANS_BLOCK_SIZE;
    constexpr uint32_t smem_codebook_size = (1u << (S + V - 1)) * sizeof(half2);

    cudaFuncSetAttribute(
        kernel_ans_fused_matvec<ProbBits, S, V, M, N, K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_codebook_size);

    kernel_ans_fused_matvec<ProbBits, S, V, M, N, K><<<grid_size, block_size, smem_codebook_size, stream>>>(
        out,
        ans_words,
        ans_states,
        ans_stream_starts,
        ans_stream_words,
        x,
        codebook,
        ans_lookup);

    gpuErrchk(cudaPeekAtLastError());
}

template <int ProbBits, uint32_t M, uint32_t N, uint32_t K>
__global__ static void __launch_bounds__(ANS_BLOCK_SIZE, 1) kernel_ans_uniform_fused_matvec(
    float* __restrict__ out,
    const uint16_t* __restrict__ ans_words,
    const uint32_t* __restrict__ ans_states,
    const uint32_t* __restrict__ ans_stream_starts,
    const uint32_t* __restrict__ ans_stream_words,
    const half2* __restrict__ x,
    const uint64_t* __restrict__ ans_lookup_u16,
    float step_size,
    int32_t symbol_offset) {
    extern __shared__ uint64_t smem_ans_lookup_u16[];

    const uint32_t thread_id = threadIdx.x;
    const uint32_t lane_id = thread_id % ANS_WARP_SIZE;
    const uint32_t warp_id = thread_id / ANS_WARP_SIZE;
    const uint32_t block_id = blockIdx.x;
    (void)step_size;
    (void)symbol_offset;

    constexpr uint32_t tile_count_m = M / ANS_MMA_M;
    constexpr uint32_t tile_count_k = K / ANS_MMA_K;
    constexpr uint32_t warps_per_block = ANS_BLOCK_SIZE / ANS_WARP_SIZE;
    constexpr uint32_t lookup_size = 1u << ProbBits;

    static_assert(tile_count_m % 2 == 0);
    constexpr uint32_t m_per_block = ROUND_UP(tile_count_m, (2 * ANS_BLOCK_COUNT));
    constexpr uint32_t k_per_block = tile_count_k / (warps_per_block * 4) * 2;
    static_assert((tile_count_k % (warps_per_block * 4)) % 4 == 0);
    constexpr uint32_t extra_warps = (tile_count_k % (warps_per_block * 4)) / 4;
    uint32_t this_warp_k = k_per_block;
    if constexpr (extra_warps > 0) {
        if (warp_id < extra_warps) {
            this_warp_k = k_per_block + 2;
        }
    }

    constexpr uint32_t f16x2_per_x_tile = ANS_MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = ANS_MMA_M;

    for (uint32_t i = thread_id; i < lookup_size; i += blockDim.x) {
        smem_ans_lookup_u16[i] = ans_lookup_u16[i];
    }
    __syncthreads();

    uint32_t tile_id_m = m_per_block * block_id;

    for (uint32_t mi = 0; mi < m_per_block; ++mi) {
        if (tile_id_m * 2 >= tile_count_m) {
            return;
        }

        const uint32_t stream_idx = tile_id_m * warps_per_block + warp_id;
        uint32_t state = ans_states[stream_idx * ANS_WARP_SIZE + lane_id];
        const uint32_t start = ans_stream_starts[stream_idx];
        const uint32_t words = ans_stream_words[stream_idx];
        const uint16_t* in = ans_words + start + words;

        float4 reg_p[2] = {};

        uint32_t x_idx = warp_id * f16x2_per_x_tile * 4 + lane_id;
        const uint32_t x_idx_step = warps_per_block * f16x2_per_x_tile * 4;

        __shared__ ditto2 x_buf[2][ANS_BLOCK_SIZE / ANS_WARP_SIZE][4][4];

#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ++ki) {
            if (ki % 2 == 0) {
                __syncwarp();
                x_buf[0][warp_id][lane_id / 8][lane_id % 4].u32[(lane_id % 8) / 4] =
                    ld_x_ans(reinterpret_cast<const uint32_t*>(x) + x_idx);
                __syncwarp();
                x_idx += x_idx_step;
            }

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; ++subki) {
                ditto2 reg_a = {};
                if (lane_id < 4) {
                    reg_a.u32x2 = x_buf[0][warp_id][ki % 2 * 2 + subki][lane_id].u32x2;
                }

#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; ++submi) {
                    ditto4 reg_w = {};
                    decode_weight_fragment_ans_uniform<ProbBits>(
                        smem_ans_lookup_u16,
                        state,
                        in,
                        lane_id,
                        reg_w);

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+f"(reg_p[submi].x), "+f"(reg_p[submi].y), "+f"(reg_p[submi].z), "+f"(reg_p[submi].w)
                        : "r"(reg_w.u32[0]),
                          "r"(reg_w.u32[1]),
                          "r"(reg_w.u32[2]),
                          "r"(reg_w.u32[3]),
                          "r"(reg_a.u32[0]),
                          "r"(reg_a.u32[1]));
                }
            }

            if (ki % 2 == 0) {
                prefetch_ans(reinterpret_cast<uint32_t*>(const_cast<half2*>(x + x_idx + x_idx_step * 4)));
            }
        }

        __shared__ __align__(16 * 8 * 32) float reduce_gather[ANS_BLOCK_SIZE / ANS_WARP_SIZE][2][16];
        if (lane_id % 4 == 0) {
            for (int pi = 0; pi < 2; ++pi) {
                reduce_gather[warp_id][pi][lane_id / 4] = reg_p[pi].x;
                reduce_gather[warp_id][pi][lane_id / 4 + 8] = reg_p[pi].z;
            }
        }
        __syncthreads();

        float reduced = 0.0f;
        if (warp_id < 1) {
            const int pi = lane_id / 16;
            for (int warpi = 0; warpi < ANS_BLOCK_SIZE / ANS_WARP_SIZE; ++warpi) {
                reduced += reduce_gather[warpi][pi][lane_id % 16];
            }

            float* out_tile = out + (tile_id_m * 2) * f32_per_out_tile;
            out_tile[lane_id] = reduced;
        }

        if constexpr (m_per_block > 1) {
            __syncthreads();
        }
        tile_id_m += 1;
    }
}

template <int ProbBits, uint32_t M, uint32_t N, uint32_t K>
__host__ static void ans_uniform_fused_matvec_ptr(
    float* __restrict__ out,
    const uint16_t* __restrict__ ans_words,
    const uint32_t* __restrict__ ans_states,
    const uint32_t* __restrict__ ans_stream_starts,
    const uint32_t* __restrict__ ans_stream_words,
    const half2* __restrict__ x,
    const uint64_t* __restrict__ ans_lookup_u16,
    float step_size,
    int32_t symbol_offset,
    CUstream_st* stream) {
    static_assert(M % ANS_MMA_M == 0);
    static_assert(N == 1);
    static_assert(K % ANS_MMA_K == 0);

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    assert(device_prop.warpSize == ANS_WARP_SIZE);

    constexpr uint32_t grid_size = ANS_BLOCK_COUNT;
    constexpr uint32_t block_size = ANS_BLOCK_SIZE;
    constexpr uint32_t smem_lookup_u16_size = (1u << ProbBits) * sizeof(uint64_t);

    cudaFuncSetAttribute(
        kernel_ans_uniform_fused_matvec<ProbBits, M, N, K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_lookup_u16_size);

    kernel_ans_uniform_fused_matvec<ProbBits, M, N, K>
        <<<grid_size, block_size, smem_lookup_u16_size, stream>>>(
        out,
        ans_words,
        ans_states,
        ans_stream_starts,
        ans_stream_words,
        x,
        ans_lookup_u16,
        step_size,
        symbol_offset);

    gpuErrchk(cudaPeekAtLastError());
}

template <int ProbBits, uint32_t M, uint32_t N, uint32_t K>
__host__ static void ans_uniform_fused_matvec_ptr_batched(
    float* __restrict__ out,
    const uint16_t* __restrict__ ans_words,
    const uint32_t* __restrict__ ans_states,
    const uint32_t* __restrict__ ans_stream_starts,
    const uint32_t* __restrict__ ans_stream_words,
    const half2* __restrict__ x,
    const uint64_t* __restrict__ ans_lookup_u16,
    float step_size,
    int32_t symbol_offset,
    int64_t batch_size,
    int64_t out_stride_f32,
    int64_t x_stride_half2,
    CUstream_st* stream) {
    static_assert(M % ANS_MMA_M == 0);
    static_assert(N == 1);
    static_assert(K % ANS_MMA_K == 0);

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    assert(device_prop.warpSize == ANS_WARP_SIZE);

    constexpr uint32_t grid_size = ANS_BLOCK_COUNT;
    constexpr uint32_t block_size = ANS_BLOCK_SIZE;
    constexpr uint32_t smem_lookup_u16_size = (1u << ProbBits) * sizeof(uint64_t);

    cudaFuncSetAttribute(
        kernel_ans_uniform_fused_matvec<ProbBits, M, N, K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_lookup_u16_size);

    for (int64_t b = 0; b < batch_size; ++b) {
        float* out_b = out + b * out_stride_f32;
        const half2* x_b = x + b * x_stride_half2;

        kernel_ans_uniform_fused_matvec<ProbBits, M, N, K>
            <<<grid_size, block_size, smem_lookup_u16_size, stream>>>(
            out_b,
            ans_words,
            ans_states,
            ans_stream_starts,
            ans_stream_words,
            x_b,
            ans_lookup_u16,
            step_size,
            symbol_offset);
    }

    gpuErrchk(cudaPeekAtLastError());
}
