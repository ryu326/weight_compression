#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/extension.h>
#include <torch/types.h>

#include <limits>

#include "my_ans_inference.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    do {               \
        CHECK_CUDA(x); \
        CHECK_CONTIGUOUS(x); \
    } while (false)

template <int ProbBits, uint32_t M, uint32_t N, uint32_t K>
__host__ static void launch_ans_fused_matvec(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup,
    torch::Tensor& x,
    torch::Tensor& codebook) {
    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 2, "out must be 2D");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out dtype must be float32");

    CHECK_INPUT(ans_words);
    TORCH_CHECK(ans_words.dim() == 1, "ans_words must be 1D");
    TORCH_CHECK(ans_words.scalar_type() == torch::kInt16, "ans_words dtype must be int16");

    CHECK_INPUT(ans_states);
    TORCH_CHECK(ans_states.dim() == 1, "ans_states must be 1D");
    TORCH_CHECK(ans_states.scalar_type() == torch::kInt32, "ans_states dtype must be int32");

    CHECK_INPUT(ans_stream_starts);
    TORCH_CHECK(ans_stream_starts.dim() == 1, "ans_stream_starts must be 1D");
    TORCH_CHECK(ans_stream_starts.scalar_type() == torch::kInt32, "ans_stream_starts dtype must be int32");

    CHECK_INPUT(ans_stream_words);
    TORCH_CHECK(ans_stream_words.dim() == 1, "ans_stream_words must be 1D");
    TORCH_CHECK(ans_stream_words.scalar_type() == torch::kInt32, "ans_stream_words dtype must be int32");

    CHECK_INPUT(ans_lookup);
    TORCH_CHECK(ans_lookup.dim() == 1, "ans_lookup must be 1D");
    TORCH_CHECK(ans_lookup.scalar_type() == torch::kInt32, "ans_lookup dtype must be int32");

    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x dtype must be float16");

    CHECK_INPUT(codebook);
    TORCH_CHECK(codebook.dim() == 1, "codebook must be 1D");
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat16, "codebook dtype must be float16");

    const int64_t m = out.size(0);
    const int64_t n = out.size(1);
    const int64_t k = x.size(0);

    TORCH_CHECK(m == static_cast<int64_t>(M), "M mismatch");
    TORCH_CHECK(n == static_cast<int64_t>(N), "N mismatch");
    TORCH_CHECK(k == static_cast<int64_t>(K), "K mismatch");
    TORCH_CHECK(x.size(1) == n, "x second dim must match out second dim");

    TORCH_CHECK(ans_lookup.size(0) == (1 << ProbBits), "ans_lookup size mismatch");
    TORCH_CHECK(codebook.size(0) == (1 << (8 + 1)), "codebook size mismatch: expected 512 (half)");

    constexpr int64_t warps_per_block = 1024 / 32;
    constexpr int64_t tile_count_m = M / 16;
    constexpr int64_t tile_pairs = tile_count_m / 2;
    constexpr int64_t required_streams = tile_pairs * warps_per_block;
    constexpr int64_t states_per_stream = 32;

    TORCH_CHECK(ans_stream_starts.numel() >= required_streams, "ans_stream_starts too small");
    TORCH_CHECK(ans_stream_words.numel() >= required_streams, "ans_stream_words too small");
    TORCH_CHECK(ans_states.numel() >= required_streams * states_per_stream, "ans_states too small");

    at::DeviceGuard guard(x.device());

    ans_fused_matvec_ptr<ProbBits, 8U, 1U, M, N, K>(
        reinterpret_cast<float*>(out.data_ptr<float>()),
        reinterpret_cast<const uint16_t*>(ans_words.data_ptr<int16_t>()),
        reinterpret_cast<const uint32_t*>(ans_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(ans_stream_starts.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(ans_stream_words.data_ptr<int32_t>()),
        reinterpret_cast<const half2*>(x.data_ptr<c10::Half>()),
        reinterpret_cast<const half2*>(codebook.data_ptr<c10::Half>()),
        reinterpret_cast<const uint32_t*>(ans_lookup.data_ptr<int32_t>()),
        at::cuda::getCurrentCUDAStream());
}

template <int ProbBits, uint32_t M, uint32_t N, uint32_t K>
__host__ static void launch_ans_uniform_fused_matvec(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup_u16,
    torch::Tensor& x,
    double step_size,
    int64_t symbol_offset) {
    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 2, "out must be 2D");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out dtype must be float32");

    CHECK_INPUT(ans_words);
    TORCH_CHECK(ans_words.dim() == 1, "ans_words must be 1D");
    TORCH_CHECK(ans_words.scalar_type() == torch::kInt16, "ans_words dtype must be int16");

    CHECK_INPUT(ans_states);
    TORCH_CHECK(ans_states.dim() == 1, "ans_states must be 1D");
    TORCH_CHECK(ans_states.scalar_type() == torch::kInt32, "ans_states dtype must be int32");

    CHECK_INPUT(ans_stream_starts);
    TORCH_CHECK(ans_stream_starts.dim() == 1, "ans_stream_starts must be 1D");
    TORCH_CHECK(ans_stream_starts.scalar_type() == torch::kInt32, "ans_stream_starts dtype must be int32");

    CHECK_INPUT(ans_stream_words);
    TORCH_CHECK(ans_stream_words.dim() == 1, "ans_stream_words must be 1D");
    TORCH_CHECK(ans_stream_words.scalar_type() == torch::kInt32, "ans_stream_words dtype must be int32");

    CHECK_INPUT(ans_lookup_u16);
    TORCH_CHECK(ans_lookup_u16.dim() == 1, "ans_lookup_u16 must be 1D");
    TORCH_CHECK(ans_lookup_u16.scalar_type() == torch::kInt64, "ans_lookup_u16 dtype must be int64");

    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x dtype must be float16");

    TORCH_CHECK(step_size > 0.0, "step_size must be positive");
    TORCH_CHECK(
        symbol_offset >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
            symbol_offset <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "symbol_offset must fit int32");

    const int64_t m = out.size(0);
    const int64_t n = out.size(1);
    const int64_t k = x.size(0);

    TORCH_CHECK(m == static_cast<int64_t>(M), "M mismatch");
    TORCH_CHECK(n == static_cast<int64_t>(N), "N mismatch");
    TORCH_CHECK(k == static_cast<int64_t>(K), "K mismatch");
    TORCH_CHECK(x.size(1) == n, "x second dim must match out second dim");

    TORCH_CHECK(ans_lookup_u16.size(0) == (1 << ProbBits), "ans_lookup_u16 size mismatch");

    constexpr int64_t warps_per_block = 1024 / 32;
    constexpr int64_t tile_count_m = M / 16;
    constexpr int64_t tile_pairs = tile_count_m / 2;
    constexpr int64_t required_streams = tile_pairs * warps_per_block;
    constexpr int64_t states_per_stream = 32;

    TORCH_CHECK(ans_stream_starts.numel() >= required_streams, "ans_stream_starts too small");
    TORCH_CHECK(ans_stream_words.numel() >= required_streams, "ans_stream_words too small");
    TORCH_CHECK(ans_states.numel() >= required_streams * states_per_stream, "ans_states too small");

    at::DeviceGuard guard(x.device());

    ans_uniform_fused_matvec_ptr<ProbBits, M, 1U, K>(
        reinterpret_cast<float*>(out.data_ptr<float>()),
        reinterpret_cast<const uint16_t*>(ans_words.data_ptr<int16_t>()),
        reinterpret_cast<const uint32_t*>(ans_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(ans_stream_starts.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(ans_stream_words.data_ptr<int32_t>()),
        reinterpret_cast<const half2*>(x.data_ptr<c10::Half>()),
        reinterpret_cast<const uint64_t*>(ans_lookup_u16.data_ptr<int64_t>()),
        static_cast<float>(step_size),
        static_cast<int32_t>(symbol_offset),
        at::cuda::getCurrentCUDAStream());
}

template <int ProbBits, uint32_t M, uint32_t N, uint32_t K>
__host__ static void launch_ans_uniform_fused_matvec_batched(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup_u16,
    torch::Tensor& x,
    double step_size,
    int64_t symbol_offset) {
    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 2, "out must be 2D [B, M]");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out dtype must be float32");

    CHECK_INPUT(ans_words);
    TORCH_CHECK(ans_words.dim() == 1, "ans_words must be 1D");
    TORCH_CHECK(ans_words.scalar_type() == torch::kInt16, "ans_words dtype must be int16");

    CHECK_INPUT(ans_states);
    TORCH_CHECK(ans_states.dim() == 1, "ans_states must be 1D");
    TORCH_CHECK(ans_states.scalar_type() == torch::kInt32, "ans_states dtype must be int32");

    CHECK_INPUT(ans_stream_starts);
    TORCH_CHECK(ans_stream_starts.dim() == 1, "ans_stream_starts must be 1D");
    TORCH_CHECK(ans_stream_starts.scalar_type() == torch::kInt32, "ans_stream_starts dtype must be int32");

    CHECK_INPUT(ans_stream_words);
    TORCH_CHECK(ans_stream_words.dim() == 1, "ans_stream_words must be 1D");
    TORCH_CHECK(ans_stream_words.scalar_type() == torch::kInt32, "ans_stream_words dtype must be int32");

    CHECK_INPUT(ans_lookup_u16);
    TORCH_CHECK(ans_lookup_u16.dim() == 1, "ans_lookup_u16 must be 1D");
    TORCH_CHECK(ans_lookup_u16.scalar_type() == torch::kInt64, "ans_lookup_u16 dtype must be int64");

    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, K]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x dtype must be float16");

    TORCH_CHECK(step_size > 0.0, "step_size must be positive");
    TORCH_CHECK(
        symbol_offset >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
            symbol_offset <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "symbol_offset must fit int32");

    const int64_t batch = x.size(0);
    const int64_t k = x.size(1);
    const int64_t out_batch = out.size(0);
    const int64_t m = out.size(1);

    TORCH_CHECK(batch == out_batch, "Batch mismatch between x and out");
    TORCH_CHECK(m == static_cast<int64_t>(M), "M mismatch");
    TORCH_CHECK(k == static_cast<int64_t>(K), "K mismatch");
    TORCH_CHECK(out.stride(1) == 1, "out must be contiguous in last dimension");
    TORCH_CHECK(x.stride(1) == 1, "x must be contiguous in last dimension");
    TORCH_CHECK((x.stride(0) % 2) == 0, "x row stride must be even for half2");

    TORCH_CHECK(ans_lookup_u16.size(0) == (1 << ProbBits), "ans_lookup_u16 size mismatch");

    constexpr int64_t warps_per_block = 1024 / 32;
    constexpr int64_t tile_count_m = M / 16;
    constexpr int64_t tile_pairs = tile_count_m / 2;
    constexpr int64_t required_streams = tile_pairs * warps_per_block;
    constexpr int64_t states_per_stream = 32;

    TORCH_CHECK(ans_stream_starts.numel() >= required_streams, "ans_stream_starts too small");
    TORCH_CHECK(ans_stream_words.numel() >= required_streams, "ans_stream_words too small");
    TORCH_CHECK(ans_states.numel() >= required_streams * states_per_stream, "ans_states too small");

    at::DeviceGuard guard(x.device());

    ans_uniform_fused_matvec_ptr_batched<ProbBits, M, 1U, K>(
        reinterpret_cast<float*>(out.data_ptr<float>()),
        reinterpret_cast<const uint16_t*>(ans_words.data_ptr<int16_t>()),
        reinterpret_cast<const uint32_t*>(ans_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(ans_stream_starts.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(ans_stream_words.data_ptr<int32_t>()),
        reinterpret_cast<const half2*>(x.data_ptr<c10::Half>()),
        reinterpret_cast<const uint64_t*>(ans_lookup_u16.data_ptr<int64_t>()),
        static_cast<float>(step_size),
        static_cast<int32_t>(symbol_offset),
        batch,
        out.stride(0),
        x.stride(0) / 2,
        at::cuda::getCurrentCUDAStream());
}

template <int ProbBits>
__host__ static void dispatch_shape_ans(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup,
    torch::Tensor& x,
    torch::Tensor& codebook) {
    const int64_t m = out.size(0);
    const int64_t k = x.size(0);

#define MY_ANS_SHAPE_CASE(M, K)                                                                                     \
    if (m == static_cast<int64_t>(M) && k == static_cast<int64_t>(K)) {                                            \
        launch_ans_fused_matvec<ProbBits, M, 1U, K>(                                                                \
            out, ans_words, ans_states, ans_stream_starts, ans_stream_words, ans_lookup, x, codebook);            \
        return;                                                                                                     \
    }

    MY_ANS_SHAPE_CASE(256U, 256U)
    MY_ANS_SHAPE_CASE(1024U, 3072U)
    MY_ANS_SHAPE_CASE(3072U, 3072U)
    MY_ANS_SHAPE_CASE(3072U, 8192U)
    MY_ANS_SHAPE_CASE(4096U, 4096U)
    MY_ANS_SHAPE_CASE(4096U, 11008U)
    MY_ANS_SHAPE_CASE(11008U, 4096U)
    MY_ANS_SHAPE_CASE(8192U, 8192U)
    MY_ANS_SHAPE_CASE(16384U, 16384U)

#undef MY_ANS_SHAPE_CASE

    TORCH_CHECK(
        false,
        "Unsupported (M, K)=(",
        m,
        ", ",
        k,
        ") for ANS fused dispatch. Add a new MY_ANS_SHAPE_CASE in my_torch_ans.cu.");
}

template <int ProbBits>
__host__ static void dispatch_shape_ans_uniform(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup_u16,
    torch::Tensor& x,
    double step_size,
    int64_t symbol_offset) {
    const int64_t m = out.size(0);
    const int64_t k = x.size(0);

#define MY_ANS_UNIFORM_SHAPE_CASE(M, K)                                                                            \
    if (m == static_cast<int64_t>(M) && k == static_cast<int64_t>(K)) {                                           \
        launch_ans_uniform_fused_matvec<ProbBits, M, 1U, K>(                                                      \
            out,                                                                                                   \
            ans_words,                                                                                             \
            ans_states,                                                                                            \
            ans_stream_starts,                                                                                     \
            ans_stream_words,                                                                                      \
            ans_lookup_u16,                                                                                        \
            x,                                                                                                     \
            step_size,                                                                                             \
            symbol_offset);                                                                                        \
        return;                                                                                                    \
    }

    MY_ANS_UNIFORM_SHAPE_CASE(256U, 256U)
    MY_ANS_UNIFORM_SHAPE_CASE(1024U, 3072U)
    MY_ANS_UNIFORM_SHAPE_CASE(3072U, 3072U)
    MY_ANS_UNIFORM_SHAPE_CASE(3072U, 8192U)
    MY_ANS_UNIFORM_SHAPE_CASE(4096U, 4096U)
    MY_ANS_UNIFORM_SHAPE_CASE(4096U, 11008U)
    MY_ANS_UNIFORM_SHAPE_CASE(11008U, 4096U)
    MY_ANS_UNIFORM_SHAPE_CASE(8192U, 8192U)
    MY_ANS_UNIFORM_SHAPE_CASE(16384U, 16384U)

#undef MY_ANS_UNIFORM_SHAPE_CASE

    TORCH_CHECK(
        false,
        "Unsupported (M, K)=(",
        m,
        ", ",
        k,
        ") for ANS uniform fused dispatch. Add a new shape case in my_torch_ans.cu.");
}

template <int ProbBits>
__host__ static void dispatch_shape_ans_uniform_batched(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup_u16,
    torch::Tensor& x,
    double step_size,
    int64_t symbol_offset) {
    const int64_t m = out.size(1);
    const int64_t k = x.size(1);

#define MY_ANS_UNIFORM_BATCHED_SHAPE_CASE(M, K)                                                                    \
    if (m == static_cast<int64_t>(M) && k == static_cast<int64_t>(K)) {                                           \
        launch_ans_uniform_fused_matvec_batched<ProbBits, M, 1U, K>(                                              \
            out,                                                                                                   \
            ans_words,                                                                                             \
            ans_states,                                                                                            \
            ans_stream_starts,                                                                                     \
            ans_stream_words,                                                                                      \
            ans_lookup_u16,                                                                                        \
            x,                                                                                                     \
            step_size,                                                                                             \
            symbol_offset);                                                                                        \
        return;                                                                                                    \
    }

    MY_ANS_UNIFORM_BATCHED_SHAPE_CASE(256U, 256U)
    MY_ANS_UNIFORM_BATCHED_SHAPE_CASE(1024U, 3072U)
    MY_ANS_UNIFORM_BATCHED_SHAPE_CASE(3072U, 3072U)
    MY_ANS_UNIFORM_BATCHED_SHAPE_CASE(3072U, 8192U)
    MY_ANS_UNIFORM_BATCHED_SHAPE_CASE(4096U, 4096U)
    MY_ANS_UNIFORM_BATCHED_SHAPE_CASE(4096U, 11008U)
    MY_ANS_UNIFORM_BATCHED_SHAPE_CASE(11008U, 4096U)
    MY_ANS_UNIFORM_BATCHED_SHAPE_CASE(8192U, 8192U)
    MY_ANS_UNIFORM_BATCHED_SHAPE_CASE(16384U, 16384U)

#undef MY_ANS_UNIFORM_BATCHED_SHAPE_CASE

    TORCH_CHECK(
        false,
        "Unsupported (M, K)=(",
        m,
        ", ",
        k,
        ") for ANS uniform batched fused dispatch. Add a new shape case in my_torch_ans.cu.");
}

__host__ void decompress_matvec_ans(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup,
    torch::Tensor& x,
    torch::Tensor& codebook,
    int64_t prob_bits) {
    switch (prob_bits) {
        case 9:
            dispatch_shape_ans<9>(
                out, ans_words, ans_states, ans_stream_starts, ans_stream_words, ans_lookup, x, codebook);
            return;
        case 10:
            dispatch_shape_ans<10>(
                out, ans_words, ans_states, ans_stream_starts, ans_stream_words, ans_lookup, x, codebook);
            return;
        case 11:
            dispatch_shape_ans<11>(
                out, ans_words, ans_states, ans_stream_starts, ans_stream_words, ans_lookup, x, codebook);
            return;
        default:
            TORCH_CHECK(false, "prob_bits must be one of {9, 10, 11}");
    }
}

__host__ void decompress_matvec_ans_uniform_batched(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup_u16,
    torch::Tensor& x,
    double step_size,
    int64_t symbol_offset,
    int64_t prob_bits) {
    switch (prob_bits) {
        case 9:
            dispatch_shape_ans_uniform_batched<9>(
                out,
                ans_words,
                ans_states,
                ans_stream_starts,
                ans_stream_words,
                ans_lookup_u16,
                x,
                step_size,
                symbol_offset);
            return;
        case 10:
            dispatch_shape_ans_uniform_batched<10>(
                out,
                ans_words,
                ans_states,
                ans_stream_starts,
                ans_stream_words,
                ans_lookup_u16,
                x,
                step_size,
                symbol_offset);
            return;
        case 11:
            dispatch_shape_ans_uniform_batched<11>(
                out,
                ans_words,
                ans_states,
                ans_stream_starts,
                ans_stream_words,
                ans_lookup_u16,
                x,
                step_size,
                symbol_offset);
            return;
        default:
            TORCH_CHECK(false, "prob_bits must be one of {9, 10, 11}");
    }
}

__host__ void decompress_matvec_ans_uniform(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup_u16,
    torch::Tensor& x,
    double step_size,
    int64_t symbol_offset,
    int64_t prob_bits) {
    switch (prob_bits) {
        case 9:
            dispatch_shape_ans_uniform<9>(
                out,
                ans_words,
                ans_states,
                ans_stream_starts,
                ans_stream_words,
                ans_lookup_u16,
                x,
                step_size,
                symbol_offset);
            return;
        case 10:
            dispatch_shape_ans_uniform<10>(
                out,
                ans_words,
                ans_states,
                ans_stream_starts,
                ans_stream_words,
                ans_lookup_u16,
                x,
                step_size,
                symbol_offset);
            return;
        case 11:
            dispatch_shape_ans_uniform<11>(
                out,
                ans_words,
                ans_states,
                ans_stream_starts,
                ans_stream_words,
                ans_lookup_u16,
                x,
                step_size,
                symbol_offset);
            return;
        default:
            TORCH_CHECK(false, "prob_bits must be one of {9, 10, 11}");
    }
}
