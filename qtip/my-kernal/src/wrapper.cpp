#include <torch/extension.h>

void decompress_matvec_ans(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup,
    torch::Tensor& x,
    torch::Tensor& codebook,
    int64_t prob_bits);

void decompress_matvec_ans_uniform(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup_u16,
    torch::Tensor& x,
    double step_size,
    int64_t symbol_offset,
    int64_t prob_bits);

void decompress_matvec_ans_uniform_batched(
    torch::Tensor& out,
    torch::Tensor& ans_words,
    torch::Tensor& ans_states,
    torch::Tensor& ans_stream_starts,
    torch::Tensor& ans_stream_words,
    torch::Tensor& ans_lookup_u16,
    torch::Tensor& x,
    double step_size,
    int64_t symbol_offset,
    int64_t prob_bits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "decompress_matvec_ans",
        &decompress_matvec_ans,
        "ANS-fused decompress+matvec kernel (my-kernal)");
    m.def(
        "decompress_matvec_ans_uniform",
        &decompress_matvec_ans_uniform,
        "ANS-fused uniform-quant decompress+matvec kernel (my-kernal)");
    m.def(
        "decompress_matvec_ans_uniform_batched",
        &decompress_matvec_ans_uniform_batched,
        "ANS-fused uniform-quant decompress+matvec kernel, batched x [B,K] -> out [B,M] (my-kernal)");
}
