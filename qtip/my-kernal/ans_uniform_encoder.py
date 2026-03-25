from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


ANS_MMA_M = 16
ANS_MMA_K = 16
ANS_WARP_SIZE = 32
ANS_BLOCK_SIZE = 1024

_K_ANS_STATE_BITS = 31
_K_ANS_ENCODED_BITS = 16
_K_ANS_MIN_STATE = 1 << (_K_ANS_STATE_BITS - _K_ANS_ENCODED_BITS)

SUPPORTED_SHAPES = {
    (256, 256),
    (1024, 3072),
    (3072, 3072),
    (3072, 8192),
    (4096, 4096),
    (4096, 11008),
    (11008, 4096),
    (8192, 8192),
    (16384, 16384),
}


@dataclass
class UniformANSEncoded:
    ans_words: torch.Tensor
    ans_states: torch.Tensor
    ans_stream_starts: torch.Tensor
    ans_stream_words: torch.Tensor
    ans_lookup_u16: torch.Tensor
    symbol_offset: int
    step_size: float
    prob_bits: int
    m: int
    k: int


def calc_entropy(q_vals: np.ndarray) -> float:
    flat = q_vals.reshape(-1)
    if flat.size == 0:
        return 0.0
    _, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _prepare_entropy_sample(
    data_np: np.ndarray,
    sample_size: int = 50000,
) -> np.ndarray:
    flat = data_np.reshape(-1)
    if flat.size <= sample_size:
        return flat
    return flat[:sample_size]


def find_step_size_for_target_bits(
    data_np: np.ndarray,
    target_bits: float,
    low: float = 1e-3,
    high: float = 10.0,
    search_iters: int = 15,
    sample_size: int = 50000,
) -> float:
    if target_bits <= 0:
        raise ValueError(f"target_bits must be > 0, got {target_bits}")
    if low <= 0 or high <= 0 or low >= high:
        raise ValueError(f"Invalid search range: low={low}, high={high}")
    if search_iters <= 0:
        raise ValueError(f"search_iters must be > 0, got {search_iters}")

    sample = _prepare_entropy_sample(data_np.astype(np.float64, copy=False), sample_size=sample_size)

    h_high = calc_entropy(np.rint(sample / high))
    expand_cnt = 0
    while h_high > target_bits and expand_cnt < 20:
        high *= 2.0
        h_high = calc_entropy(np.rint(sample / high))
        expand_cnt += 1

    best_step = high
    for _ in range(search_iters):
        mid = (low + high) / 2.0
        q_vals = np.rint(sample / mid)
        h = calc_entropy(q_vals)
        if h > target_bits:
            low = mid
        else:
            high = mid
            best_step = mid

    return float(best_step)


def estimate_entropy_for_step(
    data_np: np.ndarray,
    step_size: float,
    sample_size: int = 50000,
) -> float:
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    sample = _prepare_entropy_sample(data_np.astype(np.float64, copy=False), sample_size=sample_size)
    q_vals = np.rint(sample / step_size)
    return calc_entropy(q_vals)


def _check_shape(m: int, k: int) -> None:
    if (m, k) not in SUPPORTED_SHAPES:
        raise ValueError(
            f"Unsupported shape (M, K)=({m}, {k}). "
            f"Supported: {sorted(SUPPORTED_SHAPES)}"
        )
    if m % ANS_MMA_M != 0 or k % ANS_MMA_K != 0:
        raise ValueError("M and K must be multiples of 16.")


def _build_pdf_cdf(symbols: np.ndarray, prob_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    total_prob = 1 << prob_bits
    max_sym = int(symbols.max())
    counts = np.bincount(symbols.reshape(-1), minlength=max_sym + 1).astype(np.int64)
    present = np.flatnonzero(counts > 0)

    if present.size == 0:
        raise ValueError("No symbols to encode.")
    if present.size > total_prob:
        raise ValueError(
            f"Too many unique symbols ({present.size}) for prob_bits={prob_bits} "
            f"(max {total_prob}). Increase prob_bits or use larger step_size."
        )

    # Ensure every present symbol has at least one bucket.
    pdf = np.zeros_like(counts, dtype=np.int64)
    remaining = total_prob - int(present.size)

    if remaining == 0:
        pdf[present] = 1
    else:
        pcounts = counts[present].astype(np.float64)
        scaled = pcounts / pcounts.sum() * float(remaining)
        add = np.floor(scaled).astype(np.int64)
        pdf_present = 1 + add
        left = remaining - int(add.sum())
        if left > 0:
            residual = scaled - add
            order = np.argsort(-residual)
            pdf_present[order[:left]] += 1
        pdf[present] = pdf_present

    cdf = np.zeros_like(pdf, dtype=np.int64)
    running = 0
    for sym in present:
        cdf[sym] = running
        running += int(pdf[sym])

    if running != total_prob:
        raise RuntimeError(f"Invalid pdf sum: expected {total_prob}, got {running}")

    return pdf, cdf


def _build_lookup_u16(
    pdf: np.ndarray,
    cdf: np.ndarray,
    prob_bits: int,
    symbol_offset: int,
    step_size: float,
) -> np.ndarray:
    total_prob = 1 << prob_bits
    lookup = np.zeros(total_prob, dtype=np.int64)
    present = np.flatnonzero(pdf > 0)
    dequant_half_bits = (
        (np.arange(pdf.shape[0], dtype=np.float32) + np.float32(symbol_offset))
        * np.float32(step_size)
    ).astype(np.float16).view(np.uint16)

    for sym in present:
        p = int(pdf[sym])
        c = int(cdf[sym])
        weight_half_bits = int(dequant_half_bits[sym])
        if p > 0xFFFF:
            raise ValueError(f"pdf too large for uint16: sym={sym}, pdf={p}")
        for off in range(p):
            # [15:0] dequantized weight half bits, [31:16] pdf, [47:32] s_minus_cdf
            lookup[c + off] = weight_half_bits | (p << 16) | (off << 32)

    return lookup


def _encode_one_stream(
    symbol_steps: np.ndarray,
    pdf: np.ndarray,
    cdf: np.ndarray,
    prob_bits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # symbol_steps: [num_steps, 32], decode order
    states = np.full(ANS_WARP_SIZE, _K_ANS_MIN_STATE, dtype=np.int64)
    words = []

    for t in range(symbol_steps.shape[0] - 1, -1, -1):
        syms = symbol_steps[t]
        p = pdf[syms]
        c = cdf[syms]

        max_state = p << (_K_ANS_STATE_BITS - prob_bits)
        write_mask = states >= max_state
        if np.any(write_mask):
            lanes = np.nonzero(write_mask)[0]
            for lane in lanes:
                words.append(np.uint16(states[lane] & 0xFFFF))
            states[write_mask] >>= _K_ANS_ENCODED_BITS

        states = ((states // p) << prob_bits) + (states % p) + c

    return states.astype(np.int32), np.asarray(words, dtype=np.uint16)


def _warp_this_k(tile_count_k: int, warp_id: int, warps_per_block: int) -> int:
    k_per_block = (tile_count_k // (warps_per_block * 4)) * 2
    extra_warps = (tile_count_k % (warps_per_block * 4)) // 4
    return k_per_block + (2 if warp_id < extra_warps else 0)


def _tile_symbol_steps_for_stream(
    symbols: np.ndarray,
    tile_pair: int,
    warp_id: int,
) -> np.ndarray:
    # symbols shape: [M, K], uint16
    m, k = symbols.shape
    tile_count_k = k // ANS_MMA_K
    warps_per_block = ANS_BLOCK_SIZE // ANS_WARP_SIZE
    this_warp_k = _warp_this_k(tile_count_k, warp_id, warps_per_block)

    steps = this_warp_k * 32
    if steps == 0:
        return np.zeros((0, ANS_WARP_SIZE), dtype=np.int64)

    out = np.empty((steps, ANS_WARP_SIZE), dtype=np.int64)
    step_off = 0

    for ki in range(this_warp_k):
        # Matches x tile traversal in kernel_ans_uniform_fused_matvec.
        k_tile_base = (ki // 2) * (warps_per_block * 4) + warp_id * 4 + (ki % 2) * 2
        for subki in range(2):
            k_tile = k_tile_base + subki
            col0 = k_tile * ANS_MMA_K

            for submi in range(2):
                row0 = (tile_pair * 2 + submi) * ANS_MMA_M

                # Per-lane 8 symbols for one mma.sync m16n8k16 A-fragment:
                # e0,e1,e2,e3,e4,e5,e6,e7
                # Verified against runtime probing of the kernel.
                for lane in range(ANS_WARP_SIZE):
                    r_base = lane >> 2
                    c_base = (lane & 3) * 2

                    out[step_off + 0, lane] = symbols[row0 + r_base + 0, col0 + c_base + 0]
                    out[step_off + 1, lane] = symbols[row0 + r_base + 0, col0 + c_base + 1]
                    out[step_off + 2, lane] = symbols[row0 + r_base + 8, col0 + c_base + 0]
                    out[step_off + 3, lane] = symbols[row0 + r_base + 8, col0 + c_base + 1]
                    out[step_off + 4, lane] = symbols[row0 + r_base + 0, col0 + c_base + 8]
                    out[step_off + 5, lane] = symbols[row0 + r_base + 0, col0 + c_base + 9]
                    out[step_off + 6, lane] = symbols[row0 + r_base + 8, col0 + c_base + 8]
                    out[step_off + 7, lane] = symbols[row0 + r_base + 8, col0 + c_base + 9]

                step_off += 8

    if step_off != steps:
        raise RuntimeError(f"Invalid step count: expected {steps}, got {step_off}")

    return out


def encode_uniform_2d_to_ans(
    weight_2d: torch.Tensor,
    step_size: float,
    prob_bits: int = 9,
    device: torch.device | str | None = None,
) -> UniformANSEncoded:
    """
    Encode a 2D weight tensor for `my_kernels.decompress_matvec_ans_uniform`.

    Returns tensors:
    - ans_words: int16
    - ans_states: int32 (flattened [num_streams * 32])
    - ans_stream_starts: int32
    - ans_stream_words: int32
    - ans_lookup_u16: int64 packed decode lookup
    plus symbol_offset/step/prob_bits metadata.
    """
    if weight_2d.dim() != 2:
        raise ValueError(f"weight_2d must be 2D, got shape {tuple(weight_2d.shape)}")
    if step_size <= 0:
        raise ValueError("step_size must be > 0")
    if prob_bits not in (9, 10, 11):
        raise ValueError("prob_bits must be one of {9, 10, 11}")

    m, k = int(weight_2d.shape[0]), int(weight_2d.shape[1])
    _check_shape(m, k)

    # Quantize on CPU for deterministic integer processing.
    w_np = weight_2d.detach().to(torch.float32).cpu().numpy()
    q = np.rint(w_np / step_size).astype(np.int64)

    symbol_offset = int(q.min())
    symbols = q - symbol_offset
    if symbols.min() < 0:
        raise RuntimeError("Internal error: negative symbol after offset.")
    if symbols.max() > 0xFFFF:
        raise ValueError(
            f"symbol range too large for uint16: max symbol={int(symbols.max())}. "
            "Increase step_size."
        )

    symbols_u16 = symbols.astype(np.uint16)
    pdf, cdf = _build_pdf_cdf(symbols_u16.astype(np.int64), prob_bits)
    lookup_u16 = _build_lookup_u16(
        pdf,
        cdf,
        prob_bits,
        symbol_offset=symbol_offset,
        step_size=step_size,
    )

    tile_pairs = (m // ANS_MMA_M) // 2
    warps_per_block = ANS_BLOCK_SIZE // ANS_WARP_SIZE
    required_streams = tile_pairs * warps_per_block

    states = np.empty((required_streams, ANS_WARP_SIZE), dtype=np.int32)
    stream_starts = np.empty(required_streams, dtype=np.int32)
    stream_words = np.empty(required_streams, dtype=np.int32)
    words_chunks = []
    word_ptr = 0

    for tile_pair in range(tile_pairs):
        for warp_id in range(warps_per_block):
            stream_idx = tile_pair * warps_per_block + warp_id
            stream_starts[stream_idx] = word_ptr

            symbol_steps = _tile_symbol_steps_for_stream(symbols_u16, tile_pair, warp_id)
            if symbol_steps.shape[0] == 0:
                states[stream_idx, :] = _K_ANS_MIN_STATE
                stream_words[stream_idx] = 0
                continue

            st, wd = _encode_one_stream(symbol_steps, pdf, cdf, prob_bits)
            states[stream_idx, :] = st
            stream_words[stream_idx] = int(wd.shape[0])
            if wd.size > 0:
                words_chunks.append(wd)
                word_ptr += int(wd.shape[0])

    if words_chunks:
        all_words_u16 = np.concatenate(words_chunks, axis=0)
    else:
        all_words_u16 = np.zeros((0,), dtype=np.uint16)

    if device is None:
        if weight_2d.is_cuda:
            device = weight_2d.device
        else:
            device = torch.device("cuda")
    device = torch.device(device)

    ans_words_t = torch.from_numpy(all_words_u16.view(np.int16)).to(device=device, dtype=torch.int16)
    ans_states_t = torch.from_numpy(states.reshape(-1)).to(device=device, dtype=torch.int32)
    ans_stream_starts_t = torch.from_numpy(stream_starts).to(device=device, dtype=torch.int32)
    ans_stream_words_t = torch.from_numpy(stream_words).to(device=device, dtype=torch.int32)
    ans_lookup_u16_t = torch.from_numpy(lookup_u16).to(device=device, dtype=torch.int64)

    return UniformANSEncoded(
        ans_words=ans_words_t,
        ans_states=ans_states_t,
        ans_stream_starts=ans_stream_starts_t,
        ans_stream_words=ans_stream_words_t,
        ans_lookup_u16=ans_lookup_u16_t,
        symbol_offset=symbol_offset,
        step_size=float(step_size),
        prob_bits=int(prob_bits),
        m=m,
        k=k,
    )


def encode_uniform_2d_to_ans_target_bits(
    weight_2d: torch.Tensor,
    target_bits: float,
    prob_bits: int = 9,
    device: torch.device | str | None = None,
    search_low: float = 1e-3,
    search_high: float = 10.0,
    search_iters: int = 15,
    sample_size: int = 50000,
) -> UniformANSEncoded:
    if weight_2d.dim() != 2:
        raise ValueError(f"weight_2d must be 2D, got shape {tuple(weight_2d.shape)}")

    data_np = weight_2d.detach().to(torch.float32).cpu().numpy()
    step_size = find_step_size_for_target_bits(
        data_np=data_np,
        target_bits=target_bits,
        low=search_low,
        high=search_high,
        search_iters=search_iters,
        sample_size=sample_size,
    )
    return encode_uniform_2d_to_ans(
        weight_2d=weight_2d,
        step_size=step_size,
        prob_bits=prob_bits,
        device=device,
    )


def pack_for_kernel(encoded: UniformANSEncoded) -> Dict[str, torch.Tensor | int | float]:
    return {
        "ans_words": encoded.ans_words,
        "ans_states": encoded.ans_states,
        "ans_stream_starts": encoded.ans_stream_starts,
        "ans_stream_words": encoded.ans_stream_words,
        "ans_lookup_u16": encoded.ans_lookup_u16,
        "symbol_offset": encoded.symbol_offset,
        "step_size": encoded.step_size,
        "prob_bits": encoded.prob_bits,
    }
