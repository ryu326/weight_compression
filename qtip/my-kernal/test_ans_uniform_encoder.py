import argparse
from typing import Iterable, Tuple

import torch

import my_kernels
from ans_uniform_encoder import encode_uniform_2d_to_ans


def _run_case(
    m: int,
    k: int,
    prob_bits: int,
    step_size: float,
    seed: int,
    atol: float,
    rtol: float,
) -> Tuple[float, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    w = (torch.randn((m, k), device="cuda", dtype=torch.float32) * 0.2).contiguous()
    x = (torch.randn((k, 1), device="cuda", dtype=torch.float16) * 0.2).contiguous()

    enc = encode_uniform_2d_to_ans(
        w,
        step_size=step_size,
        prob_bits=prob_bits,
        device="cuda",
    )

    out = torch.zeros((m, 1), dtype=torch.float32, device="cuda")
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

    q = torch.round(w / step_size).to(torch.int32)
    w_ref = (q.to(torch.float32) * step_size).to(torch.float16)
    ref = (w_ref @ x).to(torch.float32)

    diff = (out - ref).abs()
    mae = float(diff.mean().item())
    maxe = float(diff.max().item())

    if not torch.allclose(out, ref, atol=atol, rtol=rtol):
        raise AssertionError(
            f"allclose failed for (M,K)=({m},{k}), prob_bits={prob_bits}, step={step_size}. "
            f"mae={mae:.6e}, maxe={maxe:.6e}, atol={atol}, rtol={rtol}"
        )

    return mae, maxe


def _default_cases(quick: bool) -> Iterable[Tuple[int, int, int, float]]:
    if quick:
        return [
            (256, 256, 9, 0.02),
        ]
    return [
        (256, 256, 9, 0.02),
        (1024, 3072, 11, 0.03),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate encode_uniform_2d_to_ans + decompress_matvec_ans_uniform"
    )
    parser.add_argument("--quick", action="store_true", help="Run only the smallest case")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--atol", type=float, default=8e-2, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=8e-2, help="Relative tolerance")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    cases = list(_default_cases(args.quick))
    for i, (m, k, prob_bits, step_size) in enumerate(cases):
        mae, maxe = _run_case(
            m=m,
            k=k,
            prob_bits=prob_bits,
            step_size=step_size,
            seed=args.seed + i,
            atol=args.atol,
            rtol=args.rtol,
        )
        print(
            f"[PASS] (M,K)=({m},{k}) prob_bits={prob_bits} step={step_size} "
            f"mae={mae:.6e} maxe={maxe:.6e}"
        )


if __name__ == "__main__":
    main()
