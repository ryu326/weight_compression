import argparse
import sys
from pathlib import Path

import torch


def _setup_paths() -> None:
    this_dir = Path(__file__).resolve().parent
    qtip_root = this_dir.parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    if str(qtip_root) not in sys.path:
        sys.path.insert(0, str(qtip_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate torch.ops.quip_lib ANS-uniform wrapper against direct batched kernel."
    )
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--prob-bits", type=int, default=9, choices=[9, 10, 11])
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=8e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    _setup_paths()
    import my_kernels  # type: ignore
    from ans_uniform_encoder import encode_uniform_2d_to_ans  # type: ignore
    from lib.codebook import ans_uniform  # noqa: F401

    op_name = f"decompress_matvec_ans_uniform_{args.m}_1_{args.k}_{args.prob_bits}"
    if not hasattr(torch.ops.quip_lib, op_name):
        raise RuntimeError(f"Missing torch op: quip_lib::{op_name}")
    op = getattr(torch.ops.quip_lib, op_name)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    w = (torch.randn((args.m, args.k), device="cuda", dtype=torch.float32) * 0.2).contiguous()
    x = (torch.randn((args.batch, args.k), device="cuda", dtype=torch.float16) * 0.2).contiguous()
    enc = encode_uniform_2d_to_ans(
        w,
        step_size=args.step_size,
        prob_bits=args.prob_bits,
        device="cuda",
    )

    out_batched = torch.empty((args.batch, args.m), dtype=torch.float32, device="cuda")
    my_kernels.decompress_matvec_ans_uniform_batched(
        out_batched,
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

    out_ops = op(
        enc.ans_words,
        enc.ans_states,
        enc.ans_stream_starts,
        enc.ans_stream_words,
        enc.ans_lookup_u16,
        x,
        float(enc.step_size),
        int(enc.symbol_offset),
    )

    q = torch.round(w / args.step_size).to(torch.int32)
    w_ref = (q.to(torch.float32) * args.step_size).to(torch.float16)
    out_ref = (x @ w_ref.T).to(torch.float32)

    maxe_kernel = float((out_batched - out_ref).abs().max().item())
    maxe_ops = float((out_ops - out_ref).abs().max().item())
    maxe_between = float((out_batched - out_ops).abs().max().item())

    if not torch.allclose(out_batched, out_ref, atol=args.atol, rtol=args.rtol):
        raise AssertionError(
            f"direct batched kernel mismatch: maxe={maxe_kernel:.6e}, atol={args.atol}, rtol={args.rtol}"
        )
    if not torch.allclose(out_ops, out_ref, atol=args.atol, rtol=args.rtol):
        raise AssertionError(
            f"torch.ops wrapper mismatch: maxe={maxe_ops:.6e}, atol={args.atol}, rtol={args.rtol}"
        )
    if maxe_between != 0.0:
        raise AssertionError(
            f"torch.ops and direct batched kernel differ: maxe_between={maxe_between:.6e}"
        )

    print(
        f"[PASS] {op_name} batch={args.batch} "
        f"maxe_kernel={maxe_kernel:.6e} maxe_ops={maxe_ops:.6e}"
    )


if __name__ == "__main__":
    main()

