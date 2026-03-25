import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch


def _setup_paths() -> None:
    this_dir = Path(__file__).resolve().parent
    qtip_root = this_dir.parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    if str(qtip_root) not in sys.path:
        sys.path.insert(0, str(qtip_root))


def _parse_cases(case_text: str) -> Iterable[Tuple[int, int]]:
    for item in case_text.split(","):
        item = item.strip()
        if not item:
            continue
        m_str, k_str = item.split("x")
        yield int(m_str), int(k_str)


def _parse_target_bits(bits_text: str) -> Iterable[float]:
    for item in bits_text.split(","):
        item = item.strip()
        if not item:
            continue
        yield float(item)


@torch.no_grad()
def _time_ms_torch_event(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(iters)


@torch.no_grad()
def _warmup_dual(fn1, fn2, warmup: int) -> None:
    for i in range(warmup):
        if i % 2 == 0:
            fn1()
        else:
            fn2()
    torch.cuda.synchronize()


@torch.no_grad()
def _time_ms_qtip_dual(fn1, fn2, iters: int) -> float:
    try:
        from cuda import cuda as cuda_driver  # type: ignore
    except Exception as exc:
        try:
            # cuda-python >= 13 often exposes driver API here.
            from cuda.bindings import driver as cuda_driver  # type: ignore
        except Exception as exc2:
            raise RuntimeError(
                "qtip-style timer requires cuda-python driver bindings "
                "(`from cuda import cuda` or `from cuda.bindings import driver`). "
                "Install/repair cuda-python or use `--timer torch_event`."
            ) from exc2

    start_events = [cuda_driver.cuEventCreate(0)[1] for _ in range(iters)]
    end_events = [cuda_driver.cuEventCreate(0)[1] for _ in range(iters)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream = torch.cuda.current_stream().cuda_stream
        for i in range(iters):
            cuda_driver.cuEventRecordWithFlags(start_events[i], stream, 1)
            if i % 2 == 0:
                fn1()
            else:
                fn2()
            cuda_driver.cuEventRecordWithFlags(end_events[i], stream, 1)

    graph.replay()
    torch.cuda.synchronize()

    elapsed_time_ms = sum(
        cuda_driver.cuEventElapsedTime(se, ee)[1]
        for se, ee in zip(start_events, end_events)
    )
    return elapsed_time_ms / float(iters)


@torch.no_grad()
def _bench_one_case(
    m: int,
    k: int,
    prob_bits: int,
    step_size: float,
    target_bits: Optional[float],
    batch: int,
    warmup: int,
    iters: int,
    seed: int,
    step_search_iters: int,
    entropy_sample_size: int,
    timer: str,
):
    import my_kernels  # type: ignore
    from ans_uniform_encoder import (  # type: ignore
        encode_uniform_2d_to_ans,
        encode_uniform_2d_to_ans_target_bits,
        estimate_entropy_for_step,
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    w_fp32 = (torch.randn((m, k), device="cuda", dtype=torch.float32) * 0.2).contiguous()
    w_fp32_2 = (torch.randn((m, k), device="cuda", dtype=torch.float32) * 0.2).contiguous()
    x_fp16 = (torch.randn((batch, k), device="cuda", dtype=torch.float16) * 0.2).contiguous()
    x_fp16_2 = (torch.randn((batch, k), device="cuda", dtype=torch.float16) * 0.2).contiguous()

    if target_bits is None:
        encoded = encode_uniform_2d_to_ans(
            w_fp32,
            step_size=step_size,
            prob_bits=prob_bits,
            device="cuda",
        )
        encoded_2 = encode_uniform_2d_to_ans(
            w_fp32_2,
            step_size=step_size,
            prob_bits=prob_bits,
            device="cuda",
        )
    else:
        encoded = encode_uniform_2d_to_ans_target_bits(
            w_fp32,
            target_bits=target_bits,
            prob_bits=prob_bits,
            device="cuda",
            search_iters=step_search_iters,
            sample_size=entropy_sample_size,
        )
        encoded_2 = encode_uniform_2d_to_ans_target_bits(
            w_fp32_2,
            target_bits=target_bits,
            prob_bits=prob_bits,
            device="cuda",
            search_iters=step_search_iters,
            sample_size=entropy_sample_size,
        )

    out_ans = torch.empty((batch, m), device="cuda", dtype=torch.float32)
    out_ans_2 = torch.empty((batch, m), device="cuda", dtype=torch.float32)
    w_t_fp16 = w_fp32.to(torch.float16).t().contiguous()
    w_t_fp16_2 = w_fp32_2.to(torch.float16).t().contiguous()
    out_fp16 = torch.empty((batch, m), device="cuda", dtype=torch.float16)
    out_fp16_2 = torch.empty((batch, m), device="cuda", dtype=torch.float16)

    def run_ans() -> None:
        my_kernels.decompress_matvec_ans_uniform_batched(
            out_ans,
            encoded.ans_words,
            encoded.ans_states,
            encoded.ans_stream_starts,
            encoded.ans_stream_words,
            encoded.ans_lookup_u16,
            x_fp16,
            encoded.step_size,
            encoded.symbol_offset,
            encoded.prob_bits,
        )

    def run_ans_2() -> None:
        my_kernels.decompress_matvec_ans_uniform_batched(
            out_ans_2,
            encoded_2.ans_words,
            encoded_2.ans_states,
            encoded_2.ans_stream_starts,
            encoded_2.ans_stream_words,
            encoded_2.ans_lookup_u16,
            x_fp16_2,
            encoded_2.step_size,
            encoded_2.symbol_offset,
            encoded_2.prob_bits,
        )

    def run_fp16() -> None:
        torch.mm(x_fp16, w_t_fp16, out=out_fp16)

    def run_fp16_2() -> None:
        torch.mm(x_fp16_2, w_t_fp16_2, out=out_fp16_2)

    if timer == "torch_event":
        ans_ms = _time_ms_torch_event(run_ans, warmup=warmup, iters=iters)
        fp16_ms = _time_ms_torch_event(run_fp16, warmup=warmup, iters=iters)
    elif timer == "qtip_dual":
        _warmup_dual(run_ans, run_ans_2, warmup=warmup)
        ans_ms = _time_ms_qtip_dual(run_ans, run_ans_2, iters=iters)
        _warmup_dual(run_fp16, run_fp16_2, warmup=warmup)
        fp16_ms = _time_ms_qtip_dual(run_fp16, run_fp16_2, iters=iters)
    else:
        raise ValueError(f"Unsupported timer: {timer}")

    speedup_vs_fp16 = fp16_ms / ans_ms
    ans_us = ans_ms * 1000.0
    fp16_us = fp16_ms * 1000.0
    avg_ans_words_bytes = int(
        (encoded.ans_words.numel() * 2 + encoded_2.ans_words.numel() * 2) / 2
    )
    ans_words_gbps = 0.0
    if ans_ms > 0.0:
        ans_words_gbps = (avg_ans_words_bytes / 1e9) / (ans_ms / 1000.0)

    entropy_est = estimate_entropy_for_step(
        w_fp32.detach().to(torch.float32).cpu().numpy(),
        step_size=encoded.step_size,
        sample_size=entropy_sample_size,
    )

    return {
        "m": m,
        "k": k,
        "batch": batch,
        "target_bits": target_bits,
        "step_size": float(encoded.step_size),
        "entropy_est": float(entropy_est),
        "ans_us": ans_us,
        "fp16_us": fp16_us,
        "speedup_vs_fp16": speedup_vs_fp16,
        "ans_words_gbps": ans_words_gbps,
        "ans_words_bytes": avg_ans_words_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ANS-uniform fused matvec vs dense FP16 matvec."
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="256x256,1024x3072,4096x4096,4096x11008,11008x4096",
        help="Comma-separated MxK list. Must be supported by ANS kernel.",
    )
    parser.add_argument("--prob-bits", type=int, default=9, choices=[9, 10, 11])
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument(
        "--target-bits",
        type=str,
        default="",
        help="Comma-separated target entropy bits (e.g., '2,4'). If set, step_size is searched per case.",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument(
        "--timer",
        type=str,
        default="torch_event",
        choices=["torch_event", "qtip_dual"],
        help="`qtip_dual` uses qtip-style CUDAGraph+cuEvent dual replay timing.",
    )
    parser.add_argument("--step-search-iters", type=int, default=15)
    parser.add_argument("--entropy-sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    _setup_paths()

    dev = torch.cuda.get_device_properties(0)
    target_bits_list = list(_parse_target_bits(args.target_bits))
    if not target_bits_list:
        target_bits_list = [None]

    print(f"GPU: {dev.name}")
    if target_bits_list == [None]:
        print(
            f"Settings: prob_bits={args.prob_bits}, step_size={args.step_size}, "
            f"batch={args.batch}, warmup={args.warmup}, iters={args.iters}, timer={args.timer}"
        )
    else:
        print(
            f"Settings: prob_bits={args.prob_bits}, target_bits={target_bits_list}, "
            f"batch={args.batch}, warmup={args.warmup}, iters={args.iters}, "
            f"step_search_iters={args.step_search_iters}, entropy_sample_size={args.entropy_sample_size}, "
            f"timer={args.timer}"
        )

    all_rows = []
    idx = 0
    for m, k in _parse_cases(args.cases):
        for target_bits in target_bits_list:
            row = _bench_one_case(
                m=m,
                k=k,
                prob_bits=args.prob_bits,
                step_size=args.step_size,
                target_bits=target_bits,
                batch=args.batch,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + idx,
                step_search_iters=args.step_search_iters,
                entropy_sample_size=args.entropy_sample_size,
                timer=args.timer,
            )
            idx += 1
            all_rows.append(row)

            target_str = (
                f"target_bits={target_bits:g}"
                if target_bits is not None
                else "target_bits=None"
            )
            print(
                f"(M,K)=({m},{k}) {target_str} "
                f"step={row['step_size']:.6f} H~{row['entropy_est']:.3f} "
                f"ANS={row['ans_us']:.2f}us "
                f"FP16={row['fp16_us']:.2f}us "
                f"speedup(fp16/ans)={row['speedup_vs_fp16']:.3f}x "
                f"ans_words={row['ans_words_bytes']}B "
                f"ans_words_bw={row['ans_words_gbps']:.2f}GB/s"
            )

    for target_bits in target_bits_list:
        rows = [r for r in all_rows if r["target_bits"] == target_bits]
        faster = [r for r in rows if r["speedup_vs_fp16"] > 1.0]
        target_str = f"{target_bits:g}" if target_bits is not None else "None"
        print(
            f"Summary target_bits={target_str}: ANS faster than FP16 in "
            f"{len(faster)}/{len(rows)} cases (speedup > 1 means ANS faster)."
        )


if __name__ == "__main__":
    main()
