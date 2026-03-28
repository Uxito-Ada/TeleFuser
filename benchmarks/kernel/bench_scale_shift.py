#!/usr/bin/env python
"""Benchmarks for fused scale and shift Triton kernel versus PyTorch."""

import argparse
import itertools
import os
import time
from typing import List

import torch

from telefuser.kernel.triton import fused_scale_shift

IS_CI = os.getenv("CI", "false").lower() == "true" or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"


def torch_fused_scale_shift(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, scale_constant: float = 1.0) -> torch.Tensor:
    """PyTorch native scale and shift."""
    return x * (scale_constant + scale) + shift


def benchmark_fn(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    return sorted(times)[len(times) // 2]


def str2int_list(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",")] if arg else []


if IS_CI:
    default_batch_sizes, default_seq_lens, default_hidden_sizes = [1, 4], [64, 128], [512, 1024]
else:
    default_batch_sizes, default_seq_lens, default_hidden_sizes = [1, 4, 16, 64], [64, 128, 256, 512, 1024], [1024, 4096, 6144]


def main():
    parser = argparse.ArgumentParser("Scale & Shift Kernel Benchmark")
    parser.add_argument("--batch_sizes", type=str2int_list, default=default_batch_sizes)
    parser.add_argument("--seq_lens", type=str2int_list, default=default_seq_lens)
    parser.add_argument("--hidden_sizes", type=str2int_list, default=default_hidden_sizes)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"\n{'#' * 80}\n Scale & Shift Kernel Benchmark\n{'#' * 80}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Seq lens: {args.seq_lens}")
    print(f"Hidden sizes: {args.hidden_sizes}")

    print(f"\n{'=' * 80}")
    print(f" Scale & Shift Results")
    print(f"{'=' * 80}")
    header = f"{'Batch':>6} {'Seq':>6} {'Hidden':>8} {'pytorch':>12} {'telefuser':>12} {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    device, dtype = torch.device("cuda"), torch.bfloat16
    for bs, sl, hs in itertools.product(args.batch_sizes, args.seq_lens, args.hidden_sizes):
        x = torch.randn(bs, sl, hs, dtype=dtype, device=device)
        scale = torch.randn(bs, hs, dtype=dtype, device=device)
        shift = torch.randn(bs, hs, dtype=dtype, device=device)

        # Expand for PyTorch
        scale_exp = scale[:, None, :].expand(bs, sl, hs)
        shift_exp = shift[:, None, :].expand(bs, sl, hs)

        t_pytorch = benchmark_fn(lambda: torch_fused_scale_shift(x, scale_exp, shift_exp), args.warmup, args.repeat)
        t_telefuser = benchmark_fn(lambda: fused_scale_shift(x, scale, shift), args.warmup, args.repeat)
        speedup = t_pytorch / t_telefuser

        print(f"{bs:>6} {sl:>6} {hs:>8} {t_pytorch:>10.1f}µs {t_telefuser:>10.1f}µs {speedup:>10.2f}x")

    print(f"{'=' * 80}\n✅ Done!\n")


if __name__ == "__main__":
    main()