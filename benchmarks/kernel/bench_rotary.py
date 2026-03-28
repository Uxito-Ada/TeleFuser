#!/usr/bin/env python
"""Benchmarks for RoPE (Rotary Position Embedding) Triton kernel versus PyTorch."""

import argparse
import itertools
import os
import time
from typing import List

import torch

from telefuser.kernel.triton import apply_rotary_embedding

IS_CI = os.getenv("CI", "false").lower() == "true" or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"


def torch_apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """PyTorch native RoPE using interleaved format (matching Triton kernel)."""
    head_dim = x.shape[-1]
    
    # Interleaved format: pairs of adjacent elements
    x1 = x[..., 0::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    # Broadcast cos/sin to match x shape
    # x: [batch, seq, num_heads, head_dim]
    # cos/sin: [seq, head_dim//2]
    # Need: [1, seq, 1, head_dim//2] for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim//2]
    
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    
    # Interleave the output
    return torch.stack([o1, o2], dim=-1).flatten(-2)


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
    default_batch_sizes, default_seq_lens = [1, 4], [64, 128]
else:
    default_batch_sizes, default_seq_lens = [1, 4, 16, 64], [64, 128, 256, 512, 1024]


def main():
    parser = argparse.ArgumentParser("RoPE Kernel Benchmark")
    parser.add_argument("--batch_sizes", type=str2int_list, default=default_batch_sizes)
    parser.add_argument("--seq_lens", type=str2int_list, default=default_seq_lens)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--head_size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"\n{'#' * 80}\n RoPE Kernel Benchmark\n{'#' * 80}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Seq lens: {args.seq_lens}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head size: {args.head_size}")

    print(f"\n{'=' * 80}")
    print(f" RoPE Results")
    print(f"{'=' * 80}")
    header = f"{'Batch':>6} {'Seq':>6} {'Heads':>6} {'HeadDim':>8} {'pytorch':>12} {'telefuser':>12} {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    device, dtype = torch.device("cuda"), torch.bfloat16
    for bs, sl in itertools.product(args.batch_sizes, args.seq_lens):
        x = torch.randn(bs, sl, args.num_heads, args.head_size, dtype=dtype, device=device)
        cos = torch.randn(sl, args.head_size // 2, dtype=dtype, device=device)
        sin = torch.randn(sl, args.head_size // 2, dtype=dtype, device=device)

        t_pytorch = benchmark_fn(lambda: torch_apply_rotary_embedding(x, cos, sin), args.warmup, args.repeat)
        t_telefuser = benchmark_fn(lambda: apply_rotary_embedding(x, cos, sin), args.warmup, args.repeat)
        speedup = t_pytorch / t_telefuser

        print(f"{bs:>6} {sl:>6} {args.num_heads:>6} {args.head_size:>8} {t_pytorch:>10.1f}µs {t_telefuser:>10.1f}µs {speedup:>10.2f}x")

    print(f"{'=' * 80}\n✅ Done!\n")


if __name__ == "__main__":
    main()