#!/usr/bin/env python
"""Benchmarks for RMSNorm Triton kernel versus PyTorch and tf-kernel."""

import argparse
import itertools
import os
import time
from typing import List, Optional, Tuple

import torch

try:
    import tf_kernel
    TF_KERNEL_AVAILABLE = True
except ImportError:
    TF_KERNEL_AVAILABLE = False

from telefuser.kernel.triton import fused_add_rms_norm, rms_norm

IS_CI = os.getenv("CI", "false").lower() == "true" or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def torch_fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_out = x + residual
    variance = residual_out.pow(2).mean(dim=-1, keepdim=True)
    return residual_out * torch.rsqrt(variance + eps) * weight, residual_out


def rmsnorm_tfkernel(x: torch.Tensor, weight: torch.Tensor, residual: Optional[torch.Tensor] = None, eps: float = 1e-6):
    if not TF_KERNEL_AVAILABLE:
        raise RuntimeError("tf_kernel not available")
    import tf_kernel as tf_ker
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    if residual is not None:
        residual_2d = residual.view(-1, residual.shape[-1])
        tf_ker.fused_add_rmsnorm(x_2d, residual_2d, weight, eps)
        return (x_2d.view(orig_shape), residual_2d.view(orig_shape))
    else:
        out = torch.empty_like(x_2d)
        tf_ker.rmsnorm(x_2d, weight, eps, out=out)
        return out.view(orig_shape)


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
        times.append((end - start) * 1e6 if (end := time.perf_counter()) else 0)
    return sorted(times)[len(times) // 2]


def str2int_list(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",")] if arg else []


if IS_CI:
    default_batch_sizes, default_seq_lens, default_hidden_sizes = [1, 4], [64, 128], [512, 1024]
else:
    default_batch_sizes, default_seq_lens, default_hidden_sizes = [1, 4, 16, 64], [64, 128, 256, 512, 1024], [1024, 4096, 6144]


def main():
    parser = argparse.ArgumentParser("RMSNorm Kernel Benchmark")
    parser.add_argument("--batch_sizes", type=str2int_list, default=default_batch_sizes)
    parser.add_argument("--seq_lens", type=str2int_list, default=default_seq_lens)
    parser.add_argument("--hidden_sizes", type=str2int_list, default=default_hidden_sizes)
    parser.add_argument("--use_residual", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"\n{'#' * 80}\n RMSNorm Kernel Benchmark\n{'#' * 80}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Seq lens: {args.seq_lens}")
    print(f"Hidden sizes: {args.hidden_sizes}")
    print(f"Use residual: {args.use_residual}")
    print(f"tf-kernel: {TF_KERNEL_AVAILABLE}")

    providers = ["pytorch", "telefuser"] + (["tf_kernel"] if TF_KERNEL_AVAILABLE else [])
    print(f"\n{'=' * 90}")
    print(f" RMSNorm {'(with residual)' if args.use_residual else ''} Results")
    print(f"{'=' * 90}")
    header = f"{'Batch':>6} {'Seq':>6} {'Hidden':>8}" + "".join([f" {p:>12}" for p in providers]) + f" {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    device, dtype = torch.device("cuda"), torch.bfloat16
    for bs, sl, hs in itertools.product(args.batch_sizes, args.seq_lens, args.hidden_sizes):
        x = torch.randn(bs, sl, hs, dtype=dtype, device=device)
        weight = torch.ones(hs, dtype=dtype, device=device)
        residual = torch.randn_like(x) if args.use_residual else None
        results = {}

        if args.use_residual and residual is not None:
            results["pytorch"] = benchmark_fn(lambda: torch_fused_add_rms_norm(x.clone(), residual.clone(), weight), args.warmup, args.repeat)
            results["telefuser"] = benchmark_fn(lambda: fused_add_rms_norm(x.clone(), residual.clone(), weight), args.warmup, args.repeat)
            if TF_KERNEL_AVAILABLE:
                results["tf_kernel"] = benchmark_fn(lambda: rmsnorm_tfkernel(x.clone(), residual.clone(), weight), args.warmup, args.repeat)
        else:
            results["pytorch"] = benchmark_fn(lambda: torch_rms_norm(x, weight), args.warmup, args.repeat)
            results["telefuser"] = benchmark_fn(lambda: rms_norm(x.clone(), weight), args.warmup, args.repeat)
            if TF_KERNEL_AVAILABLE:
                results["tf_kernel"] = benchmark_fn(lambda: rmsnorm_tfkernel(x.clone(), weight), args.warmup, args.repeat)

        row = f"{bs:>6} {sl:>6} {hs:>8}" + "".join([f" {results[p]:>10.1f}µs" if p in results else f" {'N/A':>12}" for p in providers])
        row += f" {results['pytorch'] / results['telefuser']:>10.2f}x"
        print(row)

    print(f"{'=' * 90}\n✅ Done!\n")


if __name__ == "__main__":
    main()