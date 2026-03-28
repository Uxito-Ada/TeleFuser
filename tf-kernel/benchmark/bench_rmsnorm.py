# Benchmarks tf_kernel RMSNorm kernels versus vLLM and FlashInfer across
# (batch_size, seq_len, hidden_size) and prints speed-up.
import argparse
import itertools
import os
import re
from typing import List, Optional, Tuple, Union

# Set matplotlib backend to non-interactive before importing pyplot
# to avoid blocking in environments without display
import matplotlib

matplotlib.use("Agg")

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback progress bar class
    class tqdm:
        def __init__(self, iterable=None, desc="", total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total
            self.n = 0
            if desc:
                print(f"[{desc}] Starting...")

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1

        def update(self, n=1):
            self.n += n

        def close(self):
            if self.desc:
                print(f"[{self.desc}] Completed.")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()


import torch
import torch.nn as nn
import triton
import triton.testing

import tf_kernel

# Optional imports
try:
    from flashinfer.norm import fused_add_rmsnorm, rmsnorm

    FLASHINFER_AVAILABLE = True
except ImportError:
    fused_add_rmsnorm = None
    rmsnorm = None
    FLASHINFER_AVAILABLE = False

try:
    from vllm import _custom_ops as vllm_ops

    VLLM_AVAILABLE = True
except ImportError:
    vllm_ops = None
    VLLM_AVAILABLE = False

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def str2int_list(arg: str) -> List[int]:
    if arg in ("", None):
        return []
    if re.fullmatch(r"\d+(,\d+)*", arg.strip()) is None:
        raise argparse.ArgumentTypeError(f"Bad int list: {arg}")
    return [int(x) for x in arg.split(",")]


class HuggingFaceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_flashinfer(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    if not FLASHINFER_AVAILABLE:
        # Fallback to naive implementation if FlashInfer is not available
        return rmsnorm_naive(x, weight, residual, eps)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        fused_add_rmsnorm(x, residual, weight, eps)
        output = (x, residual)
    else:
        output = rmsnorm(x, weight, eps)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    if not VLLM_AVAILABLE:
        # Fallback to naive implementation if vLLM is not available
        return rmsnorm_naive(x, weight, residual, eps)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_tfkernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        tf_kernel.fused_add_rmsnorm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        tf_kernel.rmsnorm(x, weight, eps, out=out)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def calculate_diff(batch_size, seq_len, hidden_size, use_residual=True):
    dtype = torch.bfloat16
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if use_residual else None

    output_naive = rmsnorm_naive(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_flashinfer = rmsnorm_flashinfer(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_vllm = rmsnorm_vllm(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_tfkernel = rmsnorm_tfkernel(
        x.clone(), weight, residual.clone() if residual is not None else None
    )

    if use_residual:
        output_naive = output_naive[0]
        output_flashinfer = output_flashinfer[0]
        output_vllm = output_vllm[0]
        output_tfkernel = output_tfkernel[0]

    print(f"Naive output={output_naive}")
    if FLASHINFER_AVAILABLE:
        print(f"FlashInfer output={output_flashinfer}")
    else:
        print("FlashInfer not available, skipped")
    if VLLM_AVAILABLE:
        print(f"VLLM output={output_vllm}")
    else:
        print("vLLM not available, skipped")
    print(f"TF Kernel output={output_tfkernel}")

    # Only compare available implementations
    all_match = torch.allclose(output_naive, output_tfkernel, atol=1e-2, rtol=1e-2)
    if FLASHINFER_AVAILABLE:
        all_match = all_match and torch.allclose(
            output_naive, output_flashinfer, atol=1e-2, rtol=1e-2
        )
    if VLLM_AVAILABLE:
        all_match = all_match and torch.allclose(
            output_naive, output_vllm, atol=1e-2, rtol=1e-2
        )

    if all_match:
        print("✅ All available implementations match")
    else:
        print("❌ Implementations differ")


# CI environment uses simplified parameters
if IS_CI:
    default_batch_sizes = [1]  # Single batch size for CI
    default_seq_lens = [64]  # Single sequence length for CI
    default_hidden_sizes = [4096]  # Single hidden size for CI
else:
    default_batch_sizes = [2**i for i in range(0, 7, 2)]  # 1, 4, 16, 64
    default_seq_lens = [2**i for i in range(6, 11, 1)]  # 64, 128, 256, 512, 1024
    default_hidden_sizes = [32 * 128, 48 * 128]  # 4096, 6144


def make_configs(bsizes: List[int], slens: List[int], hsizes: List[int]) -> List[Tuple]:
    return list(itertools.product(bsizes, slens, hsizes))


# Filter providers based on availability
available_providers = ["huggingface", "tfkernel"]
available_names = ["HuggingFace", "TF Kernel"]
available_styles = [("blue", "-"), ("orange", "-")]

if FLASHINFER_AVAILABLE:
    available_providers.insert(-1, "flashinfer")
    available_names.insert(-1, "FlashInfer")
    available_styles.insert(-1, ("green", "-"))

if VLLM_AVAILABLE:
    available_providers.insert(-1, "vllm")
    available_names.insert(-1, "vLLM")
    available_styles.insert(-1, ("red", "-"))


# Global progress bar instance for benchmark updates
_benchmark_pbar = None


def set_benchmark_pbar(pbar):
    """Set the global progress bar instance."""
    global _benchmark_pbar
    _benchmark_pbar = pbar


def update_benchmark_progress(n=1, desc=None):
    """Update the progress bar."""
    global _benchmark_pbar
    if _benchmark_pbar is not None:
        if desc:
            _benchmark_pbar.set_description(desc)
        _benchmark_pbar.update(n)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "hidden_size"],
        x_vals=[],
        line_arg="provider",
        line_vals=available_providers,
        line_names=available_names,
        styles=available_styles,
        ylabel="µs (median)  or  × (speed-up)",
        plot_name="rmsnorm-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, hidden_size, provider, use_residual):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x) if use_residual else None

    # timing helper
    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ms, qmin, qmax = triton.testing.do_bench_cudagraph(
            fn, quantiles=[0.5, 0.2, 0.8]
        )
        return 1000 * ms, 1000 * qmax, 1000 * qmin

    if provider == "huggingface":
        return timed(
            lambda: rmsnorm_naive(
                x.clone(),
                weight,
                residual.clone() if residual is not None else None,
            )
        )
    elif provider == "flashinfer":
        if not FLASHINFER_AVAILABLE:
            return (0, 0, 0)
        return timed(
            lambda: rmsnorm_flashinfer(
                x.clone(),
                weight,
                residual.clone() if residual is not None else None,
            )
        )
    elif provider == "vllm":
        if not VLLM_AVAILABLE:
            return (0, 0, 0)
        return timed(
            lambda: rmsnorm_vllm(
                x.clone(),
                weight,
                residual.clone() if residual is not None else None,
            )
        )
    elif provider == "tfkernel":
        return timed(
            lambda: rmsnorm_tfkernel(
                x.clone(),
                weight,
                residual.clone() if residual is not None else None,
            )
        )

    # provider == "speedup"
    if VLLM_AVAILABLE:
        t_ref, _, _ = timed(
            lambda: rmsnorm_vllm(
                x.clone(),
                weight,
                residual.clone() if residual is not None else None,
            )
        )
    else:
        t_ref, _, _ = timed(
            lambda: rmsnorm_naive(
                x.clone(),
                weight,
                residual.clone() if residual is not None else None,
            )
        )
    t_tfkernel, _, _ = timed(
        lambda: rmsnorm_tfkernel(
            x.clone(),
            weight,
            residual.clone() if residual is not None else None,
        )
    )
    spd = t_ref / t_tfkernel if t_ref > 0 else 1.0
    return (spd, spd, spd)


if __name__ == "__main__":
    p = argparse.ArgumentParser("RMSNorm kernel benchmark")
    p.add_argument("--batch_sizes", type=str2int_list, default=default_batch_sizes)
    p.add_argument("--seq_lens", type=str2int_list, default=default_seq_lens)
    p.add_argument("--hidden_sizes", type=str2int_list, default=default_hidden_sizes)
    p.add_argument(
        "--use_residual", action="store_true", help="Whether to use residual connection"
    )
    p.add_argument("--verify_only", action="store_true")
    args = p.parse_args()

    # coerce lists
    if isinstance(args.batch_sizes, str):
        args.batch_sizes = str2int_list(args.batch_sizes)
    if isinstance(args.seq_lens, str):
        args.seq_lens = str2int_list(args.seq_lens)
    if isinstance(args.hidden_sizes, str):
        args.hidden_sizes = str2int_list(args.hidden_sizes)

    # patch perf_report grid
    benchmark_grid = make_configs(args.batch_sizes, args.seq_lens, args.hidden_sizes)
    benchmark.benchmarks.x_vals = benchmark_grid

    # Calculate total iterations for progress bar
    total_configs = len(benchmark_grid)
    total_providers = len(available_providers)
    total_iterations = total_configs * total_providers

    print(f"\n📊 RMSNorm Benchmark")
    print(f"   Configs: {total_configs} (batch_size x seq_len x hidden_size)")
    print(f"   Providers: {total_providers} ({', '.join(available_providers)})")
    print(f"   Total iterations: {total_iterations}\n")

    if args.verify_only:
        ok = calculate_diff(4, 128, args.hidden_sizes[0], args.use_residual)
        print("✅ sanity pass" if ok else "❌ mismatch")
    else:
        # Monkey-patch the benchmark function to update progress
        original_fn = benchmark.fn

        def progress_wrapped_fn(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            # Extract info for progress description
            batch_size = (
                args[0] if len(args) > 0 else kwargs.get("batch_size", "unknown")
            )
            seq_len = args[1] if len(args) > 1 else kwargs.get("seq_len", "unknown")
            hidden_size = (
                args[2] if len(args) > 2 else kwargs.get("hidden_size", "unknown")
            )
            provider = args[3] if len(args) > 3 else kwargs.get("provider", "unknown")
            update_benchmark_progress(
                1,
                f"Testing batch={batch_size},seq={seq_len},hidden={hidden_size}/{provider}",
            )
            return result

        benchmark.fn = progress_wrapped_fn

        with tqdm(total=total_iterations, desc="Initializing", unit="test") as pbar:
            set_benchmark_pbar(pbar)
            benchmark.run(print_data=True, use_residual=args.use_residual)
            set_benchmark_pbar(None)

        print("\n✅ Benchmark finished!")
