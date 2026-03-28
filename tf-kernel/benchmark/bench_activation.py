# Benchmarks tf_kernel kernels versus vLLM and torch across
# (kernel, dtype, batch_size, seq_len, dim) and prints speed-up.
import argparse
import itertools
import os
import re
from typing import List, Tuple

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
import torch.nn.functional as F
import triton
import triton.testing

import tf_kernel

# Optional vLLM import
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

# gelu_quick is only available on HIP/ROCm platforms
try:
    from tf_kernel import gelu_quick

    GELU_QUICK_AVAILABLE = True
except ImportError:
    GELU_QUICK_AVAILABLE = False
    gelu_quick = None

if VLLM_AVAILABLE and not hasattr(vllm_ops, "silu_and_mul"):
    vllm_ops = torch.ops._C


def str2int_list(arg: str) -> List[int]:
    if arg in ("", None):
        return []
    if re.fullmatch(r"\d+(,\d+)*", arg.strip()) is None:
        raise argparse.ArgumentTypeError(f"Bad int list: {arg}")
    return [int(x) for x in arg.split(",")]


def calculate_diff(
    kernel: str, dtype: torch.dtype, batch_size: int, seq_len: int, dim: int
) -> bool:
    """Compare vLLM with tf_kernel for one shape."""
    device = torch.device("cuda")

    if not VLLM_AVAILABLE:
        print(
            f"[{kernel:14s} | {str(dtype):9s} | B={batch_size:3d} | "
            f"L={seq_len:3d} | D={dim:5d}] ⚠️  vLLM not available, skipping comparison"
        )
        return True

    # activation-only quick GELU
    if kernel == "gelu_quick":
        if not GELU_QUICK_AVAILABLE:
            print(
                f"[{kernel:14s} | {str(dtype):9s} | B={batch_size:3d} | "
                f"L={seq_len:3d} | D={dim:5d}] ⚠️  not available on this platform"
            )
            return True
        x = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
        ref_out = torch.zeros_like(x)
        getattr(vllm_ops, kernel)(ref_out, x)
        test_out = getattr(tf_kernel, kernel)(x)
    # fused activation x mul kernels
    else:
        x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device=device)
        ref_out = torch.zeros(batch_size, seq_len, dim, dtype=dtype, device=device)
        getattr(vllm_ops, kernel)(ref_out, x)
        test_out = getattr(tf_kernel, kernel)(x)

    ok = torch.allclose(ref_out, test_out, rtol=1e-3, atol=1e-5)
    tag = "✅ match" if ok else "❌ mismatch"
    print(
        f"[{kernel:14s} | {str(dtype):9s} | B={batch_size:3d} | "
        f"L={seq_len:3d} | D={dim:5d}] {tag}"
    )
    return ok


# CI environment uses simplified parameters for kernels and dtypes too
if IS_CI:
    kernels = ["silu_and_mul"]  # Only test one kernel in CI
    dtypes = [torch.float16]  # Only test one dtype in CI
else:
    kernels = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"]
    if GELU_QUICK_AVAILABLE:
        kernels.append("gelu_quick")
    dtypes = [torch.float16, torch.bfloat16]


def make_configs(bsizes: List[int], slens: List[int], dims_: List[int]) -> List[Tuple]:
    return list(itertools.product(kernels, dtypes, bsizes, slens, dims_))


# CI environment uses simplified parameters
if IS_CI:
    default_batch_sizes = [1]  # Single batch size for CI
    default_seq_lens = [1]  # Single sequence length for CI
    default_dims = [1024]  # Single dimension for CI
else:
    default_batch_sizes = [2**i for i in range(0, 5, 2)]  # 1,4,16
    default_seq_lens = [2**i for i in range(0, 8, 2)]  # 1,4,16,64
    default_dims = [2**i for i in range(10, 15)]  # 1024...16384


# Helper function to get torch native implementation
def get_torch_activation(x, kernel):
    """Get torch native activation function implementation."""
    if kernel == "silu_and_mul":
        # x shape: [batch, seq, 2*dim], split into [batch, seq, dim] each
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2
    elif kernel == "gelu_and_mul":
        x1, x2 = x.chunk(2, dim=-1)
        return F.gelu(x1) * x2
    elif kernel == "gelu_tanh_and_mul":
        x1, x2 = x.chunk(2, dim=-1)
        return F.gelu(x1, approximate="tanh") * x2
    elif kernel == "gelu_quick":
        return F.gelu(x, approximate="tanh")
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


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
        x_names=["kernel", "dtype", "batch_size", "seq_len", "dim"],
        x_vals=[],
        line_arg="provider",
        line_vals=["vllm", "torch", "tf_kernel", "speedup"],
        line_names=["vLLM", "Torch", "TF Kernel", "Speed-up (x)"],
        styles=[("blue", "-"), ("orange", "-"), ("green", "-"), ("red", "--")],
        ylabel="µs (median)  or  × (speed-up)",
        plot_name="activation-performance",
        args={},
    )
)
def benchmark(kernel, dtype, batch_size, seq_len, dim, provider):
    device = torch.device("cuda")
    in_mult = 1 if kernel == "gelu_quick" else 2
    x = torch.randn(batch_size, seq_len, in_mult * dim, dtype=dtype, device=device)
    y0 = torch.zeros(batch_size, seq_len, dim, dtype=dtype, device=device)

    if not VLLM_AVAILABLE and provider in ["vllm", "speedup"]:
        # Skip vLLM-related benchmarks if vLLM is not available
        return (0, 0, 0)

    if VLLM_AVAILABLE:
        vllm_kernel = getattr(vllm_ops, kernel)
    if kernel == "gelu_quick" and not GELU_QUICK_AVAILABLE:
        # Skip benchmark for gelu_quick if not available
        return (0, 0, 0)
    test_kernel = getattr(tf_kernel, kernel)

    def baseline():
        if VLLM_AVAILABLE:
            tmp = y0.clone()
            vllm_kernel(tmp, x)
            return tmp
        else:
            return torch.zeros_like(y0)

    def torch_fn():
        return get_torch_activation(x, kernel)

    def test_fn():
        return test_kernel(x)

    # timing helper
    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ms, qmin, qmax = triton.testing.do_bench_cudagraph(
            fn, quantiles=[0.5, 0.2, 0.8]
        )
        return 1000 * ms, 1000 * qmax, 1000 * qmin

    if provider == "vllm":
        return timed(baseline)
    if provider == "torch":
        return timed(torch_fn)
    if provider == "tf_kernel":
        return timed(test_fn)

    # provider == "speedup" - compute speedup vs torch
    t_torch, _, _ = timed(torch_fn)
    t_test, _, _ = timed(test_fn)
    spd = t_torch / t_test if t_torch > 0 else 1.0
    return (spd, spd, spd)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Activation kernel benchmark")
    p.add_argument("--batch_sizes", type=str2int_list, default=default_batch_sizes)
    p.add_argument("--seq_lens", type=str2int_list, default=default_seq_lens)
    p.add_argument("--dims", type=str2int_list, default=default_dims)
    p.add_argument("--verify_only", action="store_true")
    args = p.parse_args()

    # coerce lists
    if isinstance(args.batch_sizes, str):
        args.batch_sizes = str2int_list(args.batch_sizes)
    if isinstance(args.seq_lens, str):
        args.seq_lens = str2int_list(args.seq_lens)
    if isinstance(args.dims, str):
        args.dims = str2int_list(args.dims)

    # patch perf_report grid
    benchmark_grid = make_configs(args.batch_sizes, args.seq_lens, args.dims)
    benchmark.benchmarks.x_vals = benchmark_grid

    if args.verify_only:
        # Test with the first available kernel
        test_kernel = kernels[0]
        ok = calculate_diff(test_kernel, torch.float16, 1, 1, args.dims[0])
        print("✅ sanity pass" if ok else "❌ mismatch")
    else:
        # Custom run with progress bar
        total_configs = len(benchmark_grid)
        total_providers = len(benchmark.benchmarks.line_vals)
        total_iterations = total_configs * total_providers

        print(f"\n📊 Activation Kernel Benchmark")
        print(
            f"   Configurations: {total_configs} (kernels × dtypes × batch_sizes × seq_lens × dims)"
        )
        print(
            f"   Providers: {total_providers} ({', '.join(benchmark.benchmarks.line_vals)})"
        )
        print(f"   Total iterations: {total_iterations}\n")

        # Monkey-patch the benchmark function to update progress
        original_fn = benchmark.fn

        def progress_wrapped_fn(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            # Extract info for progress description
            kernel = args[0] if len(args) > 0 else kwargs.get("kernel", "unknown")
            provider = args[5] if len(args) > 5 else kwargs.get("provider", "unknown")
            update_benchmark_progress(1, f"Testing {kernel}/{provider}")
            return result

        benchmark.fn = progress_wrapped_fn

        with tqdm(total=total_iterations, desc="Initializing", unit="test") as pbar:
            set_benchmark_pbar(pbar)
            benchmark.run(print_data=True)
            set_benchmark_pbar(None)

        print(f"\n✅ Benchmark completed!")
