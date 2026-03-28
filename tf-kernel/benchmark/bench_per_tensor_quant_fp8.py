import itertools
import os
from typing import Optional, Tuple

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
import triton
import triton.testing

from tf_kernel import tf_per_tensor_quant_fp8

# Optional imports
try:
    from vllm import _custom_ops as ops

    VLLM_AVAILABLE = True
except ImportError:
    ops = None
    VLLM_AVAILABLE = False

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

fp8_type_ = torch.float8_e4m3fn


def vllm_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not VLLM_AVAILABLE:
        # Fallback to tf_kernel implementation
        return tfkernel_scaled_fp8_quant(input, scale)
    return ops.scaled_fp8_quant(input, scale)


def tfkernel_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    is_static = True
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        is_static = False
    tf_per_tensor_quant_fp8(input, output, scale, is_static)

    return output, scale


def calculate_diff(batch_size: int, seq_len: int):
    """Calculate difference between VLLM and tf_kernel implementations."""
    device = torch.device("cuda")
    x = torch.rand((batch_size, seq_len), dtype=torch.float16, device=device)

    if not VLLM_AVAILABLE:
        print("⚠️ vLLM not available, skipping comparison")
        return

    vllm_out, vllm_scale = vllm_scaled_fp8_quant(x)
    tfkernel_out, tfkernel_scale = tfkernel_scaled_fp8_quant(x)

    scale_diff = torch.abs(vllm_scale - tfkernel_scale).item()
    output_diff = torch.abs(vllm_out.float() - tfkernel_out.float()).mean().item()

    if torch.allclose(
        vllm_out.to(torch.float32), tfkernel_out.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(vllm_scale, tfkernel_scale, rtol=1e-3, atol=1e-5):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


# CI environment uses simplified parameters
if IS_CI:
    batch_size_range = [16]  # Single batch size for CI
    seq_len_range = [64]  # Single sequence length for CI
else:
    batch_size_range = [16, 32, 64, 128]
    seq_len_range = [64, 128, 256, 512, 1024, 2048]

configs = list(itertools.product(batch_size_range, seq_len_range))


if VLLM_AVAILABLE:
    line_vals = ["vllm", "tfkernel"]
    line_names = ["VLLM", "TF Kernel"]
    styles = [("blue", "-"), ("green", "-")]
else:
    line_vals = ["tfkernel"]
    line_names = ["TF Kernel"]
    styles = [("green", "-")]


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
        x_names=["batch_size", "seq_len"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="per-tensor-quant-fp8-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, provider):
    dtype = torch.float16
    device = torch.device("cuda")

    x = torch.randn(batch_size * seq_len, 4096, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm":
        fn = lambda: vllm_scaled_fp8_quant(x.clone())
    elif provider == "tfkernel":
        fn = lambda: tfkernel_scaled_fp8_quant(x.clone())

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    # Calculate total iterations for progress bar
    total_configs = len(configs)
    total_providers = len(line_vals)
    total_iterations = total_configs * total_providers

    print(f"\n📊 Per-Tensor Quant FP8 Benchmark")
    print(f"   Configs: {total_configs} (batch_size x seq_len combinations)")
    print(f"   Providers: {total_providers} ({', '.join(line_vals)})")
    print(f"   Total iterations: {total_iterations}\n")

    calculate_diff(batch_size=4, seq_len=4096)

    # Monkey-patch the benchmark function to update progress
    original_fn = benchmark.fn

    def progress_wrapped_fn(*args, **kwargs):
        result = original_fn(*args, **kwargs)
        # Extract info for progress description
        batch_size = args[0] if len(args) > 0 else kwargs.get("batch_size", "unknown")
        seq_len = args[1] if len(args) > 1 else kwargs.get("seq_len", "unknown")
        provider = args[2] if len(args) > 2 else kwargs.get("provider", "unknown")
        update_benchmark_progress(
            1, f"Testing batch={batch_size},seq={seq_len}/{provider}"
        )
        return result

    benchmark.fn = progress_wrapped_fn

    with tqdm(total=total_iterations, desc="Initializing", unit="test") as pbar:
        set_benchmark_pbar(pbar)
        benchmark.run(print_data=True)
        set_benchmark_pbar(None)

    print("\n✅ Benchmark finished!")
