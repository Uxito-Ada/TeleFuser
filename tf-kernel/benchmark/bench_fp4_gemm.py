import argparse
import csv
import os

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
from flashinfer import mm_fp4

from tf_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def get_device_capability():
    """Get the compute capability of the current GPU."""
    if not torch.cuda.is_available():
        return None, None
    props = torch.cuda.get_device_properties(0)
    return props.major, props.minor


def is_sm100_supported():
    """Check if SM100 (Blackwell) or higher is supported."""
    if not torch.cuda.is_available():
        return False
    major, minor = get_device_capability()
    return major >= 10


def get_weight_shapes(args):
    models_tps = args.tp_sizes

    if models_tps == [4]:
        return [[1024, 3584], [7168, 256], [7168, 2304], [9216, 3584]]

    if models_tps == [8]:
        return [[512, 3584], [7168, 128], [7168, 1152], [4608, 3584]]
    return [
        [1024, 3584],
        [7168, 256],
        [7168, 2304],
        [9216, 3584],
        [512, 3584],
        [7168, 128],
        [7168, 1152],
        [4608, 3584],
    ]


# CI environment uses simplified parameters
if IS_CI:
    batch_sizes = [1, 8]  # Simplified for CI
else:
    batch_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        3072,
        4096,
        8192,
        16384,
    ]


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
        x_names=["batch_size"],
        x_vals=batch_sizes,
        # x_vals = [64],
        x_log=False,
        line_arg="provider",
        line_vals=["tfkernel_cutlass", "cutlass", "cudnn", "trtllm", "auto"],
        line_names=[
            "tfkernel cutlass fp4",
            "flashinfer cutlass fp4",
            "cudnn fp4",
            "trtllm fp4",
            "auto fp4 (cudnn/cutlass)",
        ],
        styles=[
            ("red", "solid"),
            ("orange", "solid"),
            ("blue", "solid"),
            ("green", "solid"),
            ("purple", "solid"),
        ],
        ylabel="latency (ms)",
        plot_name="fp4_gemm_benchmark",
        args={},
    )
)
def benchmark(batch_size, provider, N, K, dtype, correctness, csv_file):
    M = batch_size
    packed_k = K
    K = 2 * packed_k
    a_dtype = torch.randn((M, K), dtype=dtype, device="cuda")
    b_dtype = torch.randn((N, K), dtype=dtype, device="cuda")
    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)

    alpha = 1.0 / (a_global_scale * b_global_scale)
    a_fp4, a_scale_interleaved = scaled_fp4_quant(a_dtype, a_global_scale)
    # print("a_fp4", a_fp4)
    b_fp4, b_scale_interleaved = scaled_fp4_quant(b_dtype, b_global_scale)
    res_fi = torch.empty((M, N), dtype=dtype, device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = 0.0, 0.0, 0.0

    try:
        if provider == "tfkernel_cutlass":
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: cutlass_scaled_fp4_mm(
                    a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
                ),
                quantiles=quantiles,
            )
        elif provider == "cutlass":
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: mm_fp4(
                    a_fp4,
                    b_fp4.T,
                    a_scale_interleaved,
                    b_scale_interleaved.T,
                    alpha,
                    dtype,
                    res_fi,
                    backend="cutlass",
                ),
                quantiles=quantiles,
            )
        elif provider == "cudnn":
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: mm_fp4(
                    a_fp4,
                    b_fp4.T,
                    a_scale_interleaved,
                    b_scale_interleaved.T,
                    alpha,
                    dtype,
                    res_fi,
                    backend="cudnn",
                ),
                quantiles=quantiles,
            )
        elif provider == "trtllm":
            a_scale_interleaved = a_scale_interleaved.to(torch.uint8)
            b_scale_interleaved = b_scale_interleaved.to(torch.uint8)
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: mm_fp4(
                    a_fp4,
                    b_fp4.T,
                    a_scale_interleaved,
                    b_scale_interleaved.T,
                    alpha,
                    dtype,
                    res_fi,
                    backend="trtllm",
                ),
                quantiles=quantiles,
            )
        elif provider == "auto":
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: mm_fp4(
                    a_fp4,
                    b_fp4.T,
                    a_scale_interleaved,
                    b_scale_interleaved.T,
                    alpha,
                    dtype,
                    res_fi,
                ),
                quantiles=quantiles,
            )
    except Exception as e:
        # Skip unsupported configurations
        print(f"\n⚠️  Skipping {provider} for M={M}, N={N}, K={K}: {str(e)[:50]}...")
        ms, min_ms, max_ms = 0.0, 0.0, 0.0
    if correctness and ms > 0:
        res_cutlass = cutlass_scaled_fp4_mm(
            a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
        )
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_scale_interleaved,
            b_scale_interleaved.T,
            alpha,
            dtype,
            res_fi,
            backend="cudnn",
        )
        assert torch.allclose(
            res_fi, res_cutlass, atol=1e-3, rtol=1e-3
        ), "cudnn fp4 doesn't match cutlass fp4"
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_scale_interleaved,
            b_scale_interleaved.T,
            alpha,
            dtype,
            res_fi,
            backend="trtllm",
        )
        assert torch.allclose(
            res_fi, res_cutlass, atol=1e-3, rtol=1e-3
        ), "trtllm fp4 doesn't match cutlass fp4"

    if csv_file:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([provider, M, N, K, ms])

    return ms, min_ms, max_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="List of tensor parallel sizes",
    )
    parser.add_argument(
        "--dtype",
        type=torch.dtype,
        default=torch.bfloat16,
        help="Data type",
    )
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Check correctness",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results_cutlass_cudnn.csv",
        help="CSV file to save results",
    )
    args = parser.parse_args()

    # Simplify for CI environment
    if IS_CI:
        args.tp_sizes = [args.tp_sizes[0]]  # Use only first TP size

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["provider", "m", "n", "k", "time_ms"])

    # FP4 operations require Blackwell SM100 support
    major, minor = get_device_capability()
    if not is_sm100_supported():
        print("Skipping FP4 GEMM benchmark")
        if major is not None:
            print(
                f"FP4 operations require SM100 (Blackwell), but found sm{major}{minor}"
            )
        else:
            print("Could not determine device capability")
    else:
        NKs = get_weight_shapes(args)

        # Limit iterations in CI
        if IS_CI:
            NKs = NKs[:2]  # Only test first 2 shapes in CI

        # Calculate total iterations for progress bar
        total_batch_sizes = len(batch_sizes) if not IS_CI else len([1, 8])
        total_providers = len(benchmark.benchmarks.line_vals)
        total_iterations = total_batch_sizes * total_providers * len(NKs)

        print(f"\n📊 FP4 GEMM Benchmark")
        print(f"   Weight shapes: {len(NKs)} configurations")
        print(f"   Batch sizes: {total_batch_sizes}")
        print(
            f"   Providers: {total_providers} ({', '.join(benchmark.benchmarks.line_vals)})"
        )
        print(f"   Total iterations: {total_iterations}\n")

        # Monkey-patch the benchmark function to update progress
        original_fn = benchmark.fn

        def progress_wrapped_fn(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            # Extract info for progress description
            batch_size = (
                args[0] if len(args) > 0 else kwargs.get("batch_size", "unknown")
            )
            provider = args[1] if len(args) > 1 else kwargs.get("provider", "unknown")
            update_benchmark_progress(1, f"Testing batch={batch_size}/{provider}")
            return result

        benchmark.fn = progress_wrapped_fn

        for N, K in NKs:
            print(f"DeepSeek-R1-0528-FP4 N={N} K={K}: ")
            with tqdm(
                total=total_batch_sizes * total_providers,
                desc=f"N={N}, K={K}",
                unit="test",
            ) as pbar:
                set_benchmark_pbar(pbar)
                benchmark.run(
                    print_data=True,
                    N=N,
                    K=K,
                    dtype=args.dtype,
                    correctness=args.correctness,
                    csv_file=args.csv,
                )
                set_benchmark_pbar(None)
        print("\n✅ Benchmark finished!")
