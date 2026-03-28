# Benchmarks SageAttention2 kernels versus FlashAttention (if available), PyTorch SDPA, and cuDNN
# across (batch_size, seq_len, num_heads, head_dim) configurations.
import argparse
import itertools
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
import torch.nn.functional as F
import triton
import triton.testing

# Import SageAttention from tf_kernel
from tf_kernel import sageattn

# Optional: Flash Attention
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def get_device_capability():
    """Get the compute capability of the current GPU."""
    if not torch.cuda.is_available():
        return None, None
    props = torch.cuda.get_device_properties(0)
    return props.major, props.minor


def torch_sdpa(q, k, v, causal=False, scale=None):
    """PyTorch SDPA (Scaled Dot Product Attention)."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def torch_sdpa_cudnn(q, k, v, causal=False, scale=None):
    """PyTorch SDPA with cuDNN backend."""
    try:
        # New API for PyTorch 2.0+
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, scale=scale
            )
    except ImportError:
        # Fallback to old API for older PyTorch versions
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=False,
            enable_mem_efficient=False,
            enable_cudnn=True,
        ):
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, scale=scale
            )


def sageattn_wrapper(q, k, v, causal=False, sm_scale=None):
    """SageAttention wrapper with proper format conversion."""
    # SageAttention expects HND format by default
    return sageattn(q, k, v, tensor_layout="HND", is_causal=causal, sm_scale=sm_scale)


# CI environment uses simplified parameters
if IS_CI:
    batch_sizes = [1, 2]
    seq_lens = [512, 1024]
    num_heads_list = [8]
    head_dims = [64, 128]
else:
    batch_sizes = [1, 2, 4, 8]
    seq_lens = [512, 1024, 2048, 4096, 8192]
    num_heads_list = [8, 16, 32]
    head_dims = [64, 128]

configs = list(itertools.product(batch_sizes, seq_lens, num_heads_list, head_dims))

# Filter providers based on availability
available_providers = ["torch-sdpa", "torch-sdpa-cudnn", "sageattn2"]
available_names = ["PyTorch SDPA", "PyTorch SDPA (cuDNN)", "SageAttn2"]
available_styles = [("blue", "-"), ("cyan", "-"), ("green", "-")]

if FLASH_ATTN_AVAILABLE:
    available_providers.insert(0, "flash-attn")
    available_names.insert(0, "Flash Attention")
    available_styles.insert(0, ("red", "-"))


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
        x_names=["batch_size", "seq_len", "num_heads", "head_dim"],
        x_vals=configs,
        line_arg="provider",
        line_vals=available_providers,
        line_names=available_names,
        styles=available_styles,
        ylabel="ms (median)",
        plot_name="attention-performance",
        args={},
    )
)
def benchmark(
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    provider,
    causal=False,
    dtype=torch.bfloat16,
):
    device = torch.device("cuda")

    # Create input tensors in HND format: [batch, num_heads, seq_len, head_dim]
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
    )
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
    )

    sm_scale = head_dim**-0.5
    quantiles = [0.5, 0.2, 0.8]

    try:
        if provider == "flash-attn":
            if not FLASH_ATTN_AVAILABLE:
                return (0, 0, 0)
            # Flash attention expects HND format
            fn = lambda: flash_attn_func(q, k, v, causal=causal, softmax_scale=sm_scale)
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                fn, quantiles=quantiles
            )

        elif provider == "torch-sdpa":
            fn = lambda: torch_sdpa(q, k, v, causal=causal, scale=sm_scale)
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                fn, quantiles=quantiles
            )

        elif provider == "torch-sdpa-cudnn":
            fn = lambda: torch_sdpa_cudnn(q, k, v, causal=causal, scale=sm_scale)
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                fn, quantiles=quantiles
            )

        elif provider == "sageattn2":
            fn = lambda: sageattn_wrapper(q, k, v, causal=causal, sm_scale=sm_scale)
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                fn, quantiles=quantiles
            )
        else:
            return (0, 0, 0)

    except Exception as e:
        print(
            f"\n⚠️  Skipping {provider} for B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}: {str(e)[:60]}..."
        )
        ms, min_ms, max_ms = 0.0, 0.0, 0.0

    return ms, min_ms, max_ms


def calculate_diff(
    batch_size=2,
    seq_len=1024,
    num_heads=8,
    head_dim=64,
    causal=False,
    dtype=torch.bfloat16,
):
    """Calculate difference between implementations."""
    device = torch.device("cuda")

    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
    )
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
    )

    sm_scale = head_dim**-0.5

    print(
        f"\n📊 Correctness Check: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}, causal={causal}"
    )

    # PyTorch SDPA reference
    ref_out = torch_sdpa(q, k, v, causal=causal, scale=sm_scale)

    # SageAttention
    sage_out = sageattn_wrapper(q, k, v, causal=causal, sm_scale=sm_scale)

    # Compare (SageAttention uses INT8 quantization, so tolerance needs to be higher)
    sage_diff = (ref_out - sage_out).abs().mean().item()
    sage_max_diff = (ref_out - sage_out).abs().max().item()
    sage_match = torch.allclose(ref_out, sage_out, atol=0.5, rtol=0.1)

    print(
        f"  SageAttn2 vs PyTorch SDPA: {'✅ match' if sage_match else '⚠️  differ (expected for quantized attn)'}"
    )
    print(f"     Mean diff: {sage_diff:.6f}, Max diff: {sage_max_diff:.6f}")

    if FLASH_ATTN_AVAILABLE:
        flash_out = flash_attn_func(q, k, v, causal=causal, softmax_scale=sm_scale)
        flash_match = torch.allclose(ref_out, flash_out, atol=1e-2, rtol=1e-2)
        print(
            f"  FlashAttn vs PyTorch SDPA: {'✅ match' if flash_match else '❌ differ'}"
        )

    return sage_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SageAttention2 benchmark")
    parser.add_argument(
        "--causal", action="store_true", help="Use causal attention mask"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only run correctness check"
    )
    args = parser.parse_args()

    # Check device capability
    major, minor = get_device_capability()
    if major is None:
        print("❌ CUDA not available")
        exit(1)

    print(f"\n📊 SageAttention2 Benchmark")
    print(f"   Device: sm{major}{minor}")
    print(f"   Causal: {args.causal}")
    print(f"   Configs: {len(configs)} (batch_size x seq_len x num_heads x head_dim)")
    print(
        f"   Providers: {len(available_providers)} ({', '.join(available_providers)})"
    )
    print(f"   Total iterations: {len(configs) * len(available_providers)}\n")

    if args.verify_only:
        print("Running correctness check only...")
        ok = calculate_diff(causal=args.causal)
        print("\n✅ sanity pass" if ok else "\n❌ mismatch")
    else:
        # Run correctness check first
        calculate_diff(causal=args.causal)

        # Calculate total iterations for progress bar
        total_iterations = len(configs) * len(available_providers)

        # Monkey-patch the benchmark function to update progress
        original_fn = benchmark.fn

        def progress_wrapped_fn(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            # Extract info for progress description
            batch_size = (
                args[0] if len(args) > 0 else kwargs.get("batch_size", "unknown")
            )
            seq_len = args[1] if len(args) > 1 else kwargs.get("seq_len", "unknown")
            num_heads = args[2] if len(args) > 2 else kwargs.get("num_heads", "unknown")
            head_dim = args[3] if len(args) > 3 else kwargs.get("head_dim", "unknown")
            provider = args[4] if len(args) > 4 else kwargs.get("provider", "unknown")
            update_benchmark_progress(
                1, f"B={batch_size},S={seq_len},H={num_heads},D={head_dim}/{provider}"
            )
            return result

        benchmark.fn = progress_wrapped_fn

        print("\nRunning benchmark...")
        with tqdm(total=total_iterations, desc="Initializing", unit="test") as pbar:
            set_benchmark_pbar(pbar)
            benchmark.run(print_data=True, causal=args.causal)
            set_benchmark_pbar(None)

        print("\n✅ Benchmark finished!")
