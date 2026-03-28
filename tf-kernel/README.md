# tf-kernel

English | [中文](README_zh.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange)](https://pytorch.org/)

**tf-kernel** is a high-performance CUDA kernel library for TeleFuser, providing optimized GPU operations for transformer and diffusion models. It implements custom CUDA kernels using CUTLASS and FlashInfer, with PyTorch bindings for Python accessibility.

## Features

- **Elementwise Operations**: Activation functions (SiLU, GELU), RMS normalization, rotary positional embedding (RoPE), casting
- **GEMM Operations**: FP8, INT8, and FP4 quantized matrix multiplication with various quantization schemes
- **Attention Variants**:
  - SageAttention v2: INT8 QK quantization with FP8/FP16 value
  - SageAttention v3: FP4 quantization for Blackwell (SM100+)
  - Block Sparse Attention: Efficient block-sparse pattern attention
- **Multi-Architecture Support**: SM80 (Ampere), SM90 (Hopper), SM100+ (Blackwell)

## Installation

Requires torch == 2.9.1

```bash
# Latest version
pip3 install tf-kernel --upgrade
```

## Building from Source

### Requirements

- CMake ≥3.31
- Python ≥3.10
- PyTorch == 2.9.1
- scikit-build-core
- ninja (optional)

### Development Installation

For development, install with all dev dependencies:

```bash
git clone https://github.com/YOUR_ORG/tf-kernel.git
cd tf-kernel
pip install -e ".[dev]" --no-build-isolation
```

Dependency groups available: `dev` (all), `test`, `docs`, `lint`.

### Use Makefile to build tf-kernel

```bash
# Build for all supported SM architectures (default: ALL)
make build

# Build for auto-detected GPU architecture (recommended for single-machine use)
make build-auto

# Build for specific SM architecture only
make build-sm80   # Ampere (A100, RTX 3090, etc.)
make build-sm90   # Hopper (H100)
make build-sm100  # Blackwell (RTX 5090, B100/B200)
```

### Target SM Architecture Selection

The build system supports selecting target SM architectures via the `TF_KERNEL_TARGET_SM` CMake variable:

| Option | Description |
|--------|-------------|
| `ALL` | Build for all supported SM architectures (default) |
| `AUTO` | Auto-detect local GPU and build for its architecture |
| `SM80` | Build for SM 80-89 (Ampere, Ada Lovelace) |
| `SM90` | Build for SM 90 (Hopper H100) |
| `SM100` | Build for SM 100+ (Blackwell) |

Using CMake directly:
```bash
cmake -DTF_KERNEL_TARGET_SM=AUTO ..
cmake -DTF_KERNEL_TARGET_SM=SM80 ..
```

**Note:** Building for a specific SM architecture reduces build time and binary size significantly compared to building for all architectures.

### Limit build resource usage (CPU / parallelism)

By default, `make build` uses all available CPU cores. You can override build parallelism and NVCC compile threads:

```bash
# Limit parallel jobs (controls both make and cmake parallelism)
make build MAX_JOBS=2

# Additionally limit NVCC internal threads (reduces CPU and peak memory)
make build MAX_JOBS=2 CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"
```

## Contribution

### Steps to add a new kernel:

1. Implement the kernel in [csrc](csrc)
2. Expose the interface in [include/tf_kernel_ops.h](include/tf_kernel_ops.h)
3. Create torch extension in [csrc/common_extension.cc](csrc/common_extension.cc)
4. Update [CMakeLists.txt](CMakeLists.txt) to include new CUDA source
5. Expose Python interface in [python](python/tf_kernel)
6. Add test and benchmark

### Development Tips

1. When creating torch extensions, add the function definition with `m.def`, and device binding with `m.impl`:

- How to write schema: [Schema reference](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // We need def with schema here for torch.compile
   m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, "
    "int cublas_handle) -> ()");
   m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
   ```

### Adapting C++ Native Types for Torch Compatibility

Third-party C++ libraries often use int and float, but PyTorch bindings require int64_t and double due to Python's type mapping.

Use make_pytorch_shim from tf_kernel_torch_shim.h to handle conversions automatically:

```cpp

// Add type conversion for int -> int64_t
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "value too large");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "value too small");
    return arg;
  }
};
```
```cpp
// Wrap your function
m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

### Testing & Benchmarking

1. Add pytest tests in [tests/](/tests), if you need to skip some test, please use `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
```

2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in [benchmark/](benchmark)

   **We recommend using `triton.testing.do_bench_cudagraph` for kernel benchmarking**:

   Compared to `triton.testing.do_bench`, `do_bench_cudagraph` provides:
   - Reduced CPU overhead impact for more accurate kernel performance measurements
   - Incorporation of PDL (Programmatic Dependent Launch) effects into individual kernel results
   - More realistic performance data on PDL-supported architectures (SM >= 90)

3. Run test suite

## Kernel Size Analysis

Analyze CUDA kernel sizes in compiled wheel files to identify oversized kernels and template-instantiation bloat:

This tool requires `cubloaty` (install with `pip install cubloaty`) to work.

```bash
# Install cubloaty
pip install cubloaty

# Analyze a wheel file
python analyze_whl_kernel_sizes.py path/to/tf_kernel-*.whl

# Custom output file
python analyze_whl_kernel_sizes.py path/to/tf_kernel-*.whl --output my_analysis.txt
```

The tool generates:
- A text report with:
  - Kernel groups (by name prefix)
  - Individual kernel sizes (sorted by size)

Use this to identify large kernels and potential template instantiation bloat.

## Acknowledgments

This project is built upon the excellent work of the following open-source projects:

- **[SGL-Kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)** - Part of the SGLang project, providing high-performance CUDA kernels for LLM serving
- **[SageAttention](https://github.com/thu-ml/SageAttention)** - Quantized attention implementation achieving significant speedups over standard attention mechanisms
- **[Block-Sparse-Attention](https://github.com/Dao-AILab/flash-attention)** - Block sparse attention implementation from the FlashAttention project

We sincerely thank the authors and contributors of these projects for their outstanding contributions to the open-source community.

## Contributing

We welcome contributions from the community! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting issues or pull requests.
