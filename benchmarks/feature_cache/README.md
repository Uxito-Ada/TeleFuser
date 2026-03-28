# AdaTaylorCache Taylor Order Performance Benchmark

This benchmark evaluates the performance and video quality of different Taylor expansion orders (0, 1, 2) in AdaTaylorCache for Wan2.1-T2V-1.3B video generation.

## Test Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 5090 (4x) |
| CUDA Version | 12.8+ |
| PyTorch Version | 2.6+ |
| Model | Wan2.1-T2V-1.3B |
| Attention | SAGE_ATTN_2_8_8 |

### Software Environment

```bash
Python: 3.11+
PyTorch: 2.6+
telefuser: latest (development mode)
```

## Test Script

The benchmark script is located at:

```bash
benchmarks/feature_cache/wan21_1_3b_ada_taylor_cache.py
```

### Usage

#### Basic Usage

```bash
# Single prompt test with all Taylor orders (0, 1, 2)
python benchmarks/feature_cache/wan21_1_3b_ada_taylor_cache.py \
    --prompt "A stylish little girl gently caressing her dog..." \
    --seed 42 \
    --gpu_ids 0 \
    --num_inference_steps 40 \
    --output_dir work_dirs/taylor_order_comparison
```

#### Multi-GPU Usage

Use `CUDA_VISIBLE_DEVICES` environment variable to control GPU visibility:

```bash
# Use GPU 2 only
CUDA_VISIBLE_DEVICES=2 python benchmarks/feature_cache/wan21_1_3b_ada_taylor_cache.py \
    --prompt "A cat playing piano" \
    --seed 42 \
    --num_inference_steps 40

# Use GPUs 2 and 3 (round-robin assignment for multiple prompts)
CUDA_VISIBLE_DEVICES=2,3 python benchmarks/feature_cache/wan21_1_3b_ada_taylor_cache.py \
    --num_prompts 10 \
    --gpu_ids 0,1 \
    --num_inference_steps 40
```

**Note:** The script uses a single pipeline per process. Each prompt's four orders (Original, Order 0, Order 1, Order 2) are processed sequentially on the same GPU.

#### Other Usage Examples

```bash
# Custom Taylor threshold (applied to all orders)
python benchmarks/feature_cache/wan21_1_3b_ada_taylor_cache.py \
    --prompt "Your prompt here" \
    --taylor_threshold 3 \
    --seed 42

# Multiple prompts test
python benchmarks/feature_cache/wan21_1_3b_ada_taylor_cache.py \
    --num_prompts 5 \
    --gpu_ids 0,1
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt` | None | Single prompt to evaluate |
| `--num_prompts` | 1 | Number of test prompts (1-10) |
| `--seed` | 42 | Random seed |
| `--gpu_ids` | "0" | Comma-separated GPU IDs (e.g., '0' or '0,1') |
| `--model_root` | "/dev/shm/Wan2.1-T2V-1.3B/" | Model directory |
| `--output_dir` | auto-generated | Output directory |
| `--num_inference_steps` | 50 | Inference steps |
| `--num_frames` | 81 | Number of frames |
| `--cfg_scale` | 6.0 | CFG scale |
| `--taylor_threshold` | 2 | Taylor threshold for all orders |

## Test Methodology

### Configuration Comparison

| Configuration | n_derivatives | Description |
|---------------|---------------|-------------|
| Original | N/A | No feature caching (baseline) |
| Order 0 | 0 | Residual-only caching (constant approximation) |
| Order 1 | 1 | First-order Taylor expansion (linear approximation) |
| Order 2 | 2 | Second-order Taylor expansion (quadratic approximation) |

All cache configurations use the same `taylor_threshold=2`:
- When `elapsed <= 2`: Use Taylor series expansion
- When `elapsed > 2`: Fall back to residual reuse

### Evaluation Metrics

| Metric | Description | Optimal |
|--------|-------------|---------|
| PSNR | Peak Signal-to-Noise Ratio | Higher is better |
| SSIM | Structural Similarity Index | Higher is better (range: 0-1) |
| LPIPS | Learned Perceptual Image Patch Similarity | Lower is better |
| SpeedUp | Time reduction vs Original | Higher is better |

## Test Results

### Test Case: Single Prompt (Seed 42)

**Prompt:** "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard."

**Parameters:**
- Inference Steps: 40
- Frames: 81
- Resolution: 854x480 (rounded to 864x480)
- CFG Scale: 6.0
- Taylor Threshold: 2

### Multi-Prompt Test Results (10 Prompts, Seed 42)

**Test Date:** 2026-03-05

**Parameters:**
- Inference Steps: 40
- Frames: 81
- Resolution: 854x480 (rounded to 864x480)
- CFG Scale: 6.0
- Taylor Threshold: 2

#### Aggregated Performance Results

| Configuration | Mean Time (s) | Std Time (s) | SpeedUp |
|---------------|---------------|--------------|---------|
| Original | 140.21 | 0.15 | 1.00x |
| Order 0 (n_derivatives=0) | 77.58 | 0.08 | **1.81x** |
| Order 1 (n_derivatives=1) | 77.58 | 0.11 | **1.81x** |
| Order 2 (n_derivatives=2) | 77.61 | 0.11 | **1.81x** |

#### Aggregated Quality Metrics

| Configuration | PSNR (mean±std) | SSIM (mean±std) | LPIPS (mean±std) |
|---------------|-----------------|-----------------|------------------|
| Order 0 | 25.39 ± 4.50 | 0.8466 ± 0.0774 | 0.1044 ± 0.0483 |
| Order 1 | 25.69 ± 4.55 | 0.8571 ± 0.0747 | 0.0953 ± 0.0466 |
| Order 2 | 25.41 ± 4.74 | 0.8445 ± 0.0839 | 0.1002 ± 0.0479 |

#### Multi-Prompt Analysis

**Speed Performance:**
- All three Taylor orders achieve consistent **1.81x speedup** across 10 different prompts
- Low standard deviation (~0.1s) indicates stable performance across different prompts
- Inference time reduced from ~140s to ~77s per video

**Quality Metrics (10-prompt average):**
- **Best PSNR:** Order 1 (25.69 dB) - indicates better pixel-level accuracy
- **Best SSIM:** Order 1 (0.8571) - indicates better structural similarity
- **Best LPIPS:** Order 1 (0.0953) - indicates better perceptual quality

**Key Observations:**
1. With larger sample size (10 prompts vs 1 prompt), **Order 1** shows superior quality across all metrics
2. The quality differences between orders are relatively small (PSNR within 0.3 dB, SSIM within 0.013)
3. All orders maintain consistent ~1.81x speedup with minimal variance
4. Order 1 provides the best balance of speed and quality for production use

### Performance

#### Single Prompt

| Configuration | Time (s) | SpeedUp |
|---------------|----------|---------|
| Original | ~117s | 1.00x |
| Order 0 | ~68s | **1.71x** |
| Order 1 | ~68s | **1.72x** |
| Order 2 | ~68s | **1.72x** |

