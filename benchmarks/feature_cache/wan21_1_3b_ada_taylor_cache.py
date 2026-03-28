# ruff: noqa
"""Evaluate video similarity between original and accelerated generation methods.

This script generates videos using four configurations to compare different Taylor orders:
1. Original: No feature caching (baseline reference)
2. Order 0: AdaTaylorCache with n_derivatives=0 (residual-only caching)
3. Order 1: AdaTaylorCache with n_derivatives=1 (first-order Taylor expansion)
4. Order 2: AdaTaylorCache with n_derivatives=2 (second-order Taylor expansion)

Then computes similarity metrics (PSNR, SSIM, LPIPS) comparing each accelerated
video against the original reference video.

Usage:
    # Single prompt test with all orders (0, 1, 2)
    python wan21_1_3b_ada_taylor_cache.py --prompt "A cat playing piano" --seed 42

    # Multiple prompts (10 random meaningful prompts)
    python wan21_1_3b_ada_taylor_cache.py --num_prompts 10

    # Use specific GPU
    CUDA_VISIBLE_DEVICES=2 python wan21_1_3b_ada_taylor_cache.py --prompt "A cat playing piano"

    # Custom Taylor threshold (applied to all orders)
    python wan21_1_3b_ada_taylor_cache.py --taylor_threshold 3
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from telefuser.utils.logging import logger

# ============== Metrics Functions ==============
# Re-implement metrics functions to avoid ImageReward dependency
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from telefuser.core.config import AttentionConfig, AttnImplType, FeatureCacheConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

# Global LPIPS model
_lpips_model = None


def _get_lpips_model(net="alex"):
    """Get or create LPIPS model."""
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net=net)
    return _lpips_model


def _fetch_video_frames(video_path):
    """Fetch all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def compute_video_psnr(video_true_path, video_test_path):
    """Compute average PSNR between two videos."""
    frames_true = _fetch_video_frames(video_true_path)
    frames_test = _fetch_video_frames(video_test_path)

    num_frames = min(len(frames_true), len(frames_test))
    if num_frames == 0:
        return None, 0

    psnr_values = []
    for i in range(num_frames):
        psnr = peak_signal_noise_ratio(frames_true[i], frames_test[i])
        psnr_values.append(psnr)

    return np.mean(psnr_values), num_frames


def compute_video_ssim(video_true_path, video_test_path):
    """Compute average SSIM between two videos."""
    frames_true = _fetch_video_frames(video_true_path)
    frames_test = _fetch_video_frames(video_test_path)

    num_frames = min(len(frames_true), len(frames_test))
    if num_frames == 0:
        return None, 0

    ssim_values = []
    for i in range(num_frames):
        ssim = structural_similarity(frames_true[i], frames_test[i], channel_axis=2)
        ssim_values.append(ssim)

    return np.mean(ssim_values), num_frames


def compute_video_lpips(video_true_path, video_test_path, net="alex"):
    """Compute average LPIPS between two videos."""
    from PIL import Image
    from torchvision.transforms.v2.functional import convert_image_dtype, normalize, pil_to_tensor

    frames_true = _fetch_video_frames(video_true_path)
    frames_test = _fetch_video_frames(video_test_path)

    num_frames = min(len(frames_true), len(frames_test))
    if num_frames == 0:
        return None, 0

    # Convert frames to tensors
    def frame_to_tensor(frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        img = pil_to_tensor(pil)
        img = convert_image_dtype(img, dtype=torch.float32)
        img = normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img

    model = _get_lpips_model(net)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    lpips_values = []
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Computing LPIPS"):
            img_true = frame_to_tensor(frames_true[i]).unsqueeze(0)
            img_test = frame_to_tensor(frames_test[i]).unsqueeze(0)
            img_true = img_true.to(next(model.parameters()).device)
            img_test = img_test.to(next(model.parameters()).device)
            lpips_val = model(img_true, img_test).item()
            lpips_values.append(lpips_val)

    return np.mean(lpips_values), num_frames


# Default configuration
PPL_CONFIG = dict(
    negative_prompt=(
        "Camera shake, overly saturated colors, overexposed, static, blurry details, subtitles, style, "
        "artwork, painting, frame, still, overall grayish, worst quality, low quality, "
        "JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, "
        "poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, "
        "cluttered background, three legs, crowded background, walking backwards"
    ),
    tiled=True,
    target_fps=16,
    sample_solver="euler",
    attn_impl=AttnImplType.SAGE_ATTN_2_8_8,
    model_type="Wan2.1-T2V-1.3B",
    sigma_shift=8.0,
)

# 10 meaningful test prompts
TEST_PROMPTS = [
    (
        "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. "
        "Perfect for pet and family content, or videos aiming to showcase love, style, and the bond "
        "between kids and their pets."
    ),
    (
        "A majestic eagle soaring through golden clouds at sunset, with rays of sunlight piercing "
        "through the misty mountain peaks below."
    ),
    (
        "A cozy coffee shop on a rainy day, with steam rising from a cup of hot coffee, soft jazz playing, "
        "and raindrops sliding down the window."
    ),
    (
        "A futuristic city at night with neon lights reflecting on wet streets, flying cars passing "
        "between towering skyscrapers."
    ),
    (
        "A peaceful Japanese garden in autumn, with red maple leaves gently falling onto a koi pond, "
        "creating ripples on the water surface."
    ),
    (
        "A chef in a professional kitchen plating an exquisite dish, with precise movements and "
        "artistic garnishes, steam rising from the food."
    ),
    (
        "A surfer riding a massive wave at dawn, with the golden sun rising over the horizon and "
        "spray catching the early morning light."
    ),
    (
        "An elderly couple dancing slowly in their living room, with vintage photographs on the walls "
        "and soft warm light from a fireplace."
    ),
    (
        "A time-lapse of a flower blooming in a meadow, with butterflies and bees visiting, "
        "soft clouds moving across a blue sky."
    ),
    (
        "A young artist painting a mural on a city wall, with vibrant colors and dynamic brushstrokes, "
        "people walking by and stopping to watch."
    ),
]


def get_pipeline(
    device="cuda",
    model_root="/dev/shm/Wan2.1-T2V-1.3B/",
    cache_type="none",
    cache_config=None,
    num_inference_steps=40,
):
    """Initialize the video generation pipeline with specified cache type.

    Args:
        device: Device to run on ("cuda:2", "cuda:3", etc.)
        model_root: Root directory of the model files
        cache_type: Type of feature caching to use ("none", "order_0", "order_1", "order_2")
        cache_config: Configuration dict for the cache (contains taylor_threshold)
        num_inference_steps: Total number of inference steps
    """
    # Load models
    module_manager = ModuleManager(device="cpu")
    module_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [[f"{model_root}/diffusion_pytorch_model.safetensors"]],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth"],
        torch_dtype=torch.bfloat16,
    )

    pipe = Wan21VideoPipeline(device=device, torch_dtype=torch.bfloat16)
    pipe_config = Wan21VideoPipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.enable_clip_stage = True

    # Configure feature cache at initialization time
    if cache_type in ["order_0", "order_1", "order_2"]:
        n_derivatives = 0 if cache_type == "order_0" else (1 if cache_type == "order_1" else 2)
        taylor_threshold = cache_config.get("taylor_threshold", 2) if cache_config else 2
        pipe_config.dit_config.feature_cache_config = FeatureCacheConfig(
            enabled=True,
            model_type=PPL_CONFIG["model_type"],
            n_derivatives=n_derivatives,
            taylor_threshold=taylor_threshold,
        )

    pipe.init(module_manager, pipe_config)
    return pipe


def generate_video(
    pipeline,
    prompt,
    seed=42,
    resolution="480p",
    aspect_ratio="16:9",
    num_inference_steps=40,
    num_frames=81,
    cfg_scale=6.0,
):
    """Generate video. Cache configuration is set at pipeline initialization time."""
    width, height = get_target_video_size_from_ratio(
        aspect_ratio,
        resolution=resolution,
        height_division_factor=2,
        width_division_factor=2,
    )

    kwargs = dict(
        prompt=prompt,
        negative_prompt=PPL_CONFIG["negative_prompt"],
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        cfg_scale=cfg_scale,
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
    )

    return pipeline(**kwargs)


def compute_all_metrics(video_true_path, video_test_path):
    """Compute all similarity metrics between two videos."""
    results = {}

    # PSNR (higher is better)
    psnr_value, frames = compute_video_psnr(video_true_path, video_test_path)
    results["psnr"] = psnr_value
    results["psnr_frames"] = frames

    # SSIM (higher is better, range 0-1)
    ssim_value, frames = compute_video_ssim(video_true_path, video_test_path)
    results["ssim"] = ssim_value
    results["ssim_frames"] = frames

    # LPIPS (lower is better, learned perceptual image patch similarity)
    lpips_value, frames = compute_video_lpips(video_true_path, video_test_path)
    results["lpips"] = lpips_value
    results["lpips_frames"] = frames

    return results


def run_single_evaluation(
    prompt,
    seed,
    output_dir,
    model_root,
    gpu_id,
    num_inference_steps=40,
    num_frames=81,
    cfg_scale=6.0,
    cache_configs=None,
):
    """Run evaluation for a single prompt.

    Generates videos with four configurations:
    1. Original: No feature caching (baseline)
    2. Order 0 (n_derivatives=0): Residual-only caching
    3. Order 1 (n_derivatives=1): First-order Taylor expansion
    4. Order 2 (n_derivatives=2): Second-order Taylor expansion

    Returns dict with timing and metrics for each cache type.
    """
    device = f"cuda:{gpu_id}"
    results = {
        "prompt": prompt,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "timings": {},
        "config": {},
    }

    # Create output subdirectory for this prompt
    prompt_dir = Path(output_dir) / f"seed{seed}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt to file
    with open(prompt_dir / "prompt.txt", "w") as f:
        f.write(prompt)

    # Cache types to evaluate with their display names
    cache_types = ["original", "order_0", "order_1", "order_2"]
    cache_type_names = {
        "original": "Original (No Cache)",
        "order_0": "Order 0 (n_derivatives=0)",
        "order_1": "Order 1 (n_derivatives=1)",
        "order_2": "Order 2 (n_derivatives=2)",
    }
    video_paths = {}

    for cache_type in cache_types:
        display_name = cache_type_names.get(cache_type, cache_type)
        logger.info(f"Generating video with {display_name}...")
        cache_config = cache_configs.get(cache_type, {}) if cache_configs else {}

        # Get pipeline with appropriate configuration (cache set at init time)
        pipe = get_pipeline(
            device=device,
            model_root=model_root,
            cache_type=cache_type,
            cache_config=cache_config,
            num_inference_steps=num_inference_steps,
        )

        # Generate video
        start_time = time.time()
        video = generate_video(
            pipeline=pipe,
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
        )
        elapsed_time = time.time() - start_time

        # Save video
        video_path = prompt_dir / f"{cache_type}.mp4"
        save_video(video, str(video_path), fps=16, quality=6)

        video_paths[cache_type] = str(video_path)
        results["timings"][cache_type] = elapsed_time

        # Store config info
        if cache_type == "order_0":
            results["config"][cache_type] = {
                "n_derivatives": 0,
                "taylor_threshold": cache_config.get("taylor_threshold", 2) if cache_config else 2,
            }
        elif cache_type == "order_1":
            results["config"][cache_type] = {
                "n_derivatives": 1,
                "taylor_threshold": cache_config.get("taylor_threshold", 2) if cache_config else 2,
            }
        elif cache_type == "order_2":
            results["config"][cache_type] = {
                "n_derivatives": 2,
                "taylor_threshold": cache_config.get("taylor_threshold", 2) if cache_config else 2,
            }
        else:
            results["config"][cache_type] = {"caching": False}

        logger.info(f"{display_name}: {elapsed_time:.2f}s, saved to {video_path}")

        # Clean up pipeline to free memory
        del pipe
        torch.cuda.empty_cache()

    # Compute metrics comparing each cache method against original
    original_path = video_paths["original"]
    for cache_type in ["order_0", "order_1", "order_2"]:
        display_name = cache_type_names.get(cache_type, cache_type)
        logger.info(f"Computing metrics for {display_name} vs original...")
        metrics = compute_all_metrics(original_path, video_paths[cache_type])
        results["metrics"][cache_type] = metrics
        logger.info(
            f"{display_name}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f}"
        )

    # Save results to JSON
    results_path = prompt_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_multi_prompt_evaluation(
    prompts,
    output_dir,
    model_root,
    gpu_ids,
    num_inference_steps=40,
    num_frames=81,
    cfg_scale=6.0,
    cache_configs=None,
):
    """Run evaluation across multiple prompts using multiple GPUs.

    Each prompt is processed sequentially. GPU assignment is done via round-robin.
    """
    all_results = []

    for i, prompt in enumerate(prompts):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing prompt {i + 1}/{len(prompts)}")
        logger.info(f"Prompt: {prompt[:80]}...")
        logger.info(f"{'=' * 60}")

        # Round-robin GPU assignment
        gpu_id = gpu_ids[i % len(gpu_ids)]
        seed = 42 + i  # Different seed for each prompt

        result = run_single_evaluation(
            prompt=prompt,
            seed=seed,
            output_dir=output_dir,
            model_root=model_root,
            gpu_id=gpu_id,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            cache_configs=cache_configs,
        )
        all_results.append(result)

    # Compute aggregate statistics
    aggregate = compute_aggregate_stats(all_results)

    # Save aggregate results
    aggregate_path = Path(output_dir) / "aggregate_results.json"
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Print summary table
    print_summary_table(all_results, aggregate)

    return all_results, aggregate


def compute_aggregate_stats(results):
    """Compute aggregate statistics across all prompts."""
    cache_types = ["order_0", "order_1", "order_2"]
    aggregate = {
        "num_prompts": len(results),
        "metrics": {},
        "timings": {},
    }

    for cache_type in cache_types:
        psnr_values = [r["metrics"][cache_type]["psnr"] for r in results if cache_type in r["metrics"]]
        ssim_values = [r["metrics"][cache_type]["ssim"] for r in results if cache_type in r["metrics"]]
        lpips_values = [r["metrics"][cache_type]["lpips"] for r in results if cache_type in r["metrics"]]
        timing_values = [r["timings"][cache_type] for r in results if cache_type in r["timings"]]
        original_timing_values = [r["timings"]["original"] for r in results]

        aggregate["metrics"][cache_type] = {
            "psnr_mean": np.mean(psnr_values) if psnr_values else None,
            "psnr_std": np.std(psnr_values) if psnr_values else None,
            "ssim_mean": np.mean(ssim_values) if ssim_values else None,
            "ssim_std": np.std(ssim_values) if ssim_values else None,
            "lpips_mean": np.mean(lpips_values) if lpips_values else None,
            "lpips_std": np.std(lpips_values) if lpips_values else None,
        }

        if timing_values and original_timing_values:
            avg_speedup = np.mean(original_timing_values) / np.mean(timing_values)
            aggregate["timings"][cache_type] = {
                "mean_time": np.mean(timing_values),
                "std_time": np.std(timing_values),
                "avg_speedup": avg_speedup,
            }
        else:
            aggregate["timings"][cache_type] = None

    # Original timing stats
    original_timing_values = [r["timings"]["original"] for r in results]
    aggregate["timings"]["original"] = {
        "mean_time": np.mean(original_timing_values),
        "std_time": np.std(original_timing_values),
    }

    return aggregate


def print_summary_table(results, aggregate):
    """Print a formatted summary table."""
    print("\n" + "=" * 110)
    print("VIDEO SIMILARITY EVALUATION SUMMARY - Taylor Order Comparison")
    print("=" * 110)

    # Configuration legend
    print("\nConfiguration Legend:")
    print("  - Original:  No feature caching (baseline)")
    print("  - Order 0:   AdaTaylorCache with n_derivatives=0 (residual-only caching)")
    print("  - Order 1:   AdaTaylorCache with n_derivatives=1 (first-order Taylor expansion)")
    print("  - Order 2:   AdaTaylorCache with n_derivatives=2 (second-order Taylor expansion)")
    print()

    # Per-prompt results
    print("\nPer-Prompt Results:")
    print("-" * 110)
    header = f"{'Prompt':<30} | {'Config':<18} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8} | {'Time(s)':>8}"
    print(header)
    print("-" * 110)

    cache_type_names = {
        "original": "Original",
        "order_0": "Order 0",
        "order_1": "Order 1",
        "order_2": "Order 2",
    }

    for result in results:
        prompt_short = result["prompt"][:28] + "..."
        for cache_type in ["original", "order_0", "order_1", "order_2"]:
            display_name = cache_type_names.get(cache_type, cache_type)
            if cache_type == "original":
                t = result["timings"][cache_type]
                row = f"{prompt_short:<30} | {display_name:<18} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {t:>8.2f}"
            elif cache_type in result["metrics"]:
                m = result["metrics"][cache_type]
                t = result["timings"][cache_type]
                row = (
                    f"{prompt_short:<30} | {display_name:<18} | {m['psnr']:>8.2f} | "
                    f"{m['ssim']:>8.4f} | {m['lpips']:>8.4f} | {t:>8.2f}"
                )
            else:
                continue
            print(row)
            prompt_short = ""  # Only show prompt once

    # Aggregate statistics
    print("\n" + "=" * 110)
    print("AGGREGATE STATISTICS (Mean ± Std)")
    print("=" * 110)
    print(f"{'Config':<18} | {'PSNR':>20} | {'SSIM':>20} | {'LPIPS':>20} | {'Time(s)':>12} | {'SpeedUp':>10}")
    print("-" * 110)

    # Print original timing
    orig_t = aggregate["timings"]["original"]
    print(f"{'Original':<18} | {'N/A':>20} | {'N/A':>20} | {'N/A':>20} | {orig_t['mean_time']:>12.2f} | {'1.00x':>10}")

    # Print cache methods
    for cache_type in ["order_0", "order_1", "order_2"]:
        display_name = cache_type_names.get(cache_type, cache_type)
        m = aggregate["metrics"][cache_type]
        t = aggregate["timings"][cache_type]
        if m["psnr_mean"] is not None:
            psnr_str = f"{m['psnr_mean']:.2f} ± {m['psnr_std']:.2f}"
            ssim_str = f"{m['ssim_mean']:.4f} ± {m['ssim_std']:.4f}"
            lpips_str = f"{m['lpips_mean']:.4f} ± {m['lpips_std']:.4f}"
            time_str = f"{t['mean_time']:.2f} ± {t['std_time']:.2f}"
            speedup_str = f"{t['avg_speedup']:.2f}x" if t else "N/A"
            print(
                f"{display_name:<18} | {psnr_str:>20} | {ssim_str:>20} | "
                f"{lpips_str:>20} | {time_str:>12} | {speedup_str:>10}"
            )

    print("=" * 110)
    print("\nResults saved to output_dir")


@click.command()
@click.option("--prompt", default=None, help="Single prompt to evaluate (if not specified, uses test prompts)")
@click.option("--num_prompts", default=1, help="Number of prompts to evaluate from test set (1-10)")
@click.option("--seed", default=42, help="Random seed for single prompt mode")
@click.option("--gpu_ids", default="0", help="Comma-separated GPU IDs to use (e.g., '0' or '0,1')")
@click.option("--model_root", default="/dev/shm/Wan2.1-T2V-1.3B/", help="Model root directory")
@click.option("--output_dir", default=None, help="Output directory for videos and results")
@click.option("--num_inference_steps", default=50, help="Number of inference steps (default: 50 to match calibration)")
@click.option("--num_frames", default=81, help="Number of frames")
@click.option("--cfg_scale", default=6.0, help="CFG scale")
@click.option(
    "--taylor_threshold",
    default=2,
    help=(
        "Taylor threshold for all orders (default: 2). "
        "When elapsed <= threshold: use Taylor expansion; "
        "when elapsed > threshold: use residual reuse."
    ),
)
def main(
    prompt,
    num_prompts,
    seed,
    gpu_ids,
    model_root,
    output_dir,
    num_inference_steps,
    num_frames,
    cfg_scale,
    taylor_threshold,
):
    """Evaluate video similarity between original and accelerated generation methods with different Taylor orders."""
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in gpu_ids.split(",")]
    logger.info(f"Using GPUs: {gpu_ids}")

    # Create output directory
    if output_dir is None:
        output_dir = f"work_dirs/video_similarity_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Use same taylor_threshold for all orders
    cache_configs = {
        "order_0": {"taylor_threshold": taylor_threshold},
        "order_1": {"taylor_threshold": taylor_threshold},
        "order_2": {"taylor_threshold": taylor_threshold},
    }
    logger.info(f"Cache configurations: {cache_configs}")
    logger.info(f"AdaTaylorCache using taylor_threshold={taylor_threshold} for all orders")
    logger.info(f"  - elapsed <= {taylor_threshold}: Taylor series expansion")
    logger.info(f"  - elapsed > {taylor_threshold}: Residual reuse (fallback)")

    # Determine prompts to use
    if prompt:
        prompts = [prompt]
        logger.info(f"Using single prompt mode with seed {seed}")
    else:
        num_prompts = min(max(1, num_prompts), 10)
        prompts = TEST_PROMPTS[:num_prompts]
        logger.info(f"Using {num_prompts} test prompts")

    # Run evaluation
    if len(prompts) == 1:
        result = run_single_evaluation(
            prompt=prompts[0],
            seed=seed,
            output_dir=output_dir,
            model_root=model_root,
            gpu_id=gpu_ids[0],  # Use first GPU for single prompt
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            cache_configs=cache_configs,
        )
        results = [result]
        aggregate = None
    else:
        results, aggregate = run_multi_prompt_evaluation(
            prompts=prompts,
            output_dir=output_dir,
            model_root=model_root,
            gpu_ids=gpu_ids,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            cache_configs=cache_configs,
        )

    logger.info("Evaluation complete!")

    # Print single prompt results
    if len(results) == 1:
        print("\n" + "=" * 80)
        print("SINGLE PROMPT EVALUATION RESULTS - Taylor Order Comparison")
        print("=" * 80)
        print(f"Prompt: {results[0]['prompt'][:60]}...")
        print(f"Seed: {results[0]['seed']}")
        print(f"Taylor Threshold: {taylor_threshold}")
        print("-" * 80)
        print(f"{'Cache Type':<15} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8} | {'Time(s)':>8} | {'SpeedUp':>8}")
        print("-" * 80)

        cache_type_names = {
            "order_0": "Order 0",
            "order_1": "Order 1",
            "order_2": "Order 2",
        }
        original_time = results[0]["timings"]["original"]
        print(f"{'Original':<15} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {original_time:>8.2f} | {'1.00x':>8}")
        for cache_type in ["order_0", "order_1", "order_2"]:
            display_name = cache_type_names.get(cache_type, cache_type)
            if cache_type in results[0]["metrics"] and "error" not in results[0]["metrics"][cache_type]:
                m = results[0]["metrics"][cache_type]
                t = results[0]["timings"][cache_type]
                speedup = original_time / t if t > 0 else 0
                print(
                    f"{display_name:<15} | {m['psnr']:>8.2f} | {m['ssim']:>8.4f} | "
                    f"{m['lpips']:>8.4f} | {t:>8.2f} | {speedup:>8.2f}x"
                )
            else:
                print(f"{display_name:<15} | {'ERROR':>8} | {'ERROR':>8} | {'ERROR':>8} | {'N/A':>8} | {'N/A':>8}")

        print("=" * 80)

    return results


if __name__ == "__main__":
    main()
