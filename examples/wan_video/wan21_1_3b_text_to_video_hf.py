"""
Wan2.1 T2V 1.3B with HuggingFace Diffusers format - Simple Version.

This example shows how to use from_pretrained() in a pipeline file.
The server can load this file just like any other pipeline file.

Usage with server:
    telefuser serve examples/wan_video/wan21_1_3b_text_to_video_hf_simple.py --task t2v

Usage standalone:
    python wan21_1_3b_text_to_video_hf_simple.py --model_source "Wan-AI/Wan2.1-T2V-1.3B"
"""

import os

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline
from telefuser.utils.logging import logger
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

# Configuration
PPL_CONFIG = dict(
    name="wan21_1.3B_t2v_hf_simple",
    model_root=os.getenv("WAN21_MODEL_SOURCE", "Wan-AI/Wan2.1-T2V-1.3B"),  # HF model ID or local path
    negative_prompt="Camera shake, overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=40,
    num_frames=81,
    resolution="480p",
    cfg_scale=6.0,
    seed=42,
    tiled=True,
    target_fps=16,
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    sigma_shift=8.0,
)


def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """
    Create pipeline using from_pretrained.

    This function is called by the server. It uses from_pretrained() to load
    the model from HuggingFace model ID or local HF format folder.

    Args:
        parallelism: Number of parallel GPUs (REQUIRED)
        model_root: HF model ID or local path (REQUIRED)

    Returns:
        Initialized Wan21VideoPipeline
    """
    logger.info(f"Loading model from: {model_root}")
    logger.info(f"Parallelism: {parallelism}")

    # Use from_pretrained to load model (supports both HF ID and local path)
    pipe = Wan21VideoPipeline.from_pretrained(
        model_id_or_path=model_root,
        device="cuda",
        torch_dtype=torch.bfloat16,
        attention_config=AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"]),
        enable_clip_stage=False,  # T2V model
        enable_parallel=parallelism > 1,
        parallel_devices=list(range(parallelism)) if parallelism > 1 else None,
        sample_solver=PPL_CONFIG["sample_solver"],
    )

    logger.info("Pipeline loaded successfully")
    return pipe


def run(
    pipeline,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
    aspect_ratio="16:9",
):
    """Generate video from text prompt."""
    width, height = get_target_video_size_from_ratio(
        aspect_ratio,
        resolution=resolution,
        height_division_factor=2,
        width_division_factor=2,
    )

    video = pipeline(
        prompt=prompt,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}",
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
    )
    return video


def run_with_file(
    pipeline,
    prompt,
    negative_prompt,
    seed,
    resolution,
    output_path,
    aspect_ratio: str = "16:9",
    **kwargs,
):
    """Run pipeline and save to file."""
    video = run(
        pipeline,
        prompt,
        aspect_ratio=aspect_ratio,
        negative_prompt=negative_prompt,
        seed=seed,
        resolution=resolution,
    )
    logger.info(f"Saving video to {output_path}")
    save_video(
        video,
        output_path,
        fps=PPL_CONFIG["target_fps"],
        quality=6,
    )


@click.command()
@click.option(
    "--model_root",
    default=PPL_CONFIG["model_root"],
    help="HF model ID or local path (e.g., 'Wan-AI/Wan2.1-T2V-1.3B')",
)
@click.option("--gpu_num", default=1, help="Number of GPUs to use")
@click.option(
    "--prompt",
    default="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard.",
    help="Text prompt for video generation",
)
@click.option("--negative_prompt", default="", help="Negative prompt")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
@click.option("--resolution", default=PPL_CONFIG["resolution"], help="Resolution (480p, 720p)")
@click.option("--aspect_ratio", default="16:9", help="Aspect ratio (16:9, 4:3, 1:1)")
def main(
    model_root,
    gpu_num,
    prompt,
    negative_prompt,
    seed,
    resolution,
    aspect_ratio,
):
    """
    Text to video generation using Wan2.1 1.3B with HF format loading.

    Examples:
        # Using HF Model ID
        python wan21_1_3b_text_to_video_hf.py --model_root "Wan-AI/Wan2.1-T2V-1.3B"

        # Using local HF format folder
        python wan21_1_3b_text_to_video_hf.py --model_root "/path/to/Wan2.1-T2V-1.3B"

        # Multi-GPU
        python wan21_1_3b_text_to_video_hf.py --model_root "Wan-AI/Wan2.1-T2V-1.3B" --gpu_num 2
    """
    click.echo(f"Model root: {model_root}")
    click.echo(f"GPUs: {gpu_num}")

    # Load pipeline
    click.echo("Loading pipeline...")
    pipe = get_pipeline(gpu_num, model_root)

    # Run inference
    click.echo("Generating video...")
    import time

    start = time.time()
    video = run(pipe, prompt, negative_prompt, seed, resolution, aspect_ratio)
    elapsed_time = time.time() - start

    click.echo(f"Video generation time: {elapsed_time:.2f} seconds")

    # Save results
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)

    save_video(video, output_path, fps=16, quality=6)
    click.echo(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
