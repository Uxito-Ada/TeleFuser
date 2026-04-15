"""

Cache Calibrator Example for HunyuanVideo Text-to-Video

This script runs the pipeline once to collect calibration data
and generates a parameter JSON file for AdaTaylorCache.

Usage:
    python hunyuan_video_t2v_cache_calibrate.py \
        --model_root /path/to/HunyuanVideo-1.5/ \
        --num_inference_steps 50 \
        --sigma_shift 7.0 \
        --output_path ./cache_params.json

The generated JSON file will contain:
    - K, retention_ratio, thresh: Default values (0), need to be adjusted by user
    - sigma_shift: The sigma shift value used during calibration
    - num_inference_steps: Number of inference steps used during calibration
    - cond_mag_ratios: Mag ratios for conditional path (with 1.0 prepended)
    - uncond_mag_ratios: Mag ratios for unconditional path (with 1.0 prepended)

These parameters are used by AdaTaylorCache (including when n_derivatives=0
for simple residual caching).
"""

import os
import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.hunyuan_video_dit import HunyuanVideoDiT
from telefuser.models.hunyuan_video_text_encoder import HunyuanVideoTextEncoder
from telefuser.models.hunyuan_video_vae import HunyuanVideoVAE
from telefuser.pipelines.hunyuan_video_1_5 import (
    HunyuanVideo15Pipeline,
    HunyuanVideo15PipelineConfig,
)
from telefuser.schedulers.flow_match_discrete import FlowMatchDiscreteScheduler
from telefuser.utils.logging import logger
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

# Default configuration

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")

PPL_CONFIG = dict(
    name="hunyuan_video_t2v_cache_calibrate",
    model_root=TF_MODEL_ZOO_PATH + "/HunyuanVideo-1.5",
    negative_prompt="",
    transformer_version="480p_t2v",
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    sigma_shift=5.0,
    target_fps=24,
    num_inference_steps=50,
    num_frames=121,
    cfg_scale=6.0,
)


def get_pipeline(parallelism: int = 1, model_root: str = PPL_CONFIG["model_root"]):
    """
    Create and initialize the HunyuanVideo T2V pipeline.

    Args:
        parallelism: Number of parallel GPUs for inference (REQUIRED)
        model_root: Root directory of the model checkpoints (REQUIRED)
    """
    module_manager = ModuleManager(device="cpu")

    vae_path = os.path.join(model_root, "vae")
    logger.info(f"Loading VAE from {vae_path}")
    vae = HunyuanVideoVAE.from_pretrained(vae_path, torch_dtype=torch.bfloat16)
    module_manager.add_module(vae, name="vae")

    text_encoder_path = os.path.join(model_root, "text_encoder", "llm")
    logger.info(f"Loading Text Encoder from {text_encoder_path}")
    text_encoder = HunyuanVideoTextEncoder.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
    module_manager.add_module(text_encoder, name="text_encoder")

    scheduler_path = os.path.join(model_root, "scheduler")
    logger.info(f"Loading Scheduler from {scheduler_path}")
    scheduler = FlowMatchDiscreteScheduler.from_pretrained(scheduler_path, shift=PPL_CONFIG["sigma_shift"])
    module_manager.add_module(scheduler, name="scheduler")

    transformer_path = os.path.join(model_root, "transformer", PPL_CONFIG["transformer_version"])
    logger.info(f"Loading Transformer from {transformer_path}")
    transformer = HunyuanVideoDiT.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16,
    )
    module_manager.add_module(transformer, name="hunyuan_video_dit")

    pipe = HunyuanVideo15Pipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = HunyuanVideo15PipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]

    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = 2
        pipe_config.enable_denoising_parallel = True

    pipe.init(module_manager, pipe_config)

    return pipe


def run_calibration(
    pipeline: HunyuanVideo15Pipeline,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
    model_name: str = "HunyuanVideo-T2V",
    output_path: str | None = None,
):
    """
    Run cache calibration.

    This function runs the pipeline once in calibration mode to collect
    residual data and generate cache parameters for AdaTaylorCache.

    Args:
        pipeline: Preloaded video generation pipeline
        prompt: Positive guidance text prompt
        negative_prompt: Negative guidance prompt
        seed: Random seed
        resolution: Resolution such as 720p, 480p
        aspect_ratio: Aspect ratio such as 16:9
        model_name: Model name for the output file
        output_path: Output path for the JSON file

    Returns:
        Generated video frames
    """
    width, height = get_target_video_size_from_ratio(
        aspect_ratio,
        resolution=resolution,
        height_division_factor=16,
        width_division_factor=16,
    )

    pipeline.denoise_stage.dit.set_ada_taylor_cache_calibrator(
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        sigma_shift=PPL_CONFIG["sigma_shift"],
        model_name=model_name,
        output_path=output_path,
    )

    logger.info(
        f"Starting cache calibration with {PPL_CONFIG['num_inference_steps']} steps, sigma_shift={PPL_CONFIG['sigma_shift']}"
    )
    logger.info(f"Output will be saved to: {output_path or 'default params directory'}")

    video = pipeline(
        prompt=prompt,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}".strip(),
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        height=height,
        width=width,
        rand_device="cpu",
    )

    return video


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option(
    "--prompt",
    default="一只可爱的猫咪在阳光下打盹，毛发在微风中轻轻飘动，画面温馨治愈",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--seed", default=42, help="Random seed")
@click.option("--resolution", default="480p", help="Resolution")
@click.option("--aspect_ratio", default="16:9", help="Aspect ratio")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Model checkpoint root")
@click.option("--model_name", default="HunyuanVideo-T2V", help="Model name for the output file")
@click.option("--output_path", default=None, help="Output path for the JSON file (default: params directory)")
def main(
    gpu_num: int,
    prompt: str,
    negative_prompt: str,
    seed: int,
    resolution: str,
    aspect_ratio: str,
    model_root: str,
    model_name: str,
    output_path: str | None,
):
    """Run cache calibration for HunyuanVideo Text-to-Video model."""
    logger.info("=" * 60)
    logger.info("Cache Calibrator for HunyuanVideo Text-to-Video")
    logger.info("=" * 60)

    pipe = get_pipeline(gpu_num, model_root)

    start = time.time()
    video = run_calibration(
        pipe,
        prompt,
        negative_prompt,
        seed,
        resolution,
        aspect_ratio,
        model_name,
        output_path,
    )
    elapsed_time = time.time() - start

    logger.info(f"Calibration completed in {elapsed_time:.2f} seconds")
    logger.info(f"Generated {len(video)} frames")

    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    sample_path = os.path.join(output_dir, get_example_name(__file__, "mp4"))

    save_video(video, sample_path, fps=PPL_CONFIG["target_fps"], quality=6)
    logger.info(f"Sample video saved to: {sample_path}")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
