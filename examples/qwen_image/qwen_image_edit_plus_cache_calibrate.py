"""
Cache Calibrator Example for Qwen-Image-Edit-Plus

This script runs the pipeline once to collect calibration data
and generates a parameter JSON file for AdaTaylorCache.

Usage:
    python qwen_image_edit_plus_cache_calibrate.py \
        --model_root /path/to/Qwen-Image-Edit-2511/ \
        --num_inference_steps 40 \
        --output_path ./cache_params.json

The generated JSON file will contain:
    - K, retention_ratio, thresh: Smart defaults based on num_inference_steps
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
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.qwen_image import (
    QwenImageEditPipeline,
    QwenImageEditPipelineConfig,
)
from telefuser.utils.logging import logger
from telefuser.utils.utils import get_example_name

# Default configuration
PPL_CONFIG = dict(
    name="qwen_image_edit_plus_cache_calibrate",
    negative_prompt="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
    num_inference_steps=40,
    cfg_scale=4.0,
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    model_name="Qwen-Image-Edit-Plus",
)


def get_pipeline(
    dit_path: list[str],
    vae_path: list[str],
    text_encoder_path: list[str],
    processor_path: str,
    parallelism: int = 1,
):
    """
    Create and initialize the image editing pipeline.

    Args:
        dit_path: Path to DiT model weights
        vae_path: Path to VAE model weights
        text_encoder_path: Path to text encoder weights
        processor_path: Path to processor
        parallelism: Number of parallel GPUs for inference
    """
    mm = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_model(dit_path, device="cpu", torch_dtype=torch.bfloat16)
    mm.load_model(vae_path, device="cpu", torch_dtype=torch.bfloat16)
    mm.load_model(text_encoder_path, device="cpu", torch_dtype=torch.bfloat16)
    mm.load_from_huggingface(processor_path, module_name="processor")

    pipeline = QwenImageEditPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = QwenImageEditPipelineConfig()
    pipe_config.is_edit_plus = True
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.cfg_degree = parallelism
        pipe_config.enable_denoising_parallel = True

    pipeline.init(mm, pipe_config)
    return pipeline


def run_calibration(
    pipeline: QwenImageEditPipeline,
    prompt: str,
    image: list[Image.Image],
    negative_prompt: str = "",
    seed: int = 42,
    model_name: str = "Qwen-Image-Edit-Plus",
    output_path: str | None = None,
) -> list[Image.Image]:
    """
    Run cache calibration.

    This function runs the pipeline once in calibration mode to collect
    residual data and generate cache parameters for AdaTaylorCache.

    Args:
        pipeline: Preloaded image editing pipeline
        prompt: Positive guidance text prompt
        image: List of input PIL Images for editing
        negative_prompt: Negative guidance prompt
        seed: Random seed
        model_name: Model name for the output file
        output_path: Output path for the JSON file

    Returns:
        Generated images
    """
    # Set up cache calibrator
    pipeline.denoise_stage.dit.set_ada_taylor_cache_calibrator(
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        sigma_shift=1.0,  # Qwen-Image uses FlowMatch scheduler
        model_name=model_name,
        output_path=output_path,
    )

    logger.info(f"Starting cache calibration with {PPL_CONFIG['num_inference_steps']} steps")
    logger.info(f"Output will be saved to: {output_path or 'default params directory'}")

    # Run the pipeline - calibration data will be collected during forward passes
    images = pipeline(
        prompt,
        image=image,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        rand_device="cuda",
        cfg_scale=PPL_CONFIG["cfg_scale"],
    )

    return images


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option(
    "--prompt",
    default='这个女生看着面前的电视屏幕，屏幕上面写着"阿里巴巴"',
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--seed", default=42, help="Random seed")
@click.option(
    "--image_path",
    default=None,
    help="Input image path for editing (default: examples/data/edit2511input.png)",
)
@click.option(
    "--model_root",
    default="/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-Edit-2511/",
    help="Root directory of the model files",
)
@click.option("--model_name", default="Qwen-Image-Edit-Plus", help="Model name for the output file")
@click.option("--output_path", default=None, help="Output path for the JSON file (default: params directory)")
def main(
    gpu_num: int,
    prompt: str,
    negative_prompt: str,
    seed: int,
    image_path: str | None,
    model_root: str,
    model_name: str,
    output_path: str | None,
):
    """Run cache calibration for Qwen-Image-Edit-Plus model."""
    logger.info("=" * 60)
    logger.info("Cache Calibrator for Qwen-Image-Edit-Plus")
    logger.info("=" * 60)

    if image_path is None:
        image_path = os.path.join(os.path.dirname(__file__), "../data/edit2511input.png")

    image = Image.open(image_path)
    logger.info(f"Loaded input image from: {image_path}")

    dit_path = [
        f"{model_root}/transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
    ]
    vae_path = [f"{model_root}/vae/diffusion_pytorch_model.safetensors"]
    text_encoder_path = [
        f"{model_root}/text_encoder/model-00001-of-00004.safetensors",
        f"{model_root}/text_encoder/model-00002-of-00004.safetensors",
        f"{model_root}/text_encoder/model-00003-of-00004.safetensors",
        f"{model_root}/text_encoder/model-00004-of-00004.safetensors",
    ]
    processor_path = f"{model_root}/processor"

    pipe = get_pipeline(
        dit_path=dit_path,
        vae_path=vae_path,
        text_encoder_path=text_encoder_path,
        processor_path=processor_path,
        parallelism=gpu_num,
    )

    full_negative = (
        f"{negative_prompt} {PPL_CONFIG['negative_prompt']}" if negative_prompt else PPL_CONFIG["negative_prompt"]
    )
    start = time.time()
    images = run_calibration(
        pipe,
        prompt,
        [image],
        full_negative,
        seed,
        model_name,
        output_path,
    )
    elapsed_time = time.time() - start

    logger.info(f"Calibration completed in {elapsed_time:.2f} seconds")
    logger.info(f"Generated {len(images)} images")

    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    sample_path = os.path.join(output_dir, get_example_name(__file__, "png"))

    for i, img in enumerate(images):
        img.save(sample_path.replace(".png", f"_{i}.png"))
    logger.info(f"Sample image saved to: {sample_path}")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
