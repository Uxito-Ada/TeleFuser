"""

Cache Calibrator Example for Qwen-Image

This script runs the pipeline once to collect calibration data
and generates a parameter JSON file for AdaTaylorCache.

Usage:
    python qwen_image_cache_calibrate.py \
        --model_root /path/to/Qwen-Image-2512/ \
        --num_inference_steps 50 \
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

from telefuser.core.config import AttentionConfig, AttnImplType, FeatureCacheConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.qwen_image import (
    QwenImagePipeline,
    QwenImagePipelineConfig,
)
from telefuser.pipelines.qwen_image.qwen_image import ASPECT_RATIO_TO_SIZE
from telefuser.utils.logging import logger
from telefuser.utils.utils import get_example_name

# Default configuration

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")

PPL_CONFIG = dict(
    name="qwen_image_cache_calibrate",
    model_root=TF_MODEL_ZOO_PATH + "/Qwen-Image-2512",
    negative_prompt="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
    num_inference_steps=50,
    cfg_scale=4.0,
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    model_name="Qwen-Image-2512",
)


def get_pipeline(parallelism: int = 1, model_root: str = PPL_CONFIG["model_root"]):
    """
    Create and initialize the image generation pipeline.

    Args:
        parallelism: Number of parallel GPUs for inference
        model_root: Root directory of the model files
    """
    dit_path = [
        f"{model_root}/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
        f"{model_root}/transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
    ]
    vae_path = [f"{model_root}/vae/diffusion_pytorch_model.safetensors"]
    text_encoder_path = [
        f"{model_root}/text_encoder/model-00001-of-00004.safetensors",
        f"{model_root}/text_encoder/model-00002-of-00004.safetensors",
        f"{model_root}/text_encoder/model-00003-of-00004.safetensors",
        f"{model_root}/text_encoder/model-00004-of-00004.safetensors",
    ]
    tokenizer_path = f"{model_root}/tokenizer"

    mm = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_model(dit_path, device="cpu", torch_dtype=torch.bfloat16)
    mm.load_model(vae_path, device="cpu", torch_dtype=torch.bfloat16)
    mm.load_model(text_encoder_path, device="cpu", torch_dtype=torch.bfloat16)
    mm.load_from_huggingface(tokenizer_path, "transformers", module_name="tokenizer")

    pipeline = QwenImagePipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = QwenImagePipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.cfg_degree = parallelism
        pipe_config.enable_denoising_parallel = True
        pipe_config.text_encoding_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.text_encoding_config.parallel_config.tp_degree = parallelism
        pipe_config.enable_text_encoding_parallel = True

    pipeline.init(mm, pipe_config)
    return pipeline


def run_calibration(
    pipeline: QwenImagePipeline,
    prompt: str,
    negative_prompt: str = "",
    aspect_ratio: str = "16:9",
    seed: int = 42,
    model_name: str = "Qwen-Image",
    output_path: str | None = None,
) -> list:
    """
    Run cache calibration.

    This function runs the pipeline once in calibration mode to collect
    residual data and generate cache parameters for AdaTaylorCache.

    Args:
        pipeline: Preloaded image generation pipeline
        prompt: Positive guidance text prompt
        negative_prompt: Negative guidance prompt
        aspect_ratio: Image aspect ratio such as "16:9", "1:1"
        seed: Random seed
        model_name: Model name for the output file
        output_path: Output path for the JSON file

    Returns:
        Generated images
    """
    width, height = ASPECT_RATIO_TO_SIZE[aspect_ratio]

    pipeline.denoise_stage.dit.set_ada_taylor_cache_calibrator(
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        sigma_shift=1.0,
        model_name=model_name,
        output_path=output_path,
    )

    logger.info(f"Starting cache calibration with {PPL_CONFIG['num_inference_steps']} steps")
    logger.info(f"Output will be saved to: {output_path or 'default params directory'}")

    images = pipeline(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        rand_device="cuda",
        cfg_scale=PPL_CONFIG["cfg_scale"],
        cache_config=None,
    )

    return images


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option(
    "--prompt",
    default="A beautiful sunset over the ocean with waves gently rolling onto a sandy beach.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--seed", default=42, help="Random seed")
@click.option("--aspect_ratio", default="16:9", help="Aspect ratio such as 16:9, 1:1")
@click.option(
    "--model_root",
    default=PPL_CONFIG["model_root"],
    help="Root directory of the model files",
)
@click.option("--model_name", default="Qwen-Image-2512", help="Model name for the output file")
@click.option("--output_path", default=None, help="Output path for the JSON file (default: params directory)")
def main(
    gpu_num: int,
    prompt: str,
    negative_prompt: str,
    seed: int,
    aspect_ratio: str,
    model_root: str,
    model_name: str,
    output_path: str | None,
):
    """Run cache calibration for Qwen-Image model."""
    logger.info("=" * 60)
    logger.info("Cache Calibrator for Qwen-Image")
    logger.info("=" * 60)

    pipe = get_pipeline(parallelism=gpu_num, model_root=model_root)

    full_negative = (
        f"{negative_prompt} {PPL_CONFIG['negative_prompt']}" if negative_prompt else PPL_CONFIG["negative_prompt"]
    )
    start = time.time()
    images = run_calibration(
        pipe,
        prompt,
        full_negative,
        aspect_ratio,
        seed,
        model_name,
        output_path,
    )
    elapsed_time = time.time() - start

    logger.info(f"Calibration completed in {elapsed_time:.2f} seconds")
    logger.info(f"Generated {len(images)} images")

    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    sample_path = os.path.join(output_dir, get_example_name(__file__, "png"))

    for i, image in enumerate(images):
        image.save(sample_path.replace(".png", f"_{i}.png"))
    logger.info(f"Sample image saved to: {sample_path}")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
