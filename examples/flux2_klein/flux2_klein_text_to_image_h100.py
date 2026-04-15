"""Flux2 Klein text-to-image generation example.


Usage:
    python examples/flux2_klein/flux2_klein_text_to_image_h100.py --prompt "A cat holding a sign"

For multi-GPU:
    python examples/flux2_klein/flux2_klein_text_to_image_h100.py --gpu_num 2
"""

import os
import time

import click
import torch
from diffusers import AutoencoderKLFlux2
from transformers import Qwen3ForCausalLM

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.flux2_klein import Flux2KleinPipeline, Flux2KleinPipelineConfig
from telefuser.utils.logging import logger
from telefuser.utils.utils import get_example_name

# =============================================================================
# Configuration
# =============================================================================

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="flux2_klein_t2i_h100",
    model_root=TF_MODEL_ZOO_PATH + "/FLUX.2-klein-9B",  # HF model ID or local path
    num_inference_steps=4,
    cfg_scale=1.0,
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
)


# =============================================================================
# Pipeline Loading (from_pretrained logic)
# =============================================================================


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
):
    """Load Flux2 Klein pipeline.

    Args:
        parallelism: Number of parallel GPUs (REQUIRED)
        model_root: HF model ID or local path (REQUIRED)
    """
    logger.info(f"Loading Flux2 Klein pipeline from: {model_root}")

    # Construct paths to model components
    transformer_path = os.path.join(model_root, "transformer")
    vae_path = os.path.join(model_root, "vae")
    text_encoder_path = os.path.join(model_root, "text_encoder")
    tokenizer_path = os.path.join(model_root, "tokenizer")

    logger.info(f"  Transformer: {transformer_path}")
    logger.info(f"  VAE: {vae_path}")
    logger.info(f"  Text Encoder: {text_encoder_path}")

    # Create ModuleManager to handle model loading
    mm = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")

    # 1. Load DiT (Flux2DiT with hash-based auto-detection)
    transformer_files = [
        os.path.join(transformer_path, f) for f in os.listdir(transformer_path) if f.endswith(".safetensors")
    ]
    mm.load_model(transformer_files, torch_dtype=torch.bfloat16)

    # 2. Load VAE from HuggingFace (AutoencoderKLFlux2)
    mm.load_from_huggingface(
        vae_path,
        module_source="diffusers",
        module_class=AutoencoderKLFlux2,
        module_name="vae",
        torch_dtype=torch.bfloat16,
    )

    # 3. Load TextEncoder from HuggingFace (Qwen3ForCausalLM)
    mm.load_from_huggingface(
        text_encoder_path,
        module_source="transformers",
        module_class=Qwen3ForCausalLM,
        module_name="text_encoder",
        torch_dtype=torch.bfloat16,
    )

    # 4. Load tokenizer
    mm.load_from_huggingface(
        tokenizer_path,
        module_source="transformers",
        module_name="tokenizer",
    )

    # Create pipeline instance
    pipeline = Flux2KleinPipeline(device="cuda", torch_dtype=torch.bfloat16)

    # Create pipeline configuration
    config = Flux2KleinPipelineConfig()

    attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    config.dit_config.attention_config = attention_config
    logger.info(f"Using attention implementation: {attention_config.attn_impl.name}")

    # Initialize pipeline with loaded models and config
    pipeline.init(mm, config)

    logger.info(f"Successfully loaded Flux2 Klein pipeline from {model_root}")
    return pipeline


# =============================================================================
# Inference
# =============================================================================


def run(
    pipeline,
    prompt,
    seed=42,
    height=1024,
    width=1024,
):
    """Generate image from text prompt.

    Args:
        pipeline (Flux2KleinPipeline): Preloaded pipeline.
        prompt (str): Text prompt.
        seed (int): Random seed.
        height (int): Image height (divisible by 16).
        width (int): Image width (divisible by 16).

    Returns:
        List[PIL.Image]: Generated images.
    """
    print(f"generate image with shape {width}, {height}")
    images = pipeline(
        prompt=prompt,
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        height=height,
        width=width,
    )
    return images


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option(
    "--prompt",
    default="A cat holding a sign that says hello world",
    help="Text prompt for image generation",
)
@click.option("--seed", default=42, help="Random seed")
@click.option("--height", default=1024, help="Image height (divisible by 16)")
@click.option("--width", default=1024, help="Image width (divisible by 16)")
@click.option(
    "--model_root",
    default=PPL_CONFIG["model_root"],
    help="HuggingFace model ID or local path",
)
def main(
    gpu_num,
    prompt,
    seed,
    height,
    width,
    model_root,
):
    """Text-to-image generation using Flux2 Klein model."""
    # Load pipeline
    pipe = get_pipeline(
        gpu_num,
        model_root=model_root,
    )

    # Warmup run
    run(pipe, prompt, seed, height, width)

    # Benchmark run
    start = time.time()
    images = run(pipe, prompt, seed, height, width)
    elapsed_time = time.time() - start

    print(f"Image generation time: {elapsed_time:.2f} seconds")

    # Save results
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__, "png")
    output_path = os.path.join(output_dir, filename)

    images[0].save(output_path)
    print(f"Image saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
