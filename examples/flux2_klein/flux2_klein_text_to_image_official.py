"""Flux2 Klein text-to-image generation using original diffusers pipeline.

This example uses the original diffusers Flux2KleinPipeline for comparison
with the TeleFuser implementation.

Usage:
    python examples/flux2_klein/flux2_klein_text_to_image_official.py --prompt "A cat holding a sign"
"""

import os
import time

import click
import torch
from diffusers import Flux2KleinPipeline


def get_pipeline(model_id=None, cache_dir=None):
    """Load Flux2 Klein pipeline from diffusers.

    Args:
        model_id (str): HuggingFace model ID or local path.
        cache_dir (str): Cache directory for downloads.

    Returns:
        Flux2KleinPipeline: Initialized diffusers pipeline.
    """
    model_id = model_id or "black-forest-labs/FLUX.2-klein-base-9B"

    pipeline = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    ).to("cuda")

    return pipeline


def run(
    pipeline,
    prompt,
    seed=42,
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=1.0,
):
    """Generate image from text prompt using diffusers.

    Args:
        pipeline (Flux2KleinPipeline): Preloaded diffusers pipeline.
        prompt (str): Text prompt.
        negative_prompt (str): Negative prompt for CFG.
        seed (int): Random seed.
        height (int): Image height (divisible by 16).
        width (int): Image width (divisible by 16).
        num_inference_steps (int): Number of inference steps.
        guidance_scale (float): CFG scale.

    Returns:
        List[PIL.Image]: Generated images.
    """
    # Create generator on CUDA for consistent results with TeleFuser
    generator = torch.Generator(device="cuda").manual_seed(seed)

    images = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
    ).images

    return images


@click.command()
@click.option(
    "--prompt",
    default="A cat holding a sign that says hello world",
    help="Text prompt for image generation",
)
@click.option("--seed", default=42, help="Random seed")
@click.option("--height", default=1024, help="Image height (divisible by 16)")
@click.option("--width", default=1024, help="Image width (divisible by 16)")
@click.option("--num_inference_steps", default=4, help="Number of inference steps")
@click.option("--guidance_scale", default=1.0, help="CFG guidance scale")
@click.option(
    "--model_id",
    default=None,
    help="HuggingFace model ID or local path (default: black-forest-labs/FLUX.2-klein-base-9B)",
)
@click.option("--cache_dir", default=None, help="Cache directory for downloads")
def main(
    prompt,
    seed,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    model_id,
    cache_dir,
):
    """Text-to-image generation using original diffusers Flux2 Klein pipeline."""
    print("Loading Flux2 Klein pipeline from diffusers...")
    pipe = get_pipeline(model_id, cache_dir)

    # Run inference
    print(f"Generating image with prompt: {prompt}")
    images = run(
        pipe,
        prompt,
        seed,
        height,
        width,
        num_inference_steps,
        guidance_scale,
    )
    start = time.time()
    images = run(
        pipe,
        prompt,
        seed,
        height,
        width,
        num_inference_steps,
        guidance_scale,
    )
    elapsed_time = time.time() - start

    print(f"Image generation time: {elapsed_time:.2f} seconds")

    # Save results
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    output_path = os.path.join(output_dir, "flux2_klein_official.png")

    images[0].save(output_path)
    print(f"Image saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
