import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.qwen_image.qwen_image import ASPECT_RATIO_TO_SIZE
from telefuser.pipelines.z_image import (
    ZImagePipeline,
    ZImagePipelineConfig,
)
from telefuser.utils.utils import get_example_name

PPL_CONFIG = dict(
    name="z_image_turbo_t2i",
    dit_path=[
        "/nvfile-heatstorage/model_zoo/huggingface/Z-Image-Turbo/transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Z-Image-Turbo/transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Z-Image-Turbo/transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    ],
    vae_path="/nvfile-heatstorage/model_zoo/huggingface/Z-Image-Turbo/vae",
    text_encoder_path=[
        "/nvfile-heatstorage/model_zoo/huggingface/Z-Image-Turbo/text_encoder/model-00001-of-00003.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Z-Image-Turbo/text_encoder/model-00002-of-00003.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Z-Image-Turbo/text_encoder/model-00003-of-00003.safetensors",
    ],
    tokenizer_path="/nvfile-heatstorage/model_zoo/huggingface/Z-Image-Turbo/tokenizer",
    scheduler_path="/nvfile-heatstorage/model_zoo/huggingface/Z-Image-Turbo/scheduler",
    attn_impl=AttnImplType.TORCH_SDPA,
    seed=42,
    cfg_scale=0,
    num_inference_steps=9,
)


def get_pipeline(parallelism=1):
    mm = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_model(PPL_CONFIG["dit_path"], device="cpu", torch_dtype=torch.bfloat16)
    mm.load_from_huggingface(
        PPL_CONFIG["vae_path"],
        device="cpu",
        torch_dtype=torch.bfloat16,
        module_source="diffusers",
        module_name="z_image_vae",
    )
    mm.load_model(PPL_CONFIG["text_encoder_path"], device="cpu", torch_dtype=torch.bfloat16)
    mm.load_from_huggingface(PPL_CONFIG["tokenizer_path"], "transformers", module_name="tokenizer")
    mm.load_from_huggingface(PPL_CONFIG["scheduler_path"], "diffusers", module_name="scheduler")
    pipeline = ZImagePipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = ZImagePipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipeline.init(mm, pipe_config)
    return pipeline


def run(
    pipeline: ZImagePipelineConfig,
    prompt,
    aspect_ratio: str = "16:9",
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
):
    width, height = ASPECT_RATIO_TO_SIZE[aspect_ratio]
    width = 1024
    height = 1024
    image = pipeline(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        rand_device="cuda",
        cfg_scale=PPL_CONFIG["cfg_scale"],
    )
    return image


@click.command()
@click.option("--aspect_ratio", "-ar", default="16:9", help="Image ratio such as 1:1, 16:9", type=str)
@click.option("--gpu_num", default=1, help="Number of GPUs to use", type=int)
@click.option("--prompt", default="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.", help="Custom prompt text")
@click.option("--output", default=get_example_name(__file__, "png"), help="Output image filename")
def main(aspect_ratio, gpu_num, prompt, output):
    if prompt is None:
        prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    pipeline = get_pipeline(gpu_num)
    # Warm up
    images = run(pipeline, prompt, aspect_ratio)
    # Timing run
    s = time.time()
    images = run(pipeline, prompt, aspect_ratio)
    print(f"pipe cost {time.time() - s} s")
    for i, image in enumerate(images):
        image.save(output.replace(".png", f"_{i}.png"))
    del pipeline


if __name__ == "__main__":
    main()
