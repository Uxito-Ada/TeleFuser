import os
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

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="z_image_turbo_t2i",
    model_root=TF_MODEL_ZOO_PATH + "/Z-Image-Turbo",
    dit_path="transformer/diffusion_pytorch_model-0000*-of-00003.safetensors",
    vae_path="vae",
    text_encoder_path="text_encoder/model-0000*-of-00003.safetensors",
    tokenizer_path="tokenizer",
    scheduler_path="scheduler",
    attn_impl=AttnImplType.TORCH_SDPA,
    seed=42,
    cfg_scale=0,
    num_inference_steps=9,
)


def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    mm = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_model(os.path.join(model_root, PPL_CONFIG["dit_path"]), device="cpu", torch_dtype=torch.bfloat16)
    mm.load_from_huggingface(
        os.path.join(model_root, PPL_CONFIG["vae_path"]),
        device="cpu",
        torch_dtype=torch.bfloat16,
        module_source="diffusers",
        module_name="z_image_vae",
    )
    mm.load_model(os.path.join(model_root, PPL_CONFIG["text_encoder_path"]), device="cpu", torch_dtype=torch.bfloat16)
    mm.load_from_huggingface(
        os.path.join(model_root, PPL_CONFIG["tokenizer_path"]), "transformers", module_name="tokenizer"
    )
    mm.load_from_huggingface(
        os.path.join(model_root, PPL_CONFIG["scheduler_path"]), "diffusers", module_name="scheduler"
    )
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
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Number of GPUs to use", type=str)
@click.option(
    "--prompt",
    default="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.",
    help="Custom prompt text",
)
@click.option("--output", default=get_example_name(__file__, "png"), help="Output image filename")
def main(aspect_ratio, gpu_num, prompt, output, model_root):
    pipeline = get_pipeline(gpu_num, model_root)
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
