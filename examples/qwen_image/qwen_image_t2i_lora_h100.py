import math
import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, LoraConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.qwen_image import (
    QwenImagePipeline,
    QwenImagePipelineConfig,
)
from telefuser.pipelines.qwen_image.qwen_image import ASPECT_RATIO_TO_SIZE

PPL_CONFIG = dict(
    name="qwen_image_t2i",
    dit_path=[
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
    ],
    vae_path=[
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/vae/diffusion_pytorch_model.safetensors",
    ],
    text_encoder_path=[
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/text_encoder/model-00001-of-00004.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/text_encoder/model-00002-of-00004.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/text_encoder/model-00003-of-00004.safetensors",
        "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512/text_encoder/model-00004-of-00004.safetensors",
    ],
    lora_configs=[
        LoraConfig(
            "/nvfile-heatstorage/model_zoo/huggingface/Qwen-Image-2512-Lightning/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors",
            1,
        )
    ],
    attn_impl=AttnImplType.TORCH_SDPA,
    seed=0,
    sample_solver="euler",
    cfg_scale=1.0,
    num_inference_steps=16,
)


def get_pipeline(parallelism=1):
    mm = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_model(PPL_CONFIG["dit_path"], device="cpu", torch_dtype=torch.bfloat16)
    mm.load_model(PPL_CONFIG["vae_path"], device="cpu", torch_dtype=torch.bfloat16)
    mm.load_model(PPL_CONFIG["text_encoder_path"], device="cpu", torch_dtype=torch.bfloat16)
    pipeline = QwenImagePipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = QwenImagePipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.NO_CPU_OFFLOAD
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.dit_config.lora_configs = PPL_CONFIG["lora_configs"]
    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism
        pipe_config.enable_denoising_parallel = True
    pipeline.init(mm, pipe_config)
    return pipeline


def run(
    pipeline: QwenImagePipeline,
    prompt,
    aspect_ratio: str = "1:1",
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
):
    width, height = ASPECT_RATIO_TO_SIZE[aspect_ratio]
    image = pipeline(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        rand_device="cpu",
        cfg_scale=PPL_CONFIG["cfg_scale"],
        exponential_shift_mu=math.log(3),
        shift_terminal=None,
    )
    return image


@click.command()
@click.option("--aspect_ratio", "-ar", default="19:6", help="Image ratio such as 1:1, 16:9", type=str)
@click.option("--gpu_num", default=1, help="Number of GPUs to use", type=int)
@click.option("--prompt", default="A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm.", help="Custom prompt text")
@click.option("--output", default="image.jpg", help="Output image filename")
def main(aspect_ratio, gpu_num, prompt, output):
    if prompt is None:
        prompt = """A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm."""
    negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
    pipeline = get_pipeline(gpu_num)

    # Warm up
    images = run(pipeline, prompt, aspect_ratio, negative_prompt=negative_prompt)

    # Timing run
    s = time.time()
    images = run(pipeline, prompt, aspect_ratio, negative_prompt=negative_prompt)
    print(f"pipe cost {time.time() - s} s")
    for i, image in enumerate(images):
        image.save(output.replace(".jpg", f"_{i}.jpg"))
    del pipeline


if __name__ == "__main__":
    main()
