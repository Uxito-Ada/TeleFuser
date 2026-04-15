import os
import time

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.qwen_image import (
    QwenImageEditPipeline,
    QwenImageEditPipelineConfig,
)
from telefuser.utils.utils import get_example_name

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="qwen_image_edit_2511_h100",
    model_root=TF_MODEL_ZOO_PATH + "/Qwen-Image-Edit-2511",
    negative_prompt="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
    attn_impl=AttnImplType.TORCH_SDPA,
    seed=0,
    sample_solver="euler",
    model_type="Qwen-Image-Edit-Plus",
    enable_feature_cache=False,
    cfg_scale=4.0,
    num_inference_steps=40,
)


def get_pipeline(parallelism: int = 1, model_root: str = PPL_CONFIG["model_root"]):
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
    if PPL_CONFIG["enable_feature_cache"]:
        pipe_config.dit_config.feature_cache_config.enabled = True
        pipe_config.dit_config.feature_cache_config.model_type = PPL_CONFIG["model_type"]
    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism
        pipe_config.enable_denoising_parallel = True
    pipeline.init(mm, pipe_config)
    return pipeline


def run(
    pipeline: QwenImageEditPipeline,
    prompt,
    image,
    negative_prompt=PPL_CONFIG["negative_prompt"],
    seed=PPL_CONFIG["seed"],
):
    image = pipeline(
        prompt,
        image=image,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        rand_device="cpu",
        cfg_scale=PPL_CONFIG["cfg_scale"],
    )
    return image


@click.command()
@click.option("--aspect_ratio", "-ar", default="1:1", help="Image ratio such as 1:1, 16:9", type=str)
@click.option("--gpu_num", default=1, help="Number of GPUs to use", type=int)
@click.option("--prompt", default='这个女生看着面前的电视屏幕，屏幕上面写着"阿里巴巴"', help="Custom prompt text")
@click.option("--negative_prompt", default=PPL_CONFIG["negative_prompt"], help="Negative prompt")
@click.option("--image_path", default=None, help="Custom image path")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Model root directory")
@click.option("--output", default=get_example_name(__file__, "png"), help="Output image filename")
def main(aspect_ratio, gpu_num, prompt, negative_prompt, image_path, model_root, output):
    if image_path is None:
        image_path = f"{os.path.dirname(__file__)}/../data/edit2511input.png"
    image = Image.open(image_path)
    pipeline = get_pipeline(gpu_num, model_root)
    # Warm up
    images = run(pipeline, prompt, [image], aspect_ratio)
    # Timing run
    s = time.time()
    images = run(pipeline, prompt, [image], aspect_ratio)
    print(f"pipe cost {time.time() - s} s")
    for i, image in enumerate(images):
        image.save(output.replace(".png", f"_{i}.png"))
    del pipeline


if __name__ == "__main__":
    main()
