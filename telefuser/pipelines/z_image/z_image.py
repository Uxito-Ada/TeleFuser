from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.utils.func import auto_async_call
from telefuser.utils.logging import logger
from telefuser.worker.parallel_worker import ParallelWorker

from .dit_denoising import DitDenoisingStage
from .text_encoding import TextEncodingStage
from .vae import VAEStage

# Aspect ratio to resolution mapping for Z-Image
ASPECT_RATIO_TO_SIZE = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}


@dataclass
class ZImagePipelineConfig:
    """Configuration for Z-Image generation pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    enable_metrics: bool = False


class ZImagePipeline(BasePipeline):
    """Z-Image text-to-image generation pipeline.

    Supports text-to-image generation with CFG normalization and truncation,
    tiled VAE decoding for large images, and various aspect ratios.
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        return [self.vae_stage, self.denoise_stage, self.text_encoding_stage]

    def init(self, module_manager: ModuleManager, config: ZImagePipelineConfig):
        """Initialize pipeline stages."""
        self._model_info = module_manager.get_model_info()
        self.config = config
        self.scheduler = module_manager.fetch_module("scheduler")
        self.vae_stage = VAEStage("vae", module_manager, config.vae_config)
        self.denoise_stage = DitDenoisingStage("denoise", module_manager, config.dit_config, self.scheduler)
        self.text_encoding_stage = TextEncodingStage("text_encoding", module_manager, config.text_encoding_config)

        # Auto-enable metrics if configured
        if config.enable_metrics:
            self.enable_metrics()

    def vae_output_to_images(self, vae_output: torch.Tensor) -> List[Image.Image]:
        """Convert VAE output tensor to PIL Images."""
        images = vae_output.cpu().float().permute(0, 2, 3, 1).numpy()
        images = [Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8")) for image in images]
        return images

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | List[str],
        negative_prompt: str = "",
        seed: int | None = None,
        rand_device: str | None = None,
        height: int = 1328,
        width: int = 1328,
        cfg_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        tiled: bool = False,
    ) -> List[Image.Image]:
        """Generate images from text prompt."""
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        batch_size *= num_images_per_prompt
        logger.info(f"start genereate {num_images_per_prompt}  {width}x{height} image")
        height, width = self.check_resize_height_width(height, width)
        # Initialize noise
        if rand_device is None:
            rand_device = self.device
        noise = self.generate_noise(
            (batch_size, 16, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=self.torch_dtype,
        )
        latents = noise.to(self.device, dtype=self.torch_dtype)
        # Encode prompts
        if cfg_scale <= 1:
            negative_prompt = None
        prompt_emb_list_handler = auto_async_call(
            self.text_encoding_stage.process, prompt, negative_prompt, num_images_per_prompt=num_images_per_prompt
        )
        prompt_embeds, negative_prompt_embeds = prompt_emb_list_handler()
        # denoise
        latents_handler = auto_async_call(
            self.denoise_stage.process,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            cfg_scale=cfg_scale,
            cfg_truncation=cfg_truncation,
            cfg_normalization=cfg_normalization,
            num_inference_steps=num_inference_steps,
        )
        latents = latents_handler()
        # Decode
        image_handler = auto_async_call(self.vae_stage.process, "decode", latents, tiled=tiled)
        image = image_handler()
        image = self.vae_output_to_images(image)
        return image

    def __del__(self):
        del self.vae_stage
        del self.denoise_stage
        del self.text_encoding_stage
