from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers.image_processor import VaeImageProcessor

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.utils.func import auto_async_call
from telefuser.utils.logging import logger
from telefuser.worker.parallel_worker import ParallelWorker

from .dit_denoising import DitDenoisingStage
from .text_encoding import TextEncodingStage
from .vae import VAEStage


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    """Calculate width and height from target area and aspect ratio."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


# Target sizes for condition image and VAE processing
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


@dataclass
class QwenImageEditPipelineConfig:
    """Configuration for Qwen-Image editing pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    is_edit_plus: bool = False
    enable_metrics: bool = False


class QwenImageEditPipeline(BasePipeline):
    """Qwen-Image image-to-image editing pipeline.

    Supports both standard editing and EditPlus mode (is_edit_plus=True)
    for more precise control over the editing process.
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=16)

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        return [self.vae_stage, self.denoise_stage, self.text_encoding_stage]

    def init(self, module_manager: ModuleManager, config: QwenImageEditPipelineConfig):
        """Initialize pipeline stages."""
        self._model_info = module_manager.get_model_info()
        self.config = config
        if config.sample_solver == "euler":
            self.scheduler = FlowMatchScheduler("Qwen-Image")
        else:
            raise NotImplementedError(f"solver {config.sample_solver} is not supported")
        self.vae_stage = VAEStage("vae", module_manager, config.vae_config)
        self.denoise_stage = DitDenoisingStage("denoise", module_manager, config.dit_config, self.scheduler)
        self.text_encoding_stage = TextEncodingStage("text_encoding", module_manager, config.text_encoding_config)
        if config.enable_denoising_parallel:
            self.denoise_stage = ParallelWorker(self.denoise_stage)
        if config.enable_vae_parallel:
            self.vae_stage = ParallelWorker(self.vae_stage)

        # Auto-enable metrics if configured
        if config.enable_metrics:
            self.enable_metrics()

    def vae_output_to_images(self, vae_output: torch.Tensor) -> List[Image.Image]:
        """Convert VAE output tensor to PIL Images."""
        images = vae_output.cpu().float().permute(0, 2, 3, 1).numpy()
        images = [Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8")) for image in images]
        return images

    @staticmethod
    def preprocess_image(image: Image.Image, mode: str = "RGB") -> torch.Tensor:
        """Preprocess PIL image to normalized tensor."""
        image = image.convert(mode)
        image_array = np.array(image, dtype=np.float32)
        if len(image_array.shape) == 2:
            image_array = image_array[:, :, np.newaxis]
        image = torch.Tensor((image_array / 255) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | List[str],
        negative_prompt: str = "",
        image: Image.Image | List[Image.Image] | None = None,
        seed: int | None = None,
        rand_device: str | None = None,
        height: int | None = None,
        width: int | None = None,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 30,
        exponential_shift_mu: float | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        denoising_strength: float = 1.0,
        shift_terminal: float | None = 0.02,
        num_images_per_prompt: int = 1,
    ) -> List[Image.Image]:
        """Edit input image based on text prompt."""
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        batch_size *= num_images_per_prompt
        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        if rand_device is None:
            rand_device = self.device
        if cfg_scale <= 1:
            negative_prompt = None
        if image is None:
            raise RuntimeError("image is None in qwen edie model")
        if batch_size > 1:
            raise RuntimeError(f"image edit only support batch size=1 not {batch_size} ")
        if not isinstance(image, list):
            image = [image]
        # Resize input images to standard sizes for processing
        condition_images = []
        vae_images = []
        for img in image:
            image_width, image_height = img.size
            condition_width, condition_height = calculate_dimensions(CONDITION_IMAGE_SIZE, image_width / image_height)
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
            condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
            vae_images.append(self.image_processor.resize(img, vae_height, vae_width))
            if width is None or height is None:
                width, height = vae_width, vae_height
        logger.info(f"generate image with shape width: {width}, height: {height}")
        noise = self.generate_noise(
            (batch_size, 16, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=self.torch_dtype,
        )
        latents = noise.to(self.device, dtype=self.torch_dtype)
        edit_latents_hanlder = auto_async_call(self.vae_stage.process, "encode", vae_images)
        # EditPlus uses different conditioning images
        if self.config.is_edit_plus:
            edit_images = condition_images
        else:
            edit_images = vae_images
        prompt_emb_list_handler = auto_async_call(
            self.text_encoding_stage.process,
            prompt,
            negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            edit_image=edit_images,
            is_edit_plus=self.config.is_edit_plus,
        )
        edit_latents = edit_latents_hanlder()
        prompt_emb_posi, prompt_emb_mask_posi, prompt_emb_nega, prompt_emb_mask_nega = prompt_emb_list_handler()
        # denoise
        latents_handler = auto_async_call(
            self.denoise_stage.process,
            latents,
            prompt_emb_posi=prompt_emb_posi,
            prompt_emb_mask_posi=prompt_emb_mask_posi,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            prompt_emb_nega=prompt_emb_nega,
            prompt_emb_mask_nega=prompt_emb_mask_nega,
            denoising_strength=denoising_strength,
            exponential_shift_mu=exponential_shift_mu,
            shift_terminal=shift_terminal,
            edit_latents=edit_latents,
        )
        latents = latents_handler()
        # Decode
        image_handler = auto_async_call(self.vae_stage.process, "decode", latents, **tiler_kwargs)
        image = image_handler()
        image = self.vae_output_to_images(image)
        return image

    def __del__(self):
        del self.vae_stage
        del self.denoise_stage
        del self.text_encoding_stage
