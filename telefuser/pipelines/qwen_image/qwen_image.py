from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.utils.func import auto_async_call
from telefuser.utils.hf_model_utils import (
    discover_model_components,
    resolve_hf_path,
)
from telefuser.utils.logging import logger
from telefuser.worker.parallel_worker import ParallelWorker

from .dit_denoising import DitDenoisingStage
from .text_encoding import TextEncodingStage
from .vae import VAEStage

# Aspect ratio to resolution mapping for Qwen-Image
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
class QwenImagePipelineConfig:
    """Configuration for Qwen-Image generation pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_text_encoding_parallel: bool = False
    enable_metrics: bool = False


class QwenImagePipeline(BasePipeline):
    """Qwen-Image text-to-image generation pipeline.

    Supports text-to-image generation with configurable resolution,
    classifier-free guidance, and tiled VAE decoding for large images.
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        return [self.vae_stage, self.denoise_stage, self.text_encoding_stage]

    def init(self, module_manager: ModuleManager, config: QwenImagePipelineConfig):
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
        if config.enable_text_encoding_parallel:
            self.text_encoding_stage = ParallelWorker(self.text_encoding_stage)

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
        num_inference_steps: int = 30,
        exponential_shift_mu: float | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        denoising_strength: float = 1.0,
        shift_terminal: float | None = 0.02,
        num_images_per_prompt: int = 1,
    ) -> List[Image.Image]:
        """Generate images from text prompt.

        Args:
            prompt: Text prompt or list of prompts.
            negative_prompt: Negative prompt for CFG.
            seed: Random seed.
            rand_device: Device for random number generation.
            height: Image height.
            width: Image width.
            cfg_scale: CFG scale.
            num_inference_steps: Number of inference steps.
            exponential_shift_mu: Exponential shift mu for scheduler.
            tiled: Whether to use tiled processing.
            tile_size: Tile size for tiled processing.
            tile_stride: Tile stride for tiled processing.
            denoising_strength: Denoising strength.
            shift_terminal: Shift terminal for scheduler.
            num_images_per_prompt: Number of images per prompt.

        Returns:
            List of generated images.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        batch_size *= num_images_per_prompt
        logger.info(f"start genereate  {width}x{height} image")
        height, width = self.check_resize_height_width(height, width)
        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

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

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: str | None = None,
        attention_config: AttentionConfig | None = None,
        enable_parallel: bool = False,
        parallel_devices: list[int] | None = None,
        sample_solver: str = "euler",
        enable_metrics: bool = False,
        **kwargs,
    ) -> "QwenImagePipeline":
        """Load a Qwen Image pipeline from HuggingFace model ID or local path.

        This method supports loading from:
        - HuggingFace model ID (e.g., "Qwen/Qwen-Image-2512")
        - Local path in HuggingFace Diffusers format

        The model folder should contain:
        - transformer/diffusion_pytorch_model*.safetensors (DiT weights, sharded)
        - vae/diffusion_pytorch_model.safetensors (VAE weights)
        - text_encoder/model*.safetensors (Text encoder weights, sharded)
        - tokenizer/ (Tokenizer folder)

        Example:
            >>> pipe = QwenImagePipeline.from_pretrained(
            ...     "Qwen/Qwen-Image-2512",
            ...     device="cuda",
            ...     torch_dtype=torch.bfloat16,
            ... )
            >>> images = pipe(
            ...     prompt="A beautiful sunset",
            ...     height=1024,
            ...     width=1024,
            ... )
        """
        # Resolve model path (download if needed)
        model_root = resolve_hf_path(model_id_or_path, cache_dir)
        logger.info(f"Loading Qwen Image pipeline from: {model_root}")

        # Discover model components
        components = discover_model_components(model_root)

        # Get required component paths
        transformer_path = components.get("transformer")
        vae_path = components.get("vae")
        text_encoder_path = components.get("text_encoder")
        tokenizer_path = components.get("tokenizer")

        if not transformer_path:
            raise ValueError(f"No transformer/DiT weights found in {model_root}")
        if not vae_path:
            raise ValueError(f"No VAE weights found in {model_root}")
        if not text_encoder_path:
            raise ValueError(f"No text encoder weights found in {model_root}")

        logger.info(f"  Transformer: {transformer_path}")
        logger.info(f"  VAE: {vae_path}")
        logger.info(f"  Text Encoder: {text_encoder_path}")

        # Load models using ModuleManager
        mm = ModuleManager(torch_dtype=torch_dtype, device="cpu")
        mm.load_model(transformer_path, device="cpu", torch_dtype=torch_dtype)
        mm.load_model(vae_path, device="cpu", torch_dtype=torch_dtype)
        mm.load_model(text_encoder_path, device="cpu", torch_dtype=torch_dtype)

        # Load tokenizer if available
        if tokenizer_path:
            mm.load_from_huggingface(
                tokenizer_path,
                module_source="transformers",
                module_name="tokenizer",
                torch_dtype=torch_dtype,
            )

        # Create pipeline
        pipeline = cls(device=device, torch_dtype=torch_dtype)

        # Create config
        config = QwenImagePipelineConfig()
        config.sample_solver = sample_solver
        if attention_config is None:
            attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
        config.dit_config.attention_config = attention_config

        # Configure parallelism
        if enable_parallel and parallel_devices:
            config.dit_config.parallel_config.device_ids = parallel_devices
            config.dit_config.parallel_config.cfg_degree = len(parallel_devices)
            config.enable_denoising_parallel = True
            config.text_encoding_config.parallel_config.device_ids = parallel_devices
            config.text_encoding_config.parallel_config.tp_degree = len(parallel_devices)
            config.enable_text_encoding_parallel = True

        # Configure metrics
        config.enable_metrics = enable_metrics

        # Apply additional config kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Initialize pipeline
        pipeline.init(mm, config)

        logger.info(f"Successfully loaded Qwen Image pipeline from {model_id_or_path}")
        return pipeline
