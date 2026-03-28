"""Flux2 Klein text-to-image generation pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import torch
from PIL import Image
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
from transformers import Qwen3ForCausalLM

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


@dataclass
class Flux2KleinPipelineConfig:
    """Configuration for Flux2 Klein generation pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_text_encoding_parallel: bool = False
    enable_metrics: bool = False


class Flux2KleinPipeline(BasePipeline):
    """Flux2 Klein text-to-image generation pipeline.

    Supports text-to-image generation with configurable resolution,
    classifier-free guidance, and reference image conditioning.

    Reference: https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        return [self.vae_stage, self.denoise_stage, self.text_encoding_stage]

    def init(self, module_manager: ModuleManager, config: Flux2KleinPipelineConfig):
        """Initialize pipeline stages.

        Args:
            module_manager: Module manager with loaded models
            config: Pipeline configuration
        """
        self._model_info = module_manager.get_model_info()
        self.config = config
        if config.sample_solver == "euler":
            self.scheduler = FlowMatchScheduler("FLUX.2")
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
        """Convert VAE output tensor to PIL Images.

        Args:
            vae_output: Image tensor of shape (B, 3, H, W)

        Returns:
            List of PIL Images
        """
        images = vae_output.cpu().float().permute(0, 2, 3, 1).numpy()
        images = [Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8")) for image in images]
        return images

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
        """Generate 4D position coordinates (T, H, W, L) for latent tokens.

        Args:
            latents: Latent tensor of shape (B, C, H, W)

        Returns:
            Position IDs of shape (B, H*W, 4)
        """
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0]
        h = torch.arange(height)
        w = torch.arange(width)
        layer_dim = torch.arange(1)  # [0]

        latent_ids = torch.cartesian_prod(t, h, w, layer_dim)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pack latents from (B, C, H, W) to (B, H*W, C).

        Args:
            latents: Tensor of shape (B, C, H, W)

        Returns:
            Packed tensor of shape (B, H*W, C)
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | List[str],
        negative_prompt: str | None = "",
        seed: int | None = None,
        rand_device: str | None = None,
        height: int = 1024,
        width: int = 1024,
        cfg_scale: float = 4.0,
        num_inference_steps: int = 50,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        hidden_states_layers: tuple[int, ...] = (9, 18, 27),
        reference_images: List[Image.Image] | None = None,
    ) -> List[Image.Image]:
        """Generate images from text prompt.

        Args:
            prompt: Text prompt or list of prompts.
            negative_prompt: Negative prompt for CFG.
            seed: Random seed.
            rand_device: Device for random number generation.
            height: Image height (divisible by 16).
            width: Image width (divisible by 16).
            cfg_scale: CFG scale.
            num_inference_steps: Number of inference steps.
            num_images_per_prompt: Number of images per prompt.
            max_sequence_length: Maximum sequence length for text encoding.
            hidden_states_layers: Qwen3 layer indices to extract.
            reference_images: Optional reference images for img2img.

        Returns:
            List of generated images.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        batch_size *= num_images_per_prompt
        logger.info(f"Generating {width}x{height} image")

        # Check and adjust dimensions
        height, width = self.check_resize_height_width(height, width)

        # VAE scale factor for Flux2: 8 (VAE compression) * 2 (patchification) = 16
        # The latent shape formula: (B, 128, H//16, W//16)
        # where 128 = 32 (VAE channels) * 4 (patchification)
        vae_scale_factor = 16

        # Calculate latent dimensions
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        latent_channels = 128  # 32 * 4 (patchified)

        # Initialize random device
        if rand_device is None:
            rand_device = str(self.device)

        # Generate noise
        noise = self.generate_noise(
            (batch_size, latent_channels, latent_height, latent_width),
            seed=seed,
            device=rand_device,
            dtype=self.torch_dtype,
        )
        latents = self._pack_latents(noise.to(self.device, dtype=self.torch_dtype))
        latent_ids = self._prepare_latent_ids(noise).to(self.device)

        # Handle CFG
        if cfg_scale <= 1:
            negative_prompt = None

        # Encode prompts asynchronously
        prompt_emb_handler = auto_async_call(
            self.text_encoding_stage.process,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            hidden_states_layers=hidden_states_layers,
        )
        prompt_embeds, text_ids, neg_prompt_embeds, neg_text_ids = prompt_emb_handler()

        # Process reference images if provided
        image_latents = None
        image_latent_ids = None
        if reference_images is not None:
            # Encode reference images
            ref_latent_list = []
            for ref_img in reference_images:
                ref_tensor = self.vae_stage.preprocess_image(ref_img).to(self.device, dtype=self.torch_dtype)
                ref_latent = self.vae_stage.encode_image(ref_tensor)
                ref_latent_list.append(ref_latent)

            # Concatenate reference latents
            if len(ref_latent_list) > 0:
                # Each ref_latent is (1, seq, 128), need to concatenate along seq dim
                image_latents = torch.cat([ref_lat.squeeze(0) for ref_lat in ref_latent_list], dim=0).unsqueeze(0)
                image_latents = image_latents.repeat(batch_size, 1, 1)
                # Generate position IDs for reference images
                image_latent_ids = self.denoise_stage._prepare_image_ids(
                    [ref_lat.squeeze(0).reshape(-1, 128, latent_height, latent_width) for ref_lat in ref_latent_list]
                )
                # Expand position IDs to match batch size
                image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
                image_latent_ids = image_latent_ids.to(self.device)

        # Denoising
        latents_handler = auto_async_call(
            self.denoise_stage.process,
            latents=latents,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            latent_ids=latent_ids,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_text_ids=neg_text_ids,
            image_latents=image_latents,
            image_latent_ids=image_latent_ids,
        )
        latents = latents_handler()

        # Decode
        image_handler = auto_async_call(
            self.vae_stage.process,
            "decode_latents",
            latents=latents,
            latent_ids=latent_ids,
        )
        image = image_handler()
        image = self.vae_output_to_images(image)

        return image

    def __del__(self):
        if hasattr(self, "vae_stage"):
            del self.vae_stage
        if hasattr(self, "denoise_stage"):
            del self.denoise_stage
        if hasattr(self, "text_encoding_stage"):
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
    ) -> "Flux2KleinPipeline":
        """Load a Flux2 Klein pipeline from HuggingFace model ID or local path.

        This method supports loading from:
        - HuggingFace model ID (e.g., "black-forest-labs/FLUX.2-klein-base-9B")
        - Local path in HuggingFace Diffusers format

        The model folder should contain:
        - transformer/diffusion_pytorch_model*.safetensors (DiT weights, sharded)
        - vae/diffusion_pytorch_model.safetensors (VAE weights)
        - text_encoder/model*.safetensors (Qwen3 text encoder weights)
        - tokenizer/ (Tokenizer folder)

        Example:
            >>> pipe = Flux2KleinPipeline.from_pretrained(
            ...     "black-forest-labs/FLUX.2-klein-base-9B",
            ...     device="cuda",
            ...     torch_dtype=torch.bfloat16,
            ... )
            >>> images = pipe(
            ...     prompt="A cat holding a sign that says hello world",
            ...     height=1024,
            ...     width=1024,
            ... )

        Args:
            model_id_or_path: HuggingFace model ID or local path
            device: Device to run on
            torch_dtype: Data type for models
            cache_dir: Cache directory for downloads
            attention_config: Attention configuration
            enable_parallel: Enable parallel processing
            parallel_devices: List of device IDs for parallelism
            sample_solver: Sampling solver ("euler")
            enable_metrics: Enable metrics collection
            **kwargs: Additional configuration options

        Returns:
            Initialized Flux2KleinPipeline
        """
        # Resolve model path (download if needed)
        model_root = resolve_hf_path(model_id_or_path, cache_dir)
        logger.info(f"Loading Flux2 Klein pipeline from: {model_root}")
        transformer_path = os.path.join(model_root, "transformer")
        vae_path = os.path.join(model_root, "vae")
        text_encoder_path = os.path.join(model_root, "text_encoder")
        tokenizer_path = os.path.join(model_root, "tokenizer")
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

        # Load DiT using hash-based auto-detection (requires registration in model_config.py)
        transformer_path_list = [
            os.path.join(transformer_path, item)
            for item in os.listdir(transformer_path)
            if item.endswith(".safetensors")
        ]
        mm.load_model(transformer_path_list, torch_dtype=torch.bfloat16)
        # Load VAE from HuggingFace (AutoencoderKLFlux2 has built-in BatchNorm with pretrained running_mean/running_var)
        vae_path_str = vae_path[0] if isinstance(vae_path, list) else vae_path
        mm.load_from_huggingface(
            vae_path_str,
            module_source="diffusers",
            module_class=AutoencoderKLFlux2,
            module_name="vae",
            torch_dtype=torch_dtype,
        )

        # Load TextEncoder from HuggingFace
        text_encoder_path_str = text_encoder_path[0] if isinstance(text_encoder_path, list) else text_encoder_path
        mm.load_from_huggingface(
            text_encoder_path_str,
            module_source="transformers",
            module_class=Qwen3ForCausalLM,
            module_name="text_encoder",
            torch_dtype=torch_dtype,
        )

        # Load tokenizer if available
        if tokenizer_path:
            tokenizer_path_str = tokenizer_path[0] if isinstance(tokenizer_path, list) else tokenizer_path
            mm.load_from_huggingface(
                tokenizer_path_str,
                module_source="transformers",
                module_name="tokenizer",
                torch_dtype=torch_dtype,
            )

        # Create pipeline
        pipeline = cls(device=device, torch_dtype=torch_dtype)

        # Create config
        config = Flux2KleinPipelineConfig()
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

        logger.info(f"Successfully loaded Flux2 Klein pipeline from {model_id_or_path}")
        return pipeline
