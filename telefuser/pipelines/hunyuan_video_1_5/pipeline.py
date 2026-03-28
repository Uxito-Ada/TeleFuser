"""HunyuanVideo pipeline for text-to-video and image-to-video generation."""

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

from .dit_denoising import HunyuanVideoDenoisingStage
from .image_encoding import HunyuanVideoImageEncodingStage
from .sr_dit_denoising import HunyuanVideoSRDenoisingStage
from .text_encoding import HunyuanVideoTextEncodingStage
from .upsampler import HunyuanVideoUpsamplerStage
from .vae import HunyuanVideoVAEStage


@dataclass
class HunyuanVideo15PipelineConfig:
    """Configuration for HunyuanVideo generation pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    image_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sr_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"
    enable_image_encoding: bool = False
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    # SR specific configs
    enable_sr: bool = False
    lq_noise_strength: float = 0.7
    sr_sigma_shift: float = 2.0  # SR uses different shift than base (7.0)


class HunyuanVideo15Pipeline(BasePipeline):
    """HunyuanVideo pipeline for T2V, I2V, and SR generation.

    Supports:
    - Text-to-video (T2V) generation
    - Image-to-video (I2V) generation with vision encoder conditioning
    - Super-resolution (SR) generation: 720p -> 1080p upscaling
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.base_fps = 24

        # VAE compression factors
        self.vae_spatial_compression_ratio = 16
        self.vae_temporal_compression_ratio = 4
        self.latent_channels = 32

    def init(self, module_manager: ModuleManager, config: HunyuanVideo15PipelineConfig):
        """Initialize pipeline stages."""
        self._model_info = module_manager.get_model_info()
        self.config = config

        # VAE stage
        self.vae_stage = HunyuanVideoVAEStage("vae", module_manager, config.vae_config)

        # Scheduler
        self.scheduler = module_manager.fetch_module("scheduler")

        # Base denoising stage
        self.denoise_stage = HunyuanVideoDenoisingStage("denoise", module_manager, config.dit_config, self.scheduler)

        # Text encoding stage
        self.text_encoding_stage = HunyuanVideoTextEncodingStage(
            "text_encoding", module_manager, config.text_encoding_config
        )

        # Image encoding stage (for I2V)
        if config.enable_image_encoding:
            self.image_encoding_stage = HunyuanVideoImageEncodingStage(
                "image_encoding", module_manager, config.image_encoding_config
            )
        else:
            self.image_encoding_stage = None

        # SR stages (if enabled)
        if config.enable_sr:
            self.upsampler_stage = HunyuanVideoUpsamplerStage("upsampler", module_manager, config.sr_config)
            self.sr_denoise_stage = HunyuanVideoSRDenoisingStage(
                "sr_denoise",
                module_manager,
                config.sr_config,
                self.scheduler,
                lq_noise_strength=config.lq_noise_strength,
                sigma_shift=config.sr_sigma_shift,
            )
            logger.info("SR mode enabled: base_denoise -> upsampler -> sr_denoise")
        else:
            self.upsampler_stage = None
            self.sr_denoise_stage = None

        # Parallel workers
        if config.enable_denoising_parallel:
            self.denoise_stage = ParallelWorker(self.denoise_stage)
            if self.sr_denoise_stage is not None:
                self.sr_denoise_stage = ParallelWorker(self.sr_denoise_stage)
        if config.enable_vae_parallel:
            self.vae_stage = ParallelWorker(self.vae_stage)

        logger.info("HunyuanVideoPipeline initialized")

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | List[str],
        negative_prompt: str = "",
        input_image: Image.Image | None = None,
        seed: int | None = None,
        rand_device: str | None = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 121,
        cfg_scale: float = 6.0,
        num_inference_steps: int = 50,
        embedded_guidance_scale: float | None = None,
        sr_num_inference_steps: int = 6,
    ) -> List[Image.Image]:
        """Generate video from text prompt and optional input image."""
        task_type = "i2v" if input_image is not None else "t2v"
        is_sr_mode = self.config.enable_sr

        logger.info(
            f"Starting generation: {num_frames} frames, {width}x{height}, "
            f"mode={task_type.upper()}{' + SR' if is_sr_mode else ''}"
        )

        # Check and adjust dimensions
        height, width = self.check_resize_height_width(height, width)

        # Initialize noise device
        if rand_device is None:
            rand_device = self.device

        # Calculate latent dimensions
        latent_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        # Calculate base resolution for SR mode
        if is_sr_mode:
            base_height = int(height * 2 / 3)
            base_width = int(width * 2 / 3)
            base_height = (base_height // 16) * 16
            base_width = (base_width // 16) * 16
            base_latent_height = base_height // self.vae_spatial_compression_ratio
            base_latent_width = base_width // self.vae_spatial_compression_ratio
        else:
            base_height, base_width = height, width
            base_latent_height, base_latent_width = latent_height, latent_width

        # Encode prompts
        prompt_emb_handler = auto_async_call(
            self.text_encoding_stage.process,
            prompt=prompt,
            negative_prompt=negative_prompt if cfg_scale > 1.0 else "",
            data_type="video",
            cfg_scale=cfg_scale,
        )

        # I2V: Encode image with vision encoder
        vision_states = None
        image_latents = None
        if input_image is not None and self.image_encoding_stage is not None:
            # Vision encoder expects uint8 numpy array and does resize/center_crop internally
            # Matches original: resize_and_center_crop -> encode_images
            vision_handler = auto_async_call(
                self.image_encoding_stage.process,
                image=input_image,
                target_width=width,
                target_height=height,
            )
            vision_states = vision_handler()

            # VAE encode: use _preprocess_image_for_vae for correct preprocessing
            # Matches original: Resize + CenterCrop + ToTensor + Normalize
            image_tensor = self._preprocess_image_for_vae(input_image, base_height, base_width)

        # Get prompt embeddings
        prompt_emb_result = prompt_emb_handler()
        prompt_emb_posi = prompt_emb_result["prompt_emb_posi"]
        prompt_emb_nega = prompt_emb_result.get("prompt_emb_nega")
        attention_mask = prompt_emb_result.get("attention_mask")
        nega_attention_mask = prompt_emb_result.get("nega_attention_mask")
        byt5_text_states = prompt_emb_result.get("byt5_text_states")
        byt5_text_mask = prompt_emb_result.get("byt5_text_mask")
        byt5_text_states_nega = prompt_emb_result.get("byt5_text_states_nega")
        byt5_text_mask_nega = prompt_emb_result.get("byt5_text_mask_nega")

        # Generate initial noise
        noise = self.generate_noise(
            (1, self.latent_channels, latent_frames, base_latent_height, base_latent_width),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        latents = noise.to(dtype=self.torch_dtype, device=self.device)

        # I2V: Encode image latents
        if input_image is not None and self.image_encoding_stage is not None:
            # image_tensor is already preprocessed from _preprocess_image_for_vae
            image_latents_handler = auto_async_call(self.vae_stage.process, "encode_image", image_tensor)
            image_latents = image_latents_handler()

        # Enable sparse attention if configured
        attention_config = self.config.dit_config.attention_config
        if attention_config and attention_config.is_sparse():
            sparse_config = attention_config.sparse_config
            logger.info(f"Sparse attention enabled: dense_layers={sparse_config.dense_layers}")

        # Base denoising
        latents_handler = auto_async_call(
            self.denoise_stage.process,
            latents=latents,
            num_inference_steps=num_inference_steps,
            prompt_emb_posi=prompt_emb_posi,
            prompt_emb_nega=prompt_emb_nega,
            cfg_scale=cfg_scale,
            image_latents=image_latents,
            vision_states=vision_states,
            attention_mask=attention_mask,
            nega_attention_mask=nega_attention_mask,
            task_type=task_type,
            byt5_text_states=byt5_text_states,
            byt5_text_mask=byt5_text_mask,
            byt5_text_states_nega=byt5_text_states_nega,
            byt5_text_mask_nega=byt5_text_mask_nega,
        )
        latents = latents_handler()
        logger.info(f"Base generation complete: {latents.shape}")

        # SR stages
        if is_sr_mode:
            # Upsample
            upsample_handler = auto_async_call(
                self.upsampler_stage.process,
                lq_latents=latents,
                target_shape=(latent_height, latent_width),
            )
            upsampled_latents = upsample_handler()
            logger.info(f"Upsampling complete: {upsampled_latents.shape}")

            # SR denoising
            noise = self.generate_noise(
                (1, self.latent_channels, latent_frames, latent_height, latent_width),
                seed=seed,
                device=rand_device,
                dtype=torch.float32,
            )
            target_latents = noise.to(dtype=self.torch_dtype, device=self.device)

            sr_latents_handler = auto_async_call(
                self.sr_denoise_stage.process,
                latents=target_latents,
                lq_latents=upsampled_latents,
                num_inference_steps=sr_num_inference_steps,
                prompt_emb_posi=prompt_emb_posi,
                prompt_emb_nega=prompt_emb_nega,
                attention_mask=attention_mask,
                cfg_scale=1.0,  # Distilled model doesn't use CFG
                embedded_guidance_scale=embedded_guidance_scale,
                image_cond=image_latents,
                vision_states=vision_states,
                task_type=task_type,
                byt5_text_states=byt5_text_states,
                byt5_text_mask=byt5_text_mask,
                byt5_text_states_nega=byt5_text_states_nega,
                byt5_text_mask_nega=byt5_text_mask_nega,
            )
            latents = sr_latents_handler()
            logger.info(f"SR denoising complete: {latents.shape}")

        # VAE decode
        frames_handler = auto_async_call(self.vae_stage.process, "decode_video", latents)
        frames = frames_handler()
        frames = self.tensor2video(frames[0])

        return frames

    def _preprocess_image_for_vae(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        """Preprocess image for VAE encoding.

        Matches original HunyuanVideo preprocessing:
        1. Resize to cover target size (scale factor = max(target_w/original_w, target_h/original_h))
        2. Center crop to target size
        3. ToTensor + Normalize([0.5], [0.5])
        4. Add batch and time dimensions -> (B, C, 1, H, W)
        """
        import torchvision.transforms as transforms

        original_width, original_height = image.size

        # Calculate scale factor to cover target size
        scale_factor = max(width / original_width, height / original_height)
        resize_width = int(round(original_width * scale_factor))
        resize_height = int(round(original_height * scale_factor))

        transform = transforms.Compose(
            [
                transforms.Resize((resize_height, resize_width), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # (C, H, W) -> (B, C, 1, H, W)
        image_tensor = transform(image).unsqueeze(0).unsqueeze(2)
        return image_tensor.to(device=self.device, dtype=self.torch_dtype)
