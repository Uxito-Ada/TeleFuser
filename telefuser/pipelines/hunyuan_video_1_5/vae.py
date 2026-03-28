"""VAE stage for HunyuanVideo pipeline.

This stage works with AutoencoderKLConv3D from HunyuanVideo repository:
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae import AutoencoderKLConv3D
"""

from __future__ import annotations

from typing import Optional

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.utils.logging import logger


class HunyuanVideoVAEStage(BaseStage):
    """VAE stage for HunyuanVideo using AutoencoderKLConv3D from HunyuanVideo.

    This stage wraps the VAE from hyvideo.models.autoencoders.hunyuanvideo_15_vae.
    Supports video encoding/decoding with optional tiling for memory efficiency.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.vae = module_manager.fetch_module("vae")
        self.model_names = ["vae"]

    @with_model_offload(["vae"])
    @torch.inference_mode()
    def process(self, mode: str, *args, **kwargs) -> torch.Tensor:
        """Process VAE operation based on mode.

        Args:
            mode: "encode_video", "decode_video", or "encode_image"
            *args, **kwargs: Arguments passed to the underlying method

        Returns:
            Result tensor
        """
        if mode == "encode_video":
            return self.encode_video(*args, **kwargs)
        elif mode == "decode_video":
            return self.decode_video(*args, **kwargs)
        elif mode == "encode_image":
            return self.encode_image(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video tensor to latents.

        Args:
            video: Video tensor [B, C, T, H, W] in range [-1, 1]

        Returns:
            Latent tensor
        """
        video = video.to(device=self.device, dtype=self.torch_dtype)

        # Use HunyuanVideo VAE encode
        posterior = self.vae.encode(video).latent_dist
        latents = posterior.sample() * self.vae.scaling_factor

        return latents

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video tensor.

        Args:
            latents: Latent tensor from diffusion process

        Returns:
            Video tensor [B, C, T, H, W]
        """
        latents = latents.to(device=self.device, dtype=self.torch_dtype)

        # Scale latents
        scaling_factor = getattr(self.vae, "scaling_factor", 0.47698)
        latents = latents / scaling_factor

        # Use memory efficient context if available (enables both slicing and tiling)
        # This is the same approach used in the original HunyuanVideo pipeline
        if hasattr(self.vae, "memory_efficient_context"):
            with self.vae.memory_efficient_context():
                logger.debug("Using VAE memory efficient context for decoding")
                decoded = self.vae.decode(latents).sample
        else:
            # Fallback: manually enable tiling and slicing
            original_use_spatial_tiling = getattr(self.vae, "use_spatial_tiling", False)
            original_use_slicing = getattr(self.vae, "use_slicing", False)

            if hasattr(self.vae, "enable_spatial_tiling"):
                self.vae.enable_spatial_tiling()
                logger.debug("Enabled VAE spatial tiling for memory efficient decoding")

            if hasattr(self.vae, "enable_slicing"):
                self.vae.enable_slicing()
                logger.debug("Enabled VAE slicing for memory efficient decoding")

            try:
                decoded = self.vae.decode(latents).sample
            finally:
                if hasattr(self.vae, "use_spatial_tiling"):
                    self.vae.use_spatial_tiling = original_use_spatial_tiling
                if hasattr(self.vae, "use_slicing"):
                    self.vae.use_slicing = original_use_slicing

        return decoded

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode single image to latents for I2V.

        Args:
            image: Image tensor [B, C, T=1, H, W]

        Returns:
            Image latent tensor [B, C, 1, H, W]
        """
        image = image.to(device=self.device, dtype=self.torch_dtype)

        # Encode image
        posterior = self.vae.encode(image).latent_dist
        latents = posterior.mode() * self.vae.scaling_factor

        return latents
