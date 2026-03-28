"""Upsampler stage for HunyuanVideo SR (Super-Resolution).

This stage works with Upsampler models from HunyuanVideo repository:
    from hyvideo.models.transformers.modules.upsample import SRTo720pUpsampler, SRTo1080pUpsampler

The Upsampler is used to enhance low-resolution latents before SR DiT denoising.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.utils.logging import logger


class HunyuanVideoUpsamplerStage(BaseStage):
    """Upsampler stage for HunyuanVideo SR using models from HunyuanVideo repository.

    This stage wraps the Upsampler from hyvideo.models.transformers.modules.upsample.
    Supports:
    - SRTo720pUpsampler: 480p -> 720p upscaling
    - SRTo1080pUpsampler: 720p -> 1080p upscaling

    The Upsampler performs feature enhancement on bilinear-interpolated latents.
    It's a lightweight model that runs on a single GPU without parallelization.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.upsampler = module_manager.fetch_module("upsampler")
        self.model_names = ["upsampler"]

    @with_model_offload(["upsampler"])
    @torch.inference_mode()
    def process(
        self,
        lq_latents: torch.Tensor,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Process low-quality latents through the upsampler.

        This method performs:
        1. Bilinear interpolation to target shape (if needed)
        2. Feature enhancement through the upsampler model

        Args:
            lq_latents: Low-quality latent tensor [B, C, F, H, W]
            target_shape: Target spatial shape (H, W). If None, no interpolation.

        Returns:
            Enhanced latent tensor [B, C, F, H', W']
        """
        # Move to device
        lq_latents = lq_latents.to(device=self.device)

        # Step 1: Bilinear interpolation to target shape
        if target_shape is not None and lq_latents.shape[-2:] != target_shape:
            bsz = lq_latents.shape[0]
            lq_latents = rearrange(lq_latents, "b c f h w -> (b f) c h w")
            lq_latents = F.interpolate(
                lq_latents,
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
            lq_latents = rearrange(lq_latents, "(b f) c h w -> b c f h w", b=bsz)
            logger.debug(f"Interpolated latents to target shape: {target_shape}")

        # Step 2: Pass through upsampler for feature enhancement
        # Upsampler expects float32 input for precision
        lq_latents = self.upsampler(lq_latents.to(dtype=torch.float32))

        logger.debug(f"Upsampler output shape: {lq_latents.shape}")

        return lq_latents

    def forward(
        self,
        lq_latents: torch.Tensor,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Alias for process method for compatibility."""
        return self.process(lq_latents, target_shape)
