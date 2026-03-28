from __future__ import annotations

from typing import Any

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics


class VAEStage(BaseStage):
    """VAE decoding stage for Z-Image using AutoencoderKL.

    Handles latent-to-image decoding with support for tiled processing
    and proper scaling factor/shift factor application.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.vae: AutoencoderKL = module_manager.fetch_module("z_image_vae")
        self.model_names = ["vae"]

    def decode(self, latents: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        """Decode latents to images with scaling factor and shift."""
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        latents = latents.to(self.device)
        if tiled:
            self.vae.enable_tiling()
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            images = self.vae.decode(latents, return_dict=False)[0]
        return images

    @with_model_offload(["vae"])
    @torch.inference_mode()
    @with_metrics
    def process(self, method: str, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Dispatch to specified VAE method."""
        if hasattr(self, method):
            return getattr(self, method)(*args, **kwargs)
        else:
            raise NotImplementedError(f"{method} is not supported in z_image vae")
