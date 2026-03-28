from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.models.qwen_image_vae import QwenImageVAE


class VAEStage(BaseStage):
    """VAE encoding/decoding stage for Qwen-Image.

    Handles image-to-latent encoding and latent-to-image decoding
    with support for tiled processing of large images.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.vae: QwenImageVAE = module_manager.fetch_module("qwen_image_vae")
        self.model_names = ["vae"]

    @staticmethod
    def preprocess_image(image: Image.Image, mode: str = "RGB") -> torch.Tensor:
        """Preprocess PIL image to normalized tensor."""
        image = image.convert(mode)
        image_array = np.array(image, dtype=np.float32)
        if len(image_array.shape) == 2:
            image_array = image_array[:, :, np.newaxis]
        image = torch.Tensor((image_array / 255) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
        return image

    def encode(
        self,
        images: List[Image.Image],
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
    ) -> List[torch.Tensor]:
        """Encode images to latent representations."""
        latents = []
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            for image in images:
                image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)
                latent = self.vae.encode(
                    image,
                    device=self.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                latents.append(latent)
        return latents

    def decode(
        self,
        latents: torch.Tensor,
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
    ) -> torch.Tensor:
        """Decode latent representations to images."""
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            images = self.vae.decode(
                latents,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        return images

    @with_model_offload(["vae"])
    @torch.inference_mode()
    @with_metrics
    def process(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to encode or decode method."""
        if hasattr(self, method):
            return getattr(self, method)(*args, **kwargs)
        else:
            raise NotImplementedError(f"{method} is not supported in qwen image vae")
