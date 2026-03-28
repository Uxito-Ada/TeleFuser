# SPDX-License-Identifier: Apache-2.0
"""Real-ESRGAN upscaling stage for image super-resolution."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.models.realesrgan import RealESRGAN
from telefuser.utils.profiler import ProfilingContext4Debug


class RealESRGANStage(BaseStage):
    """Image super-resolution stage using Real-ESRGAN.

    Upscales images using Real-ESRGAN model, supporting both SRVGGNetCompact
    (lightweight) and RRDBNet (heavier, higher quality) architectures.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.upscaler_model: RealESRGAN = module_manager.fetch_module("upscaler_model")  # type: ignore
        self.model_names = ["upscaler_model"]

    @with_model_offload(["upscaler_model"])
    @ProfilingContext4Debug("realesrgan_upscale")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        input_images: List[Image.Image],
    ) -> List[Image.Image]:
        """Upscale a list of PIL images.

        Args:
            input_images: List of PIL Image objects to upscale.

        Returns:
            List of upscaled PIL Image objects.
        """
        if not input_images:
            return input_images

        # Convert PIL images to tensor [N, H, W, C] in range [0, 1]
        src_tensor_list = [
            torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0 for image in input_images
        ]
        src_tensor = torch.concat(src_tensor_list, dim=0)

        # Upscale frames
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            result_tensor = self.upscaler_model.upscale_frames(src_tensor, device=self.device.type)

        # Convert back to PIL images
        frames = ((result_tensor.float()) * 255).clip(0, 255).numpy().astype(np.uint8)
        result_images = [Image.fromarray(frame) for frame in frames]
        return result_images

    @with_model_offload(["upscaler_model"])
    @ProfilingContext4Debug("realesrgan_upscale_tensor")
    @torch.inference_mode()
    def process_tensor(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Upscale a tensor of images.

        Args:
            input_tensor: Input tensor [N, H, W, C] in range [0, 1].

        Returns:
            Upscaled tensor [N, H*scale, W*scale, C] in range [0, 1].
        """
        if input_tensor.numel() == 0:
            return input_tensor

        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            result_tensor = self.upscaler_model.upscale_frames(input_tensor, device=self.device.type)

        return result_tensor
