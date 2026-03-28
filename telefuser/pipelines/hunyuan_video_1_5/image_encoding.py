"""Image encoding stage for HunyuanVideo I2V pipeline.

This stage works with HunyuanVideoImageEncoder from Telefuser:
    from telefuser.models.hunyuan_video_image_encoder import HunyuanVideoImageEncoder
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.hunyuan_video_image_encoder import resize_and_center_crop
from telefuser.utils.logging import logger


class HunyuanVideoImageEncodingStage(BaseStage):
    """Image encoding stage for HunyuanVideo I2V using VisionEncoder.

    This stage wraps HunyuanVideoImageEncoder for encoding reference images
    in image-to-video generation.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.vision_encoder = module_manager.fetch_module("vision_encoder")
        self.model_names = ["vision_encoder"]

    @with_model_offload(["vision_encoder"])
    @torch.inference_mode()
    def process(
        self,
        image: np.ndarray | Image.Image,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Encode image using HunyuanVideo VisionEncoder.

        Args:
            image: Input image as numpy array (H, W, C) uint8 or PIL Image
            target_width: Target width for resize and center crop (optional)
            target_height: Target height for resize and center crop (optional)

        Returns:
            Dictionary containing:
                - vision_states: Vision encoder output tensor (B, L, D)
        """
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Handle batch dimension
        if len(image.shape) == 4:
            image = image[0]  # Take first image if batch

        # Validate input
        assert isinstance(image, np.ndarray), f"Expected numpy array, got {type(image)}"
        assert image.ndim == 3 and image.shape[2] == 3, f"Expected (H, W, 3), got {image.shape}"
        assert image.dtype == np.uint8, f"Expected uint8, got {image.dtype}"

        # Apply resize and center crop if target size specified
        if target_width is not None and target_height is not None:
            image = resize_and_center_crop(image, target_width=target_width, target_height=target_height)

        # Use HunyuanVideoImageEncoder.encode_images
        # This matches the original HunyuanVideo interface
        vision_output = self.vision_encoder.encode_images(image)

        # Extract last_hidden_state
        if hasattr(vision_output, "last_hidden_state"):
            vision_states = vision_output.last_hidden_state
        else:
            raise ValueError(f"VisionEncoder output does not have last_hidden_state. Got: {type(vision_output)}")

        # Move to target device and dtype
        vision_states = vision_states.to(device=self.device, dtype=self.torch_dtype)
        return vision_states
