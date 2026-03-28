"""HunyuanVideo Image Encoder (Vision Encoder) for I2V generation.

Based on Siglip vision model for encoding reference images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import SiglipImageProcessor, SiglipVisionModel

from telefuser.utils.logging import logger


@dataclass
class VisionEncoderOutput:
    """Output from vision encoder.

    Args:
        last_hidden_state: Hidden states from last layer (B, L, D)
        pooler_output: Pooled output (B, D)
        hidden_states: Tuple of hidden states from all layers
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


class HunyuanVideoImageEncoder(nn.Module):
    """Vision encoder for HunyuanVideo I2V.

    Uses Siglip vision model to encode reference images into visual features
    that can be used as conditioning for the DiT model.

    Attributes:
        model: The underlying SiglipVisionModel
        processor: Image processor for preprocessing
        output_key: Key to extract from model output
    """

    def __init__(
        self,
        model: SiglipVisionModel,
        processor: SiglipImageProcessor,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        """Initialize the vision encoder with pre-loaded model.

        Args:
            model: Pre-loaded SiglipVisionModel
            processor: Pre-loaded SiglipImageProcessor
            dtype: Data type for the model
            device: Device to load model on
        """
        super().__init__()

        self.model = model.to(dtype=dtype)
        self.model.requires_grad_(False)
        self.processor = processor
        self.dtype = dtype

        if device is not None:
            self.model = self.model.to(device)

        logger.info(f"Initialized HunyuanVideoImageEncoder (dtype={dtype}, device={device})")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float16,
        device: Optional[Union[torch.device, str]] = None,
    ) -> "HunyuanVideoImageEncoder":
        """Load vision encoder from pretrained model.

        Args:
            pretrained_model_name_or_path: Path to Siglip model weights
            torch_dtype: Data type for model weights
            device: Device to load model on (can be string or torch.device)

        Returns:
            HunyuanVideoImageEncoder instance
        """
        # Convert string device to torch.device
        if isinstance(device, str):
            device = torch.device(device)

        logger.info(f"Loading HunyuanVideoImageEncoder from {pretrained_model_name_or_path}")

        # Load Siglip model
        model = SiglipVisionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="image_encoder",
        )

        # Load image processor
        processor = SiglipImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="feature_extractor",
        )

        return cls(model=model, processor=processor, dtype=torch_dtype, device=device)

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.model.parameters()).device

    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the vision encoder."""
        return self.model.config.hidden_size

    def preprocess_images(
        self,
        images: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Preprocess images for the vision encoder.

        Args:
            images: Input images as numpy array (B, H, W, C) in uint8 format

        Returns:
            Preprocessed inputs ready for the model
        """
        preprocessed = self.processor.preprocess(
            images=images,
            return_tensors="pt",
        )
        return preprocessed

    def encode_images(
        self,
        images: np.ndarray | torch.Tensor,
    ) -> VisionEncoderOutput:
        """Encode images using the vision encoder.

        Args:
            images: Input images
                - numpy array: (B, H, W, C) in uint8 format, will be preprocessed
                - torch tensor: Already preprocessed tensor

        Returns:
            VisionEncoderOutput with encoded features
        """
        # Preprocess if numpy array
        if isinstance(images, np.ndarray):
            preprocessed = self.preprocess_images(images)
            preprocessed = {k: v.to(device=self.device, dtype=self.dtype) for k, v in preprocessed.items()}
        else:
            preprocessed = images

        # Run model
        outputs = self.model(**preprocessed)

        return VisionEncoderOutput(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output if hasattr(outputs, "pooler_output") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
        )

    def forward(
        self,
        images: np.ndarray | torch.Tensor,
    ) -> VisionEncoderOutput:
        """Forward pass for image encoding.

        Args:
            images: Input images (numpy array or preprocessed tensor)

        Returns:
            VisionEncoderOutput with encoded features
        """
        return self.encode_images(images)


def resize_and_center_crop(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize and center crop image to target size.

    Args:
        image: Input image (H, W, C) in uint8 format
        target_width: Target width
        target_height: Target height

    Returns:
        Resized and cropped image
    """
    from PIL import Image

    # Early return if already correct size
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size

    # Calculate scale to cover target size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))

    # Resize
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)

    # Center crop
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    return np.array(cropped_image)
