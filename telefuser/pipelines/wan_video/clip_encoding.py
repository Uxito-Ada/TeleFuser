from __future__ import annotations

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.models.wan_video_image_encoder import WanImageEncoder
from telefuser.utils.profiler import ProfilingContext4Debug


class ClipEncodingStage(BaseStage):
    """CLIP image encoding stage for Wan video I2V (image-to-video).

    Encodes input images to CLIP features for DiT conditioning.
    Supports both single image and首尾帧 (first+last frame) generation.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.image_encoder: WanImageEncoder = module_manager.fetch_module("wan_video_image_encoder")
        self.model_names = ["image_encoder"]

    @with_model_offload(["image_encoder"])
    @ProfilingContext4Debug("cli_encoding")
    @torch.inference_mode()
    @with_metrics
    def process(self, image: torch.Tensor, end_image: torch.Tensor | None = None) -> torch.Tensor:
        """Encode image(s) to CLIP features for conditioning.

        Args:
            image: Input image tensor
            end_image: Optional end frame image for首尾帧 generation

        Returns:
            CLIP context tensor for DiT conditioning
        """
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            clip_context = self.image_encoder.encode_image([image.to(self.device)])
            if end_image is not None:
                clip_context = torch.concat(
                    [
                        clip_context,
                        self.image_encoder.encode_image([end_image.to(self.device)]),
                    ],
                    dim=1,
                )
        return clip_context
