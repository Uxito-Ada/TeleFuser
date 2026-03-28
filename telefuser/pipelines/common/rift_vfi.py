from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.models.rift_hdv3 import IFNet
from telefuser.utils.profiler import ProfilingContext4Debug


class RiftVFIStage(BaseStage):
    """Video Frame Interpolation stage using RIFE (Real-Time Intermediate Flow Estimation).

    Interpolates frames to increase video frame rate from base_fps to target_fps.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.vfi_model: IFNet = module_manager.fetch_module("vfi_model")  # type: ignore
        self.model_names = ["vfi_model"]

    @with_model_offload(["vfi_model"])
    @ProfilingContext4Debug("cli_encoding")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        input_video: List[Image.Image],
        base_fps: int,
        target_fps: int,
    ) -> List[Image.Image]:
        """Interpolate video frames from base_fps to target_fps."""
        src_tensor_list = [
            torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0 for image in input_video
        ]
        src_tensor = torch.concat(src_tensor_list, dim=0)

        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            result_tensor = self.vfi_model.interpolate_frames(src_tensor, base_fps, target_fps, device=self.device.type)

        frames = ((result_tensor.float()) * 255).clip(0, 255).cpu().numpy().astype(np.uint8)
        result_video = [Image.fromarray(frame) for frame in frames]
        return result_video
