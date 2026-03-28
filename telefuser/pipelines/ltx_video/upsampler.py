from __future__ import annotations

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx_upsampler import LTXSpatialUpsampler, upsample_video
from telefuser.models.ltx_video_vae import LTXVideoVAE
from telefuser.utils.profiler import ProfilingContext4Debug


class UpsamplerStage(BaseStage):
    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.spatial_upsampler: LTXSpatialUpsampler = module_manager.fetch_module("ltx_spatial_upsampler")
        self.vae: LTXVideoVAE = module_manager.fetch_module("ltx_video_vae")
        self.model_names = ["spatial_upsampler", "vae"]

    @with_model_offload(["spatial_upsampler", "vae"])
    @ProfilingContext4Debug("ltx upsample video latent")
    @torch.inference_mode()
    def process(self, latent: torch.Tensor) -> torch.Tensor:
        return upsample_video(latent=latent, video_encoder=self.vae.encoder, upsampler=self.spatial_upsampler)
