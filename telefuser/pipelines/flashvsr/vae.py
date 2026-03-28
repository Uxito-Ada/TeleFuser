from __future__ import annotations

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.models.TCDecoder import TAEHV
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug


class VAEStage(BaseStage):
    """VAE decoding stage for FlashVSR video super-resolution."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.vae: TAEHV = module_manager.fetch_module("wan_video_decoder")
        self.model_names = ["vae"]

    @ProfilingContext4Debug("vae decode video")
    def decode_video(self, latents: torch.Tensor, cond_video: torch.Tensor) -> torch.Tensor:
        """Decode latents to video frames with conditional input."""
        frames = (
            self.vae.decode(
                latents.transpose(1, 2).to(self.device),
                parallel=False,
                show_progress_bar=True,
                cond=cond_video.to(self.device),
                device=self.device,
            )
            .transpose(1, 2)
            .mul_(2)
            .sub_(1)
        )
        return frames

    def clean_cache(self):
        """Clean VAE memory cache."""
        logger.info("clean flashvsr vae cache")
        self.vae.clean_mem()

    @with_model_offload(["vae"])
    @torch.inference_mode()
    @with_metrics
    def process(self, method: str, *args, **kwargs) -> torch.Tensor:
        """Dispatch to the specified method."""
        if hasattr(self, method):
            return getattr(self, method)(*args, **kwargs)
        else:
            raise NotImplementedError(f"{method} is not supported in flashvsr vae")
