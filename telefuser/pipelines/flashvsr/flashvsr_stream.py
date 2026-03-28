from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.utils.func import auto_async_call
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug
from telefuser.worker.parallel_worker import ParallelWorker

from .dit_denoising import DitDenoisingStage
from .vae import VAEStage


def pil_to_tensor_neg1_1(img: Image.Image, dtype: torch.dtype = torch.bfloat16, device: str = "cuda") -> torch.Tensor:
    """Convert PIL image to tensor with values in range [-1, 1]."""
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2, 0, 1)
    return t.to(dtype)


def compute_scaled_and_target_dims(
    w0: int, h0: int, scale: float = 4.0, multiple: int = 128
) -> tuple[int, int, int, int]:
    """Compute scaled dimensions and target dimensions that are multiples of a given value."""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = int(round(((sW) / multiple)) * multiple)
    tH = int(round(((sH) / multiple)) * multiple)
    if tW == 0 or tH == 0:
        raise ValueError(f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. Increase scale (got {scale}).")

    return sW, sH, tW, tH


def upscale_then_padding_tensor(
    img_tensor: torch.Tensor, original_size: tuple[int, int], scale: float, tW: int, tH: int
) -> torch.Tensor:
    """Upscale and pad tensor using PyTorch interpolation."""
    w0, h0 = original_size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    pad_width = int((tW - sW))
    pad_height = int((tH - sH))
    # Use bicubic interpolation for upsampling
    upsampled = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0),
        size=(sH, sW),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    padded = torch.nn.functional.pad(upsampled, [0, pad_width, 0, pad_height, 0, 0], mode="constant")
    # Convert back to [-1, 1] range
    return padded / 255 * 2.0 - 1.0


@ProfilingContext4Debug("prepare input tensor")
def prepare_input_tensor(
    video: list, scale: float = 4, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"
) -> tuple[torch.Tensor, int, int, int, int]:
    """Prepare input tensor from video frames.

    Processes each frame by upscaling and padding to target dimensions,
    then stacks into a batch tensor with shape (1, C, F, H, W).
    """
    first = video[0]
    w0, h0 = first.size
    total = len(video)
    if total <= 0:
        raise RuntimeError("Input video is empty")

    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    frames = []
    for img in video:
        img = img.convert("RGB")
        img_tensor = pil_to_tensor_neg1_1(img, dtype=torch.float32, device="cpu")
        processed_tensor = upscale_then_padding_tensor(img_tensor, (w0, h0), scale=scale, tW=tW, tH=tH)
        frames.append(processed_tensor.to(dtype=dtype, device=device))

    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    return vid, tH, tW, sH, sW


@dataclass
class FlashVSRStreamPipelineConfig:
    """Configuration for FlashVSR streaming video pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    enable_denoising_parallel: bool = False
    max_chunk_num: int = 128
    enable_metrics: bool = False


class FlashVSRStreamVideoPipeline(BasePipeline):
    """FlashVSR streaming video super-resolution pipeline.

    Processes video frames in streaming chunks for memory-efficient
    4x super-resolution using diffusion-based denoising.
    """

    def __init__(self, device: str, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.pre_frames_cache = None
        self.process_chunk_idx = 0

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        return [self.vae_stage, self.denoise_stage]

    def init(self, module_manager: ModuleManager, config: FlashVSRStreamPipelineConfig) -> None:
        """Initialize pipeline stages with configuration."""
        self._model_info = module_manager.get_model_info()
        self.config = config
        self.vae_stage = VAEStage("vae", module_manager, config.vae_config)
        self.denoise_stage = DitDenoisingStage("denoise", module_manager, config.dit_config)
        if config.enable_denoising_parallel:
            self.denoise_stage = ParallelWorker(self.denoise_stage)

        # Auto-enable metrics if configured
        if config.enable_metrics:
            self.enable_metrics()

    @torch.no_grad()
    def __call__(
        self,
        seed: int | None = None,
        LQ_video: list | None = None,
        sparse_ratio: float = 2.0,
        kv_ratio: int = 3,
        local_range: int = 9,
        scale: int = 4,
        rand_device: str = "cpu",
        proj_tile: bool = False,
    ) -> list:
        """Process low-quality video through the FlashVSR pipeline."""
        num_frames = len(LQ_video)
        try:
            LQ_video, height, width, real_height, real_width = prepare_input_tensor(
                LQ_video, scale=scale, dtype=self.torch_dtype, device="cpu"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to prepare input for FlashVSR: {e}")
        logger.info(
            f"Starting generation of {num_frames}x{width}x{height} SR video "
            f"with actual dimensions {real_width}x{real_height}"
        )
        # Auto-reset pipeline after max_chunk_num to prevent memory buildup
        if self.process_chunk_idx > self.config.max_chunk_num:
            logger.info(f"{self.process_chunk_idx} > {self.config.max_chunk_num}, auto reset pipeline")
            LQ_video = torch.cat([self.pre_frames_cache, LQ_video], dim=2)
            self.clean_cache()
        dit_result_handler = auto_async_call(
            self.denoise_stage.process,
            method="denoise",
            proj_tile=proj_tile,
            LQ_video=LQ_video,
            seed=seed,
            rand_device=rand_device,
            sparse_ratio=sparse_ratio,
            local_range=local_range,
            kv_ratio=kv_ratio,
            height=height,
            width=width,
        )
        latents, ref_LQ_video = dit_result_handler()
        frames_handler = auto_async_call(self.vae_stage.process, "decode_video", latents, ref_LQ_video)
        frames = frames_handler()
        # Handle first call vs subsequent calls differently due to overlap frames
        if self.pre_frames_cache is None:
            video = frames[0][:, -(num_frames - 4) :, :real_height, :real_width]  # CTHW
        else:
            video = frames[0][:, :, :real_height, :real_width]  # CTHW
        video = self.tensor2video(video)
        self.pre_frames_cache = torch.cat([LQ_video[:, :, -8:-7], LQ_video[:, :, -8:]], dim=2)
        self.process_chunk_idx += num_frames // 8
        return video

    def clean_cache(self) -> None:
        """Clean up cached data from all pipeline stages."""
        dit_clean_handler = auto_async_call(self.denoise_stage.process, method="clean_cache")
        dit_clean_handler()
        self.vae_stage.clean_cache()
        self.process_chunk_idx = 0
