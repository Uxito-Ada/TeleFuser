from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from loguru import logger

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx_video_vae import VIDEO_SCALE_FACTORS, LTXVideoVAE
from telefuser.utils.profiler import ProfilingContext4Debug

DEFAULT_TILE_SIZE = (64, 512)
DEFAULT_TILE_STRIDE = (40, 448)
ImageSource = str | Image.Image | torch.Tensor


def _clamp_strength(strength: float) -> float:
    return float(max(0.0, min(1.0, strength)))


class VAEStage(BaseStage):
    """VAE encoding/decoding stage for LTX video."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.vae: LTXVideoVAE = module_manager.fetch_module("ltx_video_vae")
        self.model_names = ["vae"]
        if model_runtime_config.offload_config.offload_type == WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD:
            logger.info("enable sequential cpu offload for ltx vae")
            self.vae.enable_sequential_cpu_offload(device=self.device, torch_dtype=self.torch_dtype)

    @staticmethod
    def _decode_image(image: ImageSource) -> np.ndarray:
        if isinstance(image, str):
            return np.array(Image.open(image).convert("RGB"))[..., :3]
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))[..., :3]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported image source type: {type(image)!r}")

        image_tensor = image.detach().cpu()
        if image_tensor.ndim == 4:
            if image_tensor.shape[0] != 1:
                raise ValueError(f"Expected a single image tensor batch, got shape {tuple(image_tensor.shape)}.")
            image_tensor = image_tensor[0]
        if image_tensor.ndim != 3:
            raise ValueError(f"Expected an image tensor with 3 dimensions, got shape {tuple(image_tensor.shape)}.")
        if image_tensor.shape[-1] not in {1, 3, 4} and image_tensor.shape[0] in {1, 3, 4}:
            image_tensor = image_tensor.permute(1, 2, 0)
        if image_tensor.shape[-1] not in {1, 3, 4}:
            raise ValueError(f"Unsupported image tensor shape {tuple(image_tensor.shape)}.")

        if image_tensor.is_floating_point():
            if torch.amin(image_tensor) < 0:
                image_tensor = (image_tensor.clamp(-1.0, 1.0) + 1.0) * 127.5
            elif torch.amax(image_tensor) <= 1.0:
                image_tensor = image_tensor.clamp(0.0, 1.0) * 255.0
            else:
                image_tensor = image_tensor.clamp(0.0, 255.0)
        else:
            image_tensor = image_tensor.clamp(0, 255)
        return image_tensor.to(torch.uint8).numpy()[..., :3]

    @staticmethod
    def _resize_and_center_crop(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")
        _, _, src_h, src_w = tensor.shape
        scale = max(height / src_h, width / src_w)
        new_h = int(np.ceil(src_h * scale))
        new_w = int(np.ceil(src_w * scale))
        tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        crop_top = (new_h - height) // 2
        crop_left = (new_w - width) // 2
        tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]
        return tensor.unsqueeze(2) if tensor.ndim == 4 else tensor

    def _encode_image_latent(self, image: ImageSource, height: int, width: int) -> torch.Tensor:
        image_tensor = torch.tensor(self._decode_image(image), dtype=torch.float32, device=self.device)
        image_tensor = self._resize_and_center_crop(image_tensor, height, width)
        image_tensor = (image_tensor / 127.5 - 1.0).to(device=self.device, dtype=self.torch_dtype)
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            return self.vae.encode(image_tensor, tiled=False)

    @ProfilingContext4Debug("ltx vae encode video")
    def encode_image(
        self,
        image: ImageSource | None,
        height: int,
        width: int,
        frame_idx: int = 0,
        strength: float = 1.0,
    ) -> torch.Tensor | None:
        """Encode input image into conditioning latents for image-to-video generation.

        Unlike the Wan pipelines where conditioning is concatenated as extra channels, the LTX denoiser
        consumes tokenized latents. To keep conditioning transport friendly for Ray/multiprocessing,
        this method returns a single tensor containing both the conditioning mask and the latents:

        - Channel 0: denoise mask for the conditioned region (1.0 means fully denoise, 0.0 means keep clean).
        - Channels 1..: encoded latents that should be copied into the initial state.

        The temporal dimension encodes which latent frame is conditioned. Because the VAE encoder only
        returns a single latent frame for an image, the returned tensor is padded with empty frames up
        to the conditioned index.
        """
        if image is None:
            return None

        strength = _clamp_strength(strength)
        if strength <= 0.0:
            return None

        if frame_idx < 0:
            raise ValueError(f"frame_idx must be non-negative, got {frame_idx}.")

        latent_idx = frame_idx // VIDEO_SCALE_FACTORS.time
        if frame_idx % VIDEO_SCALE_FACTORS.time != 0:
            logger.warning(
                "LTX encode_image received frame_idx={} which is not aligned to latent time scale={}; "
                "conditioning snaps to latent_idx={}.",
                frame_idx,
                VIDEO_SCALE_FACTORS.time,
                latent_idx,
            )

        encoded = self._encode_image_latent(image, height, width)
        batch, channels, frames, latent_height, latent_width = encoded.shape
        if batch != 1 or frames != 1:
            raise ValueError(f"Expected encoded image latent shape (1, C, 1, H, W); got {tuple(encoded.shape)}.")

        cond_frames = latent_idx + 1
        cond_latents = torch.zeros(
            (batch, channels, cond_frames, latent_height, latent_width),
            device=encoded.device,
            dtype=encoded.dtype,
        )
        cond_latents[:, :, latent_idx : latent_idx + 1] = encoded

        mask_value = 1.0 - strength
        cond_mask = torch.ones(
            (batch, 1, cond_frames, latent_height, latent_width),
            device=encoded.device,
            dtype=torch.float32,
        )
        cond_mask[:, :, latent_idx : latent_idx + 1] = mask_value
        cond_mask = cond_mask.to(dtype=encoded.dtype)
        return torch.cat([cond_mask, cond_latents], dim=1)

    @ProfilingContext4Debug("ltx decode video")
    @torch.inference_mode()
    def decode_video(
        self,
        latents: torch.Tensor,
        generator: torch.Generator | None = None,
        tiled: bool = True,
        tile_size: tuple[int, int] = DEFAULT_TILE_SIZE,
        tile_stride: tuple[int, int] = DEFAULT_TILE_STRIDE,
    ) -> torch.Tensor:
        """Decode video latents with optional temporal/spatial tiling."""
        chunks = list(
            self.vae.decode(
                latents,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                generator=generator,
            )
        )
        return torch.cat(chunks, dim=2)

    def parallel_models(self):
        """Configure tensor parallelism for VAE."""
        self.vae.set_parallelism(self.model_runtime_config.parallel_config.world_size)

    @with_model_offload(["vae"])
    @torch.inference_mode()
    def process(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to specified VAE method."""
        if hasattr(self, method):
            return getattr(self, method)(*args, **kwargs)
        raise NotImplementedError(f"{method} is not supported in ltx vae")
