"""VAE encoding/decoding stage for Flux2 Klein."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics


class VAEStage(BaseStage):
    """VAE encoding/decoding stage for Flux2 Klein.

    Handles image-to-latent encoding and latent-to-image decoding
    with patchification and BatchNorm normalization.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.vae = module_manager.fetch_module("vae")
        self.model_names = ["vae"]
        self.vae_scale_factor = 8  # 2^(len(block_out_channels)-1)

    @staticmethod
    def preprocess_image(image: Image.Image, mode: str = "RGB") -> torch.Tensor:
        """Preprocess PIL image to normalized tensor.

        Args:
            image: PIL Image
            mode: Color mode

        Returns:
            Tensor of shape (1, C, H, W) with values in [-1, 1]
        """
        image = image.convert(mode)
        image_array = np.array(image, dtype=np.float32)
        if len(image_array.shape) == 2:
            image_array = image_array[:, :, np.newaxis]
        image = torch.Tensor((image_array / 255) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Patchify latents from (B, C, H, W) to (B, C*4, H//2, W//2).

        Args:
            latents: Tensor of shape (B, C, H, W)

        Returns:
            Patchified tensor of shape (B, C*4, H//2, W//2)
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Unpatchify latents from (B, C*4, H//2, W//2) to (B, C, H, W).

        Args:
            latents: Patchified tensor of shape (B, C*4, H//2, W//2)

        Returns:
            Unpatchified tensor of shape (B, C, H, W)
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pack latents from (B, C, H, W) to (B, H*W, C).

        Args:
            latents: Tensor of shape (B, C, H, W)

        Returns:
            Packed tensor of shape (B, H*W, C)
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _unpack_latents_with_ids(latents: torch.Tensor, latent_ids: torch.Tensor) -> torch.Tensor:
        """Unpack latents using position IDs.

        Args:
            latents: Packed tensor of shape (B, H*W, C)
            latent_ids: Position IDs of shape (B, H*W, 4)

        Returns:
            Unpacked tensor of shape (B, C, H, W)
        """
        x_list = []
        for data, pos in zip(latents, latent_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # Reshape from (H * W, C) to (C, H, W)
            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def _get_batchnorm_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get BatchNorm running mean and std from VAE.

        Returns:
            Tuple of (mean, std) tensors for latent normalization
        """
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(self.device, self.torch_dtype)
        bn_var = self.vae.bn.running_var.view(1, -1, 1, 1).to(self.device, self.torch_dtype)
        bn_std = torch.sqrt(bn_var + self.vae.config.batch_norm_eps)
        return bn_mean, bn_std

    def encode_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image to latent representation.

        Args:
            image: Image tensor of shape (B, 3, H, W)

        Returns:
            Packed latent tensor of shape (B, H*W/256, 128)
        """
        # VAE encode
        encoder_output = self.vae.encode(image)
        latents = encoder_output.latent_dist.mode()

        # Patchify
        latents = self._patchify_latents(latents)

        # BatchNorm normalization
        bn_mean, bn_std = self._get_batchnorm_params()
        latents = (latents - bn_mean) / bn_std

        # Pack for transformer
        latents = self._pack_latents(latents)

        return latents

    def decode_latents(
        self,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent representation to image.

        Args:
            latents: Packed latent tensor of shape (B, H*W, 128)
            latent_ids: Position IDs of shape (B, H*W, 4)

        Returns:
            Image tensor of shape (B, 3, H, W)
        """
        # Unpack
        latents = self._unpack_latents_with_ids(latents, latent_ids)

        # Reverse BatchNorm
        bn_mean, bn_std = self._get_batchnorm_params()
        latents = latents * bn_std + bn_mean

        # Unpatchify
        latents = self._unpatchify_latents(latents)

        # VAE decode
        image = self.vae.decode(latents, return_dict=False)[0]

        return image

    @with_model_offload(["vae"])
    @torch.inference_mode()
    @with_metrics
    def process(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to encode_image or decode_latents method.

        Args:
            method: Method name ('encode_image' or 'decode_latents')
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the called method
        """
        if hasattr(self, method):
            return getattr(self, method)(*args, **kwargs)
        else:
            raise NotImplementedError(f"{method} is not supported in Flux2 Klein VAE")
