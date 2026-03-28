"""HunyuanVideo Upsampler for Super-Resolution.

Implements SRTo720pUpsampler and SRTo1080pUpsampler for video super-resolution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from telefuser.models.hunyuan_video_vae import CausalConv3d, RMSNorm, ResnetBlock
from telefuser.utils.logging import logger


class SRResidualCausalBlock3D(nn.Module):
    """Residual block with causal 3D convolutions for super-resolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            CausalConv3d(channels, channels, kernel_size=3),
            nn.SiLU(inplace=True),
            CausalConv3d(channels, channels, kernel_size=3),
            nn.SiLU(inplace=True),
            CausalConv3d(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SRTo720pUpsampler(nn.Module):
    """Super-Resolution upsampler for 480p -> 720p upscaling.

    A lightweight residual network that enhances bilinear-upsampled latents.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        hidden_channels: Optional[int] = None,
        num_blocks: int = 6,
        global_residual: bool = False,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = 64
        self.in_conv = CausalConv3d(in_channels, hidden_channels, kernel_size=3)
        self.blocks = nn.ModuleList([SRResidualCausalBlock3D(hidden_channels) for _ in range(num_blocks)])
        self.out_conv = CausalConv3d(hidden_channels, out_channels, kernel_size=3)
        self.global_residual = bool(global_residual)

        self.config = type(
            "Config",
            (),
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "hidden_channels": hidden_channels,
                "num_blocks": num_blocks,
                "global_residual": global_residual,
            },
        )()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input latent tensor (B, C, T, H, W)

        Returns:
            Enhanced latent tensor (B, C, T, H, W)
        """
        residual = x
        y = self.in_conv(x)
        for blk in self.blocks:
            y = blk(y)
        y = self.out_conv(y)
        if self.global_residual and y.shape == residual.shape:
            y = y + residual
        return y

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float32,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> "SRTo720pUpsampler":
        """Load upsampler from pretrained checkpoint.

        Args:
            pretrained_model_name_or_path: Path to upsampler checkpoint
            torch_dtype: Data type for model weights
            device: Device to load model on

        Returns:
            SRTo720pUpsampler instance
        """
        import json
        import os

        if isinstance(device, str):
            device = torch.device(device)

        logger.info(f"Loading SRTo720pUpsampler from {pretrained_model_name_or_path}")

        # Load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default config for 720p upsampler
            config = {
                "in_channels": 32,
                "out_channels": 32,
                "hidden_channels": 64,
                "num_blocks": 6,
                "global_residual": False,
            }

        # Create model
        model = cls(
            in_channels=config.get("in_channels", 32),
            out_channels=config.get("out_channels", 32),
            hidden_channels=config.get("hidden_channels", 64),
            num_blocks=config.get("num_blocks", 6),
            global_residual=config.get("global_residual", False),
        )

        # Load weights
        weights_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.bin")

        if os.path.exists(weights_path):
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(weights_path)
            else:
                state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        else:
            raise FileNotFoundError(f"No weights found at {pretrained_model_name_or_path}")

        model = model.to(dtype=torch_dtype)
        model.requires_grad_(False)

        if device is not None:
            model = model.to(device)

        return model


class SRTo1080pUpsampler(nn.Module):
    """Super-Resolution upsampler for 720p -> 1080p upscaling.

    A more complex residual network with multiple resolution levels.
    """

    def __init__(
        self,
        z_channels: int = 32,
        out_channels: int = 32,
        block_out_channels: tuple[int, ...] = (128, 256),
        num_res_blocks: int = 2,
        is_residual: bool = False,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.block_out_channels = block_out_channels
        self.z_channels = z_channels

        block_in = block_out_channels[0]
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3)

        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            self.up.append(up)

        self.norm_out = RMSNorm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3)

        self.is_residual = is_residual

        self.config = type(
            "Config",
            (),
            {
                "z_channels": z_channels,
                "out_channels": out_channels,
                "block_out_channels": block_out_channels,
                "num_res_blocks": num_res_blocks,
                "is_residual": is_residual,
            },
        )()

    def forward(self, z: torch.Tensor, target_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Input latent tensor (B, C, T, H, W)
            target_shape: Target spatial shape (H, W) for interpolation

        Returns:
            Enhanced latent tensor (B, C, T, H', W')
        """
        if target_shape is not None and z.shape[-2:] != target_shape:
            bsz = z.shape[0]
            z = rearrange(z, "b c f h w -> (b f) c h w")
            z = F.interpolate(z, size=target_shape, mode="bilinear", align_corners=False)
            z = rearrange(z, "(b f) c h w -> b c f h w", b=bsz)

        # z to block_in
        repeats = self.block_out_channels[0] // self.z_channels
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float32,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> "SRTo1080pUpsampler":
        """Load upsampler from pretrained checkpoint.

        Args:
            pretrained_model_name_or_path: Path to upsampler checkpoint
            torch_dtype: Data type for model weights
            device: Device to load model on

        Returns:
            SRTo1080pUpsampler instance
        """
        import json
        import os

        if isinstance(device, str):
            device = torch.device(device)

        logger.info(f"Loading SRTo1080pUpsampler from {pretrained_model_name_or_path}")

        # Load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default config for 1080p upsampler
            config = {
                "z_channels": 32,
                "out_channels": 32,
                "block_out_channels": [128, 256],
                "num_res_blocks": 2,
                "is_residual": False,
            }

        # Create model
        model = cls(
            z_channels=config.get("z_channels", 32),
            out_channels=config.get("out_channels", 32),
            block_out_channels=tuple(config.get("block_out_channels", [128, 256])),
            num_res_blocks=config.get("num_res_blocks", 2),
            is_residual=config.get("is_residual", False),
        )

        # Load weights
        weights_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.bin")

        if os.path.exists(weights_path):
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(weights_path)
            else:
                state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        else:
            raise FileNotFoundError(f"No weights found at {pretrained_model_name_or_path}")

        model = model.to(dtype=torch_dtype)
        model.requires_grad_(False)

        if device is not None:
            model = model.to(device)

        return model
