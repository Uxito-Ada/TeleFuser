"""Small FP8 activation helpers used at attention boundaries."""

from __future__ import annotations

import torch
import torch.nn.functional as F

FP8_ATTENTION_BLOCK_SIZE = 64


def quantize_fp8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[B, T, H, D]`` activations with one scale per token/head."""
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("FP8 attention quantization expects a floating-point [B, T, H, D] tensor")
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = x.detach().abs().amax(dim=-1).float().clamp_min(1e-6) / fp8_max
    quantized = (x / scale.to(dtype=x.dtype).unsqueeze(-1)).to(torch.float8_e4m3fn)
    return quantized, scale


def dequantize_fp8_per_token(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Restore per-token/head FP8 activations to the kernel compute dtype."""
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError("expected torch.float8_e4m3fn activations")
    return x.to(dtype) * scale.to(dtype=dtype).unsqueeze(-1)


def quantize_fp8_per_block(
    x: torch.Tensor,
    block_size: int = FP8_ATTENTION_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[B, T, H, D]`` activations per token block and head."""
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("FP8 attention quantization expects a floating-point [B, T, H, D] tensor")
    batch, tokens, heads, head_dim = x.shape
    blocks = (tokens + block_size - 1) // block_size
    padded_tokens = blocks * block_size
    padded = F.pad(x, (0, 0, 0, 0, 0, padded_tokens - tokens))
    blocked = padded.reshape(batch, blocks, block_size, heads, head_dim)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = blocked.detach().abs().amax(dim=(2, 4)).float().clamp_min(1e-6) / fp8_max
    quantized = (blocked / scale.to(dtype=x.dtype)[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return quantized.reshape(batch, padded_tokens, heads, head_dim)[:, :tokens].contiguous(), scale


def dequantize_fp8_per_block(
    x: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = FP8_ATTENTION_BLOCK_SIZE,
) -> torch.Tensor:
    """Restore block-scaled FP8 attention activations to ``dtype``."""
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError("expected torch.float8_e4m3fn activations")
    tokens = x.shape[1]
    token_scale = scale.repeat_interleave(block_size, dim=1)[:, :tokens]
    return x.to(dtype) * token_scale.to(dtype=dtype).unsqueeze(-1)


__all__ = [
    "FP8_ATTENTION_BLOCK_SIZE",
    "dequantize_fp8_per_block",
    "dequantize_fp8_per_token",
    "quantize_fp8_per_block",
    "quantize_fp8_per_token",
]
