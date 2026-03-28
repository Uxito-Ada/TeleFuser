"""Rotary Position Embedding (RoPE) operations.

Provides optimized RoPE implementations with automatic kernel selection
based on platform and available libraries.
"""

from __future__ import annotations

import torch

from telefuser.platforms import CudaPlatform, current_platform

# Check if Triton is available (only supported on NVIDIA CUDA)
_has_triton = False
if isinstance(current_platform, CudaPlatform):
    try:
        import triton  # noqa: F401

        _has_triton = True
    except ImportError:
        pass

if _has_triton:
    from telefuser.kernel.triton import apply_rotary_embedding as _triton_apply_rotary_emb


def apply_rotary_emb(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
    sequence_dim: int = 1,
) -> torch.Tensor:
    """Apply rotary positional embeddings to input tensor.

    Automatically selects the optimal implementation based on platform:
    - NVIDIA CUDA with Triton: Uses Triton kernel for better performance
    - ROCm / Other: Falls back to PyTorch implementation

    Args:
        x: Input tensor of shape (B, S, H, D).
        freqs: Tuple of (cos, sin) tensors of shape (S, D).
        sequence_dim: Dimension for sequence (default: 1).

    Returns:
        Tensor with rotary embeddings applied.
    """
    cos, sin = freqs

    # Use Triton kernel on NVIDIA CUDA for better performance
    if _has_triton:
        # Triton kernel expects (B, S, H, D) and interleaved cos/sin
        # Current freqs format is (S, D) with repeat_interleave
        # Extract every other element to get (S, D/2) for interleaved mode
        cos_half = cos[..., ::2].contiguous()
        sin_half = sin[..., ::2].contiguous()
        return _triton_apply_rotary_emb(x, cos_half, sin_half, interleaved=True)

    # Fallback to PyTorch implementation for ROCm and other platforms
    if sequence_dim == 1:
        cos = cos.unsqueeze(0).unsqueeze(-2)
        sin = sin.unsqueeze(0).unsqueeze(-2)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    cos = cos.to(x.device)
    sin = sin.to(x.device)

    # Rotate: x_real = x[..., 0::2], x_imag = x[..., 1::2]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)

    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


__all__ = ["apply_rotary_emb"]
