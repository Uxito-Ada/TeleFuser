"""Triton kernels for Rotary Position Embedding (RoPE).

Adapted from sglang diffusion kernels for TeleFuser.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .custom_op import register_custom_op


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HS_HALF": 32}, num_warps=2),
        triton.Config({"BLOCK_HS_HALF": 64}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 128}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 256}, num_warps=8),
    ],
    key=["head_size", "interleaved"],
)
@triton.jit
def _rotary_embedding_kernel(
    output_ptr,
    x_ptr,
    cos_ptr,
    sin_ptr,
    num_heads,
    head_size,
    num_tokens,
    stride_x_row,
    stride_cos_row,
    stride_sin_row,
    interleaved: tl.constexpr,
    BLOCK_HS_HALF: tl.constexpr,
):
    """Rotary Position Embedding kernel."""
    row_idx = tl.program_id(0)
    token_idx = (row_idx // num_heads) % num_tokens

    x_row_ptr = x_ptr + row_idx * stride_x_row
    cos_row_ptr = cos_ptr + token_idx * stride_cos_row
    sin_row_ptr = sin_ptr + token_idx * stride_sin_row
    output_row_ptr = output_ptr + row_idx * stride_x_row

    head_size_half = head_size // 2

    for block_start in range(0, head_size_half, BLOCK_HS_HALF):
        offsets_half = block_start + tl.arange(0, BLOCK_HS_HALF)
        mask = offsets_half < head_size_half

        cos_vals = tl.load(cos_row_ptr + offsets_half, mask=mask, other=0.0)
        sin_vals = tl.load(sin_row_ptr + offsets_half, mask=mask, other=0.0)

        offsets_x1 = 2 * offsets_half
        offsets_x2 = 2 * offsets_half + 1

        x1_vals = tl.load(x_row_ptr + offsets_x1, mask=mask, other=0.0)
        x2_vals = tl.load(x_row_ptr + offsets_x2, mask=mask, other=0.0)

        x1_fp32 = x1_vals.to(tl.float32)
        x2_fp32 = x2_vals.to(tl.float32)
        cos_fp32 = cos_vals.to(tl.float32)
        sin_fp32 = sin_vals.to(tl.float32)
        o1_vals = tl.fma(-x2_fp32, sin_fp32, x1_fp32 * cos_fp32)
        o2_vals = tl.fma(x1_fp32, sin_fp32, x2_fp32 * cos_fp32)

        tl.store(output_row_ptr + offsets_x1, o1_vals.to(x1_vals.dtype), mask=mask)
        tl.store(output_row_ptr + offsets_x2, o2_vals.to(x2_vals.dtype), mask=mask)


@register_custom_op(
    op_name="telefuser::apply_rotary_embedding",
    mutates_args=(),
)
def _apply_rotary_embedding_impl(
    x_reshaped: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_size: int,
    num_tokens: int,
    bsz: int,
    interleaved: bool,
) -> torch.Tensor:
    """Internal implementation for apply_rotary_embedding using Triton kernel."""
    output = torch.empty_like(x_reshaped)

    grid = (bsz * num_tokens * num_heads,)

    if interleaved and cos.shape[-1] == head_size:
        cos = cos[..., ::2].contiguous()
        sin = sin[..., ::2].contiguous()
    else:
        cos = cos.contiguous()
        sin = sin.contiguous()

    _rotary_embedding_kernel[grid](
        output,
        x_reshaped,
        cos,
        sin,
        num_heads,
        head_size,
        num_tokens,
        x_reshaped.stride(0),
        cos.stride(0) if cos.dim() > 1 else 0,
        sin.stride(0) if sin.dim() > 1 else 0,
        interleaved,
    )

    return output


def apply_rotary_embedding(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to input tensor.

    Args:
        x: Input tensor of shape [batch, seq_len, num_heads, head_size] or
           [seq_len, num_heads, head_size]
        cos: Cosine values of shape [seq_len, head_size] or [batch, seq_len, head_size]
        sin: Sine values of shape [seq_len, head_size] or [batch, seq_len, head_size]
        interleaved: If True, use interleaved format where cos/sin have full head_size

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    # Handle dimension normalization outside the compiled kernel
    if x.dim() > 3:
        bsz, num_tokens, num_heads, head_size = x.shape
    else:
        num_tokens, num_heads, head_size = x.shape
        bsz = 1

    assert head_size % 2 == 0, "head_size must be divisible by 2"

    x_reshaped = x.view(-1, head_size)

    # Call the registered custom op
    output_reshaped = _apply_rotary_embedding_impl(
        x_reshaped, cos, sin, num_heads, head_size, num_tokens, bsz, interleaved
    )

    return output_reshaped.view(x.shape)


@triton.jit
def _rotary_embedding_inplace_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    num_heads,
    head_size,
    num_tokens,
    stride_x_row,
    stride_cos_row,
    stride_sin_row,
    BLOCK_HS_HALF: tl.constexpr,
):
    """In-place Rotary Position Embedding kernel."""
    row_idx = tl.program_id(0)
    token_idx = (row_idx // num_heads) % num_tokens

    x_row_ptr = x_ptr + row_idx * stride_x_row
    cos_row_ptr = cos_ptr + token_idx * stride_cos_row
    sin_row_ptr = sin_ptr + token_idx * stride_sin_row

    head_size_half = head_size // 2

    for block_start in range(0, head_size_half, BLOCK_HS_HALF):
        offsets_half = block_start + tl.arange(0, BLOCK_HS_HALF)
        mask = offsets_half < head_size_half

        cos_vals = tl.load(cos_row_ptr + offsets_half, mask=mask, other=0.0)
        sin_vals = tl.load(sin_row_ptr + offsets_half, mask=mask, other=0.0)

        offsets_x1 = 2 * offsets_half
        offsets_x2 = 2 * offsets_half + 1

        x1_vals = tl.load(x_row_ptr + offsets_x1, mask=mask, other=0.0)
        x2_vals = tl.load(x_row_ptr + offsets_x2, mask=mask, other=0.0)

        x1_fp32 = x1_vals.to(tl.float32)
        x2_fp32 = x2_vals.to(tl.float32)
        cos_fp32 = cos_vals.to(tl.float32)
        sin_fp32 = sin_vals.to(tl.float32)
        o1_vals = tl.fma(-x2_fp32, sin_fp32, x1_fp32 * cos_fp32)
        o2_vals = tl.fma(x1_fp32, sin_fp32, x2_fp32 * cos_fp32)

        tl.store(x_row_ptr + offsets_x1, o1_vals.to(x1_vals.dtype), mask=mask)
        tl.store(x_row_ptr + offsets_x2, o2_vals.to(x2_vals.dtype), mask=mask)


@register_custom_op(
    op_name="telefuser::apply_rotary_embedding_inplace",
    mutates_args=["x"],
)
def _apply_rotary_embedding_inplace_impl(
    x_reshaped: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_size: int,
    num_tokens: int,
    bsz: int,
    block_hs_half: int,
) -> None:
    """Internal implementation for in-place RoPE using Triton kernel."""
    grid = (bsz * num_tokens * num_heads,)

    _rotary_embedding_inplace_kernel[grid](
        x_reshaped,
        cos.contiguous(),
        sin.contiguous(),
        num_heads,
        head_size,
        num_tokens,
        x_reshaped.stride(0),
        cos.stride(0) if cos.dim() > 1 else 0,
        sin.stride(0) if sin.dim() > 1 else 0,
        block_hs_half,
    )


def apply_rotary_embedding_inplace(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """Apply Rotary Position Embedding (RoPE) in-place.

    Args:
        x: Input tensor to be modified in-place
        cos: Cosine values
        sin: Sine values
    """
    # Handle dimension normalization outside the compiled kernel
    if x.dim() > 3:
        bsz, num_tokens, num_heads, head_size = x.shape
    else:
        num_tokens, num_heads, head_size = x.shape
        bsz = 1

    assert head_size % 2 == 0, "head_size must be divisible by 2"

    x_reshaped = x.view(-1, head_size)

    BLOCK_HS_HALF = min(128, triton.next_power_of_2(head_size // 2))

    # Call the registered custom op
    _apply_rotary_embedding_inplace_impl(x_reshaped, cos, sin, num_heads, head_size, num_tokens, bsz, BLOCK_HS_HALF)
