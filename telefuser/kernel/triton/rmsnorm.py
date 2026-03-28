"""Single-pass RMSNorm Triton kernel.

Adapted from sglang diffusion kernels for TeleFuser.
Reference: https://github.com/ModelTC/LightX2V
"""

import torch
import triton
import triton.language as tl

from .custom_op import register_custom_op


@triton.jit
def _rms_norm_tiled_onepass(
    y_ptr,
    x_ptr,
    w_ptr,
    SEQ: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    """Single-pass RMSNorm kernel with tiled processing.

    Args:
        y_ptr: Output tensor pointer
        x_ptr: Input tensor pointer
        w_ptr: Weight tensor pointer
        SEQ: Sequence length (compile-time constant)
        DIM: Hidden dimension (compile-time constant)
        EPS: Epsilon for numerical stability
        BLOCK_SIZE_SEQ: Block size for sequence dimension
        BLOCK_SIZE_DIM: Block size for hidden dimension
    """
    seq_blk_id = tl.program_id(0)
    seq_id = seq_blk_id * BLOCK_SIZE_SEQ

    seq_offset = seq_id + tl.arange(0, BLOCK_SIZE_SEQ)[:, None]
    s_mask = seq_offset < SEQ
    d_offset = tl.arange(0, BLOCK_SIZE_DIM)[None, :]
    d_mask = d_offset < DIM
    y_blk = y_ptr + seq_offset * DIM + d_offset
    x_blk = x_ptr + seq_offset * DIM + d_offset
    mask = s_mask & d_mask

    x = tl.load(x_blk, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1, keep_dims=True) / DIM
    rstd = tl.math.rsqrt(mean_square + EPS)
    w = tl.load(w_ptr + d_offset, mask=d_mask)
    tl.store(y_blk, x * rstd * w, mask=mask)


@register_custom_op(
    op_name="telefuser::one_pass_rms_norm",
    mutates_args=["y"],
)
def _triton_one_pass_rms_norm_impl(
    x_view: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    S: int,
    D: int,
    eps: float,
    block_size_dim: int,
    block_size_seq: int,
) -> None:
    """Internal implementation for one-pass RMSNorm, wrapped as custom op."""
    grid = (triton.cdiv(S, block_size_seq),)
    torch.library.wrap_triton(_rms_norm_tiled_onepass)[grid](
        y,
        x_view,
        w,
        S,
        D,
        eps,
        BLOCK_SIZE_DIM=block_size_dim,
        BLOCK_SIZE_SEQ=block_size_seq,
    )


def triton_one_pass_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """Single-pass RMSNorm using Triton.

    This is an optimized RMSNorm implementation that processes the entire tensor
    in a single pass with tiled processing for better memory access patterns.

    Args:
        x: Input tensor of shape [*, hidden_size]
        w: Weight tensor of shape [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    shape = x.shape
    x = x.contiguous()
    y = torch.empty_like(x)
    x_view = x.reshape(-1, shape[-1])
    y_view = y.reshape(-1, shape[-1])
    S, D = x_view.shape

    BLOCK_SIZE_SEQ = min(16, triton.next_power_of_2(max(1, S // 512)))

    with torch.cuda.device(x.device):
        _triton_one_pass_rms_norm_impl(
            x_view,
            y_view,
            w,
            S,
            D,
            eps,
            triton.next_power_of_2(D),
            BLOCK_SIZE_SEQ,
        )
    return y


@triton.jit
def _rms_norm_kernel(
    Y,
    X,
    W,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Basic RMSNorm kernel.

    Args:
        Y: Output tensor pointer
        X: Input tensor pointer
        W: Weight tensor pointer
        stride: Stride for moving to next row
        N: Hidden dimension size
        eps: Epsilon for numerical stability
        BLOCK_SIZE: Block size for processing
    """
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # Compute variance
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    # RMS normalization
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Apply normalization and weight
    w = tl.load(W + cols, mask=cols < N, other=1.0).to(tl.float32)
    y = x * rstd * w

    tl.store(Y + cols, y, mask=cols < N)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Basic RMSNorm implementation using Triton.

    Args:
        x: Input tensor of shape [*, hidden_size]
        weight: Weight tensor of shape [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    shape = x.shape
    x = x.contiguous()
    y = torch.empty_like(x)

    # Flatten to 2D
    x_2d = x.view(-1, shape[-1])
    y_2d = y.view(-1, shape[-1])
    M, N = x_2d.shape

    # Determine block size
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    if N > BLOCK_SIZE:
        raise RuntimeError(f"This RMSNorm doesn't support feature dim >= {BLOCK_SIZE}.")

    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _rms_norm_kernel[(M,)](
        y_2d,
        x_2d,
        weight,
        x_2d.stride(0),
        N,
        eps,
        BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y


@triton.jit
def _fused_add_rms_norm_kernel(
    Y,
    X,
    W,
    Residual,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Add + RMSNorm kernel.

    Args:
        Y: Output tensor pointer
        X: Input tensor pointer
        W: Weight tensor pointer
        Residual: Residual tensor pointer (input and output)
        stride: Stride for moving to next row
        N: Hidden dimension size
        eps: Epsilon for numerical stability
        BLOCK_SIZE: Block size for processing
    """
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    Residual += row * stride

    cols = tl.arange(0, BLOCK_SIZE)

    # Load input and residual
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    residual = tl.load(Residual + cols, mask=cols < N, other=0.0).to(tl.float32)

    # Add residual
    x = x + residual

    # Store updated residual
    tl.store(Residual + cols, x, mask=cols < N)

    # RMS normalization
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Apply normalization and weight
    w = tl.load(W + cols, mask=cols < N, other=1.0).to(tl.float32)
    y = x * rstd * w

    tl.store(Y + cols, y, mask=cols < N)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Add + RMSNorm operation.

    Computes: residual = x + residual, y = rms_norm(residual, weight)
    This is more efficient than separate add and normalization.

    Args:
        x: Input tensor of shape [*, hidden_size]
        residual: Residual tensor (will be modified in-place)
        weight: Weight tensor of shape [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (normalized output, updated residual)
    """
    shape = x.shape
    x = x.contiguous()

    y = torch.empty_like(x)

    # Flatten to 2D
    x_2d = x.view(-1, shape[-1])
    y_2d = y.view(-1, shape[-1])
    residual_2d = residual.view(-1, shape[-1])
    M, N = x_2d.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    if N > BLOCK_SIZE:
        raise RuntimeError(f"This RMSNorm doesn't support feature dim >= {BLOCK_SIZE}.")

    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _fused_add_rms_norm_kernel[(M,)](
        y_2d,
        x_2d,
        weight,
        residual_2d,
        x_2d.stride(0),
        N,
        eps,
        BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y, residual
