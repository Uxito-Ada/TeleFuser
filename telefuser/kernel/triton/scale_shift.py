"""Triton kernels for fused scale and shift operations.

Adapted from sglang diffusion kernels for TeleFuser.
Useful for video/image generation models with adaptive normalization.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .custom_op import register_custom_op


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
    ],
    key=["inner_dim"],
)
@triton.jit
def _fused_scale_shift_4d_kernel(
    output_ptr,
    normalized_ptr,
    scale_ptr,
    shift_ptr,
    scale_constant: tl.constexpr,
    rows,
    inner_dim,
    seq_len,
    num_frames,
    frame_seqlen,
    BLOCK_N: tl.constexpr,
):
    """Fused scale and shift kernel for 4D tensors.

    Args:
        output_ptr: Output tensor pointer
        normalized_ptr: Normalized input pointer
        scale_ptr: Scale tensor pointer (per-frame)
        shift_ptr: Shift tensor pointer (per-token)
        scale_constant: Constant to add to scale (0 or 1)
        rows: Total number of rows (B*L)
        inner_dim: Inner dimension (C)
        seq_len: Sequence length (L)
        num_frames: Number of frames (F)
        frame_seqlen: Sequence length per frame (L/F)
        BLOCK_N: Block size for inner dimension
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    col_offsets = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col_offsets < inner_dim

    row_base = pid_row * inner_dim
    norm_ptrs = normalized_ptr + row_base + col_offsets
    out_ptrs = output_ptr + row_base + col_offsets

    b_idx = pid_row // seq_len
    t_idx = pid_row % seq_len
    frame_idx_in_batch = t_idx // frame_seqlen

    scale_row_idx = b_idx * num_frames + frame_idx_in_batch
    scale_ptrs = scale_ptr + scale_row_idx * inner_dim + col_offsets
    shift_ptrs = shift_ptr + pid_row * inner_dim + col_offsets

    normalized = tl.load(norm_ptrs, mask=mask, other=0.0)
    scale = tl.load(scale_ptrs, mask=mask, other=0.0)
    shift = tl.load(shift_ptrs, mask=mask, other=0.0)

    scale_const_tensor = tl.full([BLOCK_N], scale_constant, dtype=scale.dtype)
    output = normalized * (scale_const_tensor + scale) + shift

    tl.store(out_ptrs, output, mask=mask)


@triton.jit
def _fuse_scale_shift_kernel_blc(
    x_ptr,
    shift_ptr,
    scale_ptr,
    scale_constant: tl.constexpr,
    y_ptr,
    B,
    L,
    C,
    stride_x_b,
    stride_x_l,
    stride_x_c,
    stride_s_b,
    stride_s_l,
    stride_s_c,
    stride_sc_b,
    stride_sc_l,
    stride_sc_c,
    SCALE_IS_SCALAR: tl.constexpr,
    SHIFT_IS_SCALAR: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused scale and shift kernel for BLC format tensors.

    Args:
        x_ptr: Input tensor pointer
        shift_ptr: Shift tensor pointer
        scale_ptr: Scale tensor pointer
        scale_constant: Constant to add to scale
        y_ptr: Output tensor pointer
        B: Batch size
        L: Sequence length
        C: Channel dimension
        stride_*: Various strides
        SCALE_IS_SCALAR: Whether scale is a scalar
        SHIFT_IS_SCALAR: Whether shift is a scalar
        BLOCK_L: Block size for sequence dimension
        BLOCK_C: Block size for channel dimension
    """
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_l = l_offsets < L
    mask_c = c_offsets < C
    mask = mask_l[:, None] & mask_c[None, :]

    x_off = pid_b * stride_x_b + l_offsets[:, None] * stride_x_l + c_offsets[None, :] * stride_x_c
    x = tl.load(x_ptr + x_off, mask=mask, other=0)

    if SHIFT_IS_SCALAR:
        shift_val = tl.load(shift_ptr)
        shift = tl.full((BLOCK_L, BLOCK_C), shift_val, dtype=shift_val.dtype)
    else:
        s_off = pid_b * stride_s_b + l_offsets[:, None] * stride_s_l + c_offsets[None, :] * stride_s_c
        shift = tl.load(shift_ptr + s_off, mask=mask, other=0)

    if SCALE_IS_SCALAR:
        scale_val = tl.load(scale_ptr)
        scale = tl.full((BLOCK_L, BLOCK_C), scale_val, dtype=scale_val.dtype)
    else:
        sc_off = pid_b * stride_sc_b + l_offsets[:, None] * stride_sc_l + c_offsets[None, :] * stride_sc_c
        scale = tl.load(scale_ptr + sc_off, mask=mask, other=0)

    y = x * (scale_constant + scale) + shift
    tl.store(y_ptr + x_off, y, mask=mask)


def fused_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float = 1.0,
    block_l: int = 128,
    block_c: int = 128,
) -> torch.Tensor:
    """Fused scale and shift operation.

    Computes: output = x * (scale_constant + scale) + shift

    Supports multiple broadcasting patterns:
    - scale/shift shape [B, F, 1, C] (per-frame, 4D)
    - scale/shift shape [B, C] or [1, C] (per-batch, 2D)
    - scale/shift shape [B, L, C] (per-token, 3D)
    - scale/shift scalar

    Note: This function uses Triton kernels and is wrapped with
    torch.compiler.disable for compile compatibility.

    Args:
        x: Input tensor of shape [B, L, C]
        scale: Scale tensor
        shift: Shift tensor
        scale_constant: Constant to add to scale (default 1.0)
        block_l: Block size for sequence dimension
        block_c: Block size for channel dimension

    Returns:
        Output tensor of same shape as input
    """
    return _fused_scale_shift_impl(x, scale, shift, scale_constant, block_l, block_c)


@register_custom_op(
    op_name="telefuser::fused_scale_shift",
    mutates_args=(),
)
def _fused_scale_shift_impl(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float,
    block_l: int,
    block_c: int,
) -> torch.Tensor:
    """Internal implementation of fused scale and shift using Triton kernels."""
    assert x.is_cuda and scale.is_cuda
    assert x.is_contiguous()

    B, L, C = x.shape
    output = torch.empty_like(x)

    if scale.dim() == 4:
        # scale/shift: [B, F, 1, C]
        rows = B * L
        x_2d = x.view(rows, C)
        output_2d = output.view(rows, C)
        grid = lambda META: (rows, triton.cdiv(C, META["BLOCK_N"]))  # noqa
        num_frames = scale.shape[1]
        assert L % num_frames == 0, "seq_len must be divisible by num_frames"
        frame_seqlen = L // num_frames

        scale_reshaped = scale.squeeze(2).reshape(-1, C).contiguous()
        shift_reshaped = shift.reshape(rows, C).contiguous()

        _fused_scale_shift_4d_kernel[grid](
            output_2d,
            x_2d,
            scale_reshaped,
            shift_reshaped,
            scale_constant,
            rows,
            C,
            L,
            num_frames,
            frame_seqlen,
        )
    else:
        # Handle 2D/3D scale/shift
        if scale.dim() == 0 or (scale.dim() == 1 and scale.numel() == 1):
            scale_blc = scale.reshape(1)
        elif scale.dim() == 2:
            scale_blc = scale[:, None, :]
        elif scale.dim() == 3:
            scale_blc = scale
        else:
            raise ValueError("scale must be 0D/1D(1)/2D/3D or 4D")

        if shift.dim() == 0 or (shift.dim() == 1 and shift.numel() == 1):
            shift_blc = shift.reshape(1)
        elif shift.dim() == 2:
            shift_blc = shift[:, None, :]
        elif shift.dim() == 3:
            shift_blc = shift
        else:
            shift_blc = shift

        need_scale_scalar = scale_blc.dim() == 1 and scale_blc.numel() == 1
        need_shift_scalar = shift_blc.dim() == 1 and shift_blc.numel() == 1

        if not need_scale_scalar:
            scale_exp = scale_blc.expand(B, L, C)
            s_sb, s_sl, s_sc = scale_exp.stride()
        else:
            s_sb = s_sl = s_sc = 0

        if not need_shift_scalar:
            shift_exp = shift_blc.expand(B, L, C)
            sh_sb, sh_sl, sh_sc = shift_exp.stride()
        else:
            sh_sb = sh_sl = sh_sc = 0

        if need_scale_scalar and need_shift_scalar:
            if not (scale_blc.any().to("cpu", non_blocking=True) or shift_blc.any().to("cpu", non_blocking=True)):
                output.copy_(x)
                return output

        grid = (triton.cdiv(L, block_l), triton.cdiv(C, block_c), B)
        _fuse_scale_shift_kernel_blc[grid](
            x,
            shift_blc if need_shift_scalar else shift_exp,
            scale_blc if need_scale_scalar else scale_exp,
            scale_constant,
            output,
            B,
            L,
            C,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            sh_sb,
            sh_sl,
            sh_sc,
            s_sb,
            s_sl,
            s_sc,
            SCALE_IS_SCALAR=need_scale_scalar,
            SHIFT_IS_SCALAR=need_shift_scalar,
            BLOCK_L=block_l,
            BLOCK_C=block_c,
            num_warps=4,
            num_stages=2,
        )
    return output


@triton.jit
def _fused_layernorm_scale_shift_gate_select01_kernel(
    output_ptr,
    gate_out_ptr,
    x_ptr,
    weight_ptr,
    bias_ptr,
    scale0_ptr,
    shift0_ptr,
    gate0_ptr,
    scale1_ptr,
    shift1_ptr,
    gate1_ptr,
    index_ptr,
    inner_dim,
    seq_len,
    stride_x_row,
    stride_out_row,
    stride_go_row,
    stride_w,
    stride_b,
    stride_s0_b,
    stride_s0_c,
    stride_sh0_b,
    stride_sh0_c,
    stride_g0_b,
    stride_g0_c,
    stride_s1_b,
    stride_s1_c,
    stride_sh1_b,
    stride_sh1_c,
    stride_g1_b,
    stride_g1_c,
    stride_i_b,
    stride_i_l,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused LayerNorm + scale/shift + gate selection kernel.

    Combines layernorm, scale/shift, and gate selection in a single kernel
    to reduce memory bandwidth for video generation models.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < inner_dim

    x_row_ptr = x_ptr + row * stride_x_row
    out_row_ptr = output_ptr + row * stride_out_row
    gate_row_ptr = gate_out_ptr + row * stride_go_row

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / inner_dim
    xbar = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / inner_dim
    rstd = tl.rsqrt(var + eps)
    x_hat = (x - mean) * rstd

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + cols * stride_w, mask=mask, other=1.0).to(tl.float32)
        x_hat = x_hat * w
    if HAS_BIAS:
        b = tl.load(bias_ptr + cols * stride_b, mask=mask, other=0.0).to(tl.float32)
        x_hat = x_hat + b

    batch_idx = row // seq_len
    seq_idx = row % seq_len
    idx = tl.load(index_ptr + batch_idx * stride_i_b + seq_idx * stride_i_l).to(tl.int1)

    scale0 = tl.load(
        scale0_ptr + batch_idx * stride_s0_b + cols * stride_s0_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    shift0 = tl.load(
        shift0_ptr + batch_idx * stride_sh0_b + cols * stride_sh0_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate0 = tl.load(
        gate0_ptr + batch_idx * stride_g0_b + cols * stride_g0_c,
        mask=mask,
        other=0.0,
    )

    scale1 = tl.load(
        scale1_ptr + batch_idx * stride_s1_b + cols * stride_s1_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    shift1 = tl.load(
        shift1_ptr + batch_idx * stride_sh1_b + cols * stride_sh1_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate1 = tl.load(
        gate1_ptr + batch_idx * stride_g1_b + cols * stride_g1_c,
        mask=mask,
        other=0.0,
    )

    scale = tl.where(idx, scale1, scale0)
    shift = tl.where(idx, shift1, shift0)
    gate = tl.where(idx, gate1, gate0)
    y = x_hat * (1.0 + scale) + shift

    tl.store(out_row_ptr + cols, y, mask=mask)
    tl.store(gate_row_ptr + cols, gate, mask=mask)


@triton.jit
def _fused_residual_layernorm_scale_shift_gate_select01_kernel(
    output_ptr,
    residual_out_ptr,
    gate_out_ptr,
    x_ptr,
    residual_ptr,
    residual_gate_ptr,
    weight_ptr,
    bias_ptr,
    scale0_ptr,
    shift0_ptr,
    gate0_ptr,
    scale1_ptr,
    shift1_ptr,
    gate1_ptr,
    index_ptr,
    inner_dim,
    seq_len,
    stride_x_row,
    stride_res_row,
    stride_rg_row,
    stride_out_row,
    stride_res_out_row,
    stride_go_row,
    stride_w,
    stride_b,
    stride_s0_b,
    stride_s0_c,
    stride_sh0_b,
    stride_sh0_c,
    stride_g0_b,
    stride_g0_c,
    stride_s1_b,
    stride_s1_c,
    stride_sh1_b,
    stride_sh1_c,
    stride_g1_b,
    stride_g1_c,
    stride_i_b,
    stride_i_l,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused residual add + LayerNorm + scale/shift + gate selection kernel.

    Combines residual addition with gated residual, layernorm, scale/shift,
    and gate selection in a single kernel for maximum memory efficiency.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < inner_dim

    x_row_ptr = x_ptr + row * stride_x_row
    res_row_ptr = residual_ptr + row * stride_res_row
    rg_row_ptr = residual_gate_ptr + row * stride_rg_row
    out_row_ptr = output_ptr + row * stride_out_row
    res_out_row_ptr = residual_out_ptr + row * stride_res_out_row
    gate_row_ptr = gate_out_ptr + row * stride_go_row

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(res_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual_gate = tl.load(rg_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual_out = residual + residual_gate * x
    tl.store(res_out_row_ptr + cols, residual_out, mask=mask)

    mean = tl.sum(residual_out, axis=0) / inner_dim
    xbar = tl.where(mask, residual_out - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / inner_dim
    rstd = tl.rsqrt(var + eps)
    x_hat = (residual_out - mean) * rstd

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + cols * stride_w, mask=mask, other=1.0).to(tl.float32)
        x_hat = x_hat * w
    if HAS_BIAS:
        b = tl.load(bias_ptr + cols * stride_b, mask=mask, other=0.0).to(tl.float32)
        x_hat = x_hat + b

    batch_idx = row // seq_len
    seq_idx = row % seq_len
    idx = tl.load(index_ptr + batch_idx * stride_i_b + seq_idx * stride_i_l).to(tl.int1)

    scale0 = tl.load(
        scale0_ptr + batch_idx * stride_s0_b + cols * stride_s0_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    shift0 = tl.load(
        shift0_ptr + batch_idx * stride_sh0_b + cols * stride_sh0_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate0 = tl.load(
        gate0_ptr + batch_idx * stride_g0_b + cols * stride_g0_c,
        mask=mask,
        other=0.0,
    )

    scale1 = tl.load(
        scale1_ptr + batch_idx * stride_s1_b + cols * stride_s1_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    shift1 = tl.load(
        shift1_ptr + batch_idx * stride_sh1_b + cols * stride_sh1_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate1 = tl.load(
        gate1_ptr + batch_idx * stride_g1_b + cols * stride_g1_c,
        mask=mask,
        other=0.0,
    )

    scale = tl.where(idx, scale1, scale0)
    shift = tl.where(idx, shift1, shift0)
    gate = tl.where(idx, gate1, gate0)
    y = x_hat * (1.0 + scale) + shift

    tl.store(out_row_ptr + cols, y, mask=mask)
    tl.store(gate_row_ptr + cols, gate, mask=mask)


def fused_layernorm_scale_shift_gate_select01(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused LayerNorm + scale/shift + gate selection operation.

    Combines layernorm, scale/shift application, and gate selection in a
    single kernel for better memory efficiency in video generation models.

    Args:
        x: Input tensor of shape [B, L, C]
        weight: Optional LayerNorm weight tensor of shape [C]
        bias: Optional LayerNorm bias tensor of shape [C]
        scale0: First scale tensor of shape [B, C]
        shift0: First shift tensor of shape [B, C]
        gate0: First gate tensor of shape [B, C]
        scale1: Second scale tensor of shape [B, C]
        shift1: Second shift tensor of shape [B, C]
        gate1: Second gate tensor of shape [B, C]
        index: Selection index tensor of shape [B, L] (bool or 0/1)
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (output tensor, selected gate tensor)
    """
    return _fused_layernorm_scale_shift_gate_select01_impl(
        x, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index, eps
    )


@register_custom_op(
    op_name="telefuser::fused_layernorm_scale_shift_gate_select01",
    mutates_args=(),
)
def _fused_layernorm_scale_shift_gate_select01_impl(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal implementation of fused LayerNorm + scale/shift + gate selection."""
    assert x.is_cuda
    assert x.is_contiguous()
    B, L, C = x.shape
    output = torch.empty_like(x)
    gate_out = torch.empty_like(x)

    if (
        scale0.dim() != 2
        or shift0.dim() != 2
        or gate0.dim() != 2
        or scale1.dim() != 2
        or shift1.dim() != 2
        or gate1.dim() != 2
    ):
        raise ValueError("scale0/shift0/gate0/scale1/shift1/gate1 must be 2D [B, C]")
    if index.dim() != 2:
        raise ValueError("index must be 2D [B, L]")
    if weight is not None and (weight.dim() != 1 or weight.shape[0] != C):
        raise ValueError("weight must be 1D [C]")
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != C):
        raise ValueError("bias must be 1D [C]")

    x_2d = x.view(B * L, C)
    output_2d = output.view(B * L, C)
    gate_out_2d = gate_out.view(B * L, C)
    weight = weight.contiguous() if weight is not None else x_2d
    bias = bias.contiguous() if bias is not None else x_2d

    MAX_FUSED_SIZE = 65536 // x_2d.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(C))
    if C > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    grid = (B * L,)
    _fused_layernorm_scale_shift_gate_select01_kernel[grid](
        output_2d,
        gate_out_2d,
        x_2d,
        weight,
        bias,
        scale0.contiguous(),
        shift0.contiguous(),
        gate0.contiguous(),
        scale1.contiguous(),
        shift1.contiguous(),
        gate1.contiguous(),
        index.contiguous(),
        C,
        L,
        x_2d.stride(0),
        output_2d.stride(0),
        gate_out_2d.stride(0),
        weight.stride(0) if weight.dim() == 1 else 0,
        bias.stride(0) if bias.dim() == 1 else 0,
        scale0.stride(0),
        scale0.stride(1),
        shift0.stride(0),
        shift0.stride(1),
        gate0.stride(0),
        gate0.stride(1),
        scale1.stride(0),
        scale1.stride(1),
        shift1.stride(0),
        shift1.stride(1),
        gate1.stride(0),
        gate1.stride(1),
        index.stride(0),
        index.stride(1),
        eps,
        HAS_WEIGHT=weight is not x_2d,
        HAS_BIAS=bias is not x_2d,
        BLOCK_N=BLOCK_N,
    )
    return output, gate_out


def fused_residual_layernorm_scale_shift_gate_select01(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused residual add + LayerNorm + scale/shift + gate selection operation.

    Combines gated residual addition, layernorm, scale/shift application,
    and gate selection in a single kernel for maximum memory efficiency.

    Args:
        x: Input tensor of shape [B, L, C]
        residual: Residual tensor of shape [B, L, C]
        residual_gate: Gate for residual addition of shape [B, L, C]
        weight: Optional LayerNorm weight tensor of shape [C]
        bias: Optional LayerNorm bias tensor of shape [C]
        scale0: First scale tensor of shape [B, C]
        shift0: First shift tensor of shape [B, C]
        gate0: First gate tensor of shape [B, C]
        scale1: Second scale tensor of shape [B, C]
        shift1: Second shift tensor of shape [B, C]
        gate1: Second gate tensor of shape [B, C]
        index: Selection index tensor of shape [B, L] (bool or 0/1)
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (output tensor, residual_out tensor, selected gate tensor)
    """
    return _fused_residual_layernorm_scale_shift_gate_select01_impl(
        x, residual, residual_gate, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index, eps
    )


@register_custom_op(
    op_name="telefuser::fused_residual_layernorm_scale_shift_gate_select01",
    mutates_args=(),
)
def _fused_residual_layernorm_scale_shift_gate_select01_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal implementation of fused residual + LayerNorm + scale/shift + gate selection."""
    assert x.is_cuda
    assert x.is_contiguous()
    assert residual.is_contiguous()
    assert residual_gate.is_contiguous()
    B, L, C = x.shape
    output = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    gate_out = torch.empty_like(x)

    if residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x")
    if residual_gate.shape != x.shape:
        raise ValueError("residual_gate must have the same shape as x")
    if (
        scale0.dim() != 2
        or shift0.dim() != 2
        or gate0.dim() != 2
        or scale1.dim() != 2
        or shift1.dim() != 2
        or gate1.dim() != 2
    ):
        raise ValueError("scale0/shift0/gate0/scale1/shift1/gate1 must be 2D [B, C]")
    if index.dim() != 2:
        raise ValueError("index must be 2D [B, L]")
    if weight is not None and (weight.dim() != 1 or weight.shape[0] != C):
        raise ValueError("weight must be 1D [C]")
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != C):
        raise ValueError("bias must be 1D [C]")

    x_2d = x.view(B * L, C)
    residual_2d = residual.view(B * L, C)
    residual_gate_2d = residual_gate.view(B * L, C)
    output_2d = output.view(B * L, C)
    residual_out_2d = residual_out.view(B * L, C)
    gate_out_2d = gate_out.view(B * L, C)
    weight = weight.contiguous() if weight is not None else x_2d
    bias = bias.contiguous() if bias is not None else x_2d

    MAX_FUSED_SIZE = 65536 // x_2d.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(C))
    if C > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    grid = (B * L,)
    _fused_residual_layernorm_scale_shift_gate_select01_kernel[grid](
        output_2d,
        residual_out_2d,
        gate_out_2d,
        x_2d,
        residual_2d,
        residual_gate_2d,
        weight,
        bias,
        scale0.contiguous(),
        shift0.contiguous(),
        gate0.contiguous(),
        scale1.contiguous(),
        shift1.contiguous(),
        gate1.contiguous(),
        index.contiguous(),
        C,
        L,
        x_2d.stride(0),
        residual_2d.stride(0),
        residual_gate_2d.stride(0),
        output_2d.stride(0),
        residual_out_2d.stride(0),
        gate_out_2d.stride(0),
        weight.stride(0) if weight.dim() == 1 else 0,
        bias.stride(0) if bias.dim() == 1 else 0,
        scale0.stride(0),
        scale0.stride(1),
        shift0.stride(0),
        shift0.stride(1),
        gate0.stride(0),
        gate0.stride(1),
        scale1.stride(0),
        scale1.stride(1),
        shift1.stride(0),
        shift1.stride(1),
        gate1.stride(0),
        gate1.stride(1),
        index.stride(0),
        index.stride(1),
        eps,
        HAS_WEIGHT=weight is not x_2d,
        HAS_BIAS=bias is not x_2d,
        BLOCK_N=BLOCK_N,
    )
    return output, residual_out, gate_out


@triton.jit
def _fuse_scale_shift_gate_kernel_blc(
    x_ptr,
    shift0_ptr,
    scale0_ptr,
    gate0_ptr,
    shift1_ptr,
    scale1_ptr,
    gate1_ptr,
    index_ptr,
    y_ptr,
    gate_out_ptr,
    B,
    L,
    C,
    stride_x_b,
    stride_x_l,
    stride_x_c,
    stride_s0_b,
    stride_s0_c,
    stride_sc0_b,
    stride_sc0_c,
    stride_g0_b,
    stride_g0_c,
    stride_s1_b,
    stride_s1_c,
    stride_sc1_b,
    stride_sc1_c,
    stride_g1_b,
    stride_g1_c,
    stride_i_b,
    stride_i_l,
    stride_go_b,
    stride_go_l,
    stride_go_c,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused scale, shift, and gate selection kernel.

    Selects between two sets of scale/shift/gate based on index tensor.
    """
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_l = l_offsets < L
    mask_c = c_offsets < C
    mask = mask_l[:, None] & mask_c[None, :]

    x_off = pid_b * stride_x_b + l_offsets[:, None] * stride_x_l + c_offsets[None, :] * stride_x_c
    x = tl.load(x_ptr + x_off, mask=mask, other=0)

    idx_off = pid_b * stride_i_b + l_offsets * stride_i_l
    idx = tl.load(index_ptr + idx_off, mask=mask_l, other=0).to(tl.int1)[:, None]

    s0_off = pid_b * stride_s0_b + c_offsets[None, :] * stride_s0_c
    sc0_off = pid_b * stride_sc0_b + c_offsets[None, :] * stride_sc0_c
    g0_off = pid_b * stride_g0_b + c_offsets[None, :] * stride_g0_c
    s1_off = pid_b * stride_s1_b + c_offsets[None, :] * stride_s1_c
    sc1_off = pid_b * stride_sc1_b + c_offsets[None, :] * stride_sc1_c
    g1_off = pid_b * stride_g1_b + c_offsets[None, :] * stride_g1_c

    shift0 = tl.load(shift0_ptr + s0_off, mask=mask_c[None, :], other=0)
    scale0 = tl.load(scale0_ptr + sc0_off, mask=mask_c[None, :], other=0)
    gate0 = tl.load(gate0_ptr + g0_off, mask=mask_c[None, :], other=0)
    shift1 = tl.load(shift1_ptr + s1_off, mask=mask_c[None, :], other=0)
    scale1 = tl.load(scale1_ptr + sc1_off, mask=mask_c[None, :], other=0)
    gate1 = tl.load(gate1_ptr + g1_off, mask=mask_c[None, :], other=0)

    shift = tl.where(idx, shift1, shift0)
    scale = tl.where(idx, scale1, scale0)
    gate = tl.where(idx, gate1, gate0)

    y = x * (1 + scale) + shift
    tl.store(y_ptr + x_off, y, mask=mask)

    go_off = pid_b * stride_go_b + l_offsets[:, None] * stride_go_l + c_offsets[None, :] * stride_go_c
    tl.store(gate_out_ptr + go_off, gate, mask=mask)


def fused_scale_shift_gate_select(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    block_l: int = 128,
    block_c: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused scale, shift, and gate selection operation.

    Selects between two sets of (scale, shift, gate) based on index.
    Used in video generation models for adaptive normalization.

    Args:
        x: Input tensor of shape [B, L, C]
        scale0: First scale tensor of shape [B, C]
        shift0: First shift tensor of shape [B, C]
        gate0: First gate tensor of shape [B, C]
        scale1: Second scale tensor of shape [B, C]
        shift1: Second shift tensor of shape [B, C]
        gate1: Second gate tensor of shape [B, C]
        index: Selection index tensor of shape [B, L] (bool or 0/1)
        block_l: Block size for sequence dimension
        block_c: Block size for channel dimension

    Returns:
        Tuple of (output tensor, selected gate tensor)
    """
    return _fused_scale_shift_gate_select_impl(x, scale0, shift0, gate0, scale1, shift1, gate1, index, block_l, block_c)


@register_custom_op(
    op_name="telefuser::fused_scale_shift_gate_select",
    mutates_args=(),
)
def _fused_scale_shift_gate_select_impl(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    block_l: int,
    block_c: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal implementation of fused scale, shift, and gate selection."""
    assert x.is_contiguous()
    B, L, C = x.shape
    output = torch.empty_like(x)
    gate_out = torch.empty_like(x)

    if (
        scale0.dim() != 2
        or shift0.dim() != 2
        or gate0.dim() != 2
        or scale1.dim() != 2
        or shift1.dim() != 2
        or gate1.dim() != 2
    ):
        raise ValueError("scale0/shift0/gate0/scale1/shift1/gate1 must be 2D [B, C]")
    if index.dim() != 2:
        raise ValueError("index must be 2D [B, L]")

    grid = (triton.cdiv(L, block_l), triton.cdiv(C, block_c), B)
    _fuse_scale_shift_gate_kernel_blc[grid](
        x,
        shift0,
        scale0,
        gate0,
        shift1,
        scale1,
        gate1,
        index,
        output,
        gate_out,
        B,
        L,
        C,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        shift0.stride(0),
        shift0.stride(1),
        scale0.stride(0),
        scale0.stride(1),
        gate0.stride(0),
        gate0.stride(1),
        shift1.stride(0),
        shift1.stride(1),
        scale1.stride(0),
        scale1.stride(1),
        gate1.stride(0),
        gate1.stride(1),
        index.stride(0),
        index.stride(1),
        gate_out.stride(0),
        gate_out.stride(1),
        gate_out.stride(2),
        BLOCK_L=block_l,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )
    return output, gate_out
