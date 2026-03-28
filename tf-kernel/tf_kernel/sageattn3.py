"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa

# Try to import triton modules
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    triton = None
    tl = None


if _TRITON_AVAILABLE:

    @triton.jit
    def group_mean_kernel(
        q_ptr,
        q_out_ptr,
        qm_out_ptr,
        B,
        H,
        L,
        D: tl.constexpr,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_qmb,
        stride_qmh,
        stride_qml,
        stride_qmd,
        GROUP_SIZE: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_group = tl.program_id(2)

        group_start = pid_group * GROUP_SIZE
        offsets = group_start + tl.arange(0, GROUP_SIZE)

        q_offsets = (
            pid_b * stride_qb
            + pid_h * stride_qh
            + offsets[:, None] * stride_ql
            + tl.arange(0, D)[None, :] * stride_qd
        )
        q_group = tl.load(q_ptr + q_offsets)

        qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE

        q_group = q_group - qm_group
        tl.store(q_out_ptr + q_offsets, q_group)

        qm_offset = (
            pid_b * stride_qmb
            + pid_h * stride_qmh
            + pid_group * stride_qml
            + tl.arange(0, D) * stride_qmd
        )
        tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    """Compute group mean using triton if available, otherwise use PyTorch."""
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE

    if _TRITON_AVAILABLE and num_groups > 0:
        q_out = torch.empty_like(q)  # [B, H, L, D]
        qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype)

        grid = (B, H, num_groups)

        group_mean_kernel[grid](
            q,
            q_out,
            qm,
            B,
            H,
            L,
            D,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            qm.stride(0),
            qm.stride(1),
            qm.stride(2),
            qm.stride(3),
            GROUP_SIZE=GROUP_SIZE,
        )
        return q_out, qm
    else:
        # Fallback to PyTorch implementation
        qm = q.mean(dim=-2, keepdim=True)
        q_out = q - qm
        return q_out, qm


def preprocess_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True
):

    def pad_128(x):
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()

    k -= k.mean(dim=-2, keepdim=True)
    q, k, v = map(lambda x: pad_128(x), [q, k, v])
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k, v, delta_s


def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize input tensor to FP4 format.

    Args:
        x: Input tensor of shape [B, H, N, D]

    Returns:
        packed_fp4: Packed FP4 tensor of shape [B, H, N, D // 2]
        fp8_scale: Scale tensor of shape [B, H, N, D // 16]
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty(
        (B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn
    )
    torch.ops.tf_kernel.sageattn3_scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize input tensor to FP4 format with permutation.

    Args:
        x: Input tensor of shape [B, H, N, D]

    Returns:
        packed_fp4: Packed FP4 tensor of shape [B, H, N, D // 2]
        fp8_scale: Scale tensor of shape [B, H, N, D // 16]
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty(
        (B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn
    )
    torch.ops.tf_kernel.sageattn3_scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize and transpose input tensor to FP4 format.

    Args:
        x: Input tensor of shape [B, H, N, D]

    Returns:
        packed_fp4: Packed FP4 tensor of shape [B, H, D, N // 2]
        fp8_scale: Scale tensor of shape [B, H, D, N // 16]
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty(
        (B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn
    )
    torch.ops.tf_kernel.sageattn3_scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def blockscaled_fp4_attn(
    qlist: Tuple,
    klist: Tuple,
    vlist: Tuple,
    delta_s: torch.Tensor,
    KL: int,
    is_causal: bool = False,
    per_block_mean: bool = True,
    is_bf16: bool = True,
    return_lse: bool = True,
):
    """Block-scaled FP4 attention computation.

    Args:
        qlist: Tuple of (q_packed, q_scale) for query
        klist: Tuple of (k_packed, k_scale) for key
        vlist: Tuple of (v_packed, v_scale) for value
        delta_s: Delta tensor for mean correction
        KL: Key sequence length
        is_causal: Whether to apply causal mask
        per_block_mean: Whether to use per-block mean
        is_bf16: Whether to use bfloat16 output
        return_lse: Whether to return softmax_lse

    Returns:
        If return_lse=True: returns (out, softmax_lse)
        If return_lse=False: returns out
    """
    B, H, N, D_half = qlist[0].shape
    D = D_half * 2
    dtype = torch.bfloat16 if is_bf16 else torch.float16
    out = torch.empty((B, H, N, D), device=qlist[0].device, dtype=dtype)
    softmax_scale = D ** (-0.5)
    lse = torch.ops.tf_kernel.sageattn3_fp4_attn(
        qlist[0],
        klist[0],
        vlist[0],
        qlist[1],
        klist[1],
        vlist[1],
        delta_s,
        KL,
        out,
        softmax_scale,
        is_causal,
        per_block_mean,
        is_bf16,
        1 if return_lse else 0,
    )
    if return_lse:
        return out, lse
    return out


def sageattn3_blackwell(
    q, k, v, attn_mask=None, is_causal=False, per_block_mean=True, **kwargs
):
    """SageAttention3 optimized for Blackwell (SM100+) GPUs using FP4 quantization.

    Args:
        q: Query tensor of shape [B, H, L, D]
        k: Key tensor of shape [B, H, L, D]
        v: Value tensor of shape [B, H, L, D]
        attn_mask: Optional attention mask
        is_causal: Whether to apply causal mask
        per_block_mean: Whether to use per-block mean

    Returns:
        Output tensor of shape [B, H, L, D]

    Note:
        This function requires FP4 support to be compiled with ENABLE_NVFP4=ON.
    """
    # Check if FP4 operations are available
    try:
        _ = torch.ops.tf_kernel.sageattn3_fp4_attn
    except AttributeError:
        raise RuntimeError(
            "SageAttention3 FP4 operations are not available. "
            "Please compile with TF_KERNEL_ENABLE_FP4=ON."
        )

    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal=is_causal)

    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16

    # Preprocess qkv: subtract mean from k, pad to multiple of 128
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)

    # Quantize to FP4
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)

    # Run FP4 attention
    o_fp4 = blockscaled_fp4_attn(
        qlist_from_cuda,
        klist_from_cuda,
        vlist_from_cuda,
        delta_s,
        KL,
        is_causal,
        per_block_mean,
        is_bf16,
        return_lse=False,
    )[:, :, :QL, :].contiguous()

    return o_fp4
