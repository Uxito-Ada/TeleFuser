"""Triton kernel for merging attention states in ring attention.

Implements online softmax merging to combine partial attention results
from distributed computation across multiple GPUs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def fused_merge_attn_states(
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    suff_out: torch.Tensor,
    suff_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge attention states using Triton kernel.

    Combines partial attention outputs and their log-sum-exp values
    using online softmax for numerical stability.

    Args:
        prev_out: Previous attention output [B, N, H, D].
        prev_lse: Previous log-sum-exp [B, N, H, 1].
        suff_out: Current/suffix attention output [B, N, H, D].
        suff_lse: Current/suffix log-sum-exp [B, N, H, 1].

    Returns:
        Merged (output, log-sum-exp) tensors.
    """
    B, N, H, D = suff_out.shape

    # Flatten batch and sequence for parallel kernel launch
    prev_out = prev_out.flatten(0, 1).contiguous()
    suff_out = suff_out.flatten(0, 1).contiguous()
    prev_lse = prev_lse.flatten(0, 1).squeeze(-1).contiguous()
    suff_lse = suff_lse.flatten(0, 1).squeeze(-1).contiguous()

    out = torch.empty_like(suff_out).contiguous()
    lse = torch.empty_like(suff_lse).contiguous()

    _fused_merge_attn_states_kernel[(B * N, H)](
        out,
        lse,
        prev_out,
        prev_lse,
        suff_out,
        suff_lse,
        D,
        triton.next_power_of_2(D),
    )

    # Restore original shape
    out = out.view(B, N, H, D).contiguous()
    lse = lse.view(B, N, H, 1).contiguous()
    return out, lse


@triton.jit
def _fused_merge_attn_states_kernel(
    out_ptr: tl.tensor,
    lse_ptr: tl.tensor,
    prev_out_ptr: tl.tensor,
    prev_lse_ptr: tl.tensor,
    suff_out_ptr: tl.tensor,
    suff_lse_ptr: tl.tensor,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
):
    """Triton kernel for online softmax attention merging.

    Merges two partial attention results using:
    - lse = prev_lse - log(sigmoid(prev_lse - suff_lse))
    - out = prev_out - sigmoid(suff_lse - prev_lse) * (prev_out - suff_out)
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    # Use float32 for numerical stability in LSE computation
    prev_lse_val = tl.load(prev_lse_ptr + token_idx * num_heads + head_idx).to(tl.float32)
    suff_lse_val = tl.load(suff_lse_ptr + token_idx * num_heads + head_idx).to(tl.float32)

    # Handle infinity values
    prev_lse_val = float("-inf") if prev_lse_val == float("inf") else prev_lse_val
    suff_lse_val = float("-inf") if suff_lse_val == float("inf") else suff_lse_val

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE

    prev_out_val = tl.load(
        prev_out_ptr + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        mask=head_mask,
    ).to(tl.float32)
    suff_out_val = tl.load(
        suff_out_ptr + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        mask=head_mask,
    ).to(tl.float32)

    # Online softmax merge: weighted combination based on LSE difference
    out_val = prev_out_val - tl.sigmoid(suff_lse_val - prev_lse_val) * (prev_out_val - suff_out_val)
    out_val = out_val.to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        out_val,
        mask=head_mask,
    )

    # Merged log-sum-exp: combines statistics from both partial results
    lse_val = prev_lse_val - tl.log(tl.sigmoid(prev_lse_val - suff_lse_val))
    lse_val = lse_val.to(lse_ptr.dtype.element_ty)
    tl.store(lse_ptr + token_idx * num_heads + head_idx, lse_val)
