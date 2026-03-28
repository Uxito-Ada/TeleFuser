# Copyright (c) 2025, SGLang Team.
# Adapted from https://github.com/mit-han-lab/Block-Sparse-Attention

import functools

import torch


@functools.lru_cache(maxsize=128)
def _get_block_size(device_idx: int, head_dim: int, is_dropout: bool, is_causal: bool):
    """Get block size for block sparse attention.

    This function is cached to avoid repeated device queries during torch.compile.
    The cache key includes device index, head_dim, is_dropout, and is_causal.

    Args:
        device_idx: Device index (note: actual device object is not hashable, so we use index)
        head_dim: Head dimension
        is_dropout: Whether dropout is enabled
        is_causal: Whether causal mask is used

    Returns:
        Tuple of (m_block_dim, n_block_dim)
    """
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device_idx)
    is_sm8x = (
        major == 8 and minor > 0
    )  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128, 128
    if head_dim <= 64:
        return (128, 128) if not is_dropout else (128, 64)
    elif head_dim <= 96:
        return (64, 64) if (is_sm8x and is_causal) else (128, 64)
    elif head_dim <= 128:
        if is_sm8x:
            return (64, 64) if (not is_dropout and is_causal) else (128, 32)
        else:
            return 128, (64 if not is_dropout else 32)
    elif head_dim <= 160:
        if is_sm8x:
            return (128, 64) if not is_causal else (64, 64)
        else:
            return 128, 32
    elif head_dim <= 192:
        return (128, 64) if not is_dropout else (64, 64)
    elif head_dim <= 224:
        return (128, 64) if (is_sm80 or is_sm90) else (64, 64)
    elif head_dim <= 256:
        return (128, 64) if is_sm80 else (64, 64)


def convert_blockmask_row_reverse(blockmask, causal=False):
    """Convert blockmask to row-major format for forward pass."""
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(
        dim=-1, stable=True, descending=False
    )

    nonzero_idx = nonzero_sorted_rowidx
    nonzero_idx[nonzero_val == 0] = -1
    nonzero_idx = torch.flip(nonzero_idx, dims=[-1])

    return nonzero_idx.contiguous().to(dtype=torch.int32)


def convert_blockmask_col_reverse(blockmask, causal=False):
    """Convert blockmask to column-major format for backward pass."""
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(
        dim=-2, stable=True, descending=False
    )

    nonzero_idx = nonzero_sorted_rowidx
    nonzero_idx[nonzero_val == 0] = -1
    nonzero_idx = torch.flip(nonzero_idx, dims=[-2])
    nonzero_idx = torch.transpose(nonzero_idx, -1, -2)

    return nonzero_idx.contiguous().to(dtype=torch.int32)


def replace_ones_with_count(tensor):
    """Replace consecutive 1s with their counts."""
    ones_mask = tensor == 1
    ones_num = ones_mask.sum()
    count = torch.cumsum(ones_mask, dim=-1).to(tensor.dtype)
    count = count * ones_mask
    tensor = tensor.masked_scatter(ones_mask, count[ones_mask])
    return tensor, ones_num


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


class BlockSparseAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        m_block_dim,
        n_block_dim,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q_,
        max_seqlen_k_,
        p_dropout,
        softmax_scale,
        is_causal,
        exact_streaming,
        return_softmax,
        window_size_left,
        window_size_right,
        deterministic=False,
        is_grad_enabled=False,
    ):

        # Save rng_state because the backward pass will regenerate the dropout mask
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        if base_blockmask is not None:
            row_blockmask = convert_blockmask_row_reverse(base_blockmask, is_causal)
        else:
            row_blockmask = None

        if exact_streaming:
            assert streaming_info is not None
            assert is_causal

        # Call torch.ops.tf_kernel.block_sparse_attn_fwd
        out, softmax_lse, S_dmask, rng_state = (
            torch.ops.tf_kernel.block_sparse_attn_fwd(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                head_mask_type,
                streaming_info,
                row_blockmask,
                max_seqlen_q_,
                max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                window_size_left,
                window_size_right,
                m_block_dim,
                n_block_dim,
                exact_streaming,
                return_softmax and p_dropout > 0,
            )
        )

        if is_grad:
            ctx.save_for_backward(
                q,
                k,
                v,
                out,
                S_dmask,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                head_mask_type,
                streaming_info,
                base_blockmask,
                rng_state,
            )
            ctx.m_block_dim = m_block_dim
            ctx.n_block_dim = n_block_dim
            ctx.window_size_left = window_size_left
            ctx.window_size_right = window_size_right
            ctx.max_seqlen_q_ = max_seqlen_q_
            ctx.max_seqlen_k_ = max_seqlen_k_
            ctx.p_dropout = p_dropout
            ctx.softmax_scale = softmax_scale
            ctx.is_causal = is_causal
            ctx.exact_streaming = exact_streaming
            ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            S_dmask,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            base_blockmask,
            rng_state,
        ) = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        if base_blockmask is not None:
            col_blockmask = convert_blockmask_col_reverse(base_blockmask, ctx.is_causal)
        else:
            col_blockmask = None

        assert not ctx.exact_streaming, "Exact streaming not supported in backward pass"

        # Call torch.ops.tf_kernel.block_sparse_attn_bwd
        dq_out, dk_out, dv_out, softmax_d = torch.ops.tf_kernel.block_sparse_attn_bwd(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            col_blockmask,
            ctx.max_seqlen_q_,
            ctx.max_seqlen_k_,
            ctx.p_dropout,
            ctx.softmax_scale,
            True,  # zero_tensors
            ctx.is_causal,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.m_block_dim,
            ctx.n_block_dim,
            ctx.deterministic,
            rng_state,
        )

        dq = dq_out[..., : dout.shape[-1]]
        dk = dk_out[..., : dout.shape[-1]]
        dv = dv_out[..., : dout.shape[-1]]

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def block_sparse_attn_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    head_mask_type,
    streaming_info,
    base_blockmask,
    max_seqlen_q_,
    max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
):
    """Block sparse attention function.

    Args:
        q: Query tensor (total_q x num_heads x head_size)
        k: Key tensor (total_k x num_heads_k x head_size)
        v: Value tensor (total_k x num_heads_k x head_size)
        cu_seqlens_q: Cumulative sequence lengths for query (batch_size + 1)
        cu_seqlens_k: Cumulative sequence lengths for key (batch_size + 1)
        head_mask_type: Head mask type tensor (num_heads,)
        streaming_info: Streaming info tensor (optional)
        base_blockmask: Base block mask tensor (optional)
        max_seqlen_q_: Maximum sequence length for query
        max_seqlen_k_: Maximum sequence length for key
        p_dropout: Dropout probability
        deterministic: Whether to use deterministic mode
        softmax_scale: Softmax scale factor
        is_causal: Whether to use causal mask
        exact_streaming: Whether to use exact streaming
        return_attn_probs: Whether to return attention probabilities

    Returns:
        Output tensor or tuple of (output, softmax_lse, attn_probs)
    """
    head_mask_type, blocksparse_head_num = replace_ones_with_count(head_mask_type)
    if base_blockmask is not None:
        assert base_blockmask.shape[1] == blocksparse_head_num

    return BlockSparseAttnFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        128,
        128,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q_,
        max_seqlen_k_,
        p_dropout,
        softmax_scale,
        is_causal,
        exact_streaming,
        return_attn_probs,
        -1,
        -1,
        deterministic,
        torch.is_grad_enabled(),
    )


def token_streaming_attn_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    head_mask_type,
    streaming_info,
    max_seqlen_q_,
    max_seqlen_k_,
    deterministic=False,
    softmax_scale=None,
    return_attn_probs=False,
):
    """Token streaming attention function."""
    return BlockSparseAttnFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        128,
        128,
        head_mask_type,
        streaming_info,
        None,
        max_seqlen_q_,
        max_seqlen_k_,
        0.0,
        softmax_scale,
        True,
        True,
        return_attn_probs,
        -1,
        -1,
        deterministic,
        torch.is_grad_enabled(),
    )


def block_streaming_attn_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    head_mask_type,
    streaming_info,
    max_seqlen_q_,
    max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=True,
    return_attn_probs=False,
):
    """Block streaming attention function."""
    return BlockSparseAttnFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        128,
        128,
        head_mask_type,
        streaming_info,
        None,
        max_seqlen_q_,
        max_seqlen_k_,
        p_dropout,
        softmax_scale,
        is_causal,
        False,
        return_attn_probs,
        -1,
        -1,
        deterministic,
        torch.is_grad_enabled(),
    )
