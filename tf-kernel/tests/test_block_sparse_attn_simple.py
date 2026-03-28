# Copyright (c) 2025, SGLang Team.
# Simple test for block sparse attention

import torch

from tf_kernel import block_sparse_attn_func


def test_simple_block_sparse_attn():
    """Simple sanity test for block sparse attention."""
    device = "cuda:0"
    torch.manual_seed(42)

    batch_size = 1
    seqlen = 256
    nheads = 8
    d = 64

    q = torch.randn(batch_size * seqlen, nheads, d, device=device, dtype=torch.float16)
    k = torch.randn(batch_size * seqlen, nheads, d, device=device, dtype=torch.float16)
    v = torch.randn(batch_size * seqlen, nheads, d, device=device, dtype=torch.float16)

    cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

    head_mask_type = torch.tensor([0] * nheads, device=device, dtype=torch.int32)
    streaming_info = torch.tensor([1, 3] * nheads, device=device, dtype=torch.int32)

    # No block mask (dense attention)
    out, sm_lse, S_dmask = block_sparse_attn_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        None,  # No block mask
        seqlen,
        seqlen,
        0.0,  # No dropout
        deterministic=True,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=True,
    )

    assert out.shape == q.shape
    print(f"Output shape: {out.shape}")
    print(f"Simple test passed!")


def test_block_sparse_attn_with_blockmask():
    """Test block sparse attention with block mask."""
    device = "cuda:0"
    torch.manual_seed(42)

    batch_size = 1
    seqlen = 256
    nheads = 8
    d = 64
    block_size = 128

    q = torch.randn(batch_size * seqlen, nheads, d, device=device, dtype=torch.float16)
    k = torch.randn(batch_size * seqlen, nheads, d, device=device, dtype=torch.float16)
    v = torch.randn(batch_size * seqlen, nheads, d, device=device, dtype=torch.float16)

    cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

    # For block mask test, we need some heads marked with 1 (blocksparse)
    # Head mask type: 0=dense, 1=blocksparse, -1=streaming
    head_mask_type = torch.tensor(
        [0, 0, 0, 1, 1, -1, -1, -1], device=device, dtype=torch.int32
    )
    streaming_info = torch.tensor([1, 3] * nheads, device=device, dtype=torch.int32)

    # Create a simple block mask for blocksparse heads only
    # Only 2 heads are blocksparse (indices 3 and 4)
    num_blocks = (seqlen + block_size - 1) // block_size
    base_blockmask = torch.ones(
        batch_size, 2, num_blocks, num_blocks, dtype=torch.bool, device=device
    )

    out, sm_lse, S_dmask = block_sparse_attn_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        base_blockmask,
        seqlen,
        seqlen,
        0.0,
        deterministic=True,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=True,
    )

    assert out.shape == q.shape
    print(f"Output shape with blockmask: {out.shape}")
    print(f"Blockmask test passed!")


def test_token_streaming_attn():
    """Test token streaming attention."""
    device = "cuda:0"
    torch.manual_seed(42)

    batch_size = 1
    seqlen = 256
    nheads = 8
    d = 64

    q = torch.randn(batch_size * seqlen, nheads, d, device=device, dtype=torch.float16)
    k = torch.randn(batch_size * seqlen, nheads, d, device=device, dtype=torch.float16)
    v = torch.randn(batch_size * seqlen, nheads, d, device=device, dtype=torch.float16)

    cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

    head_mask_type = torch.tensor([0] * nheads, device=device, dtype=torch.int32)
    streaming_info = torch.tensor([1, 3] * nheads, device=device, dtype=torch.int32)

    from tf_kernel import token_streaming_attn_func

    out, sm_lse, S_dmask = token_streaming_attn_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        seqlen,
        seqlen,
        deterministic=True,
        softmax_scale=None,
        return_attn_probs=True,
    )

    assert out.shape == q.shape
    print(f"Output shape (token streaming): {out.shape}")
    print(f"Token streaming test passed!")


if __name__ == "__main__":
    test_simple_block_sparse_attn()
    test_block_sparse_attn_with_blockmask()
    test_token_streaming_attn()
    print("\nAll tests passed!")
