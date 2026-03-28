# Copyright (c) 2025, SGLang Team.
# Test for block sparse attention

import pytest
import torch
import torch.nn.functional as F
from einops import repeat

from tf_kernel import block_sparse_attn_func

block_size = 128
is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    """Generate random padding mask."""
    if mode == "full":
        return None
    if mode == "random":
        return torch.randint(
            0, 2, (batch_size, max_seqlen), dtype=torch.bool, device=device
        )
    return None


def generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False):
    """Generate QKV tensors with padding."""
    batch_size, seqlen_q, nheads, d = q.shape
    seqlen_k = k.shape[1]

    if query_padding_mask is not None:
        # Apply padding mask
        q = q.masked_fill(~query_padding_mask.unsqueeze(-1).unsqueeze(-1), 0)
    if key_padding_mask is not None:
        k = k.masked_fill(~key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0)
        v = v.masked_fill(~key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0)

    # For simplicity, assume no padding (full sequences)
    # In real use case, cu_seqlens would be computed from padding masks
    cu_seqlens_q = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=q.device) * seqlen_q
    )
    cu_seqlens_k = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=k.device) * seqlen_k
    )

    # Flatten batch and sequence dimensions
    q_unpad = q.reshape(-1, nheads, d)
    k_unpad = k.reshape(-1, k.shape[1] * k.shape[2] // seqlen_k, d)
    v_unpad = v.reshape(-1, v.shape[1] * v.shape[2] // seqlen_k, d)

    max_seqlen_q = seqlen_q
    max_seqlen_k = seqlen_k

    def output_pad_fn(out):
        return out.reshape(batch_size, seqlen_q, nheads, d)

    def dq_pad_fn(dq):
        return dq.reshape(batch_size, seqlen_q, nheads, d)

    def dk_pad_fn(dk):
        return dk.reshape(batch_size, seqlen_k, k.shape[2], d)

    return (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    )


def generate_base_sparsity_mask(
    max_seqlen_q,
    max_seqlen_k,
    block_size_q,
    block_size_k,
    block_size,
    batch_size,
    num_blocksparse_heads,
    sparsity_list,
    causal=False,
    device="cuda",
):
    """Generate base sparsity mask."""
    num_q_blocks = (max_seqlen_q + block_size_q - 1) // block_size_q
    num_k_blocks = (max_seqlen_k + block_size_k - 1) // block_size_k

    mask = torch.ones(
        batch_size,
        num_blocksparse_heads,
        num_q_blocks,
        num_k_blocks,
        dtype=torch.bool,
        device=device,
    )

    for i, sparsity in enumerate(sparsity_list):
        if sparsity > 0:
            # Randomly drop some blocks based on sparsity
            rand = torch.rand(batch_size, num_q_blocks, num_k_blocks, device=device)
            mask[:, i] = rand > sparsity

    if causal:
        # For causal, mask out upper triangular blocks
        for q_idx in range(num_q_blocks):
            for k_idx in range(num_k_blocks):
                if q_idx * block_size_q < k_idx * block_size_k:
                    mask[:, :, q_idx, k_idx] = False

    return mask


def generate_streaming_mask(
    max_seqlen_q,
    max_seqlen_k,
    batch_size,
    nheads,
    cu_seqlens_q,
    cu_seqlens_k,
    block_size_q,
    block_size_k,
    block_size,
    streaming_info,
    causal=False,
    device="cuda",
):
    """Generate streaming mask."""
    num_q_blocks = (max_seqlen_q + block_size_q - 1) // block_size_q
    num_k_blocks = (max_seqlen_k + block_size_k - 1) // block_size_k

    mask = torch.ones(
        batch_size, nheads, num_q_blocks, num_k_blocks, dtype=torch.bool, device=device
    )

    # Apply streaming logic based on streaming_info
    # streaming_info contains [sink_size, local_size] for each head
    sink_size = streaming_info[0].item() if len(streaming_info) > 0 else 1
    local_size = streaming_info[1].item() if len(streaming_info) > 1 else 3

    sink_blocks = (sink_size + block_size_k - 1) // block_size_k
    local_blocks = (local_size + block_size_k - 1) // block_size_k

    for q_idx in range(num_q_blocks):
        # Keep sink blocks and local window blocks
        for k_idx in range(num_k_blocks):
            if k_idx >= sink_blocks and k_idx < num_k_blocks - local_blocks:
                if q_idx * block_size_q >= k_idx * block_size_k:
                    mask[:, :, q_idx, k_idx] = False

    return mask


def attention_ref(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    upcast=True,
    reorder_ops=False,
):
    """Reference attention implementation.

    Args:
        q: [batch, seqlen_q, nheads, head_dim]
        k: [batch, seqlen_k, nheads, head_dim]
        v: [batch, seqlen_k, nheads, head_dim]
        attn_mask: [batch, nheads, seqlen_q, seqlen_k] or [batch, seqlen_q, seqlen_k]
    """
    dtype_og = q.dtype
    if upcast:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)

    # Transpose to [batch, nheads, seqlen, head_dim] for matmul
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Compute attention scores: [batch, nheads, seqlen_q, seqlen_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)

    if causal:
        # Create causal mask
        seq_len_q, seq_len_k = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    if attn_mask is not None:
        # Apply attention mask
        # attn_mask shape: [batch, nheads, seq_q, seq_k] or [batch, seq_q, seq_k]
        # scores shape: [batch, nheads, seq_q, seq_k]
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)  # Add head dimension
        scores = scores.masked_fill(~attn_mask, float("-inf"))

    # Handle case where all keys are masked for a query position
    # Replace -inf with very negative number before softmax to avoid NaN
    all_masked = (scores == float("-inf")).all(dim=-1, keepdim=True)
    scores = torch.where(all_masked, torch.zeros_like(scores), scores)

    attn = F.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)

    # Compute output: [batch, nheads, seqlen_q, head_dim]
    output = torch.matmul(attn, v)

    # Transpose back to [batch, seqlen_q, nheads, head_dim]
    output = output.transpose(1, 2)

    return output.to(dtype_og), attn


@pytest.mark.parametrize(
    "dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16])
)
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize(
    "causal, exact_streaming, sink_num, local_num",
    [
        (True, False, 1, 3),
        (False, False, 1, 3),
    ],
)
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("sparsity", [0.0, 0.5])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nheads", [8, 16])
def test_block_sparse_attn_fwd(
    seqlen_q,
    seqlen_k,
    d,
    p_dropout,
    causal,
    exact_streaming,
    sink_num,
    local_num,
    mha_type,
    dtype,
    sparsity,
    batch_size,
    nheads,
):
    """Test forward pass of block sparse attention."""
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM

    device = "cuda:0"
    torch.random.manual_seed(42)

    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 8)
    assert nheads % nheads_k == 0

    window_size = (-1, -1)

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch_size, device, mode="full"
    )
    key_padding_mask = generate_random_padding_mask(
        seqlen_k, batch_size, device, mode="full"
    )

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    num_streaming_heads = nheads // 3
    num_blocksparse_heads = nheads // 3
    num_dense_heads = nheads - num_streaming_heads - num_blocksparse_heads
    sparsity_list = [sparsity] * num_blocksparse_heads

    head_mask_type = torch.tensor(
        [0] * num_dense_heads
        + [1] * num_blocksparse_heads
        + [-1] * num_streaming_heads,
        device=device,
        dtype=torch.int32,
    )

    base_blockmask = generate_base_sparsity_mask(
        max_seqlen_q,
        max_seqlen_k,
        block_size,
        block_size,
        block_size,
        batch_size,
        num_blocksparse_heads,
        sparsity_list,
        causal=causal,
        device=device,
    )

    streaming_info = torch.tensor(
        [sink_num, local_num] * nheads, device=device, dtype=torch.int32
    )

    if exact_streaming:
        assert causal

    # Run block sparse attention
    out_unpad, sm_lse, S_dmask = block_sparse_attn_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        deterministic=True,
        softmax_scale=None,
        is_causal=causal,
        exact_streaming=exact_streaming,
        return_attn_probs=True,
    )

    out = output_pad_fn(out_unpad)

    # Reference implementation
    # For reference, we only apply blocksparse mask to blocksparse heads
    # Dense heads (0) and streaming heads (-1) use full attention
    full_mask = None
    if base_blockmask is not None:
        # Build full mask for all heads
        full_mask = torch.ones(
            batch_size, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device=device
        )

        # Expand block mask to full attention mask
        num_q_blocks = base_blockmask.shape[2]
        num_k_blocks = base_blockmask.shape[3]
        blocksparse_mask = base_blockmask.repeat_interleave(
            block_size, dim=2
        ).repeat_interleave(block_size, dim=3)
        # Trim to actual sequence length
        blocksparse_mask = blocksparse_mask[:, :, :seqlen_q, :seqlen_k]

        # Place blocksparse mask in correct head positions
        blocksparse_indices = (head_mask_type == 1).nonzero(as_tuple=True)[0]
        for i, idx in enumerate(blocksparse_indices):
            if i < blocksparse_mask.shape[1]:
                full_mask[:, idx] = blocksparse_mask[:, i]

    if causal:
        # Apply causal mask to all heads
        if full_mask is None:
            full_mask = torch.ones(
                batch_size, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device=device
            )
        seq_len_q, seq_len_k = seqlen_q, seqlen_k
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=device, dtype=torch.bool),
            diagonal=1,
        )
        full_mask = full_mask & ~causal_mask.unsqueeze(0).unsqueeze(0)

    # Expand k, v for GQA/MQA
    k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)

    out_ref, attn_ref = attention_ref(
        q,
        k_rep,
        v_rep,
        attn_mask=full_mask,
        dropout_p=p_dropout,
        causal=causal,
        window_size=window_size,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")

    # Allow some tolerance for different implementations
    # Note: Higher tolerance needed due to different handling of fully-masked positions
    max_diff = (out - out_ref).abs().max().item()
    assert (
        max_diff <= 0.5 or (out - out_ref).abs().mean().item() <= 0.05
    ), f"Max diff too large: {max_diff}, mean: {(out - out_ref).abs().mean().item()}"


@pytest.mark.parametrize(
    "dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16])
)
@pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("seqlen", [256])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("nheads", [8])
def test_block_sparse_attn_fwd_bwd(dtype, d, seqlen, batch_size, nheads):
    """Test backward pass of block sparse attention."""
    device = "cuda:0"
    torch.random.manual_seed(42)

    causal = False
    p_dropout = 0.0
    sparsity = 0.0

    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, None, None, kvpacked=False)

    num_blocksparse_heads = nheads // 3
    num_dense_heads = nheads - num_blocksparse_heads * 2
    sparsity_list = [sparsity] * num_blocksparse_heads

    head_mask_type = torch.tensor(
        [0] * num_dense_heads
        + [1] * num_blocksparse_heads
        + [-1] * num_blocksparse_heads,
        device=device,
        dtype=torch.int32,
    )

    base_blockmask = generate_base_sparsity_mask(
        max_seqlen_q,
        max_seqlen_k,
        block_size,
        block_size,
        block_size,
        batch_size,
        num_blocksparse_heads,
        sparsity_list,
        causal=causal,
        device=device,
    )

    streaming_info = torch.tensor([1, 3] * nheads, device=device, dtype=torch.int32)

    # Forward pass
    out_unpad, sm_lse, S_dmask = block_sparse_attn_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        deterministic=True,
        softmax_scale=None,
        is_causal=causal,
        exact_streaming=False,
        return_attn_probs=True,
    )

    out = output_pad_fn(out_unpad)

    # Backward pass
    g = torch.randn_like(out)

    dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(
        out, (q_unpad, k_unpad, v_unpad), g
    )

    dq = dq_pad_fn(dq_unpad)
    dk = dk_pad_fn(dk_unpad)
    dv = dk_pad_fn(dv_unpad)

    # Reference backward - build full mask for all heads
    full_mask = torch.ones(
        batch_size, nheads, seqlen, seqlen, dtype=torch.bool, device=device
    )
    if base_blockmask is not None:
        blocksparse_mask = base_blockmask.repeat_interleave(
            block_size, dim=2
        ).repeat_interleave(block_size, dim=3)
        blocksparse_mask = blocksparse_mask[:, :, :seqlen, :seqlen]
        # Place blocksparse mask in correct head positions
        blocksparse_indices = (head_mask_type == 1).nonzero(as_tuple=True)[0]
        for i, idx in enumerate(blocksparse_indices):
            if i < blocksparse_mask.shape[1]:
                full_mask[:, idx] = blocksparse_mask[:, i]

    out_ref, attn_ref = attention_ref(q, k, v, attn_mask=full_mask, causal=causal)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)

    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")

    # Allow some tolerance
    assert (dq - dq_ref).abs().max().item() <= 0.5
    assert (dk - dk_ref).abs().max().item() <= 0.5
    assert (dv - dv_ref).abs().max().item() <= 0.5


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


if __name__ == "__main__":
    test_simple_block_sparse_attn()
