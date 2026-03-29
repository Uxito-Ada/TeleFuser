"""Local window sparse attention for video.

Implements window-based sparse attention where each spatial position
only attends to neighbors within a local window, reducing complexity
from O(N^2) to O(N * window_size).
"""

from __future__ import annotations

import math

import torch
from einops import rearrange

from telefuser.distributed.device_mesh import get_ulysses_group, get_ulysses_world_size
from telefuser.distributed.ulysses_comm import ulysses_gather_heads, ulysses_scatter_heads
from telefuser.ops.attention.sparse_sage import sparse_sageattn

# Priority: tf_kernel > block_sparse_attn > sparse_sageattn
_block_sparse_attn_func = None
try:
    from tf_kernel.block_sparse_attn import block_sparse_attn_func as _block_sparse_attn_func
except ImportError:
    try:
        from block_sparse_attn import block_sparse_attn_func as _block_sparse_attn_func
    except ImportError:
        pass


@torch.no_grad()
def build_local_block_mask(
    block_h: int,
    block_w: int,
    win_h: int = 6,
    win_w: int = 6,
    include_self: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build local window mask for 2D spatial blocks.

    Creates a binary mask where each block attends to neighbors within
    window [win_h, win_w].

    Args:
        block_h: Number of blocks in height dimension.
        block_w: Number of blocks in width dimension.
        win_h: Window half-height (total window = 2*win_h+1).
        win_w: Window half-width (total window = 2*win_w+1).
        include_self: Whether to include diagonal (self-attention).
        device: Target device for mask tensor.

    Returns:
        Boolean mask [block_h*block_w, block_h*block_w].
    """
    device = device or torch.device("cpu")
    H, W = block_h, block_w

    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)

    r_half, c_half = win_h // 2, win_w // 2
    start_r, end_r = r_all - r_half, r_all - r_half + win_h - 1
    start_c, end_c = c_all - c_half, c_all - c_half + win_w - 1

    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col

    if not include_self:
        mask.fill_diagonal_(False)

    return mask


class WindowPartition3D:
    """3D window partitioning for 5D video tensors (B, F, H, W, C)."""

    @staticmethod
    def partition(x: torch.Tensor, win: tuple[int, int, int]) -> torch.Tensor:
        """Partition tensor into 3D windows.

        Args:
            x: Input tensor [B, F, H, W, C].
            win: Window size (frames, height, width).

        Returns:
            Windowed tensor [B*num_windows, window_size, C].
        """
        B, F, H, W, C = x.shape
        wf, wh, ww = win

        if F % wf != 0 or H % wh != 0 or W % ww != 0:
            raise ValueError(f"Dimensions must divide by window: ({F},{H},{W}) % ({wf},{wh},{ww})")

        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(
        windows: torch.Tensor,
        win: tuple[int, int, int],
        orig: tuple[int, int, int],
    ) -> torch.Tensor:
        """Reverse window partition to original video shape.

        Args:
            windows: Windowed tensor [B*num_windows, window_size, C].
            win: Window size (frames, height, width).
            orig: Original spatial dimensions (F, H, W).

        Returns:
            Restored tensor [B, F, H, W, C].
        """
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)

        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


@torch.no_grad()
def _compute_block_scores(
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    nheads: int,
    seqlen: int,
    local_attn_mask: torch.Tensor,
    split_k: bool = False,
) -> torch.Tensor:
    """Compute block-level attention scores for mask generation.

    Args:
        q_w: Windowed queries.
        k_w: Windowed keys.
        nheads: Number of attention heads.
        seqlen: Sequence length (temporal frames).
        local_attn_mask: Base local window mask constraint.
        split_k: Split k_w into 2 parts for refined scoring.

    Returns:
        Attention scores [heads, q_seq, k_seq].
    """
    avgpool_q = torch.mean(q_w, dim=1)
    avgpool_q = rearrange(avgpool_q, "s (h d) -> s h d", h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    D = avgpool_q.shape[-1]

    if split_k:
        k_w_split = k_w.view(k_w.shape[0], 2, 64, k_w.shape[2])
        avgpool_k_split = torch.mean(k_w_split, dim=2)
        avgpool_k_refined = rearrange(avgpool_k_split, "s two d -> (s two) d", two=2)
        avgpool_k_refined = rearrange(avgpool_k_refined, "s (h d) -> s h d", h=nheads)
        k_heads_doubled = avgpool_k_refined.permute(1, 0, 2)
        k_heads_1, k_heads_2 = torch.chunk(k_heads_doubled, 2, dim=1)
        scores_1 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_1) / math.sqrt(D)
        scores_2 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_2) / math.sqrt(D)
        scores = torch.cat([scores_1, scores_2], dim=-1)
    else:
        avgpool_k = torch.mean(k_w, dim=1)
        avgpool_k = rearrange(avgpool_k, "s (h d) -> s h d", h=nheads)
        k_heads = avgpool_k.permute(1, 0, 2)
        scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    # Apply local mask constraint
    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = (scores.shape[2] // (2 if split_k else 1)) // local_attn_mask.shape[1]

    local_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    local_mask = rearrange(local_mask, "x a y b -> (x a) (y b)")
    if split_k:
        local_mask = local_mask.repeat_interleave(2, dim=1)
    local_mask = local_mask.unsqueeze(0).repeat(repeat_head, 1, 1)

    assert scores.shape == local_mask.shape, f"Scores shape {scores.shape} != Mask shape {local_mask.shape}"

    local_mask = local_mask.to(torch.float32)
    local_mask = local_mask.masked_fill(local_mask == False, float("-inf"))
    local_mask = local_mask.masked_fill(local_mask == True, 0)
    scores = scores + local_mask

    return scores


@torch.no_grad()
def generate_draft_block_mask(
    batch_size: int,
    nheads: int,
    seqlen: int,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    topk: int = 10,
    local_attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate sparse block mask using attention scores.

    Args:
        batch_size: Batch size (must be 1).
        nheads: Number of attention heads.
        seqlen: Sequence length (temporal frames).
        q_w: Windowed queries.
        k_w: Windowed keys.
        topk: Number of top blocks to select per query.
        local_attn_mask: Base local window mask constraint.

    Returns:
        Binary block mask [B, H, Q_blocks, K_blocks].
    """
    assert batch_size == 1, "Only batch_size=1 supported"
    assert local_attn_mask is not None, "local_attn_mask required"

    scores = _compute_block_scores(q_w, k_w, nheads, seqlen, local_attn_mask, split_k=False)

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, "h (it s1) s2 -> (h it) s1 s2", it=seqlen)

    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1] - 1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(mask_new, "(h it) s1 s2 -> h (it s1) s2", it=seqlen)

    return mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)


@torch.no_grad()
def generate_draft_block_mask_sage(
    batch_size: int,
    nheads: int,
    seqlen: int,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    topk: int = 10,
    local_attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate sparse block mask optimized for Sage attention.

    Splits k_w into 2 parts for refined attention score computation.

    Args:
        batch_size: Batch size (must be 1).
        nheads: Number of attention heads.
        seqlen: Sequence length (temporal frames).
        q_w: Windowed queries.
        k_w: Windowed keys.
        topk: Number of top blocks to select per query.
        local_attn_mask: Base local window mask constraint.

    Returns:
        Binary block mask [B, H, Q_blocks, K_blocks].
    """
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"

    scores = _compute_block_scores(q_w, k_w, nheads, seqlen, local_attn_mask, split_k=True)

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, "h (it s1) s2 -> (h it) s1 s2", it=seqlen)

    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1] - 1, topk)

    if apply_topk <= 0:
        mask_new = torch.zeros_like(flat, dtype=torch.bool).reshape(loop_num, s1, s2)
    else:
        thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
        thresholds = thresholds.unsqueeze(1)
        mask_new = (flat > thresholds).reshape(loop_num, s1, s2)

    mask_new = rearrange(mask_new, "(h it) s1 s2 -> h (it s1) s2", it=seqlen)
    return mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)


@torch.compiler.disable
def block_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block sparse attention with priority: tf_kernel > block_sparse_attn > sparse_sageattn.

    Args:
        q: Query tensor [B, S, H*D].
        k: Key tensor [B, S, H*D].
        v: Value tensor [B, S, H*D].
        num_heads: Number of attention heads.
        attention_mask: Block-sparse mask.

    Returns:
        Attention output [B, S, H*D].
    """
    if attention_mask is None:
        raise ValueError("attention_mask required for block_sparse_attn")

    seqlen = q.shape[1]
    seqlen_kv = k.shape[1]

    # Priority 1 & 2: tf_kernel or block_sparse_attn library (same interface)
    if _block_sparse_attn_func is not None:
        q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)

        cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32)
        head_mask_type = torch.tensor([1] * num_heads, device=q.device, dtype=torch.int32)

        x = _block_sparse_attn_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            None,
            attention_mask,
            seqlen,
            seqlen_kv,
            0.0,
            deterministic=False,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
        ).unsqueeze(0)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        return x

    # Priority 3: sparse_sageattn (fallback)
    # sparse_sageattn_fwd expects mask shape [B, H, ceil(S/128), ceil(S/64)].
    # generate_draft_block_mask produces [B, H, num_windows, num_windows] with
    # window_size=64 tokens, so Q-dim has twice as many rows as the kernel expects.
    # Merge adjacent Q-row pairs with OR so each row covers 128 tokens.
    B_m, H_m, Q_blocks, K_blocks = attention_mask.shape
    if Q_blocks % 2 == 0:
        merged = attention_mask.view(B_m, H_m, Q_blocks // 2, 2, K_blocks)
        attention_mask = merged.any(dim=3)

    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)

    x = sparse_sageattn(q, k, v, mask_id=attention_mask.to(torch.int8), is_causal=False)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


# Global cache for local window masks
BLOCK_MASK_MAP: dict[str, torch.Tensor] = {}


def local_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    B: int,
    f: int,
    h: int,
    w: int,
    D: int,
    topk: int,
    local_range: int,
    num_heads: int,
    kv_len: int,
    pre_cache_k: torch.Tensor | None = None,
    pre_cache_v: torch.Tensor | None = None,
    win: tuple[int, int, int] = (2, 8, 8),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local window sparse attention with KV caching.

    Partitions video into 3D windows, computes block-sparse attention
    using top-k selection constrained by local window.

    Args:
        q: Query tensor [B, S, D].
        k: Key tensor [B, S, D].
        v: Value tensor [B, S, D].
        B: Batch size.
        f: Number of frames.
        h: Height in tokens.
        w: Width in tokens.
        D: Head dimension (total = num_heads * head_dim).
        topk: Top-k blocks to select.
        local_range: Local window size for spatial constraint.
        num_heads: Number of attention heads.
        kv_len: Maximum KV cache length.
        pre_cache_k: Previous cached keys.
        pre_cache_v: Previous cached values.
        win: 3D window size (frames, height, width).

    Returns:
        Tuple of (attention output, updated cache_k, updated cache_v).
    """
    # Reshape to 5D video format
    q = q.view(B, f, h, w, D)
    k = k.view(B, f, h, w, D)
    v = v.view(B, f, h, w, D)

    # Partition into 3D windows
    q_w = WindowPartition3D.partition(q, win)
    k_w = WindowPartition3D.partition(k, win)
    v_w = WindowPartition3D.partition(v, win)

    seqlen = f // win[0]
    one_len = k_w.shape[0] // B // seqlen

    # Concatenate with cached KV
    if pre_cache_k is not None and pre_cache_v is not None:
        k_w = torch.cat([pre_cache_k.to(k_w.device), k_w], dim=0)
        v_w = torch.cat([pre_cache_v.to(v_w.device), v_w], dim=0)

    block_n = q_w.shape[0] // B
    block_s = q_w.shape[1]
    block_n_kv = k_w.shape[0] // B

    reorder_q = rearrange(q_w, "(b bn) bs d -> b (bn bs) d", b=B, bn=block_n, bs=block_s)
    reorder_k = rearrange(k_w, "(b bn) bs d -> b (bn bs) d", b=B, bn=block_n_kv, bs=block_s)
    reorder_v = rearrange(v_w, "(b bn) bs d -> b (bn bs) d", b=B, bn=block_n_kv, bs=block_s)

    # Get or build local window mask
    local_mask_key = f"{h // 8}-{w // 8}-{local_range}"
    if local_mask_key not in BLOCK_MASK_MAP:
        local_attn_mask = build_local_block_mask(
            h // win[1], w // win[2], local_range, local_range, include_self=True, device="cpu"
        )
        BLOCK_MASK_MAP[local_mask_key] = local_attn_mask
    else:
        local_attn_mask = BLOCK_MASK_MAP[local_mask_key]

    # Generate sparse block mask
    if _block_sparse_attn_func is not None:
        attention_mask = generate_draft_block_mask(
            B, num_heads, seqlen, q_w, k_w, topk=topk, local_attn_mask=local_attn_mask.to(k_w.device)
        )
    else:
        attention_mask = generate_draft_block_mask_sage(
            B, num_heads, seqlen, q_w, k_w, topk=topk, local_attn_mask=local_attn_mask.to(k_w.device)
        )

    # Compute block sparse attention
    x = block_sparse_attn(q=reorder_q, k=reorder_k, v=reorder_v, num_heads=num_heads, attention_mask=attention_mask)

    # Update KV cache
    cur_block_n, cur_block_s, _ = k_w.shape
    cache_num = cur_block_n // one_len
    if cache_num > kv_len:
        cache_k = k_w[one_len:, :, :]
        cache_v = v_w[one_len:, :, :]
    else:
        cache_k, cache_v = k_w, v_w

    # Restore window format
    x = rearrange(x, "b (bn bs) d -> (b bn) bs d", bn=block_n, bs=block_s)
    x = WindowPartition3D.reverse(x, win, (f, h, w))
    x = x.view(B, f * h * w, D)

    return x, cache_k, cache_v


def distributed_local_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    B: int,
    f: int,
    h: int,
    w: int,
    D: int,
    topk: int,
    local_range: int,
    num_heads: int,
    kv_len: int,
    pre_cache_k: torch.Tensor | None = None,
    pre_cache_v: torch.Tensor | None = None,
    win: tuple[int, int, int] = (2, 8, 8),
    device_mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local sparse attention with sequence parallelism (Ulysses).

    All-to-All communication distributes heads across GPUs.
    """
    sp_size = get_ulysses_world_size(device_mesh)
    if num_heads % sp_size != 0:
        raise ValueError(f"num_heads {num_heads} must divide by sp_size {sp_size}")

    q_4d = rearrange(q, "b s (h d) -> b s h d", h=num_heads)
    k_4d = rearrange(k, "b s (h d) -> b s h d", h=num_heads)
    v_4d = rearrange(v, "b s (h d) -> b s h d", h=num_heads)

    sp_group = get_ulysses_group(device_mesh=device_mesh)
    global_num_heads = num_heads  # Save original num_heads for gather_heads
    q_wait = ulysses_scatter_heads(q_4d, sp_group)
    k_wait = ulysses_scatter_heads(k_4d, sp_group)
    v_wait = ulysses_scatter_heads(v_4d, sp_group)
    q_4d = q_wait()
    k_4d = k_wait()
    v_4d = v_wait()

    q = rearrange(q_4d, "b s h d -> b s (h d)")
    k = rearrange(k_4d, "b s h d -> b s (h d)")
    v = rearrange(v_4d, "b s h d -> b s (h d)")
    num_heads = num_heads // sp_size
    D = D // sp_size

    x, cache_k, cache_v = local_sparse_attention(
        q,
        k,
        v,
        B,
        f,
        h,
        w,
        D,
        topk,
        local_range,
        num_heads,
        kv_len,
        pre_cache_k=pre_cache_k,
        pre_cache_v=pre_cache_v,
        win=win,
    )

    x_4d = rearrange(x, "b s (h d) -> b s h d", h=num_heads)
    x_wait = ulysses_gather_heads(x_4d, sp_group, num_heads=global_num_heads)
    x_4d = x_wait()
    x = rearrange(x_4d, "b s h d -> b s (h d)")

    return x, cache_k, cache_v


__all__ = [
    "build_local_block_mask",
    "WindowPartition3D",
    "generate_draft_block_mask",
    "generate_draft_block_mask_sage",
    "block_sparse_attn",
    "local_sparse_attention",
    "distributed_local_sparse_attention",
]
