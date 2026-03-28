"""Ulysses All-to-All communication primitives for sequence parallelism.

This module provides communication primitives for Ulysses sequence parallelism,
which redistributes tensors between sequence-parallel and head-parallel layouts.

Key operations:
- scatter_heads: (B, local_seq, global_heads, D) -> (B, global_seq, local_heads, D)
- gather_heads: (B, global_seq, local_heads, D) -> (B, local_seq, global_heads, D)

The module handles head padding for cases where num_heads is not divisible by world_size.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.nn.functional as F


def _get_distributed_info(process_group: dist.ProcessGroup) -> tuple[int, int]:
    """Get rank and world size from process group.

    Args:
        process_group: PyTorch distributed process group

    Returns:
        Tuple of (rank, world_size)
    """
    world_size = dist.get_world_size(group=process_group)
    rank = dist.get_rank(group=process_group)
    return rank, world_size


def _compute_head_distribution(total_heads: int, world_size: int) -> list[int]:
    """Compute how heads are distributed across ranks.

    Args:
        total_heads: Total number of attention heads
        world_size: Number of ranks

    Returns:
        List of head counts per rank

    Example:
        total_heads=30, world_size=4 -> [8, 8, 7, 7]
    """
    base_heads = total_heads // world_size
    remainder = total_heads % world_size
    return [base_heads + 1 if i < remainder else base_heads for i in range(world_size)]


def _unpad_heads(tensor: torch.Tensor, padding_heads: int, is_last_rank: bool) -> torch.Tensor:
    """Remove head padding after communication.

    Args:
        tensor: Tensor with potentially padded head dimension
        padding_heads: Number of padded heads to remove
        is_last_rank: Whether this is the last rank

    Returns:
        Tensor with padding removed if applicable
    """
    if padding_heads > 0 and is_last_rank:
        return tensor[:, :, :-padding_heads, :].contiguous()
    return tensor


def _compute_padding(num_heads: int, world_size: int) -> tuple[int, int]:
    """Compute padding needed for head distribution.

    Args:
        num_heads: Total number of attention heads
        world_size: Number of ranks

    Returns:
        Tuple of (padding_heads, local_heads_after_padding)

    Raises:
        ValueError: If padding would exceed local head count
    """
    if num_heads % world_size == 0:
        return 0, num_heads // world_size

    padding_heads = world_size - (num_heads % world_size)
    local_heads = (num_heads + padding_heads) // world_size

    if padding_heads >= local_heads:
        raise ValueError(
            f"Cannot pad {padding_heads} heads: would exceed local head count {local_heads}. "
            "Consider reducing world_size or increasing heads."
        )
    return padding_heads, local_heads


def _wait_async_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Wait for async collective tensor to complete.

    For torch.compile compatibility with FakeTensor.

    Args:
        tensor: Potentially async collective tensor

    Returns:
        Synchronized tensor
    """
    if isinstance(tensor, fc.AsyncCollectiveTensor):
        tensor = tensor.wait()
    return tensor


def ulysses_scatter_heads(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    *,
    async_comm: bool = True,
) -> Callable[[], torch.Tensor]:
    """Scatter heads across ranks, gather sequence dimension.

    Transforms: (B, local_seq, global_heads, D) -> (B, global_seq, local_heads, D)

    Used for QKV tensors before attention computation in Ulysses sequence parallelism.
    Each rank holds a subset of sequence but all heads. After this operation,
    each rank holds full sequence but a subset of heads.

    Args:
        tensor: Input tensor of shape (batch, local_seq_len, global_heads, head_dim)
        process_group: Process group for communication
        async_comm: Use asynchronous communication (recommended for torch.compile)

    Returns:
        Callable that returns the scattered tensor when invoked

    Example:
        >>> q = torch.randn(2, 128, 32, 64)  # (B, S_local, H, D)
        >>> wait_fn = ulysses_scatter_heads(q, pg)
        >>> q_scattered = wait_fn()  # (B, S_global, H_local, D)
    """
    rank, world_size = _get_distributed_info(process_group)
    batch, local_seq_len, num_heads, head_dim = tensor.shape
    is_last_rank = rank == world_size - 1

    # Compute padding if heads not divisible by world_size
    padding_heads, local_heads = _compute_padding(num_heads, world_size)
    if padding_heads > 0:
        tensor = F.pad(tensor, (0, 0, 0, padding_heads))

    # All-to-All scatters the head dimension: each rank receives all sequence tokens
    # for its subset of heads
    tensor = tensor.reshape(batch, local_seq_len, world_size, local_heads, head_dim)
    tensor = tensor.permute(2, 1, 0, 3, 4).contiguous()
    comm_buffer_shape = tensor.shape

    if async_comm:
        tensor = tensor.flatten()
        tensor = fc.all_to_all_single(tensor, None, None, process_group)

        def wait() -> torch.Tensor:
            result = _wait_async_tensor(tensor)
            result = result.reshape(comm_buffer_shape)
            result = result.flatten(0, 1).permute(1, 0, 2, 3)
            return _unpad_heads(result, padding_heads, is_last_rank)

    else:
        tensor = tensor.flatten()
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor, None, None, group=process_group, async_op=False)

        def wait() -> torch.Tensor:
            result = output.reshape(comm_buffer_shape)
            result = result.flatten(0, 1).permute(1, 0, 2, 3)
            return _unpad_heads(result, padding_heads, is_last_rank)

    return wait


def ulysses_gather_heads(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    *,
    num_heads: int,
    async_comm: bool = True,
) -> Callable[[], torch.Tensor]:
    """Gather heads from ranks, scatter sequence dimension.

    Transforms: (B, global_seq, local_heads, D) -> (B, local_seq, global_heads, D)

    Used for output tensors after attention computation in Ulysses sequence parallelism.
    Each rank holds full sequence but a subset of heads. After this operation,
    each rank holds a subset of sequence but all heads.

    Args:
        tensor: Input tensor of shape (batch, global_seq_len, local_heads, head_dim)
        process_group: Process group for communication
        num_heads: Total number of attention heads (global)
        async_comm: Use asynchronous communication

    Returns:
        Callable that returns the gathered tensor when invoked

    Example:
        >>> out = torch.randn(2, 512, 8, 64)  # (B, S_global, H_local, D)
        >>> wait_fn = ulysses_gather_heads(out, pg, num_heads=32)
        >>> out_gathered = wait_fn()  # (B, S_local, H_global, D)
    """
    rank, world_size = _get_distributed_info(process_group)
    batch, global_seq_len, local_heads, head_dim = tensor.shape
    local_seq_len = global_seq_len // world_size
    is_last_rank = rank == world_size - 1

    # Compute padding if heads not divisible by world_size
    padding_heads, local_heads_padded = _compute_padding(num_heads, world_size)
    if padding_heads > 0 and is_last_rank:
        tensor = F.pad(tensor, (0, 0, 0, padding_heads))
        local_heads = local_heads_padded

    # All-to-All gathers the head dimension: each rank receives its subset of heads
    # for all sequence tokens
    tensor = tensor.reshape(batch, world_size, local_seq_len, local_heads, head_dim)
    tensor = tensor.permute(1, 3, 0, 2, 4).contiguous()
    comm_buffer_shape = tensor.shape

    if async_comm:
        tensor = tensor.flatten()
        tensor = fc.all_to_all_single(tensor, None, None, process_group)

        def wait() -> torch.Tensor:
            result = _wait_async_tensor(tensor)
            result = result.reshape(comm_buffer_shape)
            result = result.flatten(0, 1).permute(1, 2, 0, 3)
            # After gather, all ranks need to unpad (not just last rank)
            if padding_heads > 0:
                result = result[:, :, :-padding_heads, :].contiguous()
            return result

    else:
        tensor = tensor.flatten()
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor, None, None, group=process_group, async_op=False)

        def wait() -> torch.Tensor:
            result = output.reshape(comm_buffer_shape)
            result = result.flatten(0, 1).permute(1, 2, 0, 3)
            if padding_heads > 0:
                result = result[:, :, :-padding_heads, :].contiguous()
            return result

    return wait
