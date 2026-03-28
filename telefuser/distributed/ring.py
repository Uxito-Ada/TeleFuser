"""Ring Attention communication primitives.

This module provides P2P communication utilities for Ring Attention,
which enables processing longer sequences by distributing KV across multiple GPUs.

Ring Attention Algorithm:
1. Each GPU holds a local chunk of Q, K, V
2. K and V rotate through the ring (each GPU passes to next, receives from prev)
3. Each GPU computes attention with its Q and the current KV chunk
4. Attention outputs are merged using online softmax (log-sum-exp)
5. After world_size steps, each GPU has seen all KV chunks

References:
- Ring Attention: https://arxiv.org/abs/2310.01889
- Online softmax merging: https://arxiv.org/abs/2111.09800
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    from telefuser.kernel import fused_merge_attn_states

    _is_triton_kernel_available = True
except ImportError:
    fused_merge_attn_states = None
    _is_triton_kernel_available = False


class RingP2PComm:
    """Ring-style P2P communication for Ring Attention.

    Manages the point-to-point communication pattern where each GPU passes
    its KV chunks to the next GPU and receives from the previous GPU in
    a circular ring topology.

    Ring topology for N GPUs:
    - Rank 0 sends to 1, receives from N-1
    - Rank i sends to (i+1)%N, receives from (i-1)%N
    - Rank N-1 sends to 0, receives from N-2
    """

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops: list[dist.P2POp] = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs: list[dist.Work] | None = None

        # Ring topology: send to next rank, receive from previous rank
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: torch.Tensor | None = None) -> torch.Tensor:
        """Queue send and receive operations for ring pattern.

        Operations are batched and executed together in commit() for efficiency.

        Args:
            to_send: Tensor to send to next rank
            recv_tensor: Optional pre-allocated receive buffer

        Returns:
            Receive buffer tensor
        """
        to_send = to_send.contiguous()
        if recv_tensor is None:
            res = torch.empty_like(to_send).contiguous()
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self) -> None:
        """Execute all queued P2P operations as a batch.

        Must be called before wait().
        """
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self) -> None:
        """Wait for all pending operations to complete.

        Must be called after commit(). Resets state for next iteration.
        """
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: torch.Tensor | None = None,
        v_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Send and receive K and V tensors separately.

        Args:
            k: Key tensor to send
            v: Value tensor to send
            k_buffer: Optional buffer for receiving K
            v_buffer: Optional buffer for receiving V

        Returns:
            Received (next_k, next_v) tensors
        """
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v

    def batch_send_recv_kv(
        self,
        k: torch.Tensor,  # (B, S_LOCAL, H, D)
        v: torch.Tensor,  # (B, S_LOCAL, H, D)
        kv_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch send and receive K and V tensors with single communication.

        Concatenates K and V along sequence dimension for efficient communication,
        then splits after receiving. Reduces communication overhead vs separate sends.

        Args:
            k: Key tensor (B, S_LOCAL, H, D)
            v: Value tensor (B, S_LOCAL, H, D)
            kv_buffer: Optional pre-allocated buffer for concatenated KV

        Returns:
            Received (next_k, next_v) tensors
        """
        S = k.size(1)

        # Concatenate k and v along sequence dimension
        kv_concat = torch.cat([k, v], dim=1)  # (B, S_LOCAL*2, H, D)
        kv_recv = self.send_recv(kv_concat, kv_buffer)
        self.commit()

        # Split received tensor back into k and v (views, no copy)
        next_k, next_v = torch.split(kv_recv, [S, S], dim=1)
        return next_k, next_v


def merge_attn_states(
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    cur_out: torch.Tensor,
    cur_lse: torch.Tensor,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge attention outputs from different KV chunks using online softmax.

    Implements the online softmax merge for Ring Attention, combining partial
    attention results computed with different KV chunks.

    Formula (following cache-dit implementation):
        out = prev_out - sigmoid(cur_lse - prev_lse) * (prev_out - cur_out)
        lse = prev_lse - logsigmoid(prev_lse - cur_lse)

    Args:
        prev_out: Previous attention output (B, S_Q, H, D)
        prev_lse: Previous log-sum-exp (B, H, S_Q) from Flash Attention
        cur_out: Current attention output (B, S_Q, H, D)
        cur_lse: Current log-sum-exp (B, H, S_Q) from Flash Attention
        use_triton: Whether to use optimized Triton kernel

    Returns:
        Merged output (B, S_Q, H, D) and merged lse (B, H, S_Q)
    """
    if use_triton and _is_triton_kernel_available and fused_merge_attn_states is not None:
        return fused_merge_attn_states(prev_out, prev_lse, cur_out, cur_lse)

    # Flash Attention returns lse with shape (B, H, S_Q)
    # Convert to (B, S_Q, H, 1) for broadcasting with output (B, S_Q, H, D)
    B, S, H, D = prev_out.shape

    # Transpose LSE from (B, H, S) to (B, S, H), then add dimension
    prev_lse_4d = prev_lse.transpose(1, 2).unsqueeze(-1)  # (B, S, H, 1)
    cur_lse_4d = cur_lse.transpose(1, 2).unsqueeze(-1)  # (B, S, H, 1)

    # Use float32 for numerical stability
    prev_out_f32 = prev_out.float()
    cur_out_f32 = cur_out.float()
    prev_lse_f32 = prev_lse_4d.float()
    cur_lse_f32 = cur_lse_4d.float()

    # Online softmax merge:
    # out = prev_out - sigmoid(cur_lse - prev_lse) * (prev_out - cur_out)
    # lse = prev_lse - logsigmoid(prev_lse - cur_lse)
    sigmoid_diff = torch.sigmoid(cur_lse_f32 - prev_lse_f32)
    log_sigmoid_diff = F.logsigmoid(prev_lse_f32 - cur_lse_f32)

    merged_out = prev_out_f32 - sigmoid_diff * (prev_out_f32 - cur_out_f32)
    merged_lse = prev_lse_f32 - log_sigmoid_diff

    # Remove extra dimension and transpose back to (B, H, S)
    merged_lse = merged_lse.squeeze(-1).transpose(1, 2)

    return merged_out.to(prev_out.dtype), merged_lse.to(prev_lse.dtype)


def ring_attention_forward(
    query: torch.Tensor,  # (B, S_Q_LOCAL, H, D)
    key: torch.Tensor,  # (B, S_KV_LOCAL, H, D)
    value: torch.Tensor,  # (B, S_KV_LOCAL, H, D)
    attention_fn: callable,
    process_group: dist.ProcessGroup,
    scale: float | None = None,
    is_causal: bool = False,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward pass for Ring Attention with P2P communication.

    Algorithm:
    1. Each GPU computes local attention with its Q and current KV chunk
    2. K and V are sent to next GPU, received from previous GPU
    3. Steps 1-2 repeat for world_size iterations (each GPU sees all KV)
    4. Attention outputs are merged using online softmax
    5. Communication overlaps with computation for efficiency

    Args:
        query: Query tensor (B, S_Q_LOCAL, H, D) - stays local
        key: Key tensor (B, S_KV_LOCAL, H, D) - rotates through ring
        value: Value tensor (B, S_KV_LOCAL, H, D) - rotates through ring
        attention_fn: Function to compute local attention (e.g., Flash Attention)
        process_group: Process group for ring communication
        scale: Attention scale factor
        is_causal: Whether to use causal masking
        return_lse: Whether to return log-sum-exp

    Returns:
        Tuple of output tensor and optionally lse tensor
    """
    comm = RingP2PComm(process_group)
    world_size = comm.world_size

    prev_out: torch.Tensor | None = None
    prev_lse: torch.Tensor | None = None

    for step in range(world_size):
        # Start sending current KV to next rank (except last step)
        if step + 1 != world_size:
            next_k, next_v = comm.batch_send_recv_kv(key, value)

        # Compute local attention with current KV chunk
        result = attention_fn(
            query,
            key,
            value,
            scale=scale,
            is_causal=is_causal,
            return_lse=return_lse,
        )

        if return_lse:
            out, lse = result
        else:
            out = result
            lse = None

        # Merge with previous results using online softmax
        if prev_out is not None and prev_lse is not None and lse is not None:
            out, lse = merge_attn_states(prev_out, prev_lse, out, lse)

        prev_out = out
        prev_lse = lse

        # Wait for communication and update KV for next iteration
        if step + 1 != world_size:
            comm.wait()
            key = next_k
            value = next_v

    if prev_out is None:
        raise RuntimeError("Ring attention failed to produce output")

    out = prev_out.to(query.dtype)
    if return_lse and prev_lse is not None:
        lse = prev_lse.squeeze(-1)  # (B, S_Q, H)
        return out, lse

    return out, None


def ring_attention_allgather_forward(
    query: torch.Tensor,  # (B, S_Q_LOCAL, H, D)
    key: torch.Tensor,  # (B, S_KV_LOCAL, H, D)
    value: torch.Tensor,  # (B, S_KV_LOCAL, H, D)
    attention_fn: callable,
    process_group: dist.ProcessGroup,
    scale: float | None = None,
    is_causal: bool = False,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Ring Attention using AllGather instead of P2P.

    This variant uses AllGather to collect all KV chunks before computing attention.
    Simpler implementation but requires more memory than the P2P variant.

    Memory: O(B * S_KV_GLOBAL * H * D) vs O(B * S_KV_LOCAL * H * D) for P2P

    Args:
        query: Query tensor (B, S_Q_LOCAL, H, D) - stays local
        key: Key tensor (B, S_KV_LOCAL, H, D) - will be gathered
        value: Value tensor (B, S_KV_LOCAL, H, D) - will be gathered
        attention_fn: Function to compute local attention
        process_group: Process group for ring communication
        scale: Attention scale factor
        is_causal: Whether to use causal masking
        return_lse: Whether to return log-sum-exp

    Returns:
        Tuple of output tensor and optionally lse tensor
    """
    world_size = dist.get_world_size(process_group)

    # Gather all K and V from all ranks
    gathered_k = [torch.empty_like(key) for _ in range(world_size)]
    gathered_v = [torch.empty_like(value) for _ in range(world_size)]

    dist.all_gather(gathered_k, key, group=process_group)
    dist.all_gather(gathered_v, value, group=process_group)

    # Concatenate along sequence dimension
    key_full = torch.cat(gathered_k, dim=1)  # (B, S_KV_GLOBAL, H, D)
    value_full = torch.cat(gathered_v, dim=1)  # (B, S_KV_GLOBAL, H, D)

    # Compute attention with full KV
    result = attention_fn(
        query,
        key_full,
        value_full,
        scale=scale,
        is_causal=is_causal,
        return_lse=return_lse,
    )

    if return_lse:
        out, lse = result
        return out, lse

    return result, None
