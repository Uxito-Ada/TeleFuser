"""Unit tests for long context attention with Ulysses, Ring, and USP strategies.

This module tests that distributed attention implementations produce
results consistent with local attention.

Run with pytest:
    pytest tests/unit/ops/test_long_context_attention.py -v

Or run directly:
    python tests/unit/ops/test_long_context_attention.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Skip entire module if distributed dependencies not available (CPU-only environment)
try:
    from telefuser.core.config import AttentionConfig, AttnImplType, ParallelConfig
    from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_attention_strategy
    from telefuser.ops.attention.attention_impl import attention, long_context_attention
except ImportError as e:
    pytest.skip(f"Skipping test module due to missing dependencies: {e}", allow_module_level=True)

# Mark all tests in this module as requiring distributed setup, GPU, and multiple GPUs
pytestmark = [
    pytest.mark.distributed,
    pytest.mark.gpu,
    pytest.mark.multi_gpu,
]

# Test configuration
TOLERANCE = 0.05  # Allow 5% tolerance for numerical differences
BATCH_SIZE = 1
SEQ_LEN = 1024
NUM_HEADS = 8
HEAD_DIM = 64


def init_distributed(rank, world_size):
    """Initialize distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed environment."""
    dist.destroy_process_group()


def create_test_tensors(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
):
    """Create test Q, K, V tensors."""
    torch.manual_seed(seed)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    return q, k, v


def local_attention_reference(q, k, v, scale=None, is_causal=False):
    """Reference local attention using PyTorch SDPA."""
    # SDPA expects (B, H, S, D) layout
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=is_causal)

    # Convert back to (B, S, H, D) layout
    return output.transpose(1, 2).contiguous()


def run_ulysses_test(rank, world_size, results_queue=None):
    """Run Ulysses attention test on a single rank."""
    init_distributed(rank, world_size)

    try:
        # Create device mesh for Ulysses
        parallel_config = ParallelConfig(
            device_ids=list(range(world_size)),
            sp_ulysses_degree=world_size,
        )
        device_mesh = create_device_mesh_from_config(parallel_config)

        local_seq_len = SEQ_LEN // world_size

        # Create global tensors on rank 0
        if rank == 0:
            q_global, k_global, v_global = create_test_tensors(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)
            ref_output = local_attention_reference(q_global, k_global, v_global)
        else:
            q_global = k_global = v_global = ref_output = None

        # Create local tensors
        q_local = torch.zeros(BATCH_SIZE, local_seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k_local = torch.zeros(BATCH_SIZE, local_seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v_local = torch.zeros(BATCH_SIZE, local_seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

        # Scatter from rank 0
        if rank == 0:
            q_chunks = q_global.chunk(world_size, dim=1)
            k_chunks = k_global.chunk(world_size, dim=1)
            v_chunks = v_global.chunk(world_size, dim=1)
            q_local.copy_(q_chunks[0])
            k_local.copy_(k_chunks[0])
            v_local.copy_(v_chunks[0])
            for i in range(1, world_size):
                dist.send(q_chunks[i].contiguous(), dst=i)
                dist.send(k_chunks[i].contiguous(), dst=i)
                dist.send(v_chunks[i].contiguous(), dst=i)
        else:
            dist.recv(q_local, src=0)
            dist.recv(k_local, src=0)
            dist.recv(v_local, src=0)

        dist.barrier()

        # Run distributed attention
        attn_config = AttentionConfig(attn_impl=AttnImplType.TORCH_SDPA)
        output = long_context_attention(
            q=q_local,
            k=k_local,
            v=v_local,
            attention_config=attn_config,
            device_mesh=device_mesh,
        )

        # Gather output to rank 0
        if rank == 0:
            output_global = torch.zeros(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            output_chunks = output_global.chunk(world_size, dim=1)
            output_chunks[0].copy_(output)
            for i in range(1, world_size):
                dist.recv(output_chunks[i], src=i)

            max_diff = (output_global - ref_output).abs().max().item()
            mean_diff = (output_global - ref_output).abs().mean().item()

            if results_queue is not None:
                results_queue.put(("ulysses", max_diff, mean_diff, max_diff < TOLERANCE))
        else:
            dist.send(output.contiguous(), dst=0)

        dist.barrier()

    finally:
        cleanup_distributed()


def run_ring_test(rank, world_size, results_queue=None):
    """Run Ring attention test on a single rank."""
    init_distributed(rank, world_size)

    try:
        # Create device mesh for Ring
        parallel_config = ParallelConfig(
            device_ids=list(range(world_size)),
            sp_ring_degree=world_size,
        )
        device_mesh = create_device_mesh_from_config(parallel_config)

        local_seq_len = SEQ_LEN // world_size

        # Create global tensors on rank 0
        if rank == 0:
            q_global, k_global, v_global = create_test_tensors(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)
            ref_output = local_attention_reference(q_global, k_global, v_global)
        else:
            q_global = k_global = v_global = ref_output = None

        # Create local tensors
        q_local = torch.zeros(BATCH_SIZE, local_seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k_local = torch.zeros(BATCH_SIZE, local_seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v_local = torch.zeros(BATCH_SIZE, local_seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

        # Scatter from rank 0
        if rank == 0:
            q_chunks = q_global.chunk(world_size, dim=1)
            k_chunks = k_global.chunk(world_size, dim=1)
            v_chunks = v_global.chunk(world_size, dim=1)
            q_local.copy_(q_chunks[0])
            k_local.copy_(k_chunks[0])
            v_local.copy_(v_chunks[0])
            for i in range(1, world_size):
                dist.send(q_chunks[i].contiguous(), dst=i)
                dist.send(k_chunks[i].contiguous(), dst=i)
                dist.send(v_chunks[i].contiguous(), dst=i)
        else:
            dist.recv(q_local, src=0)
            dist.recv(k_local, src=0)
            dist.recv(v_local, src=0)

        dist.barrier()

        # Run distributed attention
        # Ring attention requires Flash Attention for lse support
        attn_config = AttentionConfig(attn_impl=AttnImplType.FLASH_ATTN_2)
        output = long_context_attention(
            q=q_local,
            k=k_local,
            v=v_local,
            attention_config=attn_config,
            device_mesh=device_mesh,
        )

        # Gather output to rank 0
        if rank == 0:
            output_global = torch.zeros(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            output_chunks = output_global.chunk(world_size, dim=1)
            output_chunks[0].copy_(output)
            for i in range(1, world_size):
                dist.recv(output_chunks[i], src=i)

            max_diff = (output_global - ref_output).abs().max().item()
            mean_diff = (output_global - ref_output).abs().mean().item()

            if results_queue is not None:
                results_queue.put(("ring", max_diff, mean_diff, max_diff < TOLERANCE))
        else:
            dist.send(output.contiguous(), dst=0)

        dist.barrier()

    finally:
        cleanup_distributed()


def run_usp_test(rank, world_size, results_queue=None):
    """Run USP attention test on a single rank."""
    init_distributed(rank, world_size)

    try:
        # Create device mesh for USP (ring_degree=2, ulysses_degree=2)
        parallel_config = ParallelConfig(
            device_ids=list(range(world_size)),
            sp_ring_degree=2,
            sp_ulysses_degree=2,
        )
        device_mesh = create_device_mesh_from_config(parallel_config)

        ring_degree = 2
        ulysses_degree = 2
        local_seq_len = SEQ_LEN // ring_degree

        # Create global tensors on rank 0
        if rank == 0:
            q_global, k_global, v_global = create_test_tensors(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)
            ref_output = local_attention_reference(q_global, k_global, v_global)
        else:
            q_global = k_global = v_global = ref_output = None

        # Create local tensors
        q_local = torch.zeros(BATCH_SIZE, local_seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k_local = torch.zeros(BATCH_SIZE, local_seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v_local = torch.zeros(BATCH_SIZE, local_seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

        # Scatter from rank 0
        if rank == 0:
            q_seq_chunks = q_global.chunk(ring_degree, dim=1)
            k_seq_chunks = k_global.chunk(ring_degree, dim=1)
            v_seq_chunks = v_global.chunk(ring_degree, dim=1)

            q_local.copy_(q_seq_chunks[0])
            k_local.copy_(k_seq_chunks[0])
            v_local.copy_(v_seq_chunks[0])

            for i in range(1, world_size):
                ring_idx = i // ulysses_degree
                dist.send(q_seq_chunks[ring_idx].contiguous(), dst=i)
                dist.send(k_seq_chunks[ring_idx].contiguous(), dst=i)
                dist.send(v_seq_chunks[ring_idx].contiguous(), dst=i)
        else:
            dist.recv(q_local, src=0)
            dist.recv(k_local, src=0)
            dist.recv(v_local, src=0)

        dist.barrier()

        # Run distributed attention
        attn_config = AttentionConfig(attn_impl=AttnImplType.FLASH_ATTN_2)
        output = long_context_attention(
            q=q_local,
            k=k_local,
            v=v_local,
            attention_config=attn_config,
            device_mesh=device_mesh,
        )

        # Gather output to rank 0
        if rank == 0:
            output_global = torch.zeros(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            output_chunks = output_global.chunk(ring_degree, dim=1)
            output_chunks[0].copy_(output)

            for i in range(1, world_size):
                ring_idx = i // ulysses_degree
                temp = torch.zeros_like(output)
                dist.recv(temp, src=i)
                output_chunks[ring_idx].copy_(temp)

            max_diff = (output_global - ref_output).abs().max().item()
            mean_diff = (output_global - ref_output).abs().mean().item()

            if results_queue is not None:
                results_queue.put(("usp", max_diff, mean_diff, max_diff < TOLERANCE))
        else:
            dist.send(output.contiguous(), dst=0)

        dist.barrier()

    finally:
        cleanup_distributed()


class TestLongContextAttention:
    """Test class for long context attention."""

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    def test_ulysses_attention(self):
        """Test Ulysses attention with 2 GPUs."""
        world_size = 2
        results_queue = mp.Queue()

        mp.spawn(
            run_ulysses_test,
            args=(world_size, results_queue),
            nprocs=world_size,
            join=True,
        )

        strategy, max_diff, mean_diff, passed = results_queue.get()
        print(f"\n[Ulysses] Max difference: {max_diff:.6e}")
        print(f"[Ulysses] Mean difference: {mean_diff:.6e}")
        assert passed, f"Ulysses attention test failed: max_diff={max_diff} > tolerance={TOLERANCE}"

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    def test_ring_attention(self):
        """Test Ring attention with 2 GPUs."""
        world_size = 2
        results_queue = mp.Queue()

        mp.spawn(
            run_ring_test,
            args=(world_size, results_queue),
            nprocs=world_size,
            join=True,
        )

        strategy, max_diff, mean_diff, passed = results_queue.get()
        print(f"\n[Ring] Max difference: {max_diff:.6e}")
        print(f"[Ring] Mean difference: {mean_diff:.6e}")
        assert passed, f"Ring attention test failed: max_diff={max_diff} > tolerance={TOLERANCE}"

    @pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
    def test_usp_attention(self):
        """Test USP attention with 4 GPUs."""
        world_size = 4
        results_queue = mp.Queue()

        mp.spawn(
            run_usp_test,
            args=(world_size, results_queue),
            nprocs=world_size,
            join=True,
        )

        strategy, max_diff, mean_diff, passed = results_queue.get()
        print(f"\n[USP] Max difference: {max_diff:.6e}")
        print(f"[USP] Mean difference: {mean_diff:.6e}")
        assert passed, f"USP attention test failed: max_diff={max_diff} > tolerance={TOLERANCE}"


def main():
    """Run all tests directly."""
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    if num_gpus < 2:
        print("Skipping tests: requires at least 2 GPUs")
        return

    # Test Ulysses (2 GPUs)
    print("\n" + "=" * 60)
    print("Testing Ulysses Attention (2 GPUs)")
    print("=" * 60)
    results_queue = mp.Queue()
    mp.spawn(run_ulysses_test, args=(2, results_queue), nprocs=2, join=True)
    strategy, max_diff, mean_diff, passed = results_queue.get()
    print(f"[Ulysses] Max difference: {max_diff:.6e}")
    print(f"[Ulysses] Mean difference: {mean_diff:.6e}")
    print(f"[Ulysses] {'✓ PASSED' if passed else '✗ FAILED'}")

    # Test Ring (2 GPUs)
    print("\n" + "=" * 60)
    print("Testing Ring Attention (2 GPUs)")
    print("=" * 60)
    results_queue = mp.Queue()
    mp.spawn(run_ring_test, args=(2, results_queue), nprocs=2, join=True)
    strategy, max_diff, mean_diff, passed = results_queue.get()
    print(f"[Ring] Max difference: {max_diff:.6e}")
    print(f"[Ring] Mean difference: {mean_diff:.6e}")
    print(f"[Ring] {'✓ PASSED' if passed else '✗ FAILED'}")

    # Test USP (4 GPUs)
    if num_gpus >= 4:
        print("\n" + "=" * 60)
        print("Testing USP Attention (4 GPUs)")
        print("=" * 60)
        results_queue = mp.Queue()
        mp.spawn(run_usp_test, args=(4, results_queue), nprocs=4, join=True)
        strategy, max_diff, mean_diff, passed = results_queue.get()
        print(f"[USP] Max difference: {max_diff:.6e}")
        print(f"[USP] Mean difference: {mean_diff:.6e}")
        print(f"[USP] {'✓ PASSED' if passed else '✗ FAILED'}")
    else:
        print("\nSkipping USP test: requires 4 GPUs")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
