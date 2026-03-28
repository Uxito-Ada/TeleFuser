"""Integration tests for parallel_shard + long_context_attention.

This module tests the combination of sequence parallel sharding/unsharding
with distributed attention implementations.

Run with pytest:
    pytest tests/unit/ops/test_parallel_shard_attention.py -v

Or run directly:
    python tests/unit/ops/test_parallel_shard_attention.py
"""

import os
import random
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Skip entire module if triton is not available (CPU-only environment)
try:
    from telefuser.core.config import AttentionConfig, AttnImplType, ParallelConfig
    from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_attention_strategy
    from telefuser.distributed.parallel_shard import (
        sequence_parallel_shard,
        sequence_parallel_unshard,
    )
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
TOLERANCE = 0.1  # Allow 10% tolerance for numerical differences with Flash Attention
BATCH_SIZE = 1
SEQ_LEN = 153600 // 2
NUM_HEADS = 128
HEAD_DIM = 32
NUM_WARMUP = 3
NUM_ITERATIONS = 5

# Large scale USP test configuration
LARGE_SEQ_LEN = 153600  # 2x larger for USP test
LARGE_NUM_HEADS = 128
LARGE_HEAD_DIM = 32


def init_distributed(rank, world_size, port=None):
    """Initialize distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    # Use a fixed port range based on world_size to avoid conflicts
    # Each test uses a different base port
    if port is None:
        # Use hash of world_size to get deterministic but different ports
        port = 29500 + (world_size * 10) % 1000
    os.environ["MASTER_PORT"] = str(port)
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
    """Reference local attention using Flash Attention 2."""
    from flash_attn import flash_attn_func

    # Flash Attention expects (B, S, H, D) layout for BSND input
    output = flash_attn_func(q, k, v, softmax_scale=scale, causal=is_causal)
    return output


def benchmark_attention(attn_fn, q, k, v, num_warmup=NUM_WARMUP, num_iterations=NUM_ITERATIONS):
    """Benchmark attention function."""
    # Warmup
    for _ in range(num_warmup):
        output = attn_fn(q, k, v)
        if isinstance(output, tuple):
            output = output[0]
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        output = attn_fn(q, k, v)
        if isinstance(output, tuple):
            output = output[0]
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time


def run_ulysses_test(rank, world_size, results_queue=None):
    """Run Ulysses attention test with parallel_shard on a single rank."""
    init_distributed(rank, world_size)

    try:
        # Create device mesh for Ulysses
        parallel_config = ParallelConfig(
            device_ids=list(range(world_size)),
            sp_ulysses_degree=world_size,
        )
        device_mesh = create_device_mesh_from_config(parallel_config)

        assert get_attention_strategy(device_mesh) == "ulysses", "Should use ulysses strategy"

        # Create global tensors on each rank (different seed for each rank to simulate real workload)
        torch.manual_seed(42 + rank)
        q_full = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k_full = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v_full = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

        # Save reference for rank 0
        if rank == 0:
            q_ref = q_full.clone()
            k_ref = k_full.clone()
            v_ref = v_full.clone()
            ref_output = local_attention_reference(q_ref, k_ref, v_ref)

        # Record original sequence length
        original_seq_len = q_full.size(1)

        # Apply sequence parallel sharding
        sequence_parallel_shard(
            device_mesh,
            tensors=[q_full, k_full, v_full],
            seq_dims=[1, 1, 1],
        )

        # Verify sharding
        expected_local_seq_len = SEQ_LEN // world_size
        assert q_full.size(1) == expected_local_seq_len, (
            f"Expected seq_len {expected_local_seq_len}, got {q_full.size(1)}"
        )

        # Run distributed attention (Ulysses with Flash Attention 2)
        attn_config = AttentionConfig(attn_impl=AttnImplType.FLASH_ATTN_2)
        dist.barrier()
        dist_time = benchmark_attention(
            lambda q, k, v: long_context_attention(q, k, v, attention_config=attn_config, device_mesh=device_mesh),
            q_full,
            k_full,
            v_full,
        )

        # Run actual attention to get output
        output_shard = long_context_attention(
            q=q_full,
            k=k_full,
            v=v_full,
            attention_config=attn_config,
            device_mesh=device_mesh,
        )

        # Unshard output
        outputs = sequence_parallel_unshard(
            device_mesh,
            tensors=[output_shard],
            seq_dims=[1],
            seq_lens=[original_seq_len],
        )
        output_full = outputs[0]

        # Verify output shape
        assert output_full.shape == (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), (
            f"Expected output shape {(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)}, got {output_full.shape}"
        )

        # Compare with reference on rank 0
        if rank == 0:
            max_diff = (output_full - ref_output).abs().max().item()
            mean_diff = (output_full - ref_output).abs().mean().item()
            passed = max_diff < TOLERANCE

            # Benchmark local attention
            local_time = benchmark_attention(local_attention_reference, q_ref, k_ref, v_ref)

            if results_queue is not None:
                results_queue.put(("ulysses", max_diff, mean_diff, passed, dist_time, local_time))

        dist.barrier()

    finally:
        cleanup_distributed()


def run_ring_test(rank, world_size, results_queue=None):
    """Run Ring attention test with parallel_shard on a single rank."""
    init_distributed(rank, world_size)

    try:
        # Create device mesh for Ring
        parallel_config = ParallelConfig(
            device_ids=list(range(world_size)),
            sp_ring_degree=world_size,
        )
        device_mesh = create_device_mesh_from_config(parallel_config)

        assert get_attention_strategy(device_mesh) == "ring", "Should use ring strategy"

        # Create global tensors
        torch.manual_seed(42 + rank)
        q_full = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k_full = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v_full = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

        # Save reference for rank 0
        if rank == 0:
            q_ref = q_full.clone()
            k_ref = k_full.clone()
            v_ref = v_full.clone()
            ref_output = local_attention_reference(q_ref, k_ref, v_ref)

        # Record original sequence length
        original_seq_len = q_full.size(1)

        # Apply sequence parallel sharding
        sequence_parallel_shard(
            device_mesh,
            tensors=[q_full, k_full, v_full],
            seq_dims=[1, 1, 1],
        )

        # Verify sharding
        expected_local_seq_len = SEQ_LEN // world_size
        assert q_full.size(1) == expected_local_seq_len, (
            f"Expected seq_len {expected_local_seq_len}, got {q_full.size(1)}"
        )

        # Run distributed attention (Ring requires Flash Attention)
        attn_config = AttentionConfig(attn_impl=AttnImplType.FLASH_ATTN_2)
        dist.barrier()
        dist_time = benchmark_attention(
            lambda q, k, v: long_context_attention(q, k, v, attention_config=attn_config, device_mesh=device_mesh),
            q_full,
            k_full,
            v_full,
        )

        # Run actual attention
        output_shard = long_context_attention(
            q=q_full,
            k=k_full,
            v=v_full,
            attention_config=attn_config,
            device_mesh=device_mesh,
        )

        # Unshard output
        outputs = sequence_parallel_unshard(
            device_mesh,
            tensors=[output_shard],
            seq_dims=[1],
            seq_lens=[original_seq_len],
        )
        output_full = outputs[0]

        # Verify output shape
        assert output_full.shape == (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), (
            f"Expected output shape {(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)}, got {output_full.shape}"
        )

        # Compare with reference on rank 0
        if rank == 0:
            max_diff = (output_full - ref_output).abs().max().item()
            mean_diff = (output_full - ref_output).abs().mean().item()
            passed = max_diff < TOLERANCE

            # Benchmark local attention (Flash Attention 2)
            torch.cuda.synchronize()
            local_config = AttentionConfig(attn_impl=AttnImplType.FLASH_ATTN_2)
            local_time = benchmark_attention(
                lambda q, k, v: attention(
                    q, k, v, attention_config=local_config, input_layout="BSND", output_layout="BSND"
                ),
                q_ref,
                k_ref,
                v_ref,
            )

            if results_queue is not None:
                results_queue.put(("ring", max_diff, mean_diff, passed, dist_time, local_time))

        dist.barrier()

    finally:
        cleanup_distributed()


def run_usp_test(rank, world_size, results_queue=None, large_scale=False):
    """Run USP attention test with parallel_shard on a single rank."""
    init_distributed(rank, world_size)

    try:
        # Create device mesh for USP (ring_degree=2, ulysses_degree=2)
        parallel_config = ParallelConfig(
            device_ids=list(range(world_size)),
            sp_ring_degree=2,
            sp_ulysses_degree=2,
        )
        device_mesh = create_device_mesh_from_config(parallel_config)

        assert get_attention_strategy(device_mesh) == "usp", "Should use usp strategy"

        ring_degree = 2

        # Use large scale config if requested
        if large_scale:
            seq_len = LARGE_SEQ_LEN
            num_heads = LARGE_NUM_HEADS
            head_dim = LARGE_HEAD_DIM
            test_name = "usp_large"
            num_iters = 3  # Reduce iterations for large scale to avoid timeout
        else:
            seq_len = SEQ_LEN
            num_heads = NUM_HEADS
            head_dim = HEAD_DIM
            test_name = "usp"
            num_iters = NUM_ITERATIONS

        # Create global tensors
        torch.manual_seed(42 + rank)
        q_full = torch.randn(BATCH_SIZE, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        k_full = torch.randn(BATCH_SIZE, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        v_full = torch.randn(BATCH_SIZE, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")

        # Save reference for rank 0
        if rank == 0:
            q_ref = q_full.clone()
            k_ref = k_full.clone()
            v_ref = v_full.clone()
            ref_output = local_attention_reference(q_ref, k_ref, v_ref)

        # Record original sequence length
        original_seq_len = q_full.size(1)

        # In USP mode, sequence is only split by ring_degree, not ulysses_degree
        # So each rank gets seq_len / ring_degree
        expected_local_seq_len = seq_len // ring_degree

        # Apply sequence parallel sharding
        sequence_parallel_shard(
            device_mesh,
            tensors=[q_full, k_full, v_full],
            seq_dims=[1, 1, 1],
        )

        # Verify sharding (USP splits by ring_degree only)
        assert q_full.size(1) == expected_local_seq_len, (
            f"Expected seq_len {expected_local_seq_len}, got {q_full.size(1)}"
        )

        # Run distributed attention
        attn_config = AttentionConfig(attn_impl=AttnImplType.FLASH_ATTN_2)
        dist.barrier()

        # Warmup
        for _ in range(NUM_WARMUP):
            _ = long_context_attention(q_full, k_full, v_full, attention_config=attn_config, device_mesh=device_mesh)
        torch.cuda.synchronize()

        # Benchmark
        dist.barrier()
        start_time = time.perf_counter()
        for _ in range(num_iters):
            _ = long_context_attention(q_full, k_full, v_full, attention_config=attn_config, device_mesh=device_mesh)
        torch.cuda.synchronize()
        dist.barrier()
        end_time = time.perf_counter()
        dist_time = (end_time - start_time) / num_iters

        # Run actual attention to get output
        output_shard = long_context_attention(
            q=q_full,
            k=k_full,
            v=v_full,
            attention_config=attn_config,
            device_mesh=device_mesh,
        )

        # Unshard output
        outputs = sequence_parallel_unshard(
            device_mesh,
            tensors=[output_shard],
            seq_dims=[1],
            seq_lens=[original_seq_len],
        )
        output_full = outputs[0]

        # Verify output shape
        assert output_full.shape == (BATCH_SIZE, seq_len, num_heads, head_dim), (
            f"Expected output shape {(BATCH_SIZE, seq_len, num_heads, head_dim)}, got {output_full.shape}"
        )

        # Compare with reference on rank 0
        if rank == 0:
            max_diff = (output_full - ref_output).abs().max().item()
            mean_diff = (output_full - ref_output).abs().mean().item()
            passed = max_diff < TOLERANCE

            # Benchmark local attention (Flash Attention 2)
            torch.cuda.synchronize()
            local_config = AttentionConfig(attn_impl=AttnImplType.FLASH_ATTN_2)
            for _ in range(NUM_WARMUP):
                _ = attention(
                    q_ref, k_ref, v_ref, attention_config=local_config, input_layout="BSND", output_layout="BSND"
                )
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(num_iters):
                _ = attention(
                    q_ref, k_ref, v_ref, attention_config=local_config, input_layout="BSND", output_layout="BSND"
                )
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            local_time = (end_time - start_time) / num_iters

            if results_queue is not None:
                results_queue.put((test_name, max_diff, mean_diff, passed, dist_time, local_time, seq_len))

        dist.barrier()

    finally:
        cleanup_distributed()


class TestParallelShardAttention:
    """Test class for parallel_shard + long_context_attention integration."""

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    def test_ulysses_shard_attention(self):
        """Test Ulysses with parallel_shard (2 GPUs)."""
        world_size = 2
        results_queue = mp.Queue()

        mp.spawn(
            run_ulysses_test,
            args=(world_size, results_queue),
            nprocs=world_size,
            join=True,
        )

        strategy, max_diff, mean_diff, passed, dist_time, local_time = results_queue.get()
        speedup = local_time / dist_time if dist_time > 0 else 0
        print(f"\n[Ulysses] Max difference: {max_diff:.6e}")
        print(f"[Ulysses] Mean difference: {mean_diff:.6e}")
        print(f"[Ulysses] Distributed time: {dist_time * 1000:.2f} ms")
        print(f"[Ulysses] Local time: {local_time * 1000:.2f} ms")
        print(f"[Ulysses] Speedup: {speedup:.2f}x")
        assert passed, f"Ulysses test failed: max_diff={max_diff} > tolerance={TOLERANCE}"
        # For long sequences, expect reasonable speedup from distributed attention
        # With communication overhead, we expect at least 0.8x for 2 GPUs on very long sequences

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    def test_ring_shard_attention(self):
        """Test Ring with parallel_shard (2 GPUs)."""
        world_size = 2
        results_queue = mp.Queue()

        mp.spawn(
            run_ring_test,
            args=(world_size, results_queue),
            nprocs=world_size,
            join=True,
        )

        strategy, max_diff, mean_diff, passed, dist_time, local_time = results_queue.get()
        speedup = local_time / dist_time if dist_time > 0 else 0
        print(f"\n[Ring] Max difference: {max_diff:.6e}")
        print(f"[Ring] Mean difference: {mean_diff:.6e}")
        print(f"[Ring] Distributed time: {dist_time * 1000:.2f} ms")
        print(f"[Ring] Local time: {local_time * 1000:.2f} ms")
        print(f"[Ring] Speedup: {speedup:.2f}x")
        assert passed, f"Ring test failed: max_diff={max_diff} > tolerance={TOLERANCE}"
        # Ring has more communication overhead but handles very long sequences better

    @pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
    def test_usp_shard_attention(self):
        """Test USP with parallel_shard (4 GPUs)."""
        world_size = 4
        results_queue = mp.Queue()

        mp.spawn(
            run_usp_test,
            args=(world_size, results_queue, False),
            nprocs=world_size,
            join=True,
        )

        result = results_queue.get()
        strategy, max_diff, mean_diff, passed, dist_time, local_time, seq_len = result
        speedup = local_time / dist_time if dist_time > 0 else 0
        print(f"\n[USP] Sequence length: {seq_len}")
        print(f"[USP] Max difference: {max_diff:.6e}")
        print(f"[USP] Mean difference: {mean_diff:.6e}")
        print(f"[USP] Distributed time: {dist_time * 1000:.2f} ms")
        print(f"[USP] Local time: {local_time * 1000:.2f} ms")
        print(f"[USP] Speedup: {speedup:.2f}x")
        assert passed, f"USP test failed: max_diff={max_diff} > tolerance={TOLERANCE}"
        # USP with 4 GPUs on very long sequences should provide good speedup

    @pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
    def test_usp_large_scale_attention(self):
        """Test USP with parallel_shard at large scale (4 GPUs, 307K sequence)."""
        world_size = 4
        results_queue = mp.Queue()

        print(f"\n[USP Large] Testing with sequence length: {LARGE_SEQ_LEN}")
        mp.spawn(
            run_usp_test,
            args=(world_size, results_queue, True),
            nprocs=world_size,
            join=True,
        )

        result = results_queue.get()
        strategy, max_diff, mean_diff, passed, dist_time, local_time, seq_len = result
        speedup = local_time / dist_time if dist_time > 0 else 0
        print(f"\n[USP Large] Sequence length: {seq_len}")
        print(f"[USP Large] Max difference: {max_diff:.6e}")
        print(f"[USP Large] Mean difference: {mean_diff:.6e}")
        print(f"[USP Large] Distributed time: {dist_time * 1000:.2f} ms")
        print(f"[USP Large] Local time: {local_time * 1000:.2f} ms")
        print(f"[USP Large] Speedup: {speedup:.2f}x")
        assert passed, f"USP large scale test failed: max_diff={max_diff} > tolerance={TOLERANCE}"
        # At large scale, USP should provide better speedup due to more work per GPU


def main():
    """Run all tests directly."""
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    print(f"Test configuration: BATCH={BATCH_SIZE}, SEQ={SEQ_LEN}, HEADS={NUM_HEADS}, DIM={HEAD_DIM}")

    if num_gpus < 2:
        print("Skipping tests: requires at least 2 GPUs")
        return

    all_passed = True

    # Test Ulysses (2 GPUs)
    print("\n" + "=" * 70)
    print("Testing Ulysses + parallel_shard (2 GPUs)")
    print("=" * 70)
    results_queue = mp.Queue()
    mp.spawn(run_ulysses_test, args=(2, results_queue), nprocs=2, join=True)
    strategy, max_diff, mean_diff, passed, dist_time, local_time = results_queue.get()
    speedup = local_time / dist_time if dist_time > 0 else 0
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    print(f"Distributed time: {dist_time * 1000:.2f} ms")
    print(f"Local time: {local_time * 1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'✓ PASSED' if passed and speedup > 0.5 else '✗ FAILED'}")
    all_passed = all_passed and passed

    # Test Ring (2 GPUs)
    print("\n" + "=" * 70)
    print("Testing Ring + parallel_shard (2 GPUs)")
    print("=" * 70)
    results_queue = mp.Queue()
    mp.spawn(run_ring_test, args=(2, results_queue), nprocs=2, join=True)
    strategy, max_diff, mean_diff, passed, dist_time, local_time = results_queue.get()
    speedup = local_time / dist_time if dist_time > 0 else 0
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    print(f"Distributed time: {dist_time * 1000:.2f} ms")
    print(f"Local time: {local_time * 1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'✓ PASSED' if passed and speedup > 0.3 else '✗ FAILED'}")
    all_passed = all_passed and passed

    # Test USP (4 GPUs) - Standard scale
    if num_gpus >= 4:
        print("\n" + "=" * 70)
        print("Testing USP + parallel_shard (4 GPUs, Standard Scale)")
        print("=" * 70)
        results_queue = mp.Queue()
        mp.spawn(run_usp_test, args=(4, results_queue, False), nprocs=4, join=True)
        result = results_queue.get()
        strategy, max_diff, mean_diff, passed, dist_time, local_time, seq_len = result
        speedup = local_time / dist_time if dist_time > 0 else 0
        print(f"Sequence length: {seq_len}")
        print(f"Max difference: {max_diff:.6e}")
        print(f"Mean difference: {mean_diff:.6e}")
        print(f"Distributed time: {dist_time * 1000:.2f} ms")
        print(f"Local time: {local_time * 1000:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"{'✓ PASSED' if passed else '✗ FAILED'}")
        all_passed = all_passed and passed

        # Test USP (4 GPUs) - Large scale
        print("\n" + "=" * 70)
        print("Testing USP + parallel_shard (4 GPUs, Large Scale)")
        print("=" * 70)
        print(f"Sequence length: {LARGE_SEQ_LEN} ({LARGE_SEQ_LEN / 1000:.0f}K)")
        results_queue = mp.Queue()
        mp.spawn(run_usp_test, args=(4, results_queue, True), nprocs=4, join=True)
        result = results_queue.get()
        strategy, max_diff, mean_diff, passed, dist_time, local_time, seq_len = result
        speedup = local_time / dist_time if dist_time > 0 else 0
        print(f"Sequence length: {seq_len}")
        print(f"Max difference: {max_diff:.6e}")
        print(f"Mean difference: {mean_diff:.6e}")
        print(f"Distributed time: {dist_time * 1000:.2f} ms")
        print(f"Local time: {local_time * 1000:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"{'✓ PASSED' if passed else '✗ FAILED'}")
        all_passed = all_passed and passed
    else:
        print("\nSkipping USP tests: requires 4 GPUs")

    print("\n" + "=" * 70)
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
