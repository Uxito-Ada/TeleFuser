"""Test Pipeline Parallel forward consistency.

This test verifies that PP=2 produces the same results as PP=1 (no PP).
Uses the real wan2.1 1.3B model configuration.
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist

# Mark all tests in this module as requiring distributed setup, GPU, and multiple GPUs
pytestmark = [
    pytest.mark.distributed,
    pytest.mark.gpu,
    pytest.mark.multi_gpu,
]

# Set GPU devices before importing telefuser
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def setup_distributed(world_size: int, rank: int):
    """Initialize distributed environment."""
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:29501",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed environment."""
    dist.destroy_process_group()


def create_wan_model(dtype=torch.bfloat16):
    """Create a WanModel with wan2.1 1.3B configuration.

    wan2.1 1.3B config:
    - dim=1536
    - num_heads=12 (head_dim=128)
    - num_layers=30
    """
    from telefuser.models.wan_video_dit import WanModel

    model = WanModel(
        dim=1536,  # wan2.1 1.3B dim
        in_dim=16,
        ffn_dim=8960,
        out_dim=16,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=12,  # head_dim = 1536/12 = 128
        num_layers=30,  # wan2.1 1.3B layers
        has_image_input=False,
    )
    model = model.to(dtype)
    model.dtype = dtype
    return model


def run_forward_no_pp(model, device: str = "cuda"):
    """Run forward without PP."""
    model = model.to(device)
    model.eval()

    batch_size = 1
    num_frames = 5  # Use small number for faster testing
    height = 64
    width = 64

    # Create input tensors
    x = torch.randn(batch_size, 16, num_frames, height, width, dtype=torch.bfloat16, device=device)
    timestep = torch.tensor([0.5], dtype=torch.float32, device=device)
    context = torch.randn(batch_size, 256, 4096, dtype=torch.bfloat16, device=device)  # text_dim=4096

    with torch.no_grad():
        output = model(
            x=x,
            timestep=timestep,
            context=context,
        )

    return output


def run_forward_with_pp(model, device: str = "cuda", rank: int = 0):
    """Run forward with PP=2."""
    from telefuser.core.config import ParallelConfig
    from telefuser.distributed.device_mesh import create_device_mesh_from_config

    # Setup parallel config for PP=2
    parallel_config = ParallelConfig(
        pp_degree=2,
        device_ids=[0, 1],  # Will be mapped to GPUs 2,3
    )

    # Create device mesh
    device_mesh = create_device_mesh_from_config(parallel_config, device_type="cuda")

    model.device_mesh = device_mesh
    model.enable_pp()
    model = model.to(device)
    model.eval()

    batch_size = 1
    num_frames = 5
    height = 64
    width = 64

    # Create input tensors (only used on first stage)
    x = torch.randn(batch_size, 16, num_frames, height, width, dtype=torch.bfloat16, device=device)
    timestep = torch.tensor([0.5], dtype=torch.float32, device=device)
    context = torch.randn(batch_size, 256, 4096, dtype=torch.bfloat16, device=device)  # text_dim=4096

    with torch.no_grad():
        output = model.pp_forward(
            x=x,
            timestep=timestep,
            context=context,
        )

    return output


def test_pp_forward_vs_no_pp():
    """Test that PP=2 produces same results as PP=1."""
    world_size = 2
    rank = int(os.environ.get("RANK", 0))

    # Setup distributed
    setup_distributed(world_size, rank)

    try:
        # Create model with same weights
        torch.manual_seed(42)
        model = create_wan_model()

        # Save model state for comparison
        model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Run with PP
        torch.manual_seed(42)
        output_pp = run_forward_with_pp(model, device="cuda", rank=rank)

        # Only compare on last stage (rank 1)
        if rank == 1:
            # Load fresh model without PP
            model_no_pp = create_wan_model()
            model_no_pp.load_state_dict(model_state)
            model_no_pp.eval()

            # Run without PP
            torch.manual_seed(42)
            output_no_pp = run_forward_no_pp(model_no_pp, device="cuda")

            # Compare outputs
            if output_pp is not None and output_no_pp is not None:
                max_diff = (output_pp - output_no_pp).abs().max().item()
                mean_diff = (output_pp - output_no_pp).abs().mean().item()

                print(f"\n{'=' * 60}")
                print(f"Output max diff: {max_diff}")
                print(f"Output mean diff: {mean_diff}")
                print(f"PP output shape: {output_pp.shape}")
                print(f"No-PP output shape: {output_no_pp.shape}")
                print(f"{'=' * 60}\n")

                if not torch.allclose(output_pp, output_no_pp, atol=1e-3):
                    print("❌ PP output differs from no-PP output")
                    print(f"PP output sample: {output_pp[0, 0, 0, 0, :5]}")
                    print(f"No-PP output sample: {output_no_pp[0, 0, 0, 0, :5]}")
                    return False
                else:
                    print("✓ PP=2 output matches PP=1 output")
                    return True
            else:
                print(f"output_pp is None: {output_pp is None}")
                print(f"output_no_pp is None: {output_no_pp is None}")
                return False

        return True

    except Exception as e:
        import traceback

        print(f"Error on rank {rank}: {e}")
        traceback.print_exc()
        return False

    finally:
        cleanup_distributed()


def run_test_worker(rank: int, world_size: int):
    """Worker function for spawned test."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"

    result = test_pp_forward_vs_no_pp()
    if rank == 1:
        print(f"\nTest result: {'PASS' if result else 'FAIL'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    # Use start method spawn for distributed training
    mp.set_start_method("spawn", force=True)

    world_size = 2
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run_test_worker, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
