"""FlashVSR Streaming Video Super-Resolution Example.

This example demonstrates streaming video super-resolution using FlashVSR model.
Supports chunk-based processing for memory-efficient inference on long videos.

Model Source:
    HuggingFace: https://huggingface.co/lzx1413/FlashVSR-v1.1-BF16
    ModelScope: https://modelscope.cn/models/lzx1413/FlashVSR-v1.1-BF16

Usage:
    python examples/flashvsr/flashvsr_stream.py \
        --input_video examples/data/dag.mp4 \
        --scale  \
        --gpu_num 1 \
        --model_root /path/to/FlashVSR-v1.1
"""

import os
import time

import click
import torch

from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.flashvsr.flashvsr_stream import (
    FlashVSRStreamPipelineConfig,
    FlashVSRStreamVideoPipeline,
)
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import VideoData, save_video

# Model filenames in the model_root directory
DIT_FILENAME = "flashvsr11_dit_streaming_dmd_5dc619.safetensors"
VAE_FILENAME = "TCDecoder.ckpt"


FIRST_CHUNK_SIZE = 25
CHUNK_SIZE = 16


def get_chunk_indices(frame_count: int) -> list[tuple[int, int]]:
    """Calculate start/end indices for all chunks.

    Args:
        frame_count: Total number of frames in the video

    Returns:
        List of (start, end) index tuples for each chunk
    """
    indices = [(0, min(FIRST_CHUNK_SIZE, frame_count))]
    offset = FIRST_CHUNK_SIZE
    while offset < frame_count:
        indices.append((offset, min(offset + CHUNK_SIZE, frame_count)))
        offset += CHUNK_SIZE
    return indices


def get_pipeline(dit_path: str, vae_path: str, parallelism: int = 1):
    """Initialize the FlashVSR streaming pipeline.

    Args:
        dit_path: Path to DiT model file
        vae_path: Path to VAE model file
        parallelism: Number of GPUs for parallel inference

    Returns:
        Initialized FlashVSRStreamVideoPipeline
    """
    mm = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([dit_path])
    mm.load_models([vae_path])

    pipe = FlashVSRStreamVideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = FlashVSRStreamPipelineConfig()

    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism
        pipe_config.dit_config.compile = True
        pipe_config.enable_denoising_parallel = True

    pipe.init(mm, pipe_config)
    return pipe


def run(pipeline: FlashVSRStreamVideoPipeline, input_video, scale: int, seed: int = 0):
    """Run video super-resolution on a video chunk.

    Args:
        pipeline: FlashVSR pipeline instance
        input_video: Input video frames
        scale: Upscaling factor
        seed: Random seed

    Returns:
        Super-resolved video frames
    """
    video = pipeline(
        seed=seed,
        LQ_video=input_video,
        scale=scale,
        rand_device="cpu",
        sparse_ratio=2,
        kv_ratio=3,
        local_range=11,
        proj_tile=False,
    )
    return video


@click.command()
@click.option("--input_video", "-i", required=True, help="Path to input low-quality video")
@click.option("--scale", "-s", default=2.25, type=float, help="Upscaling factor (default: 2.25, 480p->1080p)")
@click.option("--height", "-h", default=None, type=int, help="Input video height (default: auto-detect)")
@click.option("--width", "-w", default=None, type=int, help="Input video width (default: auto-detect)")
@click.option("--gpu_num", default=1, type=int, help="Number of GPUs to use (default: 1)")
@click.option(
    "--model_root",
    default="/dev/shm/zuoxin/flashvsr",
    help="Root directory containing model files (default: /dev/shm/zuoxin/flashvsr)",
)
@click.option("--output", "-o", default=None, help="Output video path (default: auto-generated)")
@click.option("--seed", default=0, type=int, help="Random seed (default: 0)")
def main(
    input_video: str,
    scale: float,
    height: int | None,
    width: int | None,
    gpu_num: int,
    model_root: str,
    output: str,
    seed: int,
):
    """FlashVSR Streaming Video Super-Resolution.

    Upscales low-quality videos using FlashVSR model with streaming inference.

    Examples:
        # Basic usage (auto-detect resolution)
        python flashvsr_stream.py -i input.mp4 -s 4

        # Specify input resolution
        python flashvsr_stream.py -i input.mp4 -s 4 --height 480 --width 854

        # Multi-GPU inference
        python flashvsr_stream.py -i input.mp4 -s 4 --gpu_num 2

        # Custom model path and output
        python flashvsr_stream.py -i input.mp4 -s 4 --model_root /path/to/models -o output.mp4
    """
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    dit_path = os.path.join(model_root, DIT_FILENAME)
    vae_path = os.path.join(model_root, VAE_FILENAME)

    if output is None:
        output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
        filename = get_example_name(__file__).replace(".mp4", f"_scale{scale}_{gpu_num}gpu.mp4")
        output = os.path.join(output_dir, filename)

    click.echo(f"Input video: {input_video}")
    click.echo(f"Input resolution: {width or 'auto'}x{height or 'auto'}")
    click.echo(f"Scale: {scale}x")
    click.echo(f"GPUs: {gpu_num}")
    click.echo(f"Model root: {model_root}")
    click.echo(f"Output: {output}")

    click.echo("Loading pipeline...")
    pipeline = get_pipeline(dit_path, vae_path, gpu_num)

    click.echo("Loading video...")
    LQ_video = VideoData(video_file=input_video, height=height, width=width).raw_data()
    total_frames = len(LQ_video)
    click.echo(f"Total frames: {total_frames}")

    chunk_indices = get_chunk_indices(total_frames)

    click.echo("Warmup pass...")
    for start, end in chunk_indices[:5]:
        _ = run(pipeline, LQ_video[start:end], scale=scale, seed=seed)
    pipeline.clean_cache()

    click.echo("Processing video...")
    start_time = time.time()
    final_video = []

    for i, (start, end) in enumerate(chunk_indices):
        chunk = LQ_video[start:end]
        click.echo(f"  Processing chunk {i + 1}/{len(chunk_indices)} ({len(chunk)} frames)")
        video = run(pipeline, chunk, scale=scale, seed=seed)
        final_video.extend(video)

    elapsed_time = time.time() - start_time
    click.echo(f"Processing time: {elapsed_time:.2f} seconds")

    click.echo(f"Saving to {output}...")
    save_video(final_video, output, fps=16, quality=6)
    click.echo("Done!")

    pipeline.clean_cache()
    del pipeline


if __name__ == "__main__":
    main()
