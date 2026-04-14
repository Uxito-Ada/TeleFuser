"""LiveAct Example: Use Original SoulX-LiveAct WanModel with TeleFuser VAE for Performance Comparison.

This example demonstrates how to directly use the original SoulX-LiveAct WanModel
with TeleFuser's optimized VAE (with channels_last_3d), to verify if the performance
difference is caused by TeleFuser's implementation.

Usage:
    python examples/liveact/liveact_s2v_original.py \
        --image examples/data/1.png \
        --audio examples/data/1.wav \
        --ckpt_dir /data/aigc/model_zoo/LiveAct \
        --wav2vec_dir /data/aigc/model_zoo/chinese-wav2vec2-base \
        --height 720 --width 416 --fps 20 --gpu_num 1
"""

import os
import sys
import time

import click
import torch
from PIL import Image

# Add SoulX-LiveAct to Python path for importing original WanModel
SOULX_LIVEACT_PATH = "/data/aigc/zuoxin/workspace/SoulX-LiveAct"
sys.path.insert(0, SOULX_LIVEACT_PATH)

# Import original SoulX-LiveAct WanModel (uses diffusers ModelMixin)
from model_liveact.model_memory import WanModel

from telefuser.core.config import CompileConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.wan_video_vae import WanVideoVAE
from telefuser.models.wav2vec2 import Wav2Vec2Model
from telefuser.pipelines.liveact import LiveActPipeline, LiveActPipelineConfig
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import save_video

# Import FP8 GEMM from SoulX-LiveAct
try:
    from fp8_gemm import FP8GemmOptions, enable_fp8_gemm

    FP8_GEMM_AVAILABLE = True
except ImportError:
    FP8_GEMM_AVAILABLE = False

PPL_CONFIG = dict(
    num_inference_steps=3,
    audio_cfg=1.0,
    fps=24,
    seed=42,
    height=480,
    width=832,
    compile_config=CompileConfig(enabled=True, dynamic=False),
)


def get_pipeline(gpu_num: int, ckpt_dir: str, wav2vec_dir: str):
    """Load LiveAct pipeline with original SoulX-LiveAct WanModel and TeleFuser VAE.

    Args:
        gpu_num: Number of GPUs for parallel inference (1 for single GPU test)
        ckpt_dir: Path to model checkpoints (LiveAct weights)
        wav2vec_dir: Path to wav2vec2 weights

    Returns:
        LiveActPipeline instance with original WanModel and TeleFuser VAE
    """
    torch_dtype = torch.bfloat16
    device = "cuda"

    mm = ModuleManager(torch_dtype=torch_dtype, device="cpu")

    # Load text encoder and clip weights
    mm.load_models(
        [
            os.path.join(ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        ],
        torch_dtype=torch_dtype,
    )

    # Load TeleFuser VAE (with channels_last_3d optimization)
    print("Loading TeleFuser VAE...")
    mm.load_models(
        [
            os.path.join(ckpt_dir, "Wan2.1_VAE.pth"),
        ],
        torch_dtype=torch_dtype,
    )
    print("✓ TeleFuser VAE loaded with channels_last_3d optimization")

    # Load Wav2Vec2 Audio Encoder
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True, torch_dtype=torch_dtype).eval()
    audio_encoder.feature_extractor._freeze_parameters()
    mm.add_module(audio_encoder, "wav2vec2")

    # Load original SoulX-LiveAct WanModel directly (uses diffusers ModelMixin.from_pretrained)
    print("Loading original SoulX-LiveAct WanModel...")
    original_dit = WanModel.from_pretrained(ckpt_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=False)
    original_dit = original_dit.to(dtype=torch_dtype)

    # Initialize KV index (required by original model for each attention block)
    # frame_len = (H / patch_size[1] / vae_stride[1]) * (W / patch_size[2] / vae_stride[2])
    height = PPL_CONFIG["height"]
    width = PPL_CONFIG["width"]
    vae_stride = (4, 8, 8)  # temporal, height, width
    patch_size = (1, 2, 2)  # temporal, height, width
    frame_len = (height // (patch_size[1] * vae_stride[1])) * (width // (patch_size[2] * vae_stride[2]))
    world_size = gpu_num  # 1 for single GPU test

    for n in range(40):
        original_dit.blocks[n].self_attn.init_kvidx(frame_len, world_size)

    # Enable FP8 GEMM (same as original generate.py)
    if FP8_GEMM_AVAILABLE:
        enable_fp8_gemm(original_dit, options=FP8GemmOptions())
        print("✓ FP8 GEMM enabled for DiT FFN layers")

    original_dit.eval()

    # torch.compile (same as original generate.py line 213)
    compile_config = PPL_CONFIG["compile_config"]
    if compile_config.enabled:
        original_dit = torch.compile(
            original_dit,
            mode="max-autotune-no-cudagraphs",
            backend="inductor",
            dynamic=False,
        )
        print("✓ torch.compile enabled for DiT")

    # Move freqs tensor to device (required for RoPE)
    original_dit.freqs = original_dit.freqs.to(device)
    original_dit = original_dit.to(device)

    # Freeze parameters
    for param in original_dit.parameters():
        param.requires_grad = False
    # Add original WanModel to module manager with "liveact_dit" name
    # This name is used by LiveActDenoisingStage to fetch the model
    mm.add_module(original_dit, "liveact_dit")

    # Create pipeline with minimal config (original model handles attention internally)
    pipeline = LiveActPipeline(device=device, torch_dtype=torch_dtype)
    config = LiveActPipelineConfig()
    # Original model handles attention internally via SageAttention,
    # no need to set attention_config (denoising.py now checks hasattr)
    config.dit_config.compile_config = compile_config
    # VAE compile is handled by VAEStage
    config.vae_config.compile_config = compile_config

    pipeline.init(mm, config)
    return pipeline


def run(
    pipeline: LiveActPipeline,
    image: str,
    audio_path: str,
    prompt: str,
    height: int = PPL_CONFIG["height"],
    width: int = PPL_CONFIG["width"],
    fps: int = PPL_CONFIG["fps"],
    seed: int = PPL_CONFIG["seed"],
):
    """Run LiveAct inference.

    Args:
        pipeline: LiveActPipeline instance
        image: Path to input image
        audio_path: Path to audio file
        prompt: Text prompt
        height: Video height
        width: Video width
        fps: Video fps
        seed: Random seed

    Returns:
        Generated video frames
    """
    frames = pipeline(
        prompt=prompt,
        input_image=Image.open(image).convert("RGB"),
        audio_path=audio_path,
        height=height,
        width=width,
        fps=fps,
        audio_cfg=PPL_CONFIG["audio_cfg"],
        seed=seed,
    )
    return frames


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs (default: 1 for single GPU test)")
@click.option("--ckpt_dir", required=True, help="Path to LiveAct checkpoints")
@click.option("--wav2vec_dir", required=True, help="Path to wav2vec2 weights")
@click.option("--image", required=True, help="Path to input image")
@click.option("--audio", required=True, help="Path to audio file")
@click.option("--prompt", default="A person talking naturally", help="Text prompt")
@click.option("--height", default=PPL_CONFIG["height"], help="Video height")
@click.option("--width", default=PPL_CONFIG["width"], help="Video width")
@click.option("--fps", default=PPL_CONFIG["fps"], help="Video fps")
@click.option("--output", default=None, help="Output video path")
def main(
    gpu_num: int,
    ckpt_dir: str,
    wav2vec_dir: str,
    image: str,
    audio: str,
    prompt: str,
    height: int,
    width: int,
    fps: int,
    output: str | None,
):
    """LiveAct: Generate video using original SoulX-LiveAct WanModel."""
    pipeline = get_pipeline(gpu_num, ckpt_dir, wav2vec_dir)

    start = time.time()
    frames = run(pipeline, image, audio, prompt, height, width, fps)
    elapsed = time.time() - start
    print(f"Video generation time: {elapsed:.2f} seconds")

    if output is None:
        filename = get_example_name(__file__).replace(".py", ".mp4")
        output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
        output = os.path.join(output_dir, filename)
    save_video(frames, output, fps=fps, audio_path=audio, quality=6)
    print(f"Video saved to: {output}")


if __name__ == "__main__":
    main()
