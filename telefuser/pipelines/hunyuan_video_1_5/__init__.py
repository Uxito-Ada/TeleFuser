"""HunyuanVideo pipeline for text-to-video and image-to-video generation.

This module provides a pipeline that works with models imported directly from
the HunyuanVideo repository. Users load models externally and add them to
ModuleManager, then the pipeline creates stages that use these models.

Usage:
    # 1. Load models from HunyuanVideo
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae import AutoencoderKLConv3D
    from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
    from hyvideo.models.text_encoders import TextEncoder, PROMPT_TEMPLATE
    from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

    vae = AutoencoderKLConv3D.from_pretrained(vae_path, torch_dtype=torch.float16)
    transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(transformer_path)
    text_encoder = TextEncoder(...)
    scheduler = FlowMatchDiscreteScheduler.from_pretrained(scheduler_path)

    # 2. Add to ModuleManager
    from telefuser.core.module_manager import ModuleManager
    mm = ModuleManager(torch_dtype=torch.bfloat16)
    mm.add_module(vae, name="vae")
    mm.add_module(transformer, name="transformer")
    mm.add_module(text_encoder, name="text_encoder")
    mm.add_module(scheduler, name="scheduler")

    # 3. Create pipeline
    from telefuser.pipelines.hunyuan_video import HunyuanVideoPipeline, HunyuanVideoPipelineConfig
    pipeline = HunyuanVideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipeline.init(mm, config)

    # 4. Generate
    frames = pipeline.text_to_video(prompt="...", num_frames=61)
"""

from .dit_denoising import HunyuanVideoDenoisingStage
from .image_encoding import HunyuanVideoImageEncodingStage
from .pipeline import HunyuanVideo15Pipeline, HunyuanVideo15PipelineConfig
from .sr_dit_denoising import HunyuanVideoSRDenoisingStage
from .text_encoding import HunyuanVideoTextEncodingStage
from .upsampler import HunyuanVideoUpsamplerStage
from .vae import HunyuanVideoVAEStage

__all__ = [
    # Main pipeline
    "HunyuanVideo15Pipeline",
    "HunyuanVideo15PipelineConfig",
    # Stages
    "HunyuanVideoVAEStage",
    "HunyuanVideoTextEncodingStage",
    "HunyuanVideoDenoisingStage",
    "HunyuanVideoSRDenoisingStage",
    "HunyuanVideoImageEncodingStage",
    "HunyuanVideoUpsamplerStage",
]
