from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from PIL import Image
from loguru import logger
from safetensors import safe_open

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import AttentionConfig, ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx_audio_vae import AudioDecoderConfigurator, VocoderConfigurator
from telefuser.models.ltx_video_vae import VIDEO_SCALE_FACTORS
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.utils.func import auto_async_call
from telefuser.worker.parallel_worker import ParallelWorker
from telefuser.worker.ray_worker import create_ray_worker

from .dit_denoising import AudioLatentShape, DitDenoisingStage, MultiModalGuiderParams
from .gemma_text_encoding import GemmaTextEncodingStage
from .upsampler import UpsamplerStage
from .vae import DEFAULT_TILE_SIZE, DEFAULT_TILE_STRIDE, VAEStage

DEFAULT_VIDEO_STG_BLOCKS = [28]
DEFAULT_AUDIO_STG_BLOCKS = [28]
DEFAULT_NUM_INFERENCE_STEPS = 30
LATENT_TOKEN_CHANNELS = 128


def assert_resolution(height: int, width: int, is_two_stage: bool) -> None:
    divisor = 64 if is_two_stage else 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {'two-stage' if is_two_stage else 'one-stage'} pipelines, "
            f"height and width must be multiples of {divisor}."
        )


@dataclass
class LTXVideoPipelineConfig:
    """Configuration for the LTX 2.3 two-stage video pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    upsampler_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_stage1_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_stage2_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_vae_ray: bool = False
    stage2_lora_strength: float = 1.0


class LTX23VideoPipeline(BasePipeline):
    """LTX 2.3 video generation pipeline with two-stage denoising.

    Stage 1 generates video at half of the target resolution with CFG guidance (assuming
    full model is used); then Stage 2 upsamples by 2x and refines using a distilled
    LoRA for higher quality output. Supports optional image conditioning via the
    images parameter.
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 64
        self.width_division_factor = 64
        self.base_fps = 24
        self._audio_decoder: object | None = None
        self._vocoder: object | None = None
        self._audio_sample_rate: int | None = None
        self._ltx_checkpoint_paths: str | list[str] | None = None

    def init(self, module_manager: ModuleManager, config: LTXVideoPipelineConfig):
        """Initialize pipeline stages."""
        self.module_manager = module_manager
        self.config = config
        self.vae_stage = VAEStage("vae", module_manager, config.vae_config)
        if config.sample_solver == "euler":
            self.stage1_scheduler = FlowMatchScheduler(template="LTX.2")
            self.stage2_scheduler = FlowMatchScheduler(template="LTX.2")
        elif config.sample_solver == "unipc":
            self.stage1_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            self.stage2_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
        else:
            raise NotImplementedError(f"solver {config.sample_solver} is not supported")

        self.stage1_denoise_stage = DitDenoisingStage(
            "stage1_denoise",
            module_manager,
            config.dit_stage1_config,
            "ltx_dit",
            self.stage1_scheduler,
        )
        self.stage2_denoise_stage = DitDenoisingStage(
            "stage2_denoise",
            module_manager,
            config.dit_stage2_config,
            "ltx_dit",
            self.stage2_scheduler,
        )
        self.upsampler_stage = UpsamplerStage("upsampler", module_manager, config.upsampler_config)
        self.text_encoding_stage = GemmaTextEncodingStage("text_encoding", module_manager, config.text_encoding_config)

        if config.enable_vae_ray:
            logger.info("enable ray actor for vae")
            self.vae_stage = create_ray_worker(self.vae_stage, self.config.enable_vae_parallel)
        elif config.enable_vae_parallel:
            self.vae_stage = ParallelWorker(self.vae_stage)
        if config.enable_denoising_parallel:
            self.stage1_denoise_stage = ParallelWorker(self.stage1_denoise_stage)
            self.stage2_denoise_stage = ParallelWorker(self.stage2_denoise_stage)

        checkpoint_paths = None
        fetched = module_manager.fetch_module("ltx_dit", require_model_path=True)
        if fetched is not None:
            _, checkpoint_paths = fetched
        self._ltx_checkpoint_paths = checkpoint_paths

    def _ensure_audio_decoder_loaded(self) -> None:
        """Lazy-load the LTX audio decoder + vocoder from the LTX checkpoint."""
        if self._audio_decoder is not None and self._vocoder is not None and self._audio_sample_rate is not None:
            return

        if self._ltx_checkpoint_paths is None:
            raise ValueError("LTX checkpoint paths are not available; cannot load audio decoder/vocoder.")

        checkpoint_paths: list[str]
        if isinstance(self._ltx_checkpoint_paths, str):
            checkpoint_paths = [self._ltx_checkpoint_paths]
        else:
            checkpoint_paths = list(self._ltx_checkpoint_paths)
        with safe_open(checkpoint_paths[0], framework="pt") as handle:
            meta = handle.metadata() or {}
        if "config" not in meta:
            raise ValueError("LTX checkpoint is missing `config` metadata; cannot configure audio decoder/vocoder.")

        import json

        config = json.loads(meta["config"])
        audio_decoder = AudioDecoderConfigurator.from_config(config)
        vocoder = VocoderConfigurator.from_config(config)

        from collections.abc import Callable

        def iter_filtered_state_dict(
            prefixes: tuple[str, ...],
            rename: Callable[[str], str | None],
        ) -> dict[str, torch.Tensor]:
            state_dict: dict[str, torch.Tensor] = {}
            for shard_path in checkpoint_paths:
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if not any(key.startswith(prefix) for prefix in prefixes):
                            continue
                        new_key = rename(key)
                        if new_key is None:
                            continue
                        state_dict[new_key] = handle.get_tensor(key)
            return state_dict

        def rename_audio_decoder(key: str) -> str | None:
            if key.startswith("audio_vae.decoder."):
                return key.removeprefix("audio_vae.decoder.")
            if key.startswith("audio_vae.per_channel_statistics."):
                return "per_channel_statistics." + key.removeprefix("audio_vae.per_channel_statistics.")
            return None

        def rename_vocoder(key: str) -> str | None:
            if not key.startswith("vocoder."):
                return None
            return key.removeprefix("vocoder.")

        audio_decoder_sd = iter_filtered_state_dict(
            prefixes=("audio_vae.decoder.", "audio_vae.per_channel_statistics."),
            rename=rename_audio_decoder,
        )
        vocoder_sd = iter_filtered_state_dict(prefixes=("vocoder.",), rename=rename_vocoder)

        missing, unexpected = audio_decoder.load_state_dict(audio_decoder_sd, strict=False)
        if missing or unexpected:
            logger.warning(
                "Audio decoder state dict mismatch: missing_keys={}, unexpected_keys={}",
                len(missing),
                len(unexpected),
            )
            if missing:
                logger.warning("Audio decoder missing keys (first 20): {}", missing[:20])
            if unexpected:
                logger.warning("Audio decoder unexpected keys (first 20): {}", unexpected[:20])

        missing, unexpected = vocoder.load_state_dict(vocoder_sd, strict=False)
        if missing or unexpected:
            logger.warning(
                "Vocoder state dict mismatch: missing_keys={}, unexpected_keys={}",
                len(missing),
                len(unexpected),
            )
            if missing:
                logger.warning("Vocoder missing keys (first 20): {}", missing[:20])
            if unexpected:
                logger.warning("Vocoder unexpected keys (first 20): {}", unexpected[:20])

        audio_decoder = audio_decoder.to(device=self.device, dtype=self.torch_dtype).eval()
        vocoder = vocoder.to(device=self.device, dtype=self.torch_dtype).eval()

        output_sample_rate = getattr(vocoder, "output_sampling_rate", None)
        if not isinstance(output_sample_rate, int):
            raise ValueError("Loaded vocoder does not expose an integer `output_sampling_rate`.")

        self._audio_decoder = audio_decoder
        self._vocoder = vocoder
        self._audio_sample_rate = output_sample_rate

    @torch.inference_mode()
    def _decode_audio(self, audio_latents: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Decode LTX audio latents into waveform.

        Returns:
            (waveform, sample_rate) where waveform is shaped (channels, samples).
        """
        self._ensure_audio_decoder_loaded()
        assert self._audio_decoder is not None
        assert self._vocoder is not None
        assert self._audio_sample_rate is not None

        audio_latents = audio_latents.to(device=self.device, dtype=self.torch_dtype)
        decoded_mel = self._audio_decoder(audio_latents)
        waveform = self._vocoder(decoded_mel).squeeze(0).float().detach()
        return waveform, self._audio_sample_rate

    def _init_video_noise(self, height: int, width: int, num_frames: int, seed: int) -> torch.Tensor:
        latent_frames = (num_frames - 1) // VIDEO_SCALE_FACTORS.time + 1
        latent_height = height // VIDEO_SCALE_FACTORS.height
        latent_width = width // VIDEO_SCALE_FACTORS.width
        return self.generate_noise(
            (1, latent_frames * latent_height * latent_width, LATENT_TOKEN_CHANNELS),
            seed=seed,
            device=self.device,
            dtype=torch.float32,
        )

    def _init_audio_noise(self, num_frames: int, frame_rate: float, seed: int) -> torch.Tensor:
        audio_latent_shape = AudioLatentShape.from_duration(batch=1, duration=float(num_frames) / float(frame_rate))
        return self.generate_noise(
            (1, audio_latent_shape.frames, audio_latent_shape.channels * audio_latent_shape.mel_bins),
            seed=seed,
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        input_image: Image.Image | None = None,
        seed: int | None = None,
        height: int = 1024,
        width: int = 1536,
        num_frames: int = 121,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        frame_rate: float | None = None,
        tiled: bool = True,
        tile_size: tuple[int, int] = DEFAULT_TILE_SIZE,
        tile_stride: tuple[int, int] = DEFAULT_TILE_STRIDE,
        input_image_frame_idx: int = 0,
        input_image_strength: float = 1.0,
        video_cfg_scale: float = 3.0,
        video_stg_scale: float = 1.0,
        video_rescale_scale: float = 0.7,
        video_modality_scale: float = 3.0,
        video_skip_step: int = 0,
        video_stg_blocks: Sequence[int] | None = None,
        audio_cfg_scale: float = 7.0,
        audio_stg_scale: float = 1.0,
        audio_rescale_scale: float = 0.7,
        audio_modality_scale: float = 3.0,
        audio_skip_step: int = 0,
        audio_stg_blocks: Sequence[int] | None = None,
    ) -> list[Image.Image] | tuple[list[Image.Image], torch.Tensor, int]:
        """Generate audio-video from a text prompt and optional reference image."""
        logger.info(f"start generate {num_frames} {width}x{height} frames")
        height, width = self.check_resize_height_width(height, width)
        # LTX resolution requires be divisible by 64 (for two-stage ppl) or 32 (for one-stage ppl).
        assert_resolution(height=height, width=width, is_two_stage=True)

        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        frame_rate = frame_rate or float(self.base_fps)
        # LTX uses multi-modal guidance to steer the diffusion process for both video and audio modalities.
        # Each has its own guider, allowing fine-grained control over generation quality and adherence to prompts.
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=video_cfg_scale,
            stg_scale=video_stg_scale,
            rescale_scale=video_rescale_scale,
            modality_scale=video_modality_scale,
            skip_step=video_skip_step,
            stg_blocks=list(video_stg_blocks) if video_stg_blocks is not None else list(DEFAULT_VIDEO_STG_BLOCKS),
        )
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=audio_cfg_scale,
            stg_scale=audio_stg_scale,
            rescale_scale=audio_rescale_scale,
            modality_scale=audio_modality_scale,
            skip_step=audio_skip_step,
            stg_blocks=list(audio_stg_blocks) if audio_stg_blocks is not None else list(DEFAULT_AUDIO_STG_BLOCKS),
        )
        # Initialize noise
        stage1_video_noise = self._init_video_noise(height // 2, width // 2, num_frames, seed)
        stage1_audio_noise = self._init_audio_noise(num_frames, frame_rate, seed)
        stage2_video_noise = self._init_video_noise(height, width, num_frames, seed)
        stage2_audio_noise = self._init_audio_noise(num_frames, frame_rate, seed)
        # Encode prompts
        prompt_emb_nega = None
        prompt_list = [prompt]
        if video_guider_params.cfg_scale != 1.0 and audio_guider_params.cfg_scale != 1.0:
            prompt_list.append(negative_prompt)
        prompt_emb_list_handler = auto_async_call(self.text_encoding_stage.process, prompt_list)
        prompt_emb_list = prompt_emb_list_handler()
        prompt_emb_posi = prompt_emb_list[0]
        if video_guider_params.cfg_scale != 1.0 and audio_guider_params.cfg_scale != 1.0:
            prompt_emb_nega = prompt_emb_list[1]
        # Encode image for I2V
        ref_latent = None
        if input_image is not None:
            input_image = self.preprocess_image(input_image, height, width)
            ref_latent_handler = auto_async_call(
                self.vae_stage.process,
                "encode_image",
                input_image,
                height // 2,
                width // 2,
                input_image_frame_idx,
                input_image_strength,
            )
            ref_latent = ref_latent_handler()
        # stage-1 denoising
        stage1_latents_handler = auto_async_call(
            self.stage1_denoise_stage.process,
            height=height // 2,
            width=width // 2,
            num_frames=num_frames,
            frame_rate=frame_rate,
            conditionings=ref_latent,
            positive_context=prompt_emb_posi,
            negative_context=prompt_emb_nega,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            video_noise=stage1_video_noise,
            audio_noise=stage1_audio_noise,
        )
        stage1_video_state, stage1_audio_state = stage1_latents_handler()
        # 2x upsampling
        upscaled_latent = self.upsampler_stage.process(stage1_video_state.latent[:1])
        # encode image for I2V
        stage2_conditionings = auto_async_call(
            self.vae_stage.process,
            "encode_image",
            input_image,
            height,
            width,
            input_image_frame_idx,
            input_image_strength,
            is_ray=self.config.enable_vae_ray,
        )()
        # stage-2 denoising
        stage2_sigmas = auto_async_call(self.stage2_denoise_stage.stage2_sigmas, self.stage2_denoise_stage.device)()
        stage2_states_handler = auto_async_call(
            self.stage2_denoise_stage.process,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            conditionings=stage2_conditionings,
            positive_context=prompt_emb_posi,
            negative_context=prompt_emb_nega,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            video_noise=stage2_video_noise,
            audio_noise=stage2_audio_noise,
            initial_video_latent=upscaled_latent,
            initial_audio_latent=stage1_audio_state.latent,
            sigmas=stage2_sigmas,
            noise_scale=float(stage2_sigmas[0].item()),
            simple_denoise=True,
        )
        stage2_video_state, stage2_audio_state = stage2_states_handler()

        vae_decode_generator = None
        if not self.config.enable_vae_ray:
            vae_decode_generator = torch.Generator(device=self.device).manual_seed(seed)
        frames_handler = auto_async_call(
            self.vae_stage.process, "decode_video", stage2_video_state.latent, vae_decode_generator, **tiler_kwargs
        )
        frames = frames_handler()
        frames = self.tensor2video(frames[0])

        waveform, sample_rate = self._decode_audio(stage2_audio_state.latent)
        return frames, waveform, sample_rate

    def __del__(self):
        del self.vae_stage
        del self.upsampler_stage
        del self.text_encoding_stage
        del self.stage1_denoise_stage
        del self.stage2_denoise_stage

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: str | None = None,
        quantization: object | None = None,
        attention_config: AttentionConfig | None = None,
        enable_parallel: bool = False,
        parallel_devices: list[int] | None = None,
        sample_solver: str = "euler",
        stage2_lora_strength: float = 1.0,
        **kwargs,
    ) -> "LTX23VideoPipeline":
        """Load an LTX 2.3 video pipeline from a model root or HuggingFace identifier."""
        from telefuser.models.ltx_gemma_text_encoder import GemmaTextEncoder
        from telefuser.utils.hf_model_utils import resolve_hf_path

        # Resolve model path (download if needed)
        model_root = resolve_hf_path(model_id_or_path, cache_dir)
        logger.info(f"Loading LTX2.3 pipeline from: {model_root}")
        if quantization is not None:
            logger.warning("Ignoring quantization for the registered LTX 2.3 dev loading path.")
        if not math.isclose(stage2_lora_strength, 1.0):
            logger.warning("Ignoring stage2_lora_strength because distilled loading is disabled in dev-only mode.")

        # Use analyzer to discover components
        from telefuser.utils.hf_model_analyzer import HFModelAnalyzer

        analyzer = HFModelAnalyzer(model_root)

        # Get component paths
        checkpoint_path = analyzer.get_transformer_path()
        gemma_root = analyzer.get_gemma_root_path()
        spatial_upsampler_path = analyzer.get_upsampler_path()
        distilled_lora_path = analyzer.get_distilled_lora_path()

        if not checkpoint_path:
            raise ValueError(f"No transformer/DiT weights found in {model_root}")
        if not gemma_root:
            raise ValueError(f"No Gemma root found in {model_root}")
        if not spatial_upsampler_path:
            raise ValueError(f"No spatial upsampler weights found in {model_root}")

        logger.info(f"  Transformer: {checkpoint_path}")
        logger.info(f"  Gemma root: {gemma_root}")
        logger.info(f"  Spatial upsampler: {spatial_upsampler_path}")

        if distilled_lora_path:
            logger.info(f"  Distilled LoRA detected but ignored for dev-only loading: {distilled_lora_path}")

        mm = ModuleManager(torch_dtype=torch_dtype, device="cpu")
        mm.load_model(checkpoint_path, device="cpu", torch_dtype=torch_dtype)
        mm.load_model(spatial_upsampler_path, device="cpu", torch_dtype=torch_dtype)
        mm.load_from_huggingface(
            gemma_root,
            module_source="transformers",
            module_name="ltx_text_encoder",
            module_class=GemmaTextEncoder,
            device="cpu",
            torch_dtype=torch_dtype,
        )

        pipeline = cls(device=device, torch_dtype=torch_dtype)
        config = LTXVideoPipelineConfig(stage2_lora_strength=stage2_lora_strength)
        config.sample_solver = sample_solver

        if attention_config is not None:
            config.dit_stage1_config.attention_config = attention_config
            config.dit_stage2_config.attention_config = attention_config

        if enable_parallel and parallel_devices:
            config.dit_stage1_config.parallel_config.device_ids = parallel_devices
            config.dit_stage1_config.parallel_config.sp_ulysses_degree = len(parallel_devices)
            config.dit_stage2_config.parallel_config.device_ids = parallel_devices
            config.dit_stage2_config.parallel_config.sp_ulysses_degree = len(parallel_devices)
            config.enable_denoising_parallel = True

        if "gemma_text_encoding_config" in kwargs and "text_encoding_config" not in kwargs:
            kwargs["text_encoding_config"] = kwargs.pop("gemma_text_encoding_config")

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Initialize pipeline
        pipeline.init(mm, config)

        logger.info(f"Successfully loaded LTX2.3 pipeline from {model_id_or_path}")
        return pipeline
