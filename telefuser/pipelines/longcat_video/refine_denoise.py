"""Official LongCat refinement denoising stage.

Implements the LongCat-Video refinement pipeline: load refinement LoRA,
upsample latents from base resolution to target (e.g. 480p→720p), add noise
at t_thresh level, and denoise without CFG.

Reference: https://github.com/meituan-longcat/LongCat-Video
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.distributed.fsdp import shard_model
from telefuser.metrics import with_metrics
from telefuser.models.longcat_video_dit import LongCatVideoTransformer3DModel
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.utils.lora_network import REFINE_LORA_KEY
from telefuser.utils.profiler import ProfilingContext4Debug


class LongCatRefineDenoisingStage(BaseStage):
    """Official LongCat refinement denoising stage.

    Uses refinement LoRA + no CFG + t_thresh-based noise schedule.
    Reuses the same DiT model as base denoising with dynamic LoRA switching.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        scheduler: Any,
    ):
        super().__init__(name, model_runtime_config)
        self.dit: LongCatVideoTransformer3DModel = module_manager.fetch_module("wan_video_dit")
        self.model_names = ["dit"]
        self.scheduler = scheduler
        self.num_timesteps = 1000
        self._pending_async_offload = False

        if model_runtime_config.offload_config.offload_type == WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD:
            if model_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and sequential cpu offload can not be used together")
            else:
                logger.info("enable sequential cpu offload for longcat refine dit")
                self.dit.enable_sequential_cpu_offload(self.device, self.torch_dtype)

        if model_runtime_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            if model_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and async cpu offload can not be used together")
            else:
                logger.info("enable async cpu offload for refine dit")
                self._pending_async_offload = True

    @with_model_offload(["dit"])
    @ProfilingContext4Debug("refine_dit_denoise")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        lq_latents: torch.Tensor,
        num_refine_steps: int,
        num_cond_latents: int,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        t_thresh: float = 0.5,
        attn_impl: str = "sdpa",
        seed: int | None = None,
        progress_bar_cmd: Callable = tqdm,
        enable_bsa: bool = False,
    ) -> torch.Tensor:
        """Run official LongCat refinement denoising.

        No CFG (cfg_scale=1.0). Uses refinement LoRA if loaded.

        Args:
            lq_latents: Upsampled low-quality latents (B, C, T, H, W).
            num_refine_steps: Number of refinement denoising steps.
            num_cond_latents: Number of conditioning latent frames (0 for T2V).
            prompt_embeds: Text prompt embeddings.
            prompt_attention_mask: Text attention mask.
            t_thresh: Noise threshold [0, 1]. Denoising starts from t_thresh * 1000.
            attn_impl: Attention implementation.
            seed: Random seed for noise.
            progress_bar_cmd: Progress bar callable.
            enable_bsa: Enable block sparse attention for acceleration.

        Returns:
            Refined latents (B, C, T, H, W).
        """
        if self._pending_async_offload:
            self.dit.enable_async_offload(self.device, self.model_runtime_config.offload_config)
            self._pending_async_offload = False

        # Enable refinement LoRA if loaded
        has_refine_lora = REFINE_LORA_KEY in self.dit.lora_dict
        if has_refine_lora:
            self.dit.enable_loras([REFINE_LORA_KEY])
            logger.info(f"enabled {REFINE_LORA_KEY} for refine stage")

        # Enable BSA if requested
        if enable_bsa:
            self.dit.enable_bsa()
            logger.info("enabled BSA for refine stage")

        try:
            latents = self._denoise(
                lq_latents,
                num_refine_steps,
                num_cond_latents,
                prompt_embeds,
                prompt_attention_mask,
                t_thresh,
                attn_impl,
                seed,
                progress_bar_cmd,
            )
        finally:
            # Always disable LoRA after refine
            if has_refine_lora:
                self.dit.disable_all_loras()
                logger.info(f"disabled {REFINE_LORA_KEY} after refine stage")
            if enable_bsa:
                self.dit.disable_bsa()
                logger.info("disabled BSA after refine stage")

        self.dit.clear_cache()
        return latents

    def _denoise(
        self,
        lq_latents: torch.Tensor,
        num_refine_steps: int,
        num_cond_latents: int,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        t_thresh: float,
        attn_impl: str,
        seed: int | None,
        progress_bar_cmd: Callable,
    ) -> torch.Tensor:
        """Core denoising loop."""
        # Generate noise
        generator = torch.Generator(device=lq_latents.device)
        if seed is not None:
            generator.manual_seed(seed)
        noise = torch.randn(lq_latents.shape, generator=generator, device=lq_latents.device, dtype=torch.float32)
        noise = noise.to(dtype=lq_latents.dtype)

        # SDEdit: add noise at t_thresh level
        latents = (1.0 - t_thresh) * lq_latents + t_thresh * noise

        # Set up scheduler: standard sigmas, then filter to start from t_thresh
        sigmas = torch.linspace(1, 0.001, num_refine_steps, dtype=torch.float32)
        self.scheduler.set_timesteps(num_refine_steps, sigmas=sigmas, device=self.device)

        # Filter timesteps: start from t_thresh * 1000
        timesteps = self.scheduler.timesteps
        t_thresh_value = t_thresh * self.num_timesteps
        t_thresh_tensor = torch.tensor(t_thresh_value, dtype=timesteps.dtype, device=timesteps.device)
        timesteps = torch.cat([t_thresh_tensor.unsqueeze(0), timesteps[timesteps < t_thresh_tensor]])
        self.scheduler.timesteps = timesteps
        self.scheduler.sigmas = torch.cat([timesteps / self.num_timesteps, torch.zeros(1, device=timesteps.device)])

        # Denoising loop (no CFG — only positive prompt)
        for t in progress_bar_cmd(timesteps, desc="longcat refine denoise"):
            timestep = t.expand(latents.shape[0]).to(self.torch_dtype)
            timestep = timestep.unsqueeze(-1).repeat(1, latents.shape[2])

            # Condition frames don't get denoised
            if num_cond_latents > 0:
                timestep[:, :num_cond_latents] = 0

            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                latents = latents.to(self.torch_dtype)
                noise_pred = self.dit.forward(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    attn_impl=attn_impl,
                    num_cond_latents=num_cond_latents,
                ).to(latents.device)

            # LongCat convention: negate noise prediction
            noise_pred = -noise_pred

            # Scheduler step: only update non-condition frames
            if num_cond_latents > 0:
                latents[:, :, num_cond_latents:] = self.scheduler.step(
                    noise_pred[:, :, num_cond_latents:],
                    t,
                    latents[:, :, num_cond_latents:],
                    return_dict=False,
                )[0]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents

    def parallel_models(self):
        parallel_cfg = self.model_runtime_config.parallel_config

        self.dit.device_mesh = create_device_mesh_from_config(parallel_cfg)

        if parallel_cfg.cfg_degree > 1:
            # Refine doesn't use CFG, but keep consistency with base stage
            logger.info("longcat refine dit: cfg_degree > 1 but refine uses no CFG")

        if parallel_cfg.sp_ulysses_degree > 1:
            self.dit.enable_usp()
            logger.info("longcat refine dit enabled usp")

        if parallel_cfg.enable_fsdp:
            logger.info(f"enable fsdp for {self.name}")
            if hasattr(self.dit, "quant_type") and self.dit.quant_type is not None:
                from telefuser.ops.quantized_linear import convert_params_to_buffers

                self.dit = convert_params_to_buffers(self.dit, self.dit.quant_type)
            shard_fn = partial(shard_model, wrap_module_names=self.dit.get_fsdp_module_names())
            self.dit = shard_fn(module=self.dit, device_id=self.device)
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                self.dit.cpu()
                current_platform.empty_cache()
