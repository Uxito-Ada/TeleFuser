from __future__ import annotations

from functools import partial
from typing import Any

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_cfg_rank, get_cfg_world_size
from telefuser.distributed.fsdp import shard_model
from telefuser.distributed.parallel_shard import cfg_parallel_unshard
from telefuser.feature_cache.ada_taylor_cache.ada_taylor_cache import AdaTaylorCache
from telefuser.metrics import with_metrics
from telefuser.ops.quantized_linear import convert_params_to_buffers
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.utils.lora_loader import LoRALoader
from telefuser.utils.profiler import ProfilingContext4Debug

from .latent_data_utils import parse_latent_data


class MoeDitDenoisingStage(BaseStage):
    """MoE (Mixture of Experts) DiT denoising stage for Wan2.2 video generation.

    Uses two separate DiT models: high-quality expert (dit_high) for early timesteps
    and low-quality expert (dit_low) for later timesteps. Models switch at a
    boundary timestep (default 0.875) for efficient inference.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        dit_high_runtime_config: ModelRuntimeConfig,
        dit_low_runtime_config: ModelRuntimeConfig,
        scheduler: Any,
    ) -> None:
        super().__init__(name, dit_high_runtime_config)
        self.num_train_timesteps = 1000
        self.dit_high, self.dit_low = module_manager.fetch_module(
            "wan_video_dit",
            index=2,
        )
        self.dit_high.set_attention_config(dit_high_runtime_config.attention_config)
        self.dit_low.set_attention_config(dit_low_runtime_config.attention_config)
        self.model_names = ["dit_high", "dit_low"]
        self.scheduler = scheduler
        self.dit_high_runtime_config = dit_high_runtime_config
        self.dit_low_runtime_config = dit_low_runtime_config
        self.load_loras()
        self.dit_high_onload_flag = False
        self.dit_low_onload_flag = False
        self.dit_high_cpu_offload = (
            dit_high_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD
        )
        self.dit_low_cpu_offload = (
            dit_low_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD
        )
        # Configure CPU offloading for dit_high
        if dit_high_runtime_config.offload_config.offload_type == WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD:
            if dit_high_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and sequential cpu offload can not be used together")
            else:
                logger.info("enable sequential cpu offload for dit high")
                self.dit_high.enable_sequential_cpu_offload(device=self.device, torch_dtype=self.torch_dtype)
        if dit_high_runtime_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            if dit_high_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and async cpu offload can not be used together")
            else:
                logger.info("enable async cpu offload for dit high")
                self.dit_high.enable_async_offload(self.device, dit_high_runtime_config.offload_config)

        # Configure CPU offloading for dit_low
        if dit_low_runtime_config.offload_config.offload_type == WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD:
            if dit_low_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and sequential cpu offload can not be used together")
            else:
                logger.info("enable sequential cpu offload for dit low")
                self.dit_low.enable_sequential_cpu_offload(device=self.device, torch_dtype=self.torch_dtype)
        if dit_low_runtime_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            if dit_low_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and async cpu offload can not be used together")
            else:
                logger.info("enable async cpu offload for dit low")
                self.dit_low.enable_async_offload(self.device, dit_low_runtime_config.offload_config)

        # Handle torch.compile for single GPU mode
        parallel_cfg = dit_high_runtime_config.parallel_config
        if parallel_cfg.world_size == 1 and (
            dit_high_runtime_config.compile_config.enabled or dit_low_runtime_config.compile_config.enabled
        ):
            if dit_high_runtime_config.compile_config.enabled:
                logger.info("enable torch.compile for dit_high")
                self.dit_high = torch.compile(
                    self.dit_high, **dit_high_runtime_config.compile_config.get_compile_kwargs()
                )
            if dit_low_runtime_config.compile_config.enabled:
                logger.info("enable torch.compile for dit_low")
                self.dit_low = torch.compile(self.dit_low, **dit_low_runtime_config.compile_config.get_compile_kwargs())

    def load_loras(self):
        """Load LoRA weights into both DiT models."""
        lora_configs_high = self.dit_high_runtime_config.lora_configs
        lora_loader = LoRALoader()
        for lora_config in lora_configs_high:
            lora_path = lora_config.path
            strength = lora_config.strength
            lora_loader.apply_lora(self.dit_high, lora_path, strength=strength)
            logger.info(f"Loaded LoRA: {lora_path} with strength: {strength} for dit high")
        lora_configs_low = self.dit_low_runtime_config.lora_configs
        for lora_config in lora_configs_low:
            lora_path = lora_config.path
            strength = lora_config.strength
            lora_loader.apply_lora(self.dit_low, lora_path, strength=strength)
            logger.info(f"Loaded LoRA: {lora_path} with strength: {strength} for dit low")

    def predict_noise_with_cfg(
        self,
        dit: Any,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Predict noise with classifier-free guidance."""
        if cfg_scale == 1.0:
            return dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                context=prompt_emb_posi,
            )

        # Check if CFGP is enabled
        cfgp_enabled = get_cfg_world_size(dit.device_mesh) > 1

        if cfgp_enabled:
            # CFGP mode: each rank processes one branch (posi or nega)
            cfg_rank = get_cfg_rank(dit.device_mesh)
            is_posi = cfg_rank == 0

            context = prompt_emb_posi if is_posi else prompt_emb_nega
            cond_flag = is_posi

            noise_pred = dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=cond_flag,
                context=context,
            )

            # Unshard and compute CFG
            noise_pred = cfg_parallel_unshard(dit.device_mesh, [noise_pred])[0]
            noise_pred_posi = noise_pred[0:1]
            noise_pred_nega = noise_pred[1:2]
        else:
            # Non-CFGP mode: sequential forward passes
            noise_pred_posi = dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                context=prompt_emb_posi,
            )
            noise_pred_nega = dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=False,
                context=prompt_emb_nega,
            )

        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        return noise_pred

    @ProfilingContext4Debug("moe_dit_denosing")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        ref_latent: torch.Tensor | None,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: torch.Tensor | None,
        cfg_scale_high: float,
        cfg_scale_low: float,
        sigma_shift: float,
        boundary: float,
        latent_data: dict | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Run denoising with MoE model switching at boundary timestep.

        Uses dit_high for early timesteps (> boundary * 1000) and
        dit_low for later timesteps for efficiency.

        When ``latent_data`` is provided, supports cross-request cache:
        skips the first ``skip_step`` steps using the cached latent, and
        snapshots latents at ``saved_steps`` for future reuse. Returns a
        ``(latents, payload)`` tuple in that case; otherwise returns
        ``latents`` to preserve backward compatibility.
        """
        # set mix euler scheduler
        if not isinstance(num_inference_steps, int):
            self.scheduler.set_timesteps(
                num_inference_steps,
                shift=sigma_shift,
                boundary=boundary,
            )
            num_inference_steps = num_inference_steps[0]
        else:
            self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        timesteps = self.scheduler.timesteps
        total_steps = len(timesteps)
        boundary_t = boundary * self.num_train_timesteps

        cached_latent, effective_start_step, saved_steps = parse_latent_data(
            latent_data, expected_shape=tuple(latents.shape), total_steps=total_steps
        )
        if cached_latent is not None:
            latents = cached_latent.to(device=latents.device, dtype=latents.dtype)

        saved_steps_set = frozenset(saved_steps)
        latent_states_dict: dict[int, torch.Tensor] = {}

        start_in_dit_low = effective_start_step > 0 and timesteps[effective_start_step] < boundary_t

        if start_in_dit_low:
            # Cached latent is past the MoE boundary timestep; skip dit_high entirely.
            if self.dit_high_cpu_offload and self.dit_high_onload_flag:
                logger.info("offload dit high (skip past MoE boundary)")
                self.dit_high.offload_device()
                self.dit_high_onload_flag = False
            if self.dit_low_cpu_offload or (not self.dit_low_onload_flag):
                logger.info(f"onload dit low to {self.device}")
                self.dit_low.onload_device(self.device)
                self.dit_low_onload_flag = True
            dit = self.dit_low
            cfg_scale = cfg_scale_low
            self.setup_feature_cache(
                self.dit_low,
                self.dit_low_runtime_config.feature_cache_config,
                num_inference_steps,
                init_step=effective_start_step,
            )
        else:
            # Start with dit_high; resumes mid-way when skip_step > 0.
            cache_config_dit_high = self.dit_high_runtime_config.feature_cache_config
            self.setup_feature_cache(
                self.dit_high,
                cache_config_dit_high,
                num_inference_steps,
                init_step=effective_start_step,
            )
            if self.dit_high_cpu_offload or (not self.dit_high_onload_flag):
                logger.info(f"onload dit high to {self.device}")
                self.dit_high.onload_device(self.device)
                self.dit_high_onload_flag = True
            dit = self.dit_high
            cfg_scale = cfg_scale_high

        compute_count = 0
        skip_count = 0
        cfgp_enabled = get_cfg_world_size(dit.device_mesh) > 1
        is_cond_for_me = (not cfgp_enabled) or (get_cfg_rank(dit.device_mesh) == 0)
        branch = "cond" if is_cond_for_me else "uncond"
        pbar = tqdm(timesteps[effective_start_step:], desc="dit denoise")
        for progress_id, timestep in enumerate(pbar):
            absolute_step = effective_start_step + progress_id

            # MoE boundary switch (only reachable when we started in dit_high)
            if (not start_in_dit_low) and timestep < boundary_t and dit is self.dit_high:
                if self.dit_high_cpu_offload:
                    logger.info("offload dit high")
                    self.dit_high.offload_device()
                if self.dit_low_cpu_offload or (not self.dit_low_onload_flag):
                    logger.info(f"onload dit low to {self.device}")
                    self.dit_low.onload_device(self.device)
                    self.dit_low_onload_flag = True
                dit = self.dit_low
                cfg_scale = cfg_scale_low
                # Log dit_high final stats before resetting counters for dit_low
                dit_high_cache = self.dit_high.feature_cache
                if isinstance(dit_high_cache, AdaTaylorCache):
                    state_before = dit_high_cache.cond_state if is_cond_for_me else dit_high_cache.uncond_state
                    logger.info(
                        f"[{branch}] dit_high done compute={compute_count} skip={skip_count} "
                        f"compute_steps={sorted(state_before.compute_steps)}"
                    )
                compute_count = 0
                skip_count = 0
                self.setup_feature_cache(
                    self.dit_low,
                    self.dit_low_runtime_config.feature_cache_config,
                    num_inference_steps,
                    init_step=absolute_step,
                )
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            input_latent = latents
            if ref_latent is not None:
                input_latent = torch.cat([input_latent, ref_latent], dim=1)
            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                input_latent = input_latent.to(self.torch_dtype)
                noise_pred = self.predict_noise_with_cfg(
                    dit=dit,
                    latents=input_latent,
                    timestep=timestep,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    cfg_scale=cfg_scale,
                )

            # Snapshot BEFORE scheduler.step so step k captures the input to step k.
            if absolute_step in saved_steps_set:
                latent_states_dict[absolute_step] = latents.detach().cpu()

            latents = self.scheduler.step(noise_pred, timesteps[absolute_step], latents)
            # Update feature cache compute/skip stats for progress display
            cache = dit.feature_cache
            if isinstance(cache, AdaTaylorCache):
                was_compute = cache.last_step_was_compute(is_cond=is_cond_for_me)
                if was_compute:
                    compute_count += 1
                else:
                    skip_count += 1
                logger.debug(f"[{branch}] step={absolute_step} {'compute' if was_compute else 'skip'}")
                pbar.set_postfix_str(f"[{branch}] c={compute_count} s={skip_count}")
        # After the for-loop: log dit_low final stats
        if isinstance(dit.feature_cache, AdaTaylorCache):
            final_cache = dit.feature_cache
            final_state = final_cache.cond_state if is_cond_for_me else final_cache.uncond_state
            logger.info(
                f"[{branch}] dit_low done compute={compute_count} skip={skip_count} "
                f"compute_steps={sorted(final_state.compute_steps)}"
            )
        if self.dit_low_cpu_offload:
            logger.info("offload dit low")
            self.dit_low.offload_device()

        if latent_data is not None:
            latent_payload = {
                "latent_states_dict": latent_states_dict,
                "saved_steps": saved_steps,
                "final_step": total_steps - 1,
            }
            return latents, latent_payload
        return latents

    def parallel_models(self):
        """Configure parallel processing for both DiT models."""
        # Configure dit_high parallelism
        parallel_cfg = self.dit_high_runtime_config.parallel_config
        self.dit_high.device_mesh = create_device_mesh_from_config(parallel_cfg)
        self.dit_high.set_attention_config(self.dit_high_runtime_config.attention_config)
        print(f"dit high device mesh: {self.dit_high.device_mesh}")
        # Note: CFGP is handled at stage level, no need to configure on model
        if parallel_cfg.sp_ulysses_degree > 1:
            self.dit_high.enable_usp()
        if parallel_cfg.enable_fsdp:
            logger.info(f"enable fsdp for {self.name} dit high")
            if self.dit_high.quant_type is not None:
                self.dit_high = convert_params_to_buffers(self.dit_high, self.dit_high.quant_type)
            shard_fn = partial(
                shard_model,
                wrap_module_names=self.dit_high.get_fsdp_module_names(),
                param_dtype=self.dit_high.quant_type if self.dit_high.quant_type is not None else self.torch_dtype,
            )
            self.dit_high = shard_fn(module=self.dit_high, device_id=self.device)
            if self.dit_high_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                self.dit_high.cpu()
                current_platform.empty_cache()

        # Configure dit_low parallelism
        parallel_cfg = self.dit_low_runtime_config.parallel_config
        self.dit_low.device_mesh = create_device_mesh_from_config(parallel_cfg)
        self.dit_low.set_attention_config(self.dit_low_runtime_config.attention_config)
        # Note: CFGP is handled at stage level, no need to configure on model
        if parallel_cfg.sp_ulysses_degree > 1:
            self.dit_low.enable_usp()
        if parallel_cfg.enable_fsdp:
            logger.info(f"enable fsdp for {self.name} dit low")
            if self.dit_low.quant_type is not None:
                self.dit_low = convert_params_to_buffers(self.dit_low, self.dit_low.quant_type)
            shard_fn = partial(
                shard_model,
                wrap_module_names=self.dit_low.get_fsdp_module_names(),
                param_dtype=self.dit_low.quant_type if self.dit_low.quant_type is not None else self.torch_dtype,
            )
            self.dit_low = shard_fn(module=self.dit_low, device_id=self.device)
            if self.dit_low_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                self.dit_low.cpu()
                current_platform.empty_cache()

        # Handle torch.compile for distributed mode
        if self.dit_high_runtime_config.compile_config.enabled or self.dit_low_runtime_config.compile_config.enabled:
            if self.dit_high_runtime_config.compile_config.enabled:
                logger.info("enable torch.compile for dit_high")
                self.dit_high = torch.compile(
                    self.dit_high, **self.dit_high_runtime_config.compile_config.get_compile_kwargs()
                )
            if self.dit_low_runtime_config.compile_config.enabled:
                logger.info("enable torch.compile for dit_low")
                self.dit_low = torch.compile(
                    self.dit_low, **self.dit_low_runtime_config.compile_config.get_compile_kwargs()
                )
