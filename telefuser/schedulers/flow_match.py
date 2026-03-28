"""Flow Matching scheduler for diffusion models.

Implements flow-based diffusion scheduling used by FLUX, Wan, Qwen-Image,
and other modern diffusion models.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import torch


def merge_tensors_with_threshold(A, B, T):
    A_selected = A[A >= T]
    B_selected = B[B < T]
    C = torch.cat((A_selected, B_selected))
    return C


class FlowMatchScheduler:
    """Flow matching scheduler with model-specific timestep schedules.

    Supports different scheduling strategies for various model architectures.
    """

    def __init__(
        self,
        template: Literal["FLUX.1", "Wan", "Qwen-Image", "FLUX.2", "Z-Image", "Wan-Mix", "LTX.2"] = "FLUX.1",
    ) -> None:
        self.set_timesteps_fn = {
            "FLUX.1": FlowMatchScheduler.set_timesteps_flux,
            "Wan": FlowMatchScheduler.set_timesteps_wan,
            "Qwen-Image": FlowMatchScheduler.set_timesteps_qwen_image,
            "FLUX.2": FlowMatchScheduler.set_timesteps_flux2,
            "Z-Image": FlowMatchScheduler.set_timesteps_z_image,
            "Wan-Mix": FlowMatchScheduler.set_mix_timesteps_wan,
            "LTX.2": FlowMatchScheduler.set_timesteps_ltx2,
        }.get(template, FlowMatchScheduler.set_timesteps_flux)
        self.num_train_timesteps = 1000
        self.template = template

    @staticmethod
    def set_timesteps_flux(
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        shift: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FLUX.1 timestep schedule with shift."""
        sigma_min = 0.003 / 1.002
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_mix_timesteps_wan(
        num_inference_steps: Tuple[int],
        shift: Tuple[float] | None = None,
        boundary: float = 0.9,
        denoising_strength: float = 1.0,
        num_train_timesteps: int = 1000,
    ):
        sigmas_1, timesteps_1 = FlowMatchScheduler.set_timesteps_wan(
            num_inference_steps[0], denoising_strength, shift[0]
        )
        sigmas_2, timesteps_2 = FlowMatchScheduler.set_timesteps_wan(
            num_inference_steps[1], denoising_strength, shift[1]
        )
        sigmas = merge_tensors_with_threshold(sigmas_1, sigmas_2, boundary)
        timesteps = merge_tensors_with_threshold(timesteps_1, timesteps_2, boundary * num_train_timesteps)
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_wan(
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        shift: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wan video model timestep schedule."""
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def _calculate_shift_qwen_image(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ) -> float:
        """Calculate shift for Qwen-Image based on sequence length."""
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @staticmethod
    def set_timesteps_qwen_image(
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        exponential_shift_mu: float | None = None,
        dynamic_shift_len: int | None = None,
        shift_terminal: float | None = 0.02,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Qwen-Image timestep schedule with dynamic shift."""
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        # Sigmas
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Mu
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._calculate_shift_qwen_image(
                dynamic_shift_len, base_shift=base_shift, max_shift=max_shift
            )
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # Shift terminal
        if shift_terminal is not None:
            one_minus_z = 1 - sigmas
            scale_factor = one_minus_z[-1] / (1 - shift_terminal)
            sigmas = 1 - (one_minus_z / scale_factor)
        # Timesteps
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def compute_empirical_mu(
        image_seq_len: int,
        num_steps: int,
    ) -> float:
        """Compute empirical mu for FLUX.2 scheduling."""
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)

    @staticmethod
    def set_timesteps_flux2(
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        dynamic_shift_len: int = 1024 // 16 * 1024 // 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FLUX.2 timestep schedule with empirical mu."""
        sigma_min = 1 / num_inference_steps
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        mu = FlowMatchScheduler.compute_empirical_mu(dynamic_shift_len, num_inference_steps)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_z_image(
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        shift: float | None = None,
        target_timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Z-Image timestep schedule."""
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        if target_timesteps is not None:
            target_timesteps = target_timesteps.to(dtype=timesteps.dtype, device=timesteps.device)
            for timestep in target_timesteps:
                timestep_id = torch.argmin((timesteps - timestep).abs())
                timesteps[timestep_id] = timestep
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_ltx2(
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        latent: torch.Tensor | None = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        default_number_of_tokens: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LTX2.3 timestep schedule with token-aware exponential shift."""
        tokens = math.prod(latent.shape[2:]) if latent is not None else default_number_of_tokens
        sigmas = torch.linspace(denoising_strength, 0.0, num_inference_steps + 1)

        m = (max_shift - base_shift) / (4096 - 1024)
        b = base_shift - m * 1024
        sigma_shift = tokens * m + b
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1)),
            0,
        )

        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            sigmas[non_zero_mask] = 1.0 - (one_minus_z / scale_factor)

        # Keep endpoints exact for downstream code/tests.
        sigmas[0] = float(denoising_strength)
        sigmas[-1] = 0.0

        timesteps = sigmas[:-1] * 1000
        return sigmas.to(torch.float32), timesteps.to(torch.float32)

    def set_training_weight(self) -> None:
        """Set timestep weights for training."""
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        **kwargs: float | int | torch.Tensor | None,
    ) -> None:
        """Set timesteps for inference or training."""
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def set_timesteps_with_mu(
        self,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        num_inference_steps: int | None = None,
    ) -> None:
        """Set timesteps with custom sigmas and mu for FLUX.2-style scheduling.

        Args:
            sigmas: Custom sigma values (will be shifted by mu)
            mu: Mu value for exponential shifting
            num_inference_steps: Number of steps (derived from sigmas if not provided)
        """
        import numpy as np

        if sigmas is not None:
            sigmas = np.array(sigmas).astype(np.float32)
            if num_inference_steps is None:
                num_inference_steps = len(sigmas)
        else:
            num_inference_steps = num_inference_steps or 50
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps).astype(np.float32)

        if mu is not None:
            # Apply exponential shifting: sigma = exp(mu) / (exp(mu) + (1/sigma - 1))
            sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        to_final: bool = False,
        **kwargs: float | int | torch.Tensor | None,
    ) -> torch.Tensor:
        """Single denoising step using flow matching."""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]

        if self.template == "LTX.2":
            velocity = (sample - model_output) / sigma
            prev_sample = sample.to(torch.float32) + velocity.to(torch.float32) * (sigma_ - sigma)
            return prev_sample.to(sample.dtype)

        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(
        self,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        sample_stablized: torch.Tensor,
    ) -> torch.Tensor:
        """Revert to a specific timestep (for resampling)."""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: float | torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples at given timestep."""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute training target for flow matching."""
        target = noise - sample
        return target

    def training_weight(self, timestep: float | torch.Tensor) -> torch.Tensor:
        """Get training weight for a specific timestep."""
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
