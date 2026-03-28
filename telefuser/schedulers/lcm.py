"""LCM (Latent Consistency Model) scheduler for fast inference.

Implements the LCM distillation approach for few-step generation.
"""

from __future__ import annotations

import torch

from telefuser.utils.logging import logger


class LCMScheduler:
    """LCM scheduler for accelerated diffusion sampling."""

    def __init__(self, num_train_timesteps: int, shift: float):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = 1.0
        self.sigma_min = 0.0

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str,
        shift: float | None = None,
        seed: int | None = None,
    ):
        """Set timesteps for LCM sampling.

        Args:
            num_inference_steps: Number of denoising steps (typically 4-8)
            device: Target device for tensors
            shift: Timestep shift factor
            seed: Random seed for noise generation
        """
        if shift is None:
            shift = self.shift
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        step = self.num_train_timesteps // num_inference_steps
        self.denoising_step_list = [int(self.num_train_timesteps - i * step) for i in range(num_inference_steps)]
        logger.info(f"denoising step list is {self.denoising_step_list}")
        self.num_inference_steps = num_inference_steps
        self.device = device

        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, self.num_train_timesteps + 1)[:-1]
        self.sigmas = shift * self.sigmas / (1 + (shift - 1) * self.sigmas)

        self.denoising_step_index = [self.num_train_timesteps - x for x in self.denoising_step_list]
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.timesteps = self.timesteps[self.denoising_step_index].to(device)
        self.sigmas = self.sigmas[self.denoising_step_index]

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Add noise to samples."""
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Single LCM denoising step."""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        noisy_image_or_video = sample - sigma * model_output
        if timestep_id < self.num_inference_steps - 1:
            sigma = self.sigmas[timestep_id + 1].item()
            noise = torch.randn(
                noisy_image_or_video.shape,
                dtype=torch.float32,
                device=self.device,
                generator=self.generator,
            )
            noisy_image_or_video = self.add_noise(
                noisy_image_or_video,
                noise=noise,
                sigma=self.sigmas[timestep_id + 1].item(),
            )
        return noisy_image_or_video
