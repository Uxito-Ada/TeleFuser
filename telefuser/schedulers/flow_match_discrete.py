# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.
#
# ==============================================================================
# Modified from diffusers and HunyuanVideo-1.5
# ==============================================================================

"""FlowMatch discrete scheduler for HunyuanVideo.

This implementation is based on the original HunyuanVideo-1.5 repository:
https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from telefuser.utils.logging import logger


@dataclass
class FlowMatchDiscreteSchedulerOutput:
    """Output class for the scheduler's step function output.

    Args:
        prev_sample: Computed sample (x_{t-1}) of previous timestep.
    """

    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler:
    """Euler scheduler for flow matching.

    Args:
        num_train_timesteps: The number of diffusion steps to train the model.
        shift: The shift value for the timestep schedule.
        reverse: Whether to reverse the timestep schedule.
        solver: The solver to use ("euler").
        use_flux_shift: Whether to use flux shift.
        flux_base_shift: Base shift for flux.
        flux_max_shift: Max shift for flux.
        n_tokens: Number of tokens for flux shift calculation.
        flux_base_token: Base token for flux shift.
        flux_max_token: Max token for flux shift.
        flux_shift_factor: Shift factor for flux.
    """

    _compatibles = []
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        solver: str = "euler",
        use_flux_shift: bool = False,
        flux_base_shift: float = 0.5,
        flux_max_shift: float = 1.15,
        n_tokens: Optional[int] = None,
        flux_base_token: float = 256.0,
        flux_max_token: float = 4096.0,
        flux_shift_factor: float = 1.0,
    ):
        self.config = type(
            "Config",
            (),
            {
                "num_train_timesteps": num_train_timesteps,
                "shift": shift,
                "reverse": reverse,
                "solver": solver,
                "use_flux_shift": use_flux_shift,
                "flux_base_shift": flux_base_shift,
                "flux_max_shift": flux_max_shift,
                "flux_base_token": flux_base_token,
                "flux_max_token": flux_max_token,
                "flux_shift_factor": flux_shift_factor,
            },
        )()

        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        # the value fed to model
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)

        self._step_index = None
        self._begin_index = None

        self.supported_solver = ["euler"]
        if solver not in self.supported_solver:
            raise ValueError(f"Solver {solver} not supported. Supported solvers: {self.supported_solver}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        shift: float = 1.0,
        **kwargs,
    ) -> "FlowMatchDiscreteScheduler":
        """Load scheduler from pretrained checkpoint.

        Args:
            pretrained_model_name_or_path: Path to the pretrained model directory
            shift: Shift value for the scheduler
            **kwargs: Additional arguments

        Returns:
            Loaded FlowMatchDiscreteScheduler
        """
        # Load scheduler config if available
        config_path = os.path.join(pretrained_model_name_or_path, "scheduler_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        # Use shift from argument if provided, otherwise from config
        if shift != 1.0:
            config["shift"] = shift
        elif "shift" not in config:
            config["shift"] = shift

        # Create scheduler with config
        scheduler = cls(
            num_train_timesteps=config.get("num_train_timesteps", 1000),
            shift=config.get("shift", 1.0),
            reverse=config.get("reverse", True),
            solver=config.get("solver", "euler"),
            use_flux_shift=config.get("use_flux_shift", False),
            flux_base_shift=config.get("flux_base_shift", 0.5),
            flux_max_shift=config.get("flux_max_shift", 1.15),
            flux_base_token=config.get("flux_base_token", 256.0),
            flux_max_token=config.get("flux_max_token", 4096.0),
            flux_shift_factor=config.get("flux_shift_factor", 1.0),
        )

        logger.info(f"Loaded FlowMatchDiscreteScheduler from {pretrained_model_name_or_path}")

        return scheduler

    @property
    def step_index(self):
        """The index counter for current timestep."""
        return self._step_index

    @property
    def begin_index(self):
        """The index for the first timestep."""
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """Sets the begin index for the scheduler.

        Args:
            begin_index: The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        n_tokens: int = None,
        shift: Optional[float] = None,
    ):
        """Sets the discrete timesteps used for the diffusion chain.

        Args:
            num_inference_steps: The number of diffusion steps.
            device: The device to which the timesteps should be moved.
            n_tokens: Number of tokens in the input sequence.
            shift: Optional shift value override.
        """
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)

        # Apply timestep shift
        if self.config.use_flux_shift and n_tokens is not None:
            mu = self.get_lin_function(
                x1=self.config.flux_base_token,
                x2=self.config.flux_max_token,
                y1=self.config.flux_base_shift * self.config.flux_shift_factor,
                y2=self.config.flux_max_shift * self.config.flux_shift_factor,
            )(n_tokens)
            sigmas = self.flux_time_shift(mu, 1.0, sigmas)
        elif shift is not None and shift != 1.0:
            sigmas = self.sd3_time_shift(sigmas, shift)
        elif self.config.shift != 1.0:
            sigmas = self.sd3_time_shift(sigmas, self.config.shift)

        if not self.config.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)

        # Reset step index
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """Scale model input (no-op for this scheduler)."""
        return sample

    @staticmethod
    def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
        """Get linear interpolation function."""
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @staticmethod
    def flux_time_shift(mu: float, sigma: float, t: torch.Tensor):
        """Apply flux time shift."""
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def sd3_time_shift(self, t: torch.Tensor, shift: Optional[float] = None):
        """Apply SD3 time shift."""
        if shift is None:
            shift = self.config.shift
        return (shift * t) / (1 + (shift - 1) * t)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        n_tokens: Optional[int] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        """Predict the sample from the previous timestep by reversing the SDE.

        Args:
            model_output: The direct output from learned diffusion model.
            timestep: The current discrete timestep in the diffusion chain.
            sample: A current instance of a sample created by the diffusion process.
            generator: A random number generator.
            n_tokens: Number of tokens in the input sequence.
            return_dict: Whether or not to return a dataclass output.

        Returns:
            SchedulerOutput or tuple containing the previous sample.
        """

        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]

        if self.config.solver == "euler":
            prev_sample = sample + model_output.float() * dt
        else:
            raise ValueError(f"Solver {self.config.solver} not supported. Supported solvers: {self.supported_solver}")

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
