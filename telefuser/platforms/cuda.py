"""CUDA platform implementation for NVIDIA GPUs."""

from __future__ import annotations

from typing import Any

import torch

from .interface import BasePlatform


class CudaPlatform(BasePlatform):
    """CUDA platform for NVIDIA GPU acceleration."""

    device_name: str = "cuda"
    device_type: str = "cuda"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    dispatch_key: str = "CUDA"
    dist_backend: str = "nccl"
    full_dist_backend: str = "cuda:nccl"

    @staticmethod
    def empty_cache() -> None:
        torch.cuda.empty_cache()

    @staticmethod
    def ipc_collect() -> None:
        torch.cuda.ipc_collect()

    @staticmethod
    def get_device_name() -> str:
        return torch.cuda.get_device_name()

    @staticmethod
    def device_ctx(device: int | str | torch.device) -> torch.cuda.device:
        return torch.cuda.device(device)

    @staticmethod
    def default_device() -> torch.device:
        return torch.device("cuda")

    @staticmethod
    def synchronize(device: int | str | torch.device | None = None) -> None:
        torch.cuda.synchronize(device)

    @staticmethod
    def reset_peak_memory_stats(device: int | str | torch.device | None = None) -> None:
        return torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    def max_memory_allocated(device: int | str | torch.device | None = None) -> int:
        return torch.cuda.max_memory_allocated(device)

    @staticmethod
    def get_device_properties(device: int | str | torch.device | None = None) -> Any:
        return torch.cuda.get_device_properties(device)

    @staticmethod
    def set_device(device: int | str | torch.device) -> None:
        return torch.cuda.set_device(device)

    @staticmethod
    def get_device_capability(device: int | str | torch.device | None = None) -> tuple[int, int]:
        return torch.cuda.get_device_capability(device)
