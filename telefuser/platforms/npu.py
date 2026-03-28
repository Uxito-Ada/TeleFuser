"""NPU (Ascend) platform implementation for Huawei AI accelerators."""

from __future__ import annotations

from typing import Any

import torch

from .interface import BasePlatform


class NPUPlatform(BasePlatform):
    """NPU platform for Huawei Ascend AI acceleration."""

    device_name: str = "npu"
    device_type: str = "npu"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    dispatch_key: str = "PrivateUse1"
    dist_backend: str = "hccl"
    full_dist_backend: str = "npu:hccl"

    @staticmethod
    def empty_cache() -> None:
        torch.npu.empty_cache()

    @staticmethod
    def ipc_collect() -> None:
        """No-op: torch.npu.ipc_collect() is not implemented yet."""
        pass

    @staticmethod
    def get_device_name() -> str:
        return torch.npu.get_device_name()

    @staticmethod
    def device_ctx(device: int | str | torch.device) -> Any:
        return torch.npu.device(device)

    @staticmethod
    def default_device() -> torch.device:
        return torch.device("npu")

    @staticmethod
    def synchronize(device: int | str | torch.device | None = None) -> None:
        torch.npu.synchronize(device)

    @staticmethod
    def device_count() -> int:
        return torch.npu.device_count()

    @staticmethod
    def is_accelerator_available() -> bool:
        return torch.npu.is_available()

    @staticmethod
    def current_device() -> int:
        return torch.npu.current_device()

    @staticmethod
    def reset_peak_memory_stats(device: int | str | torch.device | None = None) -> None:
        return torch.npu.reset_peak_memory_stats(device)

    @staticmethod
    def max_memory_allocated(device: int | str | torch.device | None = None) -> int:
        return torch.npu.max_memory_allocated(device)

    @staticmethod
    def get_device_properties(device: int | str | torch.device | None = None) -> Any:
        return torch.npu.get_device_properties(device)

    @staticmethod
    def set_device(device: int | str | torch.device) -> None:
        return torch.npu.set_device(device)
