"""Base platform interface for hardware abstraction.

All platform implementations (CUDA, NPU, CPU) must inherit from BasePlatform
and implement the required static methods.
"""

from __future__ import annotations

from abc import ABC


class BasePlatform(ABC):
    """Abstract base class for platform-specific operations.

    Provides unified interface for device management, memory operations,
    and distributed computing across different hardware backends.
    """

    device_name: str
    device_type: str
    device_control_env_var: str
    dispatch_key: str
    dist_backend: str
    full_dist_backend: str

    @staticmethod
    def empty_cache(*args, **kwargs):
        """Clear device memory cache."""
        raise NotImplementedError

    @staticmethod
    def ipc_collect(*args, **kwargs):
        """Collect inter-process communication resources."""
        raise NotImplementedError

    @staticmethod
    def get_device_name():
        """Get the device name."""
        raise NotImplementedError

    @staticmethod
    def device_ctx(*args, **kwargs):
        """Get device context manager."""
        raise NotImplementedError

    @staticmethod
    def default_device(*args, **kwargs):
        """Get the default device."""
        raise NotImplementedError

    @staticmethod
    def synchronize(*args, **kwargs):
        """Synchronize device operations."""
        raise NotImplementedError

    @staticmethod
    def device_count(*args, **kwargs):
        """Get number of available devices."""
        raise NotImplementedError

    @staticmethod
    def is_accelerator_available(*args, **kwargs):
        """Check if accelerator is available."""
        raise NotImplementedError

    @staticmethod
    def current_device(*args, **kwargs):
        """Get current device index."""
        raise NotImplementedError

    @staticmethod
    def reset_peak_memory_stats(*args, **kwargs):
        """Reset peak memory statistics."""
        raise NotImplementedError

    @staticmethod
    def max_memory_allocated(*args, **kwargs):
        """Get maximum memory allocated."""
        raise NotImplementedError

    @staticmethod
    def get_device_properties(*args, **kwargs):
        """Get device properties."""
        raise NotImplementedError

    @staticmethod
    def set_device(*args, **kwargs):
        """Set current device."""
        raise NotImplementedError

    @staticmethod
    def get_device_capability(*args, **kwargs):
        """Get device compute capability."""
        raise NotImplementedError

    @staticmethod
    def get_device_total_memory(*args, **kwargs):
        """Get total device memory in bytes."""
        raise NotImplementedError
