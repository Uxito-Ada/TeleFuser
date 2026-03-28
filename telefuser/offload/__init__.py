"""CPU offloading strategies for memory-constrained inference.

Provides sequential and asynchronous offloading mechanisms to fit
large models in limited GPU memory.
"""

from __future__ import annotations

from .async_offload import AsyncOffloadManager
from .model_offload import move_tensors_to_device, move_tensors_to_pinned_cpu
from .sequential_offload import (
    AutoWrappedLinear,
    AutoWrappedModule,
    WanAutoCastLayerNorm,
    enable_sequential_cpu_offload,
)

__all__ = [
    "AsyncOffloadManager",
    "AutoWrappedLinear",
    "AutoWrappedModule",
    "WanAutoCastLayerNorm",
    "enable_sequential_cpu_offload",
    "move_tensors_to_device",
    "move_tensors_to_pinned_cpu",
]
