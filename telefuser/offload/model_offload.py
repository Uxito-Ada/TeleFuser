"""Low-level tensor movement utilities for CPU offloading.

Provides helper functions for moving tensors between CPU and GPU with
pinned memory optimization for faster transfers.
"""

from __future__ import annotations

from typing import Iterable

import torch


def move_tensors_to_pinned_cpu(tensors: Iterable[torch.Tensor]) -> None:
    """Move tensors to pinned CPU memory for faster H2D transfer.

    Uses pre-allocation strategy to avoid memory doubling:
    1. Allocate pinned buffer first
    2. Copy data into it
    3. Replace original tensor data
    """
    for tensor in tensors:
        if tensor.device.type != "cpu":
            tensor.data = tensor.data.to("cpu").pin_memory()
        elif not tensor.data.is_pinned():
            pinned = torch.empty_like(tensor.data, pin_memory=True)
            pinned.copy_(tensor.data)
            tensor.data = pinned


def move_tensors_to_device(tensors: Iterable[torch.Tensor], device: torch.device, non_blocking: bool = True) -> None:
    """Move tensors to target device, optionally using non-blocking transfer."""
    for tensor in tensors:
        if tensor.device.type == "cpu":
            tensor.data = tensor.data.to(device, non_blocking=non_blocking)
