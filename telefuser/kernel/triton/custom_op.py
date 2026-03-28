"""Utilities for custom op registration for torch.compile compatibility."""

from __future__ import annotations

from typing import Callable

import torch


def register_custom_op(
    op_name: str,
    mutates_args: list[str] | None = None,
) -> Callable:
    """Register a custom operation for torch.compile compatibility.

    This decorator wraps a function to make it compatible with torch.compile
    by registering it as a custom operator. The function will be treated as
    a black box by the compiler.

    Args:
        op_name: Unique name for the custom operation (e.g., "telefuser::rms_norm")
        mutates_args: List of argument names that are mutated in-place.
            Required for correct torch.compile handling.

    Returns:
        Decorator function that registers the custom op.

    Example:
        @register_custom_op("telefuser::rms_norm", mutates_args=["out"])
        def rms_norm_impl(x, weight, out, eps):
            # kernel implementation
            pass
    """
    mutates_args = mutates_args or []

    def decorator(fn: Callable) -> Callable:
        # Try to use torch._custom_op if available (PyTorch 2.4+)
        try:
            from torch._custom_op import custom_op

            # Create the custom op - this returns a callable that wraps the function
            op = custom_op(op_name, mutates_args=mutates_args)(fn)
            return op
        except (ImportError, AttributeError):
            # Fall back to just returning the original function
            # for older PyTorch versions
            return fn

    return decorator
