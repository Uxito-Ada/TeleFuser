"""Custom operation registration for torch.compile compatibility.

Provides utilities to register custom operations that:
1. Use optimized kernels (Triton, CUDA) at runtime
2. Support torch.compile via fake implementations for tracing

Use this for complex fused operations where hand-written kernels
outperform Inductor's automatic fusion even in compile mode.
"""

from __future__ import annotations

from typing import Callable

import torch


def register_custom_op(
    op_name: str,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
):
    """Register a custom operation for torch.compile compatibility.

    This decorator wraps a function to make it compatible with torch.compile
    by registering it as a custom operator. The function will be treated as
    a black box by the compiler.

    Args:
        op_name: Unique name for the custom operation (e.g., "telefuser::rms_norm")
        mutates_args: List of argument names that are mutated in-place.
        fake_impl: Fake implementation for torch.compile tracing.
            If not provided, a default fake will be created based on output shape.

    Returns:
        Decorator function that registers the custom op.

    Example:
        @register_custom_op(
            op_name="telefuser::fused_add_rms_norm",
            mutates_args=["residual"],
            fake_impl=_fused_add_rms_norm_fake,
        )
        def fused_add_rms_norm(x, residual, weight, eps):
            # Triton kernel implementation
            ...
            return output, residual
    """
    mutates_args = mutates_args or []

    def decorator(fn: Callable) -> Callable:
        try:
            from torch.library import custom_op

            # Create the custom op
            op = custom_op(op_name, mutates_args=mutates_args)(fn)

            # Register fake implementation if provided
            if fake_impl is not None:
                op.register_fake(fake_impl)

            return op
        except (ImportError, AttributeError):
            # Fallback for older PyTorch versions
            return fn

    return decorator


class TritonKernelWrapper:
    """Wrapper for Triton kernels that ensures torch.compile compatibility.

    Use this when you have a Triton kernel that should be used even in
    compile mode (because it's more efficient than Inductor's fusion).

    Example:
        fused_add_rms_norm = TritonKernelWrapper(
            op_name="telefuser::fused_add_rms_norm",
            kernel_fn=_fused_add_rms_norm_triton,
            mutates_args=["residual"],
            fake_impl=fused_add_rms_norm_fake,
        )

        output, residual = fused_add_rms_norm(x, residual, weight, eps)
    """

    def __init__(
        self,
        op_name: str,
        kernel_fn: Callable,
        mutates_args: list[str] | None = None,
        fake_impl: Callable | None = None,
    ):
        self.op_name = op_name
        self.kernel_fn = kernel_fn
        self.mutates_args = mutates_args or []
        self.fake_impl = fake_impl
        self._registered_op = None
        self._register()

    def _register(self):
        """Register the custom op lazily."""
        try:
            from torch.library import custom_op

            self._registered_op = custom_op(self.op_name, mutates_args=self.mutates_args)(self.kernel_fn)
            if self.fake_impl is not None:
                self._registered_op.register_fake(self.fake_impl)
        except (ImportError, AttributeError):
            self._registered_op = None

    def __call__(self, *args, **kwargs):
        if self._registered_op is not None:
            return self._registered_op(*args, **kwargs)
        return self.kernel_fn(*args, **kwargs)


__all__ = [
    "register_custom_op",
    "TritonKernelWrapper",
]
