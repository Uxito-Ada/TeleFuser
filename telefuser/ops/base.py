"""Base class for platform-aware operations with torch.compile support.

Provides automatic dispatch between optimized kernels (CUDA/Triton) and
PyTorch-native implementations based on runtime conditions:

1. torch.compile mode -> PyTorch native (compile-friendly)
2. Eager mode + CUDA -> Optimized Triton kernel
3. Eager mode + other -> PyTorch native fallback
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from telefuser.platforms import CudaPlatform, NPUPlatform, RocmPlatform, current_platform


class CustomOp(nn.Module):
    """Base class for custom operations with automatic platform dispatch.

    Subclasses should implement:
    - forward_native(): PyTorch-native implementation (required, compile-safe)
    - forward_cuda(): CUDA-optimized implementation (optional, for Triton kernels)
    - forward_npu(): NPU-optimized implementation (optional)
    - forward_rocm(): ROCm-optimized implementation (optional)

    The forward method automatically selects the appropriate implementation:
    - In torch.compile mode: always uses forward_native()
    - In eager mode: selects based on current platform

    Example:
        class RMSNorm(CustomOp):
            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps

            def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
                from telefuser.kernel.triton import rms_norm
                return rms_norm(x, self.weight, self.eps)

            def forward_native(self, x: torch.Tensor) -> torch.Tensor:
                variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
                return (x * torch.rsqrt(variance + self.eps) * self.weight).to(x.dtype)
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate implementation based on compile state and platform.

        torch.compiler.is_compiling() is lightweight and will be constant-folded
        after compilation, so there's no runtime overhead in compiled mode.
        """
        # In torch.compile mode, always use native implementation
        # This allows Inductor to optimize and potentially fuse with other ops
        if torch.compiler.is_compiling():
            return self.forward_native(*args, **kwargs)

        # In eager mode, select based on platform
        if isinstance(current_platform, CudaPlatform):
            if hasattr(self, "forward_cuda"):
                return self.forward_cuda(*args, **kwargs)
        elif isinstance(current_platform, RocmPlatform):
            if hasattr(self, "forward_rocm"):
                return self.forward_rocm(*args, **kwargs)
            # ROCm can often use CUDA kernels (Triton supports ROCm)
            if hasattr(self, "forward_cuda"):
                return self.forward_cuda(*args, **kwargs)
        elif isinstance(current_platform, NPUPlatform):
            if hasattr(self, "forward_npu"):
                return self.forward_npu(*args, **kwargs)

        # Fallback to native implementation
        return self.forward_native(*args, **kwargs)

    def forward_native(self, *args, **kwargs) -> Any:
        """PyTorch-native implementation (required).

        This implementation must be compatible with torch.compile.
        It should use only standard PyTorch operations without custom kernels.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.forward_native() must be implemented")


class CustomOpFunction:
    """Function-level custom op dispatcher for stateless operations.

    Use this for operations that don't need to hold parameters (like apply_rotary_emb).

    Example:
        def _apply_rotary_emb_cuda(x, cos, sin):
            from telefuser.kernel.triton import apply_rotary_embedding
            return apply_rotary_embedding(x, cos, sin)

        def _apply_rotary_emb_native(x, cos, sin):
            # PyTorch implementation
            ...

        apply_rotary_emb = CustomOpFunction(
            name="apply_rotary_emb",
            native_impl=_apply_rotary_emb_native,
            cuda_impl=_apply_rotary_emb_cuda,
        )
    """

    def __init__(
        self,
        name: str,
        native_impl: Callable,
        cuda_impl: Callable | None = None,
        npu_impl: Callable | None = None,
        rocm_impl: Callable | None = None,
    ):
        self.name = name
        self._native_impl = native_impl
        self._cuda_impl = cuda_impl
        self._npu_impl = npu_impl
        self._rocm_impl = rocm_impl

    def __call__(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate implementation."""
        # In torch.compile mode, always use native implementation
        if torch.compiler.is_compiling():
            return self._native_impl(*args, **kwargs)

        # In eager mode, select based on platform
        if isinstance(current_platform, CudaPlatform) and self._cuda_impl is not None:
            return self._cuda_impl(*args, **kwargs)
        if isinstance(current_platform, RocmPlatform):
            if self._rocm_impl is not None:
                return self._rocm_impl(*args, **kwargs)
            if self._cuda_impl is not None:
                return self._cuda_impl(*args, **kwargs)
        if isinstance(current_platform, NPUPlatform) and self._npu_impl is not None:
            return self._npu_impl(*args, **kwargs)

        return self._native_impl(*args, **kwargs)


__all__ = ["CustomOp", "CustomOpFunction"]
