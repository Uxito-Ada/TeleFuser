"""Normalization layers including RMSNorm, LayerNorm, and adaptive layer norm."""

from __future__ import annotations

import torch
import torch.nn as nn

# Try to import tf_kernel rmsnorm for better performance on CUDA
try:
    from tf_kernel import rmsnorm as tf_kernel_rmsnorm

    TF_KERNEL_AVAILABLE = True
except ImportError:
    TF_KERNEL_AVAILABLE = False

    def tf_kernel_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Stub function when tf_kernel is not available."""
        raise ImportError("tf_kernel is required for rmsnorm but not installed")


# Try to import triton kernels
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    from telefuser.kernel.triton import fused_scale_shift, layer_norm_fn, rms_norm
else:

    def _make_stub(name: str):
        def stub(*args, **kwargs):
            raise ImportError(f"triton is required for {name} but not installed")

        stub.__name__ = name
        return stub

    fused_scale_shift = _make_stub("fused_scale_shift")
    rms_norm = _make_stub("rms_norm")
    layer_norm_fn = _make_stub("layer_norm_fn")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply modulation: x * (1 + scale) + shift.

    Uses Triton kernel when on CUDA for better performance.

    Args:
        x: Input tensor to modulate.
        shift: Shift tensor for additive modulation.
        scale: Scale tensor for multiplicative modulation.

    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    # Triton kernel requires tensor to be on CUDA device
    if x.device.type == "cuda" and HAS_TRITON:
        return fused_scale_shift(x, scale, shift)
    return x * (1 + scale) + shift


class AdaLayerNormContinuous(nn.Module):
    """Adaptive layer normalization with continuous conditioning.

    Applies scale and shift from conditioning embedding after normalization.

    Args:
        embedding_dim: Dimension to normalize.
        conditioning_embedding_dim: Conditioning input dimension.
        elementwise_affine: Whether norm has learnable affine parameters.
        eps: Numerical stability epsilon.
        bias: Whether to use bias in conditioning projection.
        norm_type: "layer_norm" or "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        # Output scale + shift (2x embedding_dim)
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # Cast conditioning to input dtype for mixed-precision compatibility
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        # Apply scale-shift: (1 + scale) for multiplicative, + shift for additive
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: https://huggingface.co/papers/1910.07467

    Uses tf_kernel rmsnorm when available on CUDA for best performance,
    falls back to Triton kernel, then to PyTorch for non-CUDA tensors.

    Args:
        dim: Dimension for learnable weights.
        eps: Epsilon for numerical stability.
        elementwise_affine: Whether to use learnable weights.
        bias: Whether to use learnable bias.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use PyTorch implementation when compiling (better for torch.compile optimization)
        if torch.compiler.is_compiling():
            return self._forward_pytorch(hidden_states)

        # Fast path: use optimized kernels on CUDA device in eager mode
        # Triton and tf_kernel only support CUDA platform
        if hidden_states.device.type == "cuda" and self.elementwise_affine and self.bias is None:
            assert self.weight is not None, "weight should not be None when elementwise_affine=True"
            # Prefer tf_kernel if available (optimized CUDA kernel)
            if TF_KERNEL_AVAILABLE:
                input_shape = hidden_states.shape
                if hidden_states.ndim > 2:
                    hidden_states_2d = hidden_states.view(-1, input_shape[-1])
                    out = tf_kernel_rmsnorm(hidden_states_2d, self.weight, self.eps)
                    return out.view(input_shape)
                return tf_kernel_rmsnorm(hidden_states, self.weight, self.eps)
            # Fallback to Triton kernel
            if HAS_TRITON:
                return rms_norm(hidden_states, self.weight, self.eps)

        return self._forward_pytorch(hidden_states)

    def _forward_pytorch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation for compilation compatibility."""
        input_dtype = hidden_states.dtype
        # FP32 computation for numerical stability
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # Cast to weight dtype for half-precision support
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class LayerNorm(nn.Module):
    """Layer Normalization with optimized Triton kernel support on CUDA.

    Uses Triton kernel when on CUDA for better performance in eager mode.
    Falls back to PyTorch for non-CUDA tensors, non-affine case, or when compiling.

    Args:
        dim: Dimension for learnable weights.
        eps: Epsilon for numerical stability.
        elementwise_affine: Whether to use learnable weights and bias.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use PyTorch built-in layer_norm when compiling (better for torch.compile optimization)
        # or when not on CUDA device, or when non-affine
        # Triton kernels only support CUDA platform
        if torch.compiler.is_compiling() or x.device.type != "cuda" or not self.elementwise_affine or not HAS_TRITON:
            if self.elementwise_affine:
                return nn.functional.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
            return nn.functional.layer_norm(x, (x.shape[-1],), eps=self.eps)

        # Use Triton kernel in eager mode for better performance
        return layer_norm_fn(x, self.weight, self.bias, eps=self.eps)
