"""Custom CUDA kernels and Triton implementations."""

from __future__ import annotations

# Check if triton is available
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Lazy imports for triton-dependent modules
_per_token_dequant_fp8 = None
_per_token_quant_fp8 = None
_fused_merge_attn_states = None


def per_token_dequant_fp8(*args, **kwargs):
    """Lazily import and call per_token_dequant_fp8."""
    global _per_token_dequant_fp8
    if _per_token_dequant_fp8 is None:
        if not HAS_TRITON:
            raise ImportError("triton is required for per_token_dequant_fp8 but not installed")
        from .triton.quant import per_token_dequant_fp8 as _func

        _per_token_dequant_fp8 = _func
    return _per_token_dequant_fp8(*args, **kwargs)


def per_token_quant_fp8(*args, **kwargs):
    """Lazily import and call per_token_quant_fp8."""
    global _per_token_quant_fp8
    if _per_token_quant_fp8 is None:
        if not HAS_TRITON:
            raise ImportError("triton is required for per_token_quant_fp8 but not installed")
        from .triton.quant import per_token_quant_fp8 as _func

        _per_token_quant_fp8 = _func
    return _per_token_quant_fp8(*args, **kwargs)


def fused_merge_attn_states(*args, **kwargs):
    """Lazily import and call fused_merge_attn_states."""
    global _fused_merge_attn_states
    if _fused_merge_attn_states is None:
        if not HAS_TRITON:
            raise ImportError("triton is required for fused_merge_attn_states but not installed")
        from .triton.merge_attn_states import fused_merge_attn_states as _func

        _fused_merge_attn_states = _func
    return _fused_merge_attn_states(*args, **kwargs)


__all__ = ["per_token_dequant_fp8", "per_token_quant_fp8", "fused_merge_attn_states", "HAS_TRITON"]
