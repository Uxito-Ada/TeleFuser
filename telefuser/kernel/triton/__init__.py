"""Triton kernel implementations for TeleFuser.

This module provides optimized Triton kernels for:
- Normalization: LayerNorm, RMSNorm, fused add + RMSNorm
- Position Encoding: Rotary Position Embedding (RoPE)
- Element-wise Operations: Fused scale and shift
- Quantization: FP8 per-token quantization
- Attention: Merge attention states for Ring Attention

Note: All functions in this module require triton to be installed.
Importing this module will fail if triton is not available.
"""

from __future__ import annotations

# Check if triton is available before importing any modules
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    # Existing kernels
    from .merge_attn_states import fused_merge_attn_states

    # Normalization kernels
    from .norm import (
        layer_norm_fn,
        norm_infer,
        rms_norm_fn,
    )
    from .quant import per_token_dequant_fp8, per_token_quant_fp8

    # RMSNorm kernels
    from .rmsnorm import (
        fused_add_rms_norm,
        rms_norm,
        triton_one_pass_rms_norm,
    )

    # Rotary Position Embedding kernels
    from .rotary import (
        apply_rotary_embedding,
        apply_rotary_embedding_inplace,
    )

    # Scale and shift kernels
    from .scale_shift import (
        fused_layernorm_scale_shift_gate_select01,
        fused_residual_layernorm_scale_shift_gate_select01,
        fused_scale_shift,
        fused_scale_shift_gate_select,
    )

    __all__ = [
        # Normalization
        "layer_norm_fn",
        "norm_infer",
        "rms_norm_fn",
        "rms_norm",
        "triton_one_pass_rms_norm",
        "fused_add_rms_norm",
        # Rotary Position Embedding
        "apply_rotary_embedding",
        "apply_rotary_embedding_inplace",
        # Scale and Shift
        "fused_scale_shift",
        "fused_scale_shift_gate_select",
        "fused_layernorm_scale_shift_gate_select01",
        "fused_residual_layernorm_scale_shift_gate_select01",
        # Attention
        "fused_merge_attn_states",
        # Quantization
        "per_token_quant_fp8",
        "per_token_dequant_fp8",
    ]
else:
    # When triton is not available, provide stub functions that raise ImportError
    def _make_stub(name):
        def stub(*args, **kwargs):
            raise ImportError(f"triton is required for {name} but not installed")

        stub.__name__ = name
        return stub

    # Define all expected names as stubs
    layer_norm_fn = _make_stub("layer_norm_fn")
    norm_infer = _make_stub("norm_infer")
    rms_norm_fn = _make_stub("rms_norm_fn")
    rms_norm = _make_stub("rms_norm")
    triton_one_pass_rms_norm = _make_stub("triton_one_pass_rms_norm")
    fused_add_rms_norm = _make_stub("fused_add_rms_norm")
    apply_rotary_embedding = _make_stub("apply_rotary_embedding")
    apply_rotary_embedding_inplace = _make_stub("apply_rotary_embedding_inplace")
    fused_scale_shift = _make_stub("fused_scale_shift")
    fused_scale_shift_gate_select = _make_stub("fused_scale_shift_gate_select")
    fused_layernorm_scale_shift_gate_select01 = _make_stub("fused_layernorm_scale_shift_gate_select01")
    fused_residual_layernorm_scale_shift_gate_select01 = _make_stub(
        "fused_residual_layernorm_scale_shift_gate_select01"
    )
    fused_merge_attn_states = _make_stub("fused_merge_attn_states")
    per_token_quant_fp8 = _make_stub("per_token_quant_fp8")
    per_token_dequant_fp8 = _make_stub("per_token_dequant_fp8")

    __all__ = [
        "layer_norm_fn",
        "norm_infer",
        "rms_norm_fn",
        "rms_norm",
        "triton_one_pass_rms_norm",
        "fused_add_rms_norm",
        "apply_rotary_embedding",
        "apply_rotary_embedding_inplace",
        "fused_scale_shift",
        "fused_scale_shift_gate_select",
        "fused_layernorm_scale_shift_gate_select01",
        "fused_residual_layernorm_scale_shift_gate_select01",
        "fused_merge_attn_states",
        "per_token_quant_fp8",
        "per_token_dequant_fp8",
        "HAS_TRITON",
    ]
