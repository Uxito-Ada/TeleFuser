"""Attention implementations for dense and sparse patterns.

Exports the main attention interface and configuration types.
"""

from __future__ import annotations

from .attention_impl import (
    SparseAttentionState,
    attention,
    long_context_attention,
)
from .sparse_patterns import (
    MaskMap,
    clear_radial_mask_cache,
    create_radial_mask_map,
    sparse_attention,
)

__all__ = [
    "attention",
    "long_context_attention",
    "SparseAttentionState",
    # Sparse attention utilities
    "MaskMap",
    "sparse_attention",
    "create_radial_mask_map",
    "clear_radial_mask_cache",
]
