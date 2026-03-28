"""Custom operations module for TeleFuser.

Provides activation functions, feed-forward networks, normalization layers,
rotary position embeddings, and attention implementations optimized for video generation.
"""

from __future__ import annotations

from .rotary import apply_rotary_emb

__all__ = ["apply_rotary_emb"]
