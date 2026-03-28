"""AdaTaylorCache: Adaptive Taylor Cache for Diffusion Transformers.

This module implements AdaTaylorCache which combines adaptive skip logic
with Taylor series approximation for efficient and accurate feature caching
in diffusion models.

When n_derivatives=0, AdaTaylorCache reduces to simple residual caching.
"""

from __future__ import annotations

from .ada_taylor_cache import (
    AdaTaylorCache,
    AdaTaylorCacheCalibrator,
    AdaTaylorCacheConfig,
    AdaTaylorCacheState,
    load_cache_params,
    nearest_interp,
)

__all__ = [
    "AdaTaylorCache",
    "AdaTaylorCacheConfig",
    "AdaTaylorCacheState",
    "AdaTaylorCacheCalibrator",
    "load_cache_params",
    "nearest_interp",
]
