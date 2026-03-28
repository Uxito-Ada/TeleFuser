"""Feature Cache module for diffusion transformers.

This module provides various feature caching strategies for accelerating
diffusion model inference:

- AdaTaylorCache: Combines adaptive skip logic with Taylor approximation.
  When n_derivatives=0, it reduces to simple residual caching (no Taylor expansion).

Hook-based interface:
- FeatureCacheHook: Abstract base class for cache hooks
- FeatureCacheHookManager: Manages hooks for a DiT model
- AdaTaylorCacheHook, AdaTaylorCacheCalibratorHook: Concrete implementations
"""

from __future__ import annotations

from .ada_taylor_cache import (
    AdaTaylorCache,
    AdaTaylorCacheCalibrator,
    AdaTaylorCacheConfig,
    AdaTaylorCacheState,
)
from .hooks import (
    AdaTaylorCacheCalibratorHook,
    AdaTaylorCacheHook,
    FeatureCacheHook,
    FeatureCacheHookManager,
    NoOpHook,
    ResidualAnalyzerHook,
)

__all__ = [
    # AdaTaylorCache
    "AdaTaylorCache",
    "AdaTaylorCacheConfig",
    "AdaTaylorCacheState",
    "AdaTaylorCacheCalibrator",
    # Hooks
    "FeatureCacheHook",
    "FeatureCacheHookManager",
    "NoOpHook",
    "AdaTaylorCacheHook",
    "AdaTaylorCacheCalibratorHook",
    "ResidualAnalyzerHook",
]
