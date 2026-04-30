from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["CacheConfig", "CacheResult", "LatentCache"]


def __getattr__(name: str) -> Any:
    """Lazily expose heavy symbols to keep lightweight imports usable."""
    if name == "CacheResult":
        module = import_module("telefuser.cache_mem.cache_types")
        return getattr(module, "CacheResult")
    if name == "LatentCache":
        module = import_module("telefuser.cache_mem.latent_cache")
        return getattr(module, "LatentCache")
    if name == "CacheConfig":
        try:
            module = import_module("telefuser.cache_mem.config")
            return getattr(module, "CacheConfig")
        except (ImportError, ModuleNotFoundError):
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
