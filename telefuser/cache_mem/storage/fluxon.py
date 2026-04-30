from __future__ import annotations

from typing import Any, Optional

from .interfaces import KVStore


class FluxonKVStore(KVStore):
    """Fluxon KV backend stub — not available in MVP."""

    def __init__(self, config_path: Optional[str] = None, store: Optional[Any] = None):
        raise NotImplementedError("FluxonKV backend not available in MVP. Planned for v2.")

    def get(self, key: str) -> Optional[bytes]:
        raise NotImplementedError("FluxonKV backend not available in MVP. Planned for v2.")

    def put(self, key: str, value: bytes) -> None:
        raise NotImplementedError("FluxonKV backend not available in MVP. Planned for v2.")

    def remove(self, key: str) -> None:
        raise NotImplementedError("FluxonKV backend not available in MVP. Planned for v2.")

    def list_keys(self) -> list[str]:
        raise NotImplementedError("FluxonKV backend not available in MVP. Planned for v2.")
