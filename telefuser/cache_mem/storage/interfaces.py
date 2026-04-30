from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class KVStore(ABC):
    """键值存储接口。"""

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        pass

    @abstractmethod
    def put(self, key: str, value: bytes) -> None:
        pass

    @abstractmethod
    def remove(self, key: str) -> None:
        pass

    @abstractmethod
    def list_keys(self) -> list[str]:
        """列出当前存储的 key 列表。"""
        pass
