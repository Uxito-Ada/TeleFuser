from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..cache_types import VectorSearchResult


class VectorStore(ABC):
    """向量存储接口。"""

    @abstractmethod
    def search(
        self,
        collection: str,
        vector: List[float],
        limit: int = 1,
        score_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        pass

    @abstractmethod
    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        pass

    @abstractmethod
    def delete(self, collection: str, point_ids: List[str]) -> None:
        pass

    @abstractmethod
    def ensure_collection(self, collection: str, vector_dim: int) -> None:
        pass

    @abstractmethod
    def get_vector_size(self, collection: str) -> Optional[int]:
        pass
