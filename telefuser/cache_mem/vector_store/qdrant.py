from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..cache_types import VectorSearchResult
from .interfaces import VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant vector store stub — not available in MVP."""

    def __init__(
        self,
        url: str = "",
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout: int = 30,
    ) -> None:
        raise NotImplementedError("Qdrant backend not available in MVP. Planned for v2.")

    def search(
        self,
        collection: str,
        vector: List[float],
        limit: int = 1,
        score_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        raise NotImplementedError("Qdrant backend not available in MVP. Planned for v2.")

    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        raise NotImplementedError("Qdrant backend not available in MVP. Planned for v2.")

    def delete(self, collection: str, point_ids: List[str]) -> None:
        raise NotImplementedError("Qdrant backend not available in MVP. Planned for v2.")

    def ensure_collection(self, collection: str, vector_dim: int) -> None:
        raise NotImplementedError("Qdrant backend not available in MVP. Planned for v2.")

    def get_vector_size(self, collection: str) -> Optional[int]:
        raise NotImplementedError("Qdrant backend not available in MVP. Planned for v2.")
