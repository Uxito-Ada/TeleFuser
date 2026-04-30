from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..cache_types import IndexEntry


class CacheMetadataManager(ABC):
    """缓存元数据管理器接口。"""

    @abstractmethod
    def register_cache(
        self,
        cache_id: str,
        prompt: str,
        saved_steps: List[int],
        size_mb: float,
        num_frames: int,
        cache_type: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def lookup_prompt(self, prompt: str, cache_type: Optional[str] = None) -> Optional[IndexEntry]:
        pass

    @abstractmethod
    def record_access(self, cache_id: str) -> None:
        pass

    @abstractmethod
    def plan_eviction(self, required_mb: float, limit_mb: float) -> List[tuple]:
        pass

    @abstractmethod
    def remove_cache(self, cache_id: str) -> None:
        pass

    @abstractmethod
    def record_hit_pair(
        self,
        request_prompt: str,
        cache_id: str,
        cached_prompt: str,
        similarity: float,
        task_type: str,
        cache_type: str,
        skip_step: int,
    ) -> None:
        pass

    @abstractmethod
    def record_similarity_scores(
        self,
        request_prompt: str,
        task_type: str,
        cache_type: str,
        stage: str,
        candidates: List[dict],
    ) -> None:
        pass

    @abstractmethod
    def get_cache_meta(self, cache_id: str) -> Optional[dict]:
        """获取指定 cache_id 的元数据，用于排查一致性问题。"""
        pass
