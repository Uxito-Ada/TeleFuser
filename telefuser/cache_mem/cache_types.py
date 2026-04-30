from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class CacheResult:
    """缓存查询结果。"""

    hit: bool
    skip_step: int = 0
    cache_type: str = "none"  # "approximate", "continue", "exact", "none"
    similarity: float = 0.0
    latent_state: Optional[torch.Tensor] = None
    cached_prompt: str = ""
    session_id: Optional[str] = None


@dataclass
class IndexEntry:
    """索引条目。"""

    cache_id: str
    prompt: str
    saved_steps: List[int]
    cache_type: str = "approximate_cache"


@dataclass
class VectorSearchResult:
    """向量检索结果。"""

    cache_id: str
    similarity: float
    prompt: str
    saved_steps: List[int]
    payload: Dict[str, Any]
