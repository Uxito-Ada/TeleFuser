"""Unit tests for cache data types and config.

Tests CacheResult, IndexEntry, VectorSearchResult construction and
CacheConfig defaults / field parsing. Pure CPU, no GPU required.
"""

from __future__ import annotations

import torch

from telefuser.cache_mem.cache_types import CacheResult, IndexEntry, VectorSearchResult
from telefuser.cache_mem.config import CacheConfig, CacheMode


class TestCacheResult:
    def test_default_miss(self):
        r = CacheResult(hit=False)
        assert not r.hit
        assert r.skip_step == 0
        assert r.cache_type == "none"
        assert r.latent_state is None

    def test_hit_with_latent(self):
        t = torch.randn(1, 16, 5, 32, 32)
        r = CacheResult(
            hit=True,
            skip_step=5,
            cache_type="approximate",
            similarity=0.95,
            latent_state=t,
            cached_prompt="cached",
        )
        assert r.hit
        assert r.skip_step == 5
        assert r.latent_state is t


class TestIndexEntry:
    def test_construction(self):
        e = IndexEntry(cache_id="abc123", prompt="hello", saved_steps=[3, 5])
        assert e.cache_id == "abc123"
        assert e.cache_type == "approximate_cache"

    def test_custom_cache_type(self):
        e = IndexEntry(cache_id="x", prompt="p", saved_steps=[], cache_type="video")
        assert e.cache_type == "video"


class TestVectorSearchResult:
    def test_construction(self):
        r = VectorSearchResult(
            cache_id="v1",
            similarity=0.88,
            prompt="search query",
            saved_steps=[1, 2],
            payload={"extra": "data"},
        )
        assert r.similarity == 0.88
        assert r.payload["extra"] == "data"


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.enable_latent_cache is False
        assert cfg.cache_mode == CacheMode.READ_WRITE
        assert cfg.kv_store_type == "local_file"
        assert cfg.vector_store_type == "faiss"
        assert cfg.vector_dim == 2048
        assert cfg.save_async_enabled is True

    def test_custom_values(self):
        cfg = CacheConfig(
            enable_latent_cache=True,
            cache_mode=CacheMode.READ_ONLY,
            kv_store_type="fluxon",
            vector_store_type="qdrant",
            vector_dim=1024,
            video_similarity_threshold=0.25,
            rerank_enabled=True,
        )
        assert cfg.enable_latent_cache is True
        assert cfg.cache_mode == CacheMode.READ_ONLY
        assert cfg.kv_store_type == "fluxon"
        assert cfg.vector_dim == 1024
        assert cfg.rerank_enabled is True


class TestCacheMode:
    def test_enum_values(self):
        assert CacheMode.READ_WRITE.value == "read_write"
        assert CacheMode.READ_ONLY.value == "read_only"
        assert CacheMode.WRITE_ONLY.value == "write_only"

    def test_from_string(self):
        assert CacheMode("read_only") == CacheMode.READ_ONLY
