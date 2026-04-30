"""Unit tests for LocalCacheMetadataManager.

Tests CRUD operations, eviction planning, access tracking, and
audit logging. Pure CPU, no GPU required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from telefuser.cache_mem.metadata import LocalCacheMetadataManager


@pytest.fixture()
def mgr(tmp_path: Path) -> LocalCacheMetadataManager:
    return LocalCacheMetadataManager(tmp_path / "meta")


class TestRegisterAndLookup:
    def test_register_and_lookup_by_prompt(self, mgr: LocalCacheMetadataManager):
        mgr.register_cache("c1", "hello world", saved_steps=[3, 5], size_mb=1.0, num_frames=8)
        entry = mgr.lookup_prompt("hello world")
        assert entry is not None
        assert entry.cache_id == "c1"
        assert entry.saved_steps == [3, 5]

    def test_lookup_nonexistent_prompt(self, mgr: LocalCacheMetadataManager):
        assert mgr.lookup_prompt("does not exist") is None

    def test_register_with_cache_type(self, mgr: LocalCacheMetadataManager):
        mgr.register_cache("c2", "typed", saved_steps=[1], size_mb=0.5, num_frames=4, cache_type="video")
        entry = mgr.lookup_prompt("typed", cache_type="video")
        assert entry is not None
        assert entry.cache_type == "video"
        # Should not appear in default type
        assert mgr.lookup_prompt("typed", cache_type="nonexistent") is None

    def test_duplicate_steps_are_deduplicated(self, mgr: LocalCacheMetadataManager):
        mgr.register_cache("c3", "dup", saved_steps=[5, 3, 5, 3], size_mb=1.0, num_frames=4)
        entry = mgr.lookup_prompt("dup")
        assert entry is not None
        assert entry.saved_steps == [3, 5]


class TestRemoveCache:
    def test_remove_existing(self, mgr: LocalCacheMetadataManager):
        mgr.register_cache("c1", "to delete", saved_steps=[1], size_mb=0.5, num_frames=4)
        mgr.remove_cache("c1")
        assert mgr.lookup_prompt("to delete") is None
        assert mgr.get_cache_meta("c1") is None

    def test_remove_nonexistent_is_noop(self, mgr: LocalCacheMetadataManager):
        mgr.remove_cache("nonexistent")  # should not raise


class TestGetCacheMeta:
    def test_returns_meta_dict(self, mgr: LocalCacheMetadataManager):
        mgr.register_cache("c1", "meta test", saved_steps=[2], size_mb=1.5, num_frames=16)
        meta = mgr.get_cache_meta("c1")
        assert meta is not None
        assert meta["prompt"] == "meta test"
        assert meta["size_mb"] == 1.5
        assert meta["num_frames"] == 16

    def test_returns_none_for_missing(self, mgr: LocalCacheMetadataManager):
        assert mgr.get_cache_meta("missing") is None


class TestRecordAccess:
    def test_increments_access_count(self, mgr: LocalCacheMetadataManager):
        mgr.register_cache("c1", "access", saved_steps=[1], size_mb=0.5, num_frames=4)
        mgr.record_access("c1")
        mgr.record_access("c1")
        meta = mgr.get_cache_meta("c1")
        assert meta is not None
        assert meta["access_count"] == 2

    def test_noop_for_missing(self, mgr: LocalCacheMetadataManager):
        mgr.record_access("missing")  # should not raise


class TestPlanEviction:
    def test_no_eviction_needed(self, mgr: LocalCacheMetadataManager):
        mgr.register_cache("c1", "small", saved_steps=[1], size_mb=1.0, num_frames=4)
        result = mgr.plan_eviction(required_mb=1.0, limit_mb=10.0)
        assert result == []

    def test_evicts_oldest_first(self, mgr: LocalCacheMetadataManager):
        mgr.register_cache("old", "old", saved_steps=[1], size_mb=5.0, num_frames=4)
        # Access "old" to set its timestamp, then register "new" which gets a newer timestamp
        mgr.register_cache("new", "new", saved_steps=[1], size_mb=5.0, num_frames=4)
        result = mgr.plan_eviction(required_mb=3.0, limit_mb=10.0)
        assert len(result) > 0
        # Oldest entry (by last_access_time) should be evicted first
        evicted_ids = [cid for cid, _ in result]
        assert "old" in evicted_ids


class TestRecordHitPair:
    def test_writes_jsonl(self, mgr: LocalCacheMetadataManager):
        mgr.record_hit_pair(
            request_prompt="new prompt",
            cache_id="c1",
            cached_prompt="old prompt",
            similarity=0.95,
            task_type="t2v",
            cache_type="approximate",
            skip_step=5,
        )
        log_path = mgr.metadata_cache_dir / "hit_pairs.jsonl"
        assert log_path.exists()
        line = json.loads(log_path.read_text().strip())
        assert line["similarity"] == 0.95
        assert line["skip_step"] == 5


class TestRecordSimilarityScores:
    def test_writes_jsonl(self, mgr: LocalCacheMetadataManager):
        mgr.record_similarity_scores(
            request_prompt="query",
            task_type="t2v",
            cache_type="video",
            stage="rerank",
            candidates=[{"id": "c1", "score": 0.9}],
        )
        log_path = mgr.metadata_cache_dir / "similarity_scores.jsonl"
        assert log_path.exists()
        line = json.loads(log_path.read_text().strip())
        assert line["stage"] == "rerank"
        assert len(line["candidates"]) == 1


class TestPersistence:
    def test_survives_reload(self, tmp_path: Path):
        meta_dir = tmp_path / "persist"
        mgr1 = LocalCacheMetadataManager(meta_dir)
        mgr1.register_cache("c1", "persist", saved_steps=[1, 3], size_mb=2.0, num_frames=8)

        # Create a new manager pointing to the same directory
        mgr2 = LocalCacheMetadataManager(meta_dir)
        entry = mgr2.lookup_prompt("persist")
        assert entry is not None
        assert entry.cache_id == "c1"
        assert entry.saved_steps == [1, 3]
        meta = mgr2.get_cache_meta("c1")
        assert meta is not None
        assert meta["size_mb"] == 2.0
