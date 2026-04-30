"""Unit tests for KV storage backends.

Tests LocalFileKVStore and InMemoryKVStore. Pure CPU, no GPU required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from telefuser.cache_mem.storage.local_file import LocalFileKVStore
from telefuser.cache_mem.storage.memory import InMemoryKVStore


class TestInMemoryKVStore:
    def test_put_and_get(self):
        store = InMemoryKVStore()
        store.put("k1", b"value1")
        assert store.get("k1") == b"value1"

    def test_get_missing_returns_none(self):
        store = InMemoryKVStore()
        assert store.get("missing") is None

    def test_remove(self):
        store = InMemoryKVStore()
        store.put("k1", b"value1")
        store.remove("k1")
        assert store.get("k1") is None

    def test_remove_missing_is_noop(self):
        store = InMemoryKVStore()
        store.remove("nonexistent")  # should not raise

    def test_list_keys(self):
        store = InMemoryKVStore()
        store.put("a", b"1")
        store.put("b", b"2")
        assert sorted(store.list_keys()) == ["a", "b"]

    def test_list_keys_empty(self):
        store = InMemoryKVStore()
        assert store.list_keys() == []

    def test_overwrite(self):
        store = InMemoryKVStore()
        store.put("k1", b"old")
        store.put("k1", b"new")
        assert store.get("k1") == b"new"


class TestLocalFileKVStore:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> LocalFileKVStore:
        return LocalFileKVStore(tmp_path / "kv")

    def test_put_and_get(self, store: LocalFileKVStore):
        store.put("k1", b"hello")
        assert store.get("k1") == b"hello"

    def test_get_missing_returns_none(self, store: LocalFileKVStore):
        assert store.get("missing") is None

    def test_remove(self, store: LocalFileKVStore):
        store.put("k1", b"data")
        store.remove("k1")
        assert store.get("k1") is None

    def test_remove_missing_is_noop(self, store: LocalFileKVStore):
        store.remove("nonexistent")  # should not raise

    def test_list_keys(self, store: LocalFileKVStore):
        store.put("x", b"1")
        store.put("y", b"2")
        assert sorted(store.list_keys()) == ["x", "y"]

    def test_overwrite(self, store: LocalFileKVStore):
        store.put("k1", b"old")
        store.put("k1", b"new")
        assert store.get("k1") == b"new"

    def test_binary_data(self, store: LocalFileKVStore):
        data = bytes(range(256))
        store.put("binary", data)
        assert store.get("binary") == data

    def test_persistence_across_instances(self, tmp_path: Path):
        kv_dir = tmp_path / "persist_kv"
        s1 = LocalFileKVStore(kv_dir)
        s1.put("persist_key", b"persist_value")

        s2 = LocalFileKVStore(kv_dir)
        assert s2.get("persist_key") == b"persist_value"

    def test_remove_cleans_file(self, store: LocalFileKVStore):
        store.put("to_del", b"data")
        # Verify file exists
        filename = store._index.get("to_del")
        assert filename is not None
        file_path = store.root_dir / filename
        assert file_path.exists()
        # Remove and verify file is gone
        store.remove("to_del")
        assert not file_path.exists()
