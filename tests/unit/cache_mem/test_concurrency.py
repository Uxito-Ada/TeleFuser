"""Concurrency tests for cache_mem stores guarded by ``threading.RLock``.

These tests lock in the invariant that under heavy multi-threaded
concurrent access:

* No registered/put/upsert entry is lost.
* Read paths never raise mid-mutation (e.g. no ``RuntimeError: dictionary
  changed size during iteration``).
* On-disk artifacts (``kv_index.json`` / ``prompt_index.json`` /
  ``cache_meta.json`` / ``<col>.faiss`` + ``<col>.json``) remain valid
  and reload cleanly into a fresh instance.

Pure CPU, no GPU. ``ThreadPoolExecutor`` is used to surface races on a
multi-core machine while still finishing in well under 2 s each.
"""

from __future__ import annotations

import os
import sys

# faiss-cpu and torch each ship their own OpenMP runtime; on macOS loading
# both into the same process aborts inside libomp unless this is set.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json  # noqa: E402
import random  # noqa: E402
import threading  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, wait  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from telefuser.cache_mem.metadata import LocalCacheMetadataManager  # noqa: E402
from telefuser.cache_mem.storage.local_file import LocalFileKVStore  # noqa: E402

NUM_OPS = 200


def _run_with_barrier(fns: list) -> list:
    """Run callables in parallel, releasing them simultaneously via a barrier.

    The pool must have at least ``len(fns)`` workers so every callable can
    actually start and reach ``barrier.wait()``; otherwise the barrier
    deadlocks because pool workers block on the barrier before yielding
    back to pick up queued tasks.

    Returns the list of futures (already completed). Re-raises any worker
    exception via ``future.result()``.
    """
    n = len(fns)
    barrier = threading.Barrier(n)

    def _wrapped(fn):
        def _inner():
            barrier.wait()
            return fn()

        return _inner

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_wrapped(fn)) for fn in fns]
        wait(futures)
    return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# LocalCacheMetadataManager
# ---------------------------------------------------------------------------


class TestMetadataManagerConcurrency:
    """Concurrency invariants for ``LocalCacheMetadataManager``."""

    def test_no_loss_register(self, tmp_path: Path) -> None:
        """200 unique register_cache calls — every entry survives, on disk too."""
        meta_dir = tmp_path / "meta_no_loss"
        mgr = LocalCacheMetadataManager(meta_dir)

        def make_fn(i: int):
            def _fn() -> None:
                mgr.register_cache(
                    cache_id=f"c{i}",
                    prompt=f"p{i}",
                    saved_steps=[i % 10],
                    size_mb=1.0,
                    num_frames=4,
                )

            return _fn

        _run_with_barrier([make_fn(i) for i in range(NUM_OPS)])

        # In-memory: all 200 entries present.
        assert len(mgr._meta) == NUM_OPS
        for i in range(NUM_OPS):
            entry = mgr.lookup_prompt(f"p{i}")
            assert entry is not None, f"missing entry for p{i}"
            assert entry.cache_id == f"c{i}"

        # On-disk: reload via a second manager.
        mgr2 = LocalCacheMetadataManager(meta_dir)
        assert len(mgr2._meta) == NUM_OPS
        for i in range(NUM_OPS):
            assert mgr2.lookup_prompt(f"p{i}") is not None, f"missing on reload p{i}"

    def test_atomic_access_counter(self, tmp_path: Path) -> None:
        """200 record_access calls on the same cache_id — final count is exactly 200.

        This is the load-bearing test: Python ``+=`` on a dict value is not
        atomic, so without the RLock this test fails.
        """
        mgr = LocalCacheMetadataManager(tmp_path / "meta_counter")
        mgr.register_cache("c0", "prompt0", saved_steps=[0], size_mb=1.0, num_frames=4)

        fns = [lambda: mgr.record_access("c0") for _ in range(NUM_OPS)]
        _run_with_barrier(fns)

        meta = mgr.get_cache_meta("c0")
        assert meta is not None
        assert meta["access_count"] == NUM_OPS, f"expected {NUM_OPS}, got {meta['access_count']}"

    def test_mixed_read_write(self, tmp_path: Path) -> None:
        """Half the threads register fresh ids, half iterate lookup_prompt.

        Reads must never raise (e.g. dictionary-changed-size errors).
        """
        mgr = LocalCacheMetadataManager(tmp_path / "meta_mixed")
        # Pre-register a base set so readers have non-empty maps to iterate.
        base_count = 50
        for i in range(base_count):
            mgr.register_cache(
                cache_id=f"base{i}",
                prompt=f"base_p{i}",
                saved_steps=[i % 5],
                size_mb=0.5,
                num_frames=4,
            )

        def make_writer(i: int):
            def _fn() -> None:
                mgr.register_cache(
                    cache_id=f"new{i}",
                    prompt=f"new_p{i}",
                    saved_steps=[i % 5],
                    size_mb=0.5,
                    num_frames=4,
                )

            return _fn

        def make_reader(i: int):
            def _fn() -> None:
                # Look up an existing base entry; should always succeed.
                target = f"base_p{i % base_count}"
                entry = mgr.lookup_prompt(target)
                assert entry is not None, f"reader could not find {target}"

            return _fn

        fns = []
        for i in range(NUM_OPS // 2):
            fns.append(make_writer(i))
            fns.append(make_reader(i))
        _run_with_barrier(fns)

        # Writers' entries must all be there.
        assert len(mgr._meta) == base_count + NUM_OPS // 2


# ---------------------------------------------------------------------------
# LocalFileKVStore
# ---------------------------------------------------------------------------


class TestLocalFileKVStoreConcurrency:
    """Concurrency invariants for ``LocalFileKVStore``."""

    def test_no_loss_put(self, tmp_path: Path) -> None:
        """200 unique put calls — all keys end up in the index and survive reload."""
        kv_dir = tmp_path / "kv_no_loss"
        store = LocalFileKVStore(kv_dir)

        def make_fn(i: int):
            def _fn() -> None:
                store.put(f"k{i}", f"v{i}".encode("utf-8"))

            return _fn

        _run_with_barrier([make_fn(i) for i in range(NUM_OPS)])

        expected_keys = {f"k{i}" for i in range(NUM_OPS)}
        assert set(store.list_keys()) == expected_keys

        # kv_index.json must be valid JSON containing all keys.
        index_path = kv_dir / "kv_index.json"
        raw = json.loads(index_path.read_text())
        assert set(raw.keys()) == expected_keys

        # Reload via a second store on the same dir.
        store2 = LocalFileKVStore(kv_dir)
        assert set(store2.list_keys()) == expected_keys
        # Spot-check some values round-trip correctly.
        for i in (0, NUM_OPS // 2, NUM_OPS - 1):
            assert store2.get(f"k{i}") == f"v{i}".encode("utf-8")

    def test_concurrent_overwrite_same_key(self, tmp_path: Path) -> None:
        """N threads put the same key — final value is one of theirs (not torn)."""
        kv_dir = tmp_path / "kv_overwrite"
        store = LocalFileKVStore(kv_dir)

        values = [f"v_{i}".encode("utf-8") for i in range(NUM_OPS)]

        def make_fn(payload: bytes):
            def _fn() -> None:
                store.put("k", payload)

            return _fn

        _run_with_barrier([make_fn(v) for v in values])

        final = store.get("k")
        assert final in values, f"final value {final!r} is not one of the writes"

        # Index file must still be valid JSON containing exactly "k".
        raw = json.loads((kv_dir / "kv_index.json").read_text())
        assert list(raw.keys()) == ["k"]

        # Reload yields the same single key.
        store2 = LocalFileKVStore(kv_dir)
        assert store2.list_keys() == ["k"]
        assert store2.get("k") == final


# ---------------------------------------------------------------------------
# FAISSVectorStore (skipped if faiss unavailable)
# ---------------------------------------------------------------------------


pytest.importorskip("faiss")  # noqa: E402


# On macOS, torch and faiss-cpu each ship their own libomp; loaded into the
# same process they crash inside faiss.search regardless of KMP_DUPLICATE_LIB_OK.
# The RLock invariants we want to lock in are platform-independent, so the
# Linux CI run is what actually gates correctness — skipping locally on Darwin
# avoids a known-bad infra setup, not a real bug in our wrapper.
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="faiss-cpu + torch OpenMP collision on macOS aborts inside faiss.search",
)
class TestFAISSVectorStoreConcurrency:
    """Concurrency invariants for ``FAISSVectorStore``."""

    VECTOR_DIM = 8
    COLLECTION = "test_col"

    @staticmethod
    def _random_vec(rng: random.Random, dim: int) -> list[float]:
        # Random unit-ish vector; exact magnitude does not matter for L2 search.
        return [rng.random() for _ in range(dim)]

    def test_no_loss_upsert(self, tmp_path: Path) -> None:
        """200 unique upserts — every point survives and on-disk metadata is consistent."""
        from telefuser.cache_mem.vector_store.faiss import FAISSVectorStore

        index_dir = tmp_path / "faiss_no_loss"
        store = FAISSVectorStore(index_dir=index_dir, vector_dim=self.VECTOR_DIM, index_type="L2")
        store.ensure_collection(self.COLLECTION, self.VECTOR_DIM)

        # Pre-generate vectors so threads do no shared RNG work.
        rng = random.Random(0)
        points = [(f"pid{i}", self._random_vec(rng, self.VECTOR_DIM)) for i in range(NUM_OPS)]

        def make_fn(pid: str, vec: list[float]):
            def _fn() -> None:
                store.upsert(self.COLLECTION, pid, vec, payload={"prompt": pid, "saved_steps": [0]})

            return _fn

        _run_with_barrier([make_fn(pid, vec) for pid, vec in points])

        # Search returns at most NUM_OPS unique results.
        query = self._random_vec(random.Random(1), self.VECTOR_DIM)
        results = store.search(self.COLLECTION, query, limit=NUM_OPS)
        assert len(results) <= NUM_OPS
        # All returned cache_ids belong to the set we inserted.
        inserted_ids = {pid for pid, _ in points}
        for r in results:
            assert r.cache_id in inserted_ids

        # On-disk: id_map length must equal index.ntotal — that is exactly the
        # consistency invariant ``_load_index`` asserts on reload.
        meta_path = index_dir / f"{self.COLLECTION}.json"
        on_disk_meta = json.loads(meta_path.read_text())
        id_map = on_disk_meta.get("id_map", {})
        # Reload via a second store; the constructor + first _load_index call
        # will raise if id_map length disagrees with index.ntotal.
        store2 = FAISSVectorStore(index_dir=index_dir, vector_dim=self.VECTOR_DIM, index_type="L2")
        # Touch the collection to trigger _load_index validation.
        size = store2.get_vector_size(self.COLLECTION)
        assert size == self.VECTOR_DIM
        # Full upsert survival: every inserted id must remain in the on-disk map.
        assert set(id_map.keys()) == inserted_ids

    def test_concurrent_upsert_and_search(self, tmp_path: Path) -> None:
        """Half upsert, half search — search never raises mid-mutation."""
        from telefuser.cache_mem.vector_store.faiss import FAISSVectorStore

        index_dir = tmp_path / "faiss_mixed"
        store = FAISSVectorStore(index_dir=index_dir, vector_dim=self.VECTOR_DIM, index_type="L2")
        store.ensure_collection(self.COLLECTION, self.VECTOR_DIM)

        # Seed the collection so searches have something to find.
        seed_rng = random.Random(42)
        for i in range(20):
            store.upsert(
                self.COLLECTION,
                f"seed{i}",
                self._random_vec(seed_rng, self.VECTOR_DIM),
                payload={"prompt": f"seed{i}", "saved_steps": [0]},
            )

        rng = random.Random(7)
        upsert_points = [(f"new{i}", self._random_vec(rng, self.VECTOR_DIM)) for i in range(NUM_OPS // 2)]
        query_vecs = [self._random_vec(rng, self.VECTOR_DIM) for _ in range(NUM_OPS // 2)]

        def make_upsert(pid: str, vec: list[float]):
            def _fn() -> None:
                store.upsert(self.COLLECTION, pid, vec, payload={"prompt": pid, "saved_steps": [0]})

            return _fn

        def make_search(vec: list[float]):
            def _fn() -> list:
                results = store.search(self.COLLECTION, vec, limit=5)
                # Search must always return a list (possibly empty), never crash.
                assert isinstance(results, list)
                return results

            return _fn

        fns = []
        for (pid, vec), qvec in zip(upsert_points, query_vecs):
            fns.append(make_upsert(pid, vec))
            fns.append(make_search(qvec))
        _run_with_barrier(fns)

        # All upserts succeeded.
        meta_path = index_dir / f"{self.COLLECTION}.json"
        on_disk_meta = json.loads(meta_path.read_text())
        id_map = on_disk_meta.get("id_map", {})
        for pid, _ in upsert_points:
            assert pid in id_map, f"upsert for {pid} was lost"
