from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .storage.interfaces import KVStore
from .vector_store.interfaces import VectorStore


class ConnectionManager:
    def __init__(
        self,
        config: Any,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self._config = config
        self._storage_dir = Path(storage_dir) if storage_dir else None
        self._lock = threading.Lock()

        self._vector_store: Optional[VectorStore] = None
        self._vector_store_created = False
        self._kv_store: Optional[KVStore] = None
        self._kv_store_created = False

    # ── public properties ──────────────────────────────────────────

    @property
    def vector_store(self) -> Optional[VectorStore]:
        """延迟创建 VectorStore 连接（Qdrant / FAISS）。"""
        if not self._vector_store_created:
            with self._lock:
                if not self._vector_store_created:
                    self._vector_store = self._create_vector_store()
                    self._vector_store_created = True
        return self._vector_store

    @property
    def kv_store(self) -> Optional[KVStore]:
        """延迟创建 KVStore 连接（Fluxon / LocalFile）。"""
        if not self._kv_store_created:
            with self._lock:
                if not self._kv_store_created:
                    self._kv_store = self._create_kv_store()
                    self._kv_store_created = True
        return self._kv_store

    # ── health check ───────────────────────────────────────────────

    def health_check(self) -> dict:
        result: dict = {}

        # vector_store
        vs = self._vector_store
        if vs is None and not self._vector_store_created:
            result["vector_store"] = {"status": "not_initialized"}
        elif vs is None:
            result["vector_store"] = {"status": "disabled"}
        else:
            vs_status: dict = {
                "status": "connected",
                "type": type(vs).__name__,
            }
            # Qdrant: 尝试获取 collections 列表验证连通性
            if hasattr(vs, "client"):
                try:
                    vs.client.get_collections()
                    vs_status["reachable"] = True
                except Exception as exc:
                    logger.exception(
                        "ConnectionManager.health_check vector_store reachability failed: {}",
                        exc,
                    )
                    vs_status["reachable"] = False
                    vs_status["error"] = str(exc)
            result["vector_store"] = vs_status

        # kv_store
        kvs = self._kv_store
        if kvs is None and not self._kv_store_created:
            result["kv_store"] = {"status": "not_initialized"}
        elif kvs is None:
            result["kv_store"] = {"status": "disabled"}
        else:
            result["kv_store"] = {
                "status": "connected",
                "type": type(kvs).__name__,
            }

        return result

    # ── shutdown ───────────────────────────────────────────────────

    def shutdown(self) -> None:
        with self._lock:
            for name, store in [
                ("vector_store", self._vector_store),
                ("kv_store", self._kv_store),
            ]:
                if store is None:
                    continue
                for method_name in ("shutdown", "close"):
                    if hasattr(store, method_name):
                        try:
                            getattr(store, method_name)()
                        except Exception as exc:
                            logger.exception(
                                "ConnectionManager.{}.{} failed: {}",
                                name,
                                method_name,
                                exc,
                            )
                        break
            self._vector_store = None
            self._vector_store_created = False
            self._kv_store = None
            self._kv_store_created = False

    # ── private: 创建逻辑（从 LatentCache._build_* 迁移） ──────────

    def _create_vector_store(self) -> Optional[VectorStore]:
        from .vector_store.qdrant import QdrantVectorStore

        config = self._config
        store_type = (getattr(config, "vector_store_type", "") or "").lower()

        if store_type == "faiss":
            return self._build_faiss_store()

        if store_type == "qdrant":
            qdrant_url = getattr(config, "qdrant_url", None)
            if not qdrant_url:
                logger.debug("Qdrant vector store selected without qdrant_url; using in-memory Qdrant")
            try:
                return QdrantVectorStore(
                    url=qdrant_url or "",
                    api_key=getattr(config, "qdrant_api_key", None),
                )
            except NotImplementedError as exc:
                # TODO(qdrant): drop this fallback once QdrantVectorStore lands;
                # otherwise it silently masks regressions in the qdrant backend.
                logger.warning(
                    "Qdrant vector store is not implemented yet ({}); "
                    "falling back to FAISSVectorStore. "
                    "Set vector_store_type='faiss' in CacheConfig to silence this warning.",
                    exc,
                )
                return self._build_faiss_store()

        if store_type:
            logger.debug(
                "Unknown vector_store_type '{}'; vector store disabled",
                store_type,
            )
        else:
            logger.debug("vector_store_type not set; vector store disabled")
        return None

    def _build_faiss_store(self) -> "VectorStore":
        from .vector_store.faiss import FAISSVectorStore

        config = self._config
        cache_dir = self._storage_dir.parent if self._storage_dir else Path(".")
        index_dir = getattr(config, "faiss_index_dir", None) or str(cache_dir / "faiss")
        vector_dim = int(getattr(config, "vector_dim", 512))
        return FAISSVectorStore(Path(index_dir), vector_dim=vector_dim, index_type="L2")

    def _create_kv_store(self) -> KVStore:
        from .storage.fluxon import FluxonKVStore
        from .storage.local_file import LocalFileKVStore

        config = self._config
        store_type = (getattr(config, "kv_store_type", "") or "").lower()

        if store_type == "fluxon":
            config_path = getattr(config, "fluxon_config_path", None)
            try:
                return FluxonKVStore(config_path=config_path)
            except Exception as exc:
                logger.exception(
                    "FluxonKVStore init failed config_path={}: {}",
                    config_path,
                    exc,
                )
                raise RuntimeError(
                    f"FluxonKVStore init failed config_path={config_path} err_type={type(exc).__name__} err={exc}"
                ) from exc

        if store_type and store_type not in {"local", "local_file"}:
            logger.debug(
                "Unknown kv_store_type '{}'; falling back to LocalFileKVStore",
                store_type,
            )
        storage_dir = self._storage_dir or Path("./storage")
        return LocalFileKVStore(storage_dir)
