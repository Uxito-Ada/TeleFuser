from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from .cache_types import CacheResult
from .connection import ConnectionManager
from .metadata import LocalCacheMetadataManager
from .state.interfaces import CacheMetadataManager
from .storage.interfaces import KVStore
from .strategies import BaseCacheStrategy, get_strategy_class
from .vector_store.interfaces import VectorStore

try:
    from telefuser.cache_mem.config import CacheConfig
except (ImportError, ModuleNotFoundError):  # optional dependency for cache service
    CacheConfig = None


class LatentCache:
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        config: Optional["CacheConfig"] = None,
        kv_store: Optional[KVStore] = None,
        vector_store: Optional[VectorStore] = None,
        metadata_manager: Optional[CacheMetadataManager] = None,
        strategy: Optional[BaseCacheStrategy] = None,
    ):
        # Initialize config and directories.
        if config is None:
            if CacheConfig is None:
                raise ValueError("LatentCache requires CacheConfig but it is unavailable")
            config = CacheConfig()
        self.config = config
        if hasattr(self.config, "latent_cache_dir"):
            self.cache_dir = Path(cache_dir or self.config.latent_cache_dir)
        else:
            self.cache_dir = Path(cache_dir or self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir = self.cache_dir / "storage"
        self.dit_cache_dir = self.cache_dir / "dit_cache"
        self.metadata_dir = self.cache_dir / "metadata"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dit_cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self._conn_mgr = ConnectionManager(
            config,
            storage_dir=self.storage_dir,
        )
        self._managed_kv = kv_store is None
        self._managed_vector = vector_store is None

        # Initialize kv_store
        self.kv_store = kv_store or self._conn_mgr.kv_store

        # Initialize metadata_manager
        self.metadata_manager = metadata_manager or LocalCacheMetadataManager(self.metadata_dir)

        # Initialize vector_store
        self.vector_store = vector_store or self._conn_mgr.vector_store

        # Initialize strategy
        if strategy is not None:
            self.strategy = strategy
        else:
            strategy_name = getattr(self.config, "cache_strategy_type", "")
            strategy_cls = get_strategy_class(strategy_name) if strategy_name else None
            if strategy_cls is not None:
                self.strategy = strategy_cls(
                    self.config,
                    self.kv_store,
                    self.vector_store,
                    self.metadata_manager,
                )
            else:
                self.strategy = None

    async def lookup(self, task_request: Any) -> CacheResult:
        task_type = getattr(task_request, "task", "t2v")
        prompt = getattr(task_request, "prompt", "")
        if self.strategy is None:
            return CacheResult(hit=False)
        result = await self.strategy.lookup(prompt, task_type)
        return await self.strategy.apply(result)

    async def save(
        self,
        task_request: Any,
        latent_states_dict: Dict[int, torch.Tensor],
        num_frames: int,
        final_step: int,
        saved_steps: List[int],
        embedding_video_frames: Optional[List[Any]] = None,
    ) -> None:
        task_type = getattr(task_request, "task", "t2v")
        prompt = getattr(task_request, "prompt", "")
        if self.strategy is None:
            return
        await self.strategy.save(
            prompt,
            latent_states_dict,
            num_frames,
            task_type,
            saved_steps,
            embedding_video_frames=embedding_video_frames,
        )

    def shutdown(self) -> None:
        """Release internal resources (best effort)."""
        # Shut down strategy and metadata_manager directly.
        for name in ("strategy", "metadata_manager"):
            obj = getattr(self, name, None)
            if obj is None:
                continue
            for method_name in ("shutdown", "close"):
                if hasattr(obj, method_name):
                    try:
                        getattr(obj, method_name)()
                    except Exception as exc:
                        logger.exception(f"LatentCache.{name}.{method_name} failed: {exc}")
                    break

        if self._conn_mgr is not None and (self._managed_kv or self._managed_vector):
            try:
                self._conn_mgr.shutdown()
            except Exception as exc:
                logger.exception(f"LatentCache.ConnectionManager.shutdown failed: {exc}")

        for name, managed in [
            ("kv_store", self._managed_kv),
            ("vector_store", self._managed_vector),
        ]:
            if managed:
                continue  # Already handled by ConnectionManager
            obj = getattr(self, name, None)
            if obj is None:
                continue
            for method_name in ("shutdown", "close"):
                if hasattr(obj, method_name):
                    try:
                        getattr(obj, method_name)()
                    except Exception as exc:
                        logger.exception(f"LatentCache.{name}.{method_name} failed: {exc}")
                    break

        self.strategy = None
        self.vector_store = None
        self.kv_store = None
        self.metadata_manager = None

    def purge_by_prompt(self, prompt: str, collection: str = "whole") -> bool:
        """Delete cache by prompt (metadata / vector_store / kv_store)."""
        prompt = prompt or ""
        if not prompt:
            return False
        entry = self.metadata_manager.lookup_prompt(
            prompt,
            cache_type="video_approximate_cache",
        )
        if entry is None:
            return False
        cache_id = entry.cache_id
        errors: List[str] = []
        for step in entry.saved_steps:
            try:
                self.kv_store.remove(f"{cache_id}_step{int(step)}")
            except Exception as exc:
                logger.exception(
                    "LatentCache.purge_by_prompt kv remove failed prompt={} cache_id={} step={} err={}",
                    prompt,
                    cache_id,
                    int(step),
                    exc,
                )
                errors.append(
                    f"kv remove failed cache_id={cache_id} step={int(step)} type={type(exc).__name__} err={exc}"
                )
        if self.vector_store is not None:
            try:
                self.vector_store.delete(collection, [cache_id])
            except Exception as exc:
                logger.exception(
                    "LatentCache.purge_by_prompt vector delete failed prompt={} collection={} cache_id={} err={}",
                    prompt,
                    collection,
                    cache_id,
                    exc,
                )
                errors.append(
                    "vector delete failed "
                    f"collection={collection} cache_id={cache_id} "
                    f"type={type(exc).__name__} err={exc}"
                )
        try:
            self.metadata_manager.remove_cache(cache_id)
        except Exception as exc:
            logger.exception(
                "LatentCache.purge_by_prompt metadata remove failed prompt={} cache_id={} err={}",
                prompt,
                cache_id,
                exc,
            )
            errors.append(f"metadata remove failed cache_id={cache_id} type={type(exc).__name__} err={exc}")
        if errors:
            raise RuntimeError(
                f"LatentCache.purge_by_prompt failed prompt={prompt!r} cache_id={cache_id}: {'; '.join(errors)}"
            )
        return True
