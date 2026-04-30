import asyncio
import time
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Optional

import torch.multiprocessing as mp
from loguru import logger

from telefuser.utils.profiler import ProfilingContext4Debug

try:
    from telefuser.cache_mem.config import CacheConfig, CacheMode
    from telefuser.cache_mem.latent_cache import LatentCache
except Exception:  # optional dependency for cache service
    LatentCache = Any
    CacheConfig = Any
    CacheMode = Any

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class CacheService:
    """缓存服务（lookup + writeback）。

    默认策略：缓存层任何异常都不影响主链路，仅记录告警并降级。
    """

    def __init__(
        self,
        latent_cache: Optional[LatentCache] = None,
        build_latent_data_func: Optional[callable] = None,
        cache_mode: Optional[Any] = None,
        app_cache_config: Optional[Any] = None,
    ) -> None:
        self.latent_cache = latent_cache
        self.app_cache_config = app_cache_config  # CacheConfig from telefuser.cache_mem.config
        self.build_latent_data_func = build_latent_data_func
        self.cache_mode = cache_mode or (CacheMode.READ_WRITE if CacheMode is not Any else None)

        # 确定缓存模式。
        if self.app_cache_config and hasattr(self.app_cache_config, "cache_mode"):
            self.cache_mode = self.app_cache_config.cache_mode
        elif CacheConfig is not Any:
            self.cache_mode = CacheConfig().cache_mode

        # 异步保存相关配置（按 CacheConfig 字段读取，保持文档接口一致）。
        cache_config = self.app_cache_config
        if cache_config is None and CacheConfig is not Any:
            try:
                cache_config = CacheConfig()
            except Exception:
                cache_config = None

        self.save_async_enabled = bool(getattr(cache_config, "save_async_enabled", False))
        self.save_queue_size = int(getattr(cache_config, "save_queue_size", 0) or 0)
        self.save_on_full = str(getattr(cache_config, "save_on_full", "drop") or "drop").lower()
        self.save_queue_warn_threshold = int(getattr(cache_config, "save_queue_warn_threshold", 0) or 0)
        self.vector_wait_warn_s = float(getattr(cache_config, "vector_wait_warn_s", 0) or 0)
        self.vector_wait_poll_s = float(getattr(cache_config, "vector_wait_poll_s", 0) or 0)
        self.vector_wait_timeout_s = float(getattr(cache_config, "vector_wait_timeout_s", 0) or 0)
        self.flush_on_shutdown = bool(getattr(cache_config, "flush_on_shutdown", False))

        self.save_queue: Optional[Queue] = None
        self.save_worker: Optional[Thread] = None
        self._save_stop_event = Event()
        self.pending_vector_updates = 0
        self._pending_lock = Lock()
        self.vector_update_idle = Event()
        self.vector_update_idle.set()

        if self.save_async_enabled:
            maxsize = max(1, self.save_queue_size) if self.save_queue_size else 0
            self.save_queue = Queue(maxsize=maxsize)
            self.save_worker = Thread(
                target=self._start_save_worker,
                name="cache-save-worker",
                daemon=True,
            )
            self.save_worker.start()

    def _reserve_vector_update(self) -> None:
        with self._pending_lock:
            self.pending_vector_updates += 1
            self.vector_update_idle.clear()

    def _release_vector_update(self) -> None:
        with self._pending_lock:
            self.pending_vector_updates = max(0, self.pending_vector_updates - 1)
            if self.pending_vector_updates == 0:
                self.vector_update_idle.set()

    def set_build_latent_data_func(self, func: Optional[callable]) -> None:
        """Set build_latent_data function imported from ppl file."""

        self.build_latent_data_func = func

    def _start_save_worker(self) -> None:
        """后台保存线程入口：循环消费队列并执行保存。"""
        if self.save_queue is None:
            logger.warning("cache-save-worker start skipped: save_queue is None")
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("cache-save-worker started")
        try:
            while not self._save_stop_event.is_set():
                try:
                    task = self.save_queue.get(timeout=0.5)
                except Empty:
                    continue
                try:
                    if task is None:
                        continue
                    task_request, latent_payload = task
                    coro = self._save_latent_payload_impl(task_request, latent_payload, vector_update_reserved=True)
                    if asyncio.iscoroutine(coro):
                        loop.run_until_complete(coro)
                except Exception as exc:
                    logger.exception(f"cache-save-worker save failed: {exc}")
                finally:
                    self.save_queue.task_done()
        finally:
            loop.close()
            logger.info("cache-save-worker stopped")

    async def _save_latent_payload_impl(
        self,
        task_request: Any,
        latent_payload: Optional[dict],
        vector_update_reserved: bool = False,
    ) -> None:
        """后台保存实现（后续步骤补充 pending_vector_updates 逻辑）。"""
        if self.latent_cache is None or not latent_payload:
            return

        latent_states_dict = latent_payload.get("latent_states_dict")
        saved_steps = latent_payload.get("saved_steps") or []
        final_step = latent_payload.get("final_step")
        num_frames = latent_payload.get("num_frames", 0)
        embedding_video_frames = latent_payload.get("embedding_video_frames")
        if latent_states_dict is None or final_step is None or not saved_steps:
            return

        need_vector_update = bool(embedding_video_frames)
        try:
            with ProfilingContext4Debug("save_latent_payload"):
                if need_vector_update:
                    if not vector_update_reserved:
                        self._reserve_vector_update()
                try:
                    task_id = getattr(task_request, "task_id", None)
                    logger.info("cache-save-worker save start task_id={}", task_id)
                except Exception:
                    pass
                await self.latent_cache.save(
                    task_request,
                    latent_states_dict,
                    num_frames,
                    int(final_step),
                    list(saved_steps),
                    embedding_video_frames=embedding_video_frames,
                )
                try:
                    task_id = getattr(task_request, "task_id", None)
                    logger.info("cache-save-worker save end task_id={}", task_id)
                except Exception:
                    pass
        except Exception as exc:
            logger.exception(f"Cache writeback failed, ignored: {exc}")
        finally:
            if need_vector_update:
                self._release_vector_update()

    async def _wait_vector_updates_done(self, task_request: Any = None) -> None:
        """等待向量更新栅栏释放，避免 lookup 读到未完成的向量更新。"""
        if not self.save_async_enabled:
            return
        if self.vector_update_idle.is_set():
            return
        poll_s = self.vector_wait_poll_s if self.vector_wait_poll_s > 0 else 0.05
        warn_s = self.vector_wait_warn_s if self.vector_wait_warn_s > 0 else 0.0
        timeout_s = self.vector_wait_timeout_s if self.vector_wait_timeout_s > 0 else 0.0
        task_id = getattr(task_request, "task_id", None)
        start = time.monotonic()
        warned = False
        logger.info(
            "CacheService.build_latent_data wait vector_update_idle start task_id={} pending={}",
            task_id,
            self.pending_vector_updates,
        )
        while not self.vector_update_idle.is_set():
            elapsed = time.monotonic() - start
            if warn_s and not warned and elapsed >= warn_s:
                logger.warning(
                    "CacheService.build_latent_data wait vector_update_idle exceeded {:.2f}s task_id={} pending={}",
                    warn_s,
                    task_id,
                    self.pending_vector_updates,
                )
                warned = True
            if timeout_s and elapsed >= timeout_s:
                logger.warning(
                    "CacheService.build_latent_data wait vector_update_idle timeout {:.2f}s "
                    "task_id={} pending={}; continue with lookup",
                    timeout_s,
                    task_id,
                    self.pending_vector_updates,
                )
                return
            await asyncio.sleep(poll_s)
        logger.info(
            "CacheService.build_latent_data wait vector_update_idle end task_id={} elapsed={:.2f}s",
            task_id,
            time.monotonic() - start,
        )

    async def build_latent_data(self, task_request: Any, task_data: dict) -> Optional[dict]:
        """构建 latent_data，用于传递给 pipeline。

        默认降级：缓存 lookup / build 任何异常都返回安全的 miss 结构。
        """
        with ProfilingContext4Debug("build_latent_data"):
            cache_config = self.app_cache_config
            if cache_config is None and CacheConfig is not Any:
                try:
                    cache_config = CacheConfig()
                except Exception:
                    cache_config = None

            cache_result = None
            if self.latent_cache is not None and self.cache_mode in [CacheMode.READ_WRITE, CacheMode.READ_ONLY]:
                try:
                    await self._wait_vector_updates_done(task_request)
                    cache_result = await self.latent_cache.lookup(task_request)
                except Exception as exc:
                    logger.exception(f"Cache lookup failed, degrade to miss: {exc}")
                    cache_result = None

            if self.build_latent_data_func is not None:
                try:
                    latent_data = self.build_latent_data_func(task_data=task_data, cache_result=cache_result)
                    if latent_data is not None:
                        return latent_data
                except Exception as exc:
                    logger.exception(f"build_latent_data_func failed, fallback to default: {exc}")

            cached_latent = None
            skip_step = 0
            if cache_result is not None and getattr(cache_result, "hit", False):
                cached_latent = getattr(cache_result, "latent_state", None)
                skip_step = int(getattr(cache_result, "skip_step", 0) or 0)

            saved_steps = []
            if cache_config is not None:
                saved_steps = list(getattr(cache_config, "key_steps", []) or [])
            saved_steps = [int(step) for step in saved_steps]

            return {
                "hit": bool(cached_latent is not None and skip_step > 0),
                "skip_step": skip_step,
                "cached_latent": cached_latent,
                "saved_steps": saved_steps,
            }

    async def save_latent_payload(self, task_request: Any, latent_payload: Optional[dict]) -> None:
        """保存 latent_payload 到缓存。

        默认降级：缓存 writeback 任何异常都记录告警并忽略。
        """

        if self.cache_mode == CacheMode.READ_ONLY or self.latent_cache is None or not latent_payload:
            return

        latent_states_dict = latent_payload.get("latent_states_dict")
        saved_steps = latent_payload.get("saved_steps") or []
        final_step = latent_payload.get("final_step")
        embedding_video_frames = latent_payload.get("embedding_video_frames")
        need_vector_update = bool(embedding_video_frames)
        if latent_states_dict is None or final_step is None or not saved_steps:
            return

        if self.save_async_enabled:
            if self.save_queue is None:
                logger.warning("Cache save enqueue skipped: save_queue is not initialized")
                return
            if self.save_queue_warn_threshold > 0:
                try:
                    if self.save_queue.qsize() >= self.save_queue_warn_threshold:
                        logger.warning(
                            "Cache save queue length warning: {}",
                            self.save_queue.qsize(),
                        )
                except Exception:
                    pass
            if not self.save_queue.full():
                reserved_vector_update = False
                try:
                    if need_vector_update:
                        self._reserve_vector_update()
                        reserved_vector_update = True
                    self.save_queue.put_nowait((task_request, latent_payload))
                    try:
                        task_id = getattr(task_request, "task_id", None)
                        logger.info(
                            "Cache save enqueue task_id={} qsize={}",
                            task_id,
                            self.save_queue.qsize(),
                        )
                    except Exception:
                        pass
                except Exception as exc:
                    if reserved_vector_update:
                        self._release_vector_update()
                    logger.exception(f"Cache save enqueue failed, ignored: {exc}")
                return

            policy = (self.save_on_full or "drop").lower()
            if policy == "drop":
                logger.warning("Cache save queue full: drop task")
                return
            if policy == "downgrade":
                # TODO: implement latent-only downgrade/eviction when the async
                # save queue is full. For now, keep the behavior explicit and
                # avoid running a partial path that pretends to persist data.
                logger.warning("Cache save queue full: downgrade policy is TODO; drop task")
                return
            if policy == "sync":
                logger.warning("Cache save queue full: fallback to sync save")
            else:
                logger.warning("Cache save queue full: unknown policy={}, drop task", policy)
                return

        # Sync fallback: route through _save_latent_payload_impl so the
        # vector_update_idle barrier is respected (otherwise concurrent
        # lookups can race the in-flight upsert and read stale state).
        try:
            await self._save_latent_payload_impl(
                task_request,
                latent_payload,
                vector_update_reserved=False,
            )
        except Exception as exc:
            logger.exception(f"Cache writeback failed, ignored: {exc}")

    async def _save_latent_payload_downgrade(self, task_request: Any, latent_payload: Optional[dict]) -> None:
        """TODO: save latent-only cache entries when full-queue downgrade is implemented."""
        del task_request, latent_payload
        logger.warning("Cache save downgrade is TODO; task dropped")

    def shutdown(self) -> None:
        """释放缓存服务资源（尽力而为）。"""

        if self.save_worker is not None:
            try:
                if self.flush_on_shutdown and self.save_queue is not None:
                    try:
                        self.save_queue.join()
                    except Exception as exc:
                        logger.exception(f"CacheService.flush failed: {exc}")
                self._save_stop_event.set()
                if self.save_queue is not None:
                    try:
                        self.save_queue.put_nowait(None)
                    except Exception:
                        pass
                self.save_worker.join(timeout=5)
            except Exception as exc:
                logger.exception(f"CacheService.stop worker failed: {exc}")

        if self.latent_cache is not None and hasattr(self.latent_cache, "shutdown"):
            try:
                self.latent_cache.shutdown()
            except Exception as exc:
                logger.exception(f"CacheService.shutdown failed: {exc}")
        self.latent_cache = None
        self.save_worker = None
        self.save_queue = None

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
        if hasattr(self, "pipeline") and self.pipeline is not None:
            del self.pipeline
