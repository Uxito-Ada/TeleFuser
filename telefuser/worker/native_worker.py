"""Native worker using threading for single-device execution.

Simple worker implementation for pipelines that don't require
multi-process or distributed execution.
"""

from __future__ import annotations

import queue
import threading
from queue import Empty
from typing import Any

import torch

from telefuser.core.base_stage import BaseStage
from telefuser.utils.logging import logger


class NativeFuture:
    """Simple future for native worker results."""

    def __init__(self):
        self._q = queue.Queue(maxsize=1)

    def set(self, value: Any) -> None:
        self._q.put(value)

    def get(self, timeout: float | None = None) -> Any:
        res = self._q.get(timeout=timeout)
        if isinstance(res, Exception):
            raise res
        return res


class NativeStageWorker:
    """Thread-based worker for single-device pipeline stages."""

    def __init__(self, stage: BaseStage):
        parallel_config = stage.model_runtime_config.parallel_config
        parallel_config.validate()
        if parallel_config.device_ids:
            logger.warning("native worker is designed for single-device stage.")

        self.name = f"Native Worker {stage.name}"
        self.timeout = parallel_config.timeout
        self._queue = queue.Queue()
        self._alive = True

        self._stage = stage

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        """Main worker loop processing tasks."""
        while self._alive:
            item = self._queue.get()
            if item is None:
                break

            name, args, kwargs, future = item
            if not hasattr(self._stage, name):
                raise AttributeError(f'{self._stage.__class__.__name__} has no attribute "{name}"')
            try:
                with torch.no_grad():
                    if not isinstance(args, list):
                        args = [args]
                    result = getattr(self._stage, name)(*args, **kwargs)
                future.set(result)
            except Exception as e:
                logger.exception(f"{self.name} error in {name}")
                future.set(e)

    def __call__(self, *args, **kwargs):
        """Submit __call__ task to worker."""
        sync = kwargs.pop("sync", False)
        future = NativeFuture()
        self._queue.put(("__call__", args, kwargs, future))

        def wait():
            try:
                return future.get(timeout=self.timeout)
            except Empty:
                logger.error(f"NativelWorker:{self.name} __call__ timeout")
                raise RuntimeError(f"NativelWorker:{self.name} __call__ timeout")
            except Exception as e:
                logger.error(f"NativelWorker:{self.name} __call__ error: {e}")
                raise RuntimeError(f"NativelWorker:{self.name} __call__ error: {e}")

        if sync:
            return wait()
        else:
            return wait

    def __getattr__(self, name: str) -> Any:
        """Submit arbitrary method call to worker."""

        def wrapped(*args, **kwargs):
            sync = kwargs.pop("sync", False)
            future = NativeFuture()
            self._queue.put((name, args, kwargs, future))

            def wait():
                try:
                    return future.get(timeout=self.timeout)
                except Empty:
                    logger.error(f"NativelWorker:{self.name} {name} timeout")
                    raise RuntimeError(f"NativelWorker:{self.name} {name} timeout")
                except Exception as e:
                    logger.error(f"ParallelWorker:{self.name} {name} error: {e}")
                    raise RuntimeError(f"ParallelWorker:{self.name} {name} error: {e}")

            if sync:
                return wait()
            else:
                return wait

        return wrapped

    def close(self):
        """Shutdown the worker."""
        self._alive = False
        self._queue.put(None)
        self._thread.join()
