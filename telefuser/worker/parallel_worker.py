"""Parallel worker using multiprocessing for multi-GPU execution.

Spawns separate processes for distributed execution across multiple GPUs
with proper process group initialization.
"""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable
from datetime import timedelta
from queue import Empty
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.multiprocessing.spawn

from telefuser.core.base_stage import BaseStage
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.utils.system import PortAllocator

if TYPE_CHECKING:
    from telefuser.metrics import StageMetricContext


def to_device(data: Any, device: str | torch.device) -> Any:
    """Recursively move data to target device."""
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, tuple | list):
        return [to_device(t, device) for t in data]
    elif isinstance(data, torch.Tensor):
        if data.device == device:
            # Already on target device - ensure sharing for CPU tensors
            if device == "cpu" and not data.is_shared():
                data.share_memory_()
            return data
        tensor = data.clone().to(device)
        if device == "cpu" and not tensor.is_shared():
            tensor.share_memory_()
        return tensor
    else:
        return data


def _worker_loop(
    rank: int,
    world_size: int,
    queue_in: list[mp.Queue],
    queue_out: mp.Queue,
    stage: BaseStage,
    master_port: int,
) -> None:
    """Worker process main loop.

    Initializes distributed process group and processes tasks.
    """
    try:
        device = stage.device
        if world_size > 1:
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(master_port)
            parallel_config = stage.model_runtime_config.parallel_config
            device_ids = parallel_config.device_ids
            device_id = rank
            if device_ids is not None:
                device_id = device_ids[rank]
            device = torch.device(type=current_platform.device_type, index=device_id)
            stage.model_runtime_config.device_id = device_id
            stage.device = device
            if current_platform.device_type == "cuda":
                # torch.distributed.all_reduce does not free the input tensor until
                # the synchronization point. This causes the memory usage to grow
                # as the number of all_reduce calls increases. This env var disables
                # this behavior.
                # Related issue:
                # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
                os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            timeout = timedelta(seconds=600)
            dist.init_process_group(
                backend=current_platform.dist_backend,
                init_method="env://",
                timeout=timeout,
                world_size=world_size,
                rank=rank,
                device_id=device,
            )
            current_platform.set_device(device)
            stage.parallel_models()

        while True:
            data = queue_in[rank].get()
            name, args, kwargs = data
            del data
            if name == "exit":
                logger.info(f"parallel worker {stage.name} on rank {rank} exits")
                break
            if not hasattr(stage, name):
                raise AttributeError(f'{stage.__class__.__name__} has no attribute "{name}"')
            kwargs = to_device(kwargs, device)
            args = to_device(args, device)
            with torch.no_grad():
                if not isinstance(args, list):
                    args = [args]
                y = getattr(stage, name)(*args, **kwargs)
            del kwargs, args
            current_platform.empty_cache()
            # Always output results when world_size=1
            if world_size == 1 or rank == 0:
                queue_out.put(y)
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Error in worker loop (rank {rank}): {e}")
        queue_out.put(e)  # any exception caught in the worker will be raised to the main process
    finally:
        del stage, args, kwargs
        current_platform.synchronize()
        gc.collect()
        current_platform.empty_cache()
        current_platform.ipc_collect()
        if world_size > 1:
            dist.destroy_process_group()


class ParallelWorker:
    """Multi-process worker for distributed execution across multiple GPUs."""

    def __init__(
        self,
        stage: BaseStage,
    ) -> None:
        parallel_config = stage.model_runtime_config.parallel_config
        parallel_config.validate()

        self._stage: BaseStage = stage  # Keep reference to wrapped stage for metrics proxying
        self.world_size: int = parallel_config.world_size
        self.device: str = current_platform.device_type
        self.device_ids: list[int] = parallel_config.device_ids
        if self.device_ids is None:
            self.device_ids = list(range(self.world_size))

        self.name: str = f"Parallel Worker {stage.name}"
        self.queue_with_cpu: bool = parallel_config.queue_with_cpu
        self.timeout: int = parallel_config.timeout

        # Use spawn to start processes regardless of world_size
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None or current_method != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError as e:
                raise RuntimeError("Failed to set start method to spawn:", e)

        spawn_ctx = mp.get_context("spawn")
        self.queue_in: list[mp.Queue] = [spawn_ctx.Queue() for _ in range(self.world_size)]
        self.queue_out: mp.Queue = spawn_ctx.Queue()

        master_port = PortAllocator().get_free_port_in_interval()
        logger.info(f"parallel worker {self.name} with port {master_port}, world_size={self.world_size}")

        # For world_size=1 case, still use spawn to start one process
        self.ctx = mp.spawn(
            _worker_loop,
            args=(
                self.world_size,
                self.queue_in,
                self.queue_out,
                stage,
                master_port,
            ),
            nprocs=self.world_size,
            join=False,
        )

    def enable_metrics(self, registry: Any | None = None) -> None:
        """Enable metrics collection on the wrapped stage.

        Note: Metrics are collected in worker subprocesses but aggregated
        metrics won't be visible in the main process. For distributed
        execution, consider using external metrics collection (e.g., Prometheus).
        """
        self._stage.enable_metrics(registry)

    def disable_metrics(self) -> None:
        """Disable metrics collection on the wrapped stage."""
        self._stage.disable_metrics()

    @property
    def _metrics_hook(self) -> StageMetricContext | None:
        """Proxy metrics hook from wrapped stage."""
        return self._stage._metrics_hook

    @_metrics_hook.setter
    def _metrics_hook(self, value: StageMetricContext | None) -> None:
        """Proxy metrics hook setter to wrapped stage."""
        self._stage._metrics_hook = value

    def put_data(self, data: Any) -> None:
        """Send data to all worker processes."""
        if self.queue_with_cpu:
            data = to_device(data, "cpu")
        for i, q in enumerate(self.queue_in):
            data = to_device(data, device=f"{self.device}:{self.device_ids[i]}")
            q.put(data)

    def __call__(self, *args: Any, **kwargs: Any) -> Any | Callable[[], Any]:
        """Submit __call__ task to all workers."""
        sync = kwargs.pop("sync", False)
        data = ["__call__", args, kwargs]
        self.put_data(data)

        def wait() -> Any:
            try:
                res = self.queue_out.get(timeout=self.timeout)
                if isinstance(res, Exception):
                    raise res
            except Empty:
                logger.error(f"ParallelWorker:{self.name} __call__ timeout")
                raise RuntimeError(f"ParallelWorker:{self.name} __call__ timeout")
            except Exception as e:
                logger.error(f"ParallelWorker:{self.name} __call__ error: {e}")
                raise RuntimeError(f"ParallelWorker:{self.name} __call__ error: {e}")
            return res

        if sync:
            return wait()
        else:
            return wait

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Submit arbitrary method call to all workers."""

        def wrapped_func(*args: Any, **kwargs: Any) -> Any | Callable[[], Any]:
            sync = kwargs.pop("sync", False)
            data = [name, args, kwargs]
            self.put_data(data)

            hook = self._metrics_hook
            if hook is not None:
                hook.enter()

            def wait() -> Any:
                start_time = time.perf_counter()
                try:
                    res = self.queue_out.get(timeout=self.timeout)
                    if isinstance(res, Exception):
                        raise res
                except Empty:
                    logger.error(f"ParallelWorker:{self.name} {name} timeout")
                    raise RuntimeError(f"ParallelWorker:{self.name} {name} timeout")
                except Exception as e:
                    logger.error(f"ParallelWorker:{self.name} {name} error: {e}")
                    raise RuntimeError(f"ParallelWorker:{self.name} {name} error: {e}")
                finally:
                    if hook is not None:
                        duration = time.perf_counter() - start_time
                        hook.record_execution(duration, success=True)
                        hook.exit()

                logger.info(f"ParallelWorker:{self.name} {name} done")
                return res

            if sync:
                return wait()
            else:
                return wait

        return wrapped_func

    def __del__(self) -> None:
        """Cleanup worker processes on deletion."""
        if hasattr(self, "ctx"):
            self.put_data(["exit", None, None])
            for p in self.ctx.processes:
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()
            for q in self.queue_in:
                q.close()
            self.queue_out.close()
