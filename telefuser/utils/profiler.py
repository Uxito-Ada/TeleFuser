"""Profiling utilities for performance monitoring."""

from __future__ import annotations

import asyncio
import os
import time
from functools import wraps
from pathlib import Path
from types import TracebackType
from typing import Any, Callable

import torch

from telefuser.platforms import current_platform
from telefuser.utils.logging import logger


class _ProfilingContext:
    """Context manager for profiling function execution."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        self.rank_info = f"Rank {self.rank} - "

        # PyTorch Profiler configuration
        self.enable_profiler = self._should_enable_profiler()
        self.profiler: torch.profiler.profile | None = None
        self.profiler_output_dir = Path(os.getenv("PROFILER_OUTPUT_DIR", "./profiler_output"))
        self.profiler_run_count = 0

    def _should_enable_profiler(self) -> bool:
        """Determine whether to enable profiler based on environment variables."""
        enabled_names = os.getenv("ENABLE_PROFILER_NAMES", "").split(",")
        enabled_names = {name.strip() for name in enabled_names if name.strip()}
        return self.name in enabled_names

    def _get_profiler_output_path(self) -> Path:
        """Generate profiler output file path."""
        self.profiler_run_count += 1
        filename = f"{self.name}_rank{self.rank}_run{self.profiler_run_count}.json"
        return self.profiler_output_dir / filename

    def __enter__(self) -> _ProfilingContext: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    async def __aenter__(self) -> _ProfilingContext: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self:
                    return func(*args, **kwargs)

            return sync_wrapper


class _NullContext:
    """No-op context manager for when profiling is disabled."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    async def __aenter__(self) -> _NullContext:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        return func


ProfilingContext = _ProfilingContext
ENABLE_PROFILING_DEBUG = os.getenv("ENABLE_PROFILING_DEBUG", "false").lower() == "true"
ProfilingContext4Debug = _ProfilingContext if ENABLE_PROFILING_DEBUG else _NullContext


def enable_profiler_for_names(names: str) -> None:
    """Set the list of names to enable profiler for."""
    os.environ["ENABLE_PROFILER_NAMES"] = names


def set_profiler_output_dir(path: str) -> None:
    """Set profiler output directory."""
    os.environ["PROFILER_OUTPUT_DIR"] = path


def get_enabled_profiler_names() -> set[str]:
    """Get the set of currently enabled profiler names."""
    enabled_names = os.getenv("ENABLE_PROFILER_NAMES", "").split(",")
    return {name.strip() for name in enabled_names if name.strip()}
