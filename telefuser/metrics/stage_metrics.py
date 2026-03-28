"""
Stage metrics integration for TeleFuser.

Provides decorators and hooks for stage-level metrics collection.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from .registry import MetricRegistry, StageMetricContext, get_metrics_registry

F = TypeVar("F", bound=Callable[..., Any])


def with_metrics(func: F) -> F:
    """Decorator to wrap stage methods with metrics collection.

    This decorator should be used on BaseStage process methods to
    automatically record execution metrics.

    The stage must have a `_metrics_hook` attribute (StageMetricContext).

    Example:
        class MyStage(BaseStage):
            @with_metrics
            def process(self, data):
                return result
    """

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        hook = getattr(self, "_metrics_hook", None)
        if hook is None:
            return func(self, *args, **kwargs)

        hook.enter()
        start_time = time.perf_counter()
        try:
            result = func(self, *args, **kwargs)
            duration = time.perf_counter() - start_time
            hook.record_execution(duration, success=True)
            return result
        except Exception:
            duration = time.perf_counter() - start_time
            hook.record_execution(duration, success=False)
            raise
        finally:
            hook.exit()

    return wrapper  # type: ignore


async def with_metrics_async(func: F) -> F:
    """Async decorator to wrap stage methods with metrics collection.

    This decorator should be used on async BaseStage process methods.

    Example:
        class MyStage(BaseStage):
            @with_metrics_async
            async def process(self, data):
                return await some_async_operation(data)
    """

    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        hook = getattr(self, "_metrics_hook", None)
        if hook is None:
            return await func(self, *args, **kwargs)

        hook.enter()
        start_time = time.perf_counter()
        try:
            result = await func(self, *args, **kwargs)
            duration = time.perf_counter() - start_time
            hook.record_execution(duration, success=True)
            return result
        except Exception:
            duration = time.perf_counter() - start_time
            hook.record_execution(duration, success=False)
            raise
        finally:
            hook.exit()

    return wrapper  # type: ignore


class StageMetricsManager:
    """Manager for stage metrics across a pipeline.

    Provides centralized management for enabling/disabling metrics
    on multiple stages.
    """

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        """Initialize the stage metrics manager.

        Args:
            registry: Optional metrics registry. Uses global registry if not provided.
        """
        self._registry = registry or get_metrics_registry()
        self._enabled_stages: set[str] = set()

    def enable_stage(self, stage: Any) -> None:
        """Enable metrics collection for a stage.

        Args:
            stage: The stage instance (must have 'name' attribute).
        """
        if not hasattr(stage, "name"):
            raise ValueError("Stage must have a 'name' attribute")

        stage_name = stage.name
        hook = self._registry.register_stage(stage_name)
        stage._metrics_hook = hook
        self._enabled_stages.add(stage_name)

    def disable_stage(self, stage: Any) -> None:
        """Disable metrics collection for a stage.

        Args:
            stage: The stage instance.
        """
        if hasattr(stage, "_metrics_hook"):
            stage_name = getattr(stage, "name", "unknown")
            self._registry.unregister_stage(stage_name)
            stage._metrics_hook = None
            self._enabled_stages.discard(stage_name)

    def enable_all_stages(self, stages: list[Any]) -> None:
        """Enable metrics for multiple stages.

        Args:
            stages: List of stage instances.
        """
        for stage in stages:
            self.enable_stage(stage)

    def disable_all_stages(self) -> None:
        """Disable metrics for all enabled stages."""
        for stage_name in list(self._enabled_stages):
            self._registry.unregister_stage(stage_name)
        self._enabled_stages.clear()

    def get_stage_context(self, stage_name: str) -> StageMetricContext | None:
        """Get the metrics context for a stage.

        Args:
            stage_name: Name of the stage.

        Returns:
            The StageMetricContext if enabled, None otherwise.
        """
        return self._registry.get_stage(stage_name)

    @property
    def registry(self) -> MetricRegistry:
        """Get the underlying metrics registry."""
        return self._registry

    @property
    def enabled_stages(self) -> set[str]:
        """Get the set of enabled stage names."""
        return self._enabled_stages.copy()


def enable_stage_metrics(stage: Any, registry: MetricRegistry | None = None) -> StageMetricContext:
    """Enable metrics collection for a single stage.

    Convenience function for enabling metrics on a stage without
    using a StageMetricsManager.

    Args:
        stage: The stage instance (must have 'name' attribute).
        registry: Optional metrics registry. Uses global registry if not provided.

    Returns:
        The StageMetricContext for the stage.

    Example:
        stage = TextEncodingStage("text_encoding", config)
        context = enable_stage_metrics(stage)
    """
    if not hasattr(stage, "name"):
        raise ValueError("Stage must have a 'name' attribute")

    reg = registry or get_metrics_registry()
    hook = reg.register_stage(stage.name)
    stage._metrics_hook = hook
    return hook


def disable_stage_metrics(stage: Any, registry: MetricRegistry | None = None) -> None:
    """Disable metrics collection for a single stage.

    Args:
        stage: The stage instance.
        registry: Optional metrics registry. Uses global registry if not provided.
    """
    if hasattr(stage, "_metrics_hook"):
        stage_name = getattr(stage, "name", "unknown")
        reg = registry or get_metrics_registry()
        reg.unregister_stage(stage_name)
        stage._metrics_hook = None
