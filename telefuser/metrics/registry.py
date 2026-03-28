"""
Metric Registry for TeleFuser metrics module.

Provides centralized management and registration of metrics.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from .collector import (
    Counter,
    Gauge,
    Histogram,
    Labels,
    Metric,
    Summary,
    create_metric,
)


class MetricAlreadyRegisteredError(Exception):
    """Raised when attempting to register a metric with a name that already exists."""

    pass


class MetricNotFoundError(Exception):
    """Raised when attempting to get a metric that doesn't exist."""

    pass


class StageMetricContext:
    """Context object returned when registering a stage for metrics collection.

    Provides convenient access to common stage metrics.
    """

    def __init__(self, stage_name: str, registry: "MetricRegistry") -> None:
        self.stage_name = stage_name
        self._registry = registry

        # Create standard stage metrics
        self._duration = registry.histogram(
            f"telefuser_stage_{stage_name}_duration_seconds",
            f"Duration of {stage_name} stage execution in seconds",
        )

        self._total = registry.counter(
            f"telefuser_stage_{stage_name}_total",
            f"Total executions of {stage_name} stage",
        )

        self._errors = registry.counter(
            f"telefuser_stage_{stage_name}_errors_total",
            f"Total errors in {stage_name} stage",
        )

    @property
    def duration(self) -> Histogram:
        """Get duration histogram."""
        return self._duration

    @property
    def total(self) -> Counter:
        """Get total counter."""
        return self._total

    @property
    def errors(self) -> Counter:
        """Get errors counter."""
        return self._errors

    def record_execution(
        self,
        duration: float,
        success: bool,
    ) -> None:
        """Record a stage execution.

        Args:
            duration: Execution duration in seconds.
            success: Whether the execution was successful.
        """
        self._duration.observe(duration)
        self._total.inc()

        if not success:
            self._errors.inc()

    def enter(self) -> None:
        """Mark the start of an execution."""
        pass

    def exit(self) -> None:
        """Mark the end of an execution."""
        pass


class MetricRegistry:
    """Central registry for all metrics.

    Thread-safe implementation for concurrent access.
    Supports stage registration for pipeline monitoring.
    """

    def __init__(self, namespace: str = "telefuser") -> None:
        """Initialize the metric registry.

        Args:
            namespace: Prefix for all metrics (default: "telefuser").
        """
        self._namespace = namespace
        self._metrics: dict[str, Metric] = {}
        self._stages: dict[str, StageMetricContext] = {}
        self._lock = threading.RLock()
        self._custom_collectors: list[Callable[[], list[str]]] = []

    @property
    def namespace(self) -> str:
        """Get the metric namespace."""
        return self._namespace

    def _make_full_name(self, name: str) -> str:
        """Create full metric name with namespace prefix."""
        if name.startswith(self._namespace):
            return name
        return f"{self._namespace}_{name}"

    def register(
        self,
        metric: Metric,
        override: bool = False,
    ) -> Metric:
        """Register a metric with the registry.

        Args:
            metric: The metric to register.
            override: If True, replace existing metric with same name.

        Returns:
            The registered metric.

        Raises:
            MetricAlreadyRegisteredError: If metric with same name exists and override is False.
        """
        full_name = self._make_full_name(metric.name)
        with self._lock:
            if full_name in self._metrics and not override:
                raise MetricAlreadyRegisteredError(f"Metric '{full_name}' already registered")

            # Update metric name to full name
            metric.name = full_name
            self._metrics[full_name] = metric
            return metric

    def counter(
        self,
        name: str,
        description: str,
        labels: Labels = None,
    ) -> Counter:
        """Create and register a counter metric.

        Args:
            name: Metric name (without namespace prefix).
            description: Metric description.
            labels: Optional labels.

        Returns:
            The created counter.
        """
        full_name = self._make_full_name(name)
        with self._lock:
            if full_name in self._metrics:
                return self._metrics[full_name]  # type: ignore

            metric = Counter(name=full_name, description=description, labels=labels)
            self._metrics[full_name] = metric
            return metric

    def gauge(
        self,
        name: str,
        description: str,
        labels: Labels = None,
    ) -> Gauge:
        """Create and register a gauge metric.

        Args:
            name: Metric name (without namespace prefix).
            description: Metric description.
            labels: Optional labels.

        Returns:
            The created gauge.
        """
        full_name = self._make_full_name(name)
        with self._lock:
            if full_name in self._metrics:
                return self._metrics[full_name]  # type: ignore

            metric = Gauge(name=full_name, description=description, labels=labels)
            self._metrics[full_name] = metric
            return metric

    def histogram(
        self,
        name: str,
        description: str,
        labels: Labels = None,
        buckets: list[float] | None = None,
    ) -> Histogram:
        """Create and register a histogram metric.

        Args:
            name: Metric name (without namespace prefix).
            description: Metric description.
            labels: Optional labels.
            buckets: Optional histogram buckets.

        Returns:
            The created histogram.
        """
        full_name = self._make_full_name(name)
        with self._lock:
            if full_name in self._metrics:
                return self._metrics[full_name]  # type: ignore

            metric = Histogram(name=full_name, description=description, labels=labels, buckets=buckets)
            self._metrics[full_name] = metric
            return metric

    def summary(
        self,
        name: str,
        description: str,
        labels: Labels = None,
        quantiles: list[float] | None = None,
    ) -> Summary:
        """Create and register a summary metric.

        Args:
            name: Metric name (without namespace prefix).
            description: Metric description.
            labels: Optional labels.
            quantiles: Optional quantiles.

        Returns:
            The created summary.
        """
        full_name = self._make_full_name(name)
        with self._lock:
            if full_name in self._metrics:
                return self._metrics[full_name]  # type: ignore

            metric = Summary(name=full_name, description=description, labels=labels, quantiles=quantiles)
            self._metrics[full_name] = metric
            return metric

    def register_stage(self, stage_name: str) -> StageMetricContext:
        """Register a stage for metrics collection.

        Args:
            stage_name: Name of the stage.

        Returns:
            A StageMetricContext for recording stage metrics.
        """
        with self._lock:
            if stage_name in self._stages:
                return self._stages[stage_name]

            context = StageMetricContext(stage_name, self)
            self._stages[stage_name] = context
            return context

    def unregister_stage(self, stage_name: str) -> None:
        """Unregister a stage from metrics collection.

        Args:
            stage_name: Name of the stage to unregister.
        """
        with self._lock:
            if stage_name in self._stages:
                del self._stages[stage_name]

    def get_stage(self, stage_name: str) -> StageMetricContext | None:
        """Get the metrics context for a stage.

        Args:
            stage_name: Name of the stage.

        Returns:
            The StageMetricContext if registered, None otherwise.
        """
        with self._lock:
            return self._stages.get(stage_name)

    def get_metric(self, name: str) -> Metric:
        """Get a metric by name.

        Args:
            name: Full metric name (with namespace prefix).

        Returns:
            The metric.

        Raises:
            MetricNotFoundError: If metric doesn't exist.
        """
        full_name = self._make_full_name(name)
        with self._lock:
            if full_name not in self._metrics:
                raise MetricNotFoundError(f"Metric '{full_name}' not found")
            return self._metrics[full_name]

    def get_or_create_counter(
        self,
        name: str,
        description: str,
        labels: Labels = None,
    ) -> Counter:
        """Get existing counter or create new one.

        Args:
            name: Metric name (without namespace prefix).
            description: Metric description (used only if creating new).
            labels: Optional labels.

        Returns:
            The counter.
        """
        return self.counter(name, description, labels)

    def get_or_create_gauge(
        self,
        name: str,
        description: str,
        labels: Labels = None,
    ) -> Gauge:
        """Get existing gauge or create new one.

        Args:
            name: Metric name (without namespace prefix).
            description: Metric description (used only if creating new).
            labels: Optional labels.

        Returns:
            The gauge.
        """
        return self.gauge(name, description, labels)

    def get_or_create_histogram(
        self,
        name: str,
        description: str,
        labels: Labels = None,
        buckets: list[float] | None = None,
    ) -> Histogram:
        """Get existing histogram or create new one.

        Args:
            name: Metric name (without namespace prefix).
            description: Metric description (used only if creating new).
            labels: Optional labels.
            buckets: Optional histogram buckets.

        Returns:
            The histogram.
        """
        return self.histogram(name, description, labels, buckets)

    def add_custom_collector(self, collector: Callable[[], list[str]]) -> None:
        """Add a custom collector function.

        Custom collectors are called during Prometheus export to add
        additional metrics from external sources.

        Args:
            collector: Function that returns list of Prometheus-formatted lines.
        """
        with self._lock:
            self._custom_collectors.append(collector)

    def remove_custom_collector(self, collector: Callable[[], list[str]]) -> None:
        """Remove a custom collector function.

        Args:
            collector: The collector function to remove.
        """
        with self._lock:
            if collector in self._custom_collectors:
                self._custom_collectors.remove(collector)

    def get_prometheus_format(self) -> str:
        """Export all metrics in Prometheus text exposition format.

        Returns:
            String containing all metrics in Prometheus format.
        """
        lines: list[str] = []

        with self._lock:
            # Export all registered metrics
            for metric in self._metrics.values():
                lines.extend(metric.to_prometheus())
                lines.append("")  # Empty line between metrics

            # Call custom collectors
            for collector in self._custom_collectors:
                try:
                    custom_lines = collector()
                    lines.extend(custom_lines)
                except Exception:
                    # Log error but don't fail export
                    pass

        return "\n".join(lines)

    def reset_all(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()

    def clear(self) -> None:
        """Clear all registered metrics."""
        with self._lock:
            self._metrics.clear()
            self._stages.clear()
            self._custom_collectors.clear()

    def list_metrics(self) -> list[str]:
        """List all registered metric names.

        Returns:
            List of metric names.
        """
        with self._lock:
            return list(self._metrics.keys())

    def list_stages(self) -> list[str]:
        """List all registered stage names.

        Returns:
            List of stage names.
        """
        with self._lock:
            return list(self._stages.keys())


# Global registry instance
_global_registry: MetricRegistry | None = None
_registry_lock = threading.Lock()


def get_metrics_registry(namespace: str = "telefuser") -> MetricRegistry:
    """Get or create the global metrics registry.

    Args:
        namespace: Namespace for metrics (only used when creating new registry).

    Returns:
        The global MetricRegistry instance.
    """
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = MetricRegistry(namespace=namespace)
        return _global_registry


def reset_global_registry() -> None:
    """Reset the global metrics registry.

    Useful for testing or when you need to clear all metrics.
    """
    global _global_registry
    with _registry_lock:
        if _global_registry is not None:
            _global_registry.clear()
        _global_registry = None
