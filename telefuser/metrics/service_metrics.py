"""
Service-level metrics for TeleFuser.

Provides comprehensive metrics collection for service monitoring including
task metrics, queue metrics, and GPU metrics.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

from .config import MetricsConfig
from .registry import MetricRegistry, get_metrics_registry


class GPUMetricsCollector(ABC):
    """Abstract base class for GPU metrics collectors."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if GPU monitoring is available.

        Returns:
            True if the GPU platform is available for monitoring.
        """
        pass

    @abstractmethod
    def collect(self) -> list[dict[str, Any]]:
        """Collect GPU metrics.

        Returns:
            List of dictionaries containing GPU metrics per device.
            Each dict should contain:
            - device_id: GPU device ID
            - name: GPU name
            - memory_used: Used memory in bytes
            - memory_total: Total memory in bytes
            - utilization: GPU utilization as percentage (0-100)
            - temperature: GPU temperature in Celsius
        """
        pass


class NvidiaGPUMetricsCollector(GPUMetricsCollector):
    """NVIDIA GPU metrics collector using pynvml."""

    def __init__(self) -> None:
        """Initialize NVIDIA GPU collector."""
        self._pynvml = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Ensure pynvml is initialized."""
        if self._initialized:
            return self._pynvml is not None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._initialized = True
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if NVIDIA GPU monitoring is available."""
        return self._ensure_initialized()

    def collect(self) -> list[dict[str, Any]]:
        """Collect NVIDIA GPU metrics."""
        if not self._ensure_initialized():
            return []

        metrics = []
        try:
            device_count = self._pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get device name
                name = self._pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                # Get memory info
                mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Get utilization
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except Exception:
                    gpu_util = 0

                # Get temperature
                try:
                    temp = self._pynvml.nvmlDeviceGetTemperature(handle, self._pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = 0

                metrics.append(
                    {
                        "device_id": i,
                        "name": name,
                        "memory_used": mem_info.used,
                        "memory_total": mem_info.total,
                        "utilization": gpu_util,
                        "temperature": temp,
                    }
                )

        except Exception:
            pass

        return metrics

    def __del__(self) -> None:
        """Cleanup pynvml on destruction."""
        if self._pynvml is not None and self._initialized:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


class AMDGPUMetricsCollector(GPUMetricsCollector):
    """AMD GPU metrics collector - placeholder for future implementation."""

    def is_available(self) -> bool:
        """Check if AMD GPU monitoring is available."""
        # TODO: Implement using AMD SMI or similar
        return False

    def collect(self) -> list[dict[str, Any]]:
        """Collect AMD GPU metrics."""
        # TODO: Implement using AMD SMI or similar
        return []


def get_gpu_collector(platform: str = "auto") -> GPUMetricsCollector | None:
    """Get the appropriate GPU metrics collector for the platform.

    Args:
        platform: GPU platform ('nvidia', 'amd', 'auto').

    Returns:
        A GPUMetricsCollector instance, or None if unavailable.
    """
    collectors: list[tuple[str, type[GPUMetricsCollector]]] = [
        ("nvidia", NvidiaGPUMetricsCollector),
        ("amd", AMDGPUMetricsCollector),
    ]

    if platform != "auto":
        # Try specific platform first
        for plat, collector_cls in collectors:
            if plat == platform:
                collector = collector_cls()
                if collector.is_available():
                    return collector
                return None

    # Auto-detect
    for plat, collector_cls in collectors:
        collector = collector_cls()
        if collector.is_available():
            return collector

    return None


class ServiceMetrics:
    """Service-level metrics collection.

    Provides comprehensive metrics for monitoring TeleFuser service
    including task metrics, queue metrics, and GPU metrics.
    """

    def __init__(
        self,
        registry: MetricRegistry | None = None,
        config: MetricsConfig | None = None,
    ) -> None:
        """Initialize service metrics.

        Args:
            registry: Optional metrics registry. Uses global registry if not provided.
            config: Optional metrics configuration.
        """
        self._registry = registry or get_metrics_registry()
        self._config = config or MetricsConfig()
        self._start_time = time.time()

        # GPU collector
        self._gpu_collector: GPUMetricsCollector | None = None
        if self._config.enable_gpu_metrics:
            self._gpu_collector = get_gpu_collector(self._config.gpu_platform)

        # Initialize metrics
        self._init_task_metrics()
        self._init_queue_metrics()
        self._init_gpu_metrics()
        self._init_service_metrics()

    def _init_task_metrics(self) -> None:
        """Initialize task-related metrics."""
        self.tasks_created = self._registry.counter(
            "tasks_created_total",
            "Total number of tasks created",
        )

        self.tasks_completed = self._registry.counter(
            "tasks_completed_total",
            "Total number of tasks completed successfully",
        )

        self.tasks_failed = self._registry.counter(
            "tasks_failed_total",
            "Total number of tasks that failed",
        )

        self.tasks_cancelled = self._registry.counter(
            "tasks_cancelled_total",
            "Total number of tasks cancelled",
        )

        self.task_duration = self._registry.histogram(
            "task_duration_seconds",
            "Duration of task execution in seconds",
            buckets=self._config.histogram_buckets,
        )

        self.task_queue_wait = self._registry.histogram(
            "task_queue_wait_seconds",
            "Time tasks spend waiting in queue in seconds",
            buckets=self._config.histogram_buckets,
        )

    def _init_queue_metrics(self) -> None:
        """Initialize queue-related metrics."""
        self.queue_size = self._registry.gauge(
            "queue_size",
            "Current number of tasks in queue",
        )

        self.queue_pending = self._registry.gauge(
            "queue_pending",
            "Current number of pending tasks",
        )

        self.queue_processing = self._registry.gauge(
            "queue_processing",
            "Current number of tasks being processed",
        )

    def _init_gpu_metrics(self) -> None:
        """Initialize GPU-related metrics."""
        # These are set dynamically during collection
        self._gpu_metrics_initialized = False

        if self._gpu_collector is None:
            return

        # Create metrics for each GPU device
        self.gpu_memory_used: dict[int, Any] = {}
        self.gpu_memory_total: dict[int, Any] = {}
        self.gpu_utilization: dict[int, Any] = {}
        self.gpu_temperature: dict[int, Any] = {}

    def _init_service_metrics(self) -> None:
        """Initialize service-level metrics."""
        self.service_uptime = self._registry.gauge(
            "service_uptime_seconds",
            "Service uptime in seconds",
        )

        self.service_info = self._registry.gauge(
            "service_info",
            "Service information",
            labels={"version": "1.0.0"},
        )

    def record_task_created(self) -> None:
        """Record a new task creation."""
        self.tasks_created.inc()

    def record_task_completed(self, duration: float) -> None:
        """Record a task completion.

        Args:
            duration: Task execution duration in seconds.
        """
        self.tasks_completed.inc()
        self.task_duration.observe(duration)

    def record_task_failed(self) -> None:
        """Record a task failure."""
        self.tasks_failed.inc()

    def record_task_cancelled(self) -> None:
        """Record a task cancellation."""
        self.tasks_cancelled.inc()

    def record_queue_wait(self, wait_time: float) -> None:
        """Record queue wait time.

        Args:
            wait_time: Time spent in queue in seconds.
        """
        self.task_queue_wait.observe(wait_time)

    def update_queue_metrics(
        self,
        size: int,
        pending: int,
        processing: int,
    ) -> None:
        """Update queue metrics.

        Args:
            size: Total queue size.
            pending: Number of pending tasks.
            processing: Number of tasks being processed.
        """
        self.queue_size.set(size)
        self.queue_pending.set(pending)
        self.queue_processing.set(processing)

    def collect_gpu_metrics(self) -> None:
        """Collect and update GPU metrics."""
        if self._gpu_collector is None:
            return

        gpu_data = self._gpu_collector.collect()
        if not gpu_data:
            return

        for gpu_info in gpu_data:
            device_id = gpu_info["device_id"]

            # Initialize metrics for this device if needed
            if device_id not in self.gpu_memory_used:
                self.gpu_memory_used[device_id] = self._registry.gauge(
                    f"gpu_{device_id}_memory_used_bytes",
                    f"GPU {device_id} memory used in bytes",
                )
                self.gpu_memory_total[device_id] = self._registry.gauge(
                    f"gpu_{device_id}_memory_total_bytes",
                    f"GPU {device_id} total memory in bytes",
                )
                self.gpu_utilization[device_id] = self._registry.gauge(
                    f"gpu_{device_id}_utilization_ratio",
                    f"GPU {device_id} utilization ratio (0-1)",
                )
                self.gpu_temperature[device_id] = self._registry.gauge(
                    f"gpu_{device_id}_temperature_celsius",
                    f"GPU {device_id} temperature in Celsius",
                )

            # Update metrics
            self.gpu_memory_used[device_id].set(gpu_info["memory_used"])
            self.gpu_memory_total[device_id].set(gpu_info["memory_total"])
            self.gpu_utilization[device_id].set(gpu_info["utilization"] / 100.0)
            self.gpu_temperature[device_id].set(gpu_info["temperature"])

    def update_service_metrics(self) -> None:
        """Update service-level metrics."""
        self.service_uptime.set(time.time() - self._start_time)

    async def start_periodic_collection(self) -> None:
        """Start periodic metrics collection (for GPU metrics).

        This should be run as a background task.
        """
        while True:
            try:
                self.collect_gpu_metrics()
                self.update_service_metrics()
            except Exception:
                pass

            await asyncio.sleep(self._config.gpu_metrics_interval)

    def get_prometheus_format(self) -> str:
        """Export all metrics in Prometheus format."""
        # Update service metrics before export
        self.update_service_metrics()
        return self._registry.get_prometheus_format()

    @property
    def registry(self) -> MetricRegistry:
        """Get the underlying metrics registry."""
        return self._registry

    @property
    def gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._gpu_collector is not None and self._gpu_collector.is_available()


# Global service metrics instance
_service_metrics: ServiceMetrics | None = None


def get_service_metrics(
    registry: MetricRegistry | None = None,
    config: MetricsConfig | None = None,
) -> ServiceMetrics:
    """Get or create the global service metrics instance.

    Args:
        registry: Optional metrics registry.
        config: Optional metrics configuration.

    Returns:
        The global ServiceMetrics instance.
    """
    global _service_metrics
    if _service_metrics is None:
        _service_metrics = ServiceMetrics(registry=registry, config=config)
    return _service_metrics


def reset_service_metrics() -> None:
    """Reset the global service metrics instance."""
    global _service_metrics
    _service_metrics = None
