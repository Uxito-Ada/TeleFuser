"""
Metrics configuration for TeleFuser.

Provides configuration options for metrics collection and export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MetricsConfig:
    """Configuration for metrics collection.

    Attributes:
        enabled: Whether metrics collection is enabled.
        enable_stage_metrics: Whether to collect stage-level metrics.
        enable_gpu_metrics: Whether to collect GPU metrics.
        enable_http_metrics: Whether to collect HTTP request metrics.
        enable_queue_metrics: Whether to collect queue metrics.
        gpu_metrics_interval: Interval in seconds for GPU metrics collection.
        histogram_buckets: Default buckets for histogram metrics.
        metrics_path: HTTP path for Prometheus metrics endpoint.
        namespace: Prefix for all metric names.
        port: Port for metrics HTTP server (if separate from main server).
    """

    enabled: bool = True

    # Feature flags
    enable_stage_metrics: bool = True
    enable_gpu_metrics: bool = True
    enable_http_metrics: bool = True
    enable_queue_metrics: bool = True

    # Timing
    gpu_metrics_interval: float = 5.0  # seconds

    # Histogram configuration
    histogram_buckets: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )

    # HTTP configuration
    metrics_path: str = "/metrics"

    # Naming
    namespace: str = "telefuser"

    # GPU configuration
    gpu_platform: Literal["nvidia", "amd", "auto"] = "auto"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.gpu_metrics_interval <= 0:
            raise ValueError("gpu_metrics_interval must be positive")

        if not self.histogram_buckets:
            raise ValueError("histogram_buckets cannot be empty")

        # Ensure buckets are sorted
        self.histogram_buckets = sorted(self.histogram_buckets)


# Default configuration
default_metrics_config = MetricsConfig()


def create_metrics_config(
    enabled: bool = True,
    enable_stage_metrics: bool = True,
    enable_gpu_metrics: bool = True,
    enable_http_metrics: bool = True,
    enable_queue_metrics: bool = True,
    gpu_metrics_interval: float = 5.0,
    histogram_buckets: list[float] | None = None,
    metrics_path: str = "/metrics",
    namespace: str = "telefuser",
    gpu_platform: Literal["nvidia", "amd", "auto"] = "auto",
) -> MetricsConfig:
    """Create a metrics configuration with custom settings.

    Args:
        enabled: Whether metrics collection is enabled.
        enable_stage_metrics: Whether to collect stage-level metrics.
        enable_gpu_metrics: Whether to collect GPU metrics.
        enable_http_metrics: Whether to collect HTTP request metrics.
        enable_queue_metrics: Whether to collect queue metrics.
        gpu_metrics_interval: Interval in seconds for GPU metrics collection.
        histogram_buckets: Default buckets for histogram metrics.
        metrics_path: HTTP path for Prometheus metrics endpoint.
        namespace: Prefix for all metric names.
        gpu_platform: GPU platform to use for metrics.

    Returns:
        A MetricsConfig instance.
    """
    return MetricsConfig(
        enabled=enabled,
        enable_stage_metrics=enable_stage_metrics,
        enable_gpu_metrics=enable_gpu_metrics,
        enable_http_metrics=enable_http_metrics,
        enable_queue_metrics=enable_queue_metrics,
        gpu_metrics_interval=gpu_metrics_interval,
        histogram_buckets=histogram_buckets or default_metrics_config.histogram_buckets,
        metrics_path=metrics_path,
        namespace=namespace,
        gpu_platform=gpu_platform,
    )
