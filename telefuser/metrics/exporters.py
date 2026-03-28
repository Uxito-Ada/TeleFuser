"""
Metrics exporters for TeleFuser.

Provides exporters for different monitoring systems.
Currently supports Prometheus format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import MetricRegistry


class MetricsExporter(ABC):
    """Abstract base class for metrics exporters."""

    @abstractmethod
    def export(self, registry: MetricRegistry) -> str:
        """Export metrics in specific format.

        Args:
            registry: The metric registry to export.

        Returns:
            String containing exported metrics.
        """
        pass


class PrometheusExporter(MetricsExporter):
    """Prometheus text exposition format exporter."""

    def export(self, registry: MetricRegistry) -> str:
        """Export metrics in Prometheus text exposition format.

        Args:
            registry: The metric registry to export.

        Returns:
            String containing Prometheus-formatted metrics.
        """
        return registry.get_prometheus_format()


class OpenTelemetryExporter(MetricsExporter):
    """OpenTelemetry format exporter - placeholder for future implementation."""

    def export(self, registry: MetricRegistry) -> str:
        """Export metrics in OpenTelemetry format.

        Args:
            registry: The metric registry to export.

        Returns:
            String containing OpenTelemetry-formatted metrics.
        """
        # TODO: Implement OpenTelemetry export
        raise NotImplementedError("OpenTelemetry export not yet implemented")


class StatsDExporter(MetricsExporter):
    """StatsD format exporter - placeholder for future implementation."""

    def __init__(self, host: str = "localhost", port: int = 8125) -> None:
        """Initialize StatsD exporter.

        Args:
            host: StatsD server host.
            port: StatsD server port.
        """
        self.host = host
        self.port = port

    def export(self, registry: MetricRegistry) -> str:
        """Export metrics in StatsD format.

        Args:
            registry: The metric registry to export.

        Returns:
            String containing StatsD-formatted metrics.
        """
        # TODO: Implement StatsD export
        raise NotImplementedError("StatsD export not yet implemented")


def get_exporter(exporter_type: str = "prometheus", **kwargs) -> MetricsExporter:
    """Get an exporter by type.

    Args:
        exporter_type: Type of exporter ('prometheus', 'opentelemetry', 'statsd').
        **kwargs: Additional arguments for specific exporters.

    Returns:
        A MetricsExporter instance.

    Raises:
        ValueError: If exporter type is unknown.
    """
    exporters = {
        "prometheus": PrometheusExporter,
        "opentelemetry": OpenTelemetryExporter,
        "statsd": StatsDExporter,
    }

    exporter_cls = exporters.get(exporter_type.lower())
    if exporter_cls is None:
        raise ValueError(f"Unknown exporter type: {exporter_type}")

    return exporter_cls(**kwargs)
