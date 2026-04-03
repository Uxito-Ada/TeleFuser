"""
Memory Snapshot Module for GPU Memory Statistics.

This module provides utilities for capturing and analyzing GPU memory usage
during inference. Adapted from sglang's memory tracking implementation.

Usage:
    from telefuser.utils.memory_snapshot import (
        MemorySnapshot,
        capture_memory_snapshot,
        record_memory_snapshot,
        get_memory_analysis,
    )

    # Capture a snapshot
    snapshot = capture_memory_snapshot()
    print(f"Allocated: {snapshot.allocated_mb:.1f} MB")

    # Record at checkpoints
    record_memory_snapshot("before_forward")
    # ... inference ...
    record_memory_snapshot("after_forward")

    # Get analysis
    analysis = get_memory_analysis()
"""

from __future__ import annotations

import torch

from telefuser.platforms import current_platform
from telefuser.utils.logging import logger

# =============================================================================
# Memory Snapshot Dataclass
# =============================================================================


class MemorySnapshot:
    """
    Snapshot of GPU memory usage at a point in time.

    Captures both current and peak memory metrics for detailed analysis.

    Attributes:
        allocated_mb: Current allocated memory (actual tensor data)
        reserved_mb: Current reserved memory (actual VRAM usage including cache)
        peak_allocated_mb: Peak allocated memory since last reset
        peak_reserved_mb: Peak reserved memory since last reset
    """

    __slots__ = (
        "allocated_mb",
        "reserved_mb",
        "peak_allocated_mb",
        "peak_reserved_mb",
    )

    def __init__(
        self,
        allocated_mb: float,
        reserved_mb: float,
        peak_allocated_mb: float,
        peak_reserved_mb: float,
    ):
        self.allocated_mb = allocated_mb
        self.reserved_mb = reserved_mb
        self.peak_allocated_mb = peak_allocated_mb
        self.peak_reserved_mb = peak_reserved_mb

    def to_dict(self) -> dict[str, float]:
        """Convert to JSON-serializable dict."""
        return {
            "allocated_mb": round(self.allocated_mb, 2),
            "reserved_mb": round(self.reserved_mb, 2),
            "peak_allocated_mb": round(self.peak_allocated_mb, 2),
            "peak_reserved_mb": round(self.peak_reserved_mb, 2),
        }

    @property
    def allocated_gb(self) -> float:
        """Allocated memory in GB."""
        return self.allocated_mb / 1024

    @property
    def reserved_gb(self) -> float:
        """Reserved memory in GB."""
        return self.reserved_mb / 1024

    @property
    def peak_allocated_gb(self) -> float:
        """Peak allocated memory in GB."""
        return self.peak_allocated_mb / 1024

    @property
    def peak_reserved_gb(self) -> float:
        """Peak reserved memory in GB."""
        return self.peak_reserved_mb / 1024

    @property
    def pool_overhead_mb(self) -> float:
        """Memory pool overhead in MB (reserved - allocated)."""
        return self.reserved_mb - self.allocated_mb

    @property
    def pool_overhead_gb(self) -> float:
        """Memory pool overhead in GB."""
        return self.pool_overhead_mb / 1024

    @property
    def pool_overhead_pct(self) -> float:
        """Memory pool overhead as percentage of reserved memory."""
        if self.reserved_mb == 0:
            return 0.0
        return (self.pool_overhead_mb / self.reserved_mb) * 100

    def __repr__(self) -> str:
        return (
            f"MemorySnapshot("
            f"allocated={self.allocated_mb:.1f}MB, "
            f"reserved={self.reserved_mb:.1f}MB, "
            f"peak_allocated={self.peak_allocated_mb:.1f}MB, "
            f"peak_reserved={self.peak_reserved_mb:.1f}MB)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemorySnapshot):
            return NotImplemented
        return (
            self.allocated_mb == other.allocated_mb
            and self.reserved_mb == other.reserved_mb
            and self.peak_allocated_mb == other.peak_allocated_mb
            and self.peak_reserved_mb == other.peak_reserved_mb
        )


# =============================================================================
# Memory Capture Functions
# =============================================================================


def capture_memory_snapshot(device: torch.device | str | None = None) -> MemorySnapshot:
    """
    Capture a snapshot of current GPU memory usage.

    Args:
        device: Device to capture memory for. None uses current device.

    Returns:
        MemorySnapshot with current and peak memory metrics
    """
    device_module = torch.get_device_module()

    if not device_module.is_available():
        return MemorySnapshot(
            allocated_mb=0.0,
            reserved_mb=0.0,
            peak_allocated_mb=0.0,
            peak_reserved_mb=0.0,
        )

    allocated = device_module.memory_allocated(device)
    reserved = device_module.memory_reserved(device)
    peak_allocated = device_module.max_memory_allocated(device)
    peak_reserved = device_module.max_memory_reserved(device)

    return MemorySnapshot(
        allocated_mb=allocated / (1024**2),
        reserved_mb=reserved / (1024**2),
        peak_allocated_mb=peak_allocated / (1024**2),
        peak_reserved_mb=peak_reserved / (1024**2),
    )


def reset_peak_memory_stats(device: torch.device | str | None = None) -> None:
    """
    Reset peak memory statistics for the device.

    Args:
        device: Device to reset stats for. None uses current device.
    """
    device_module = torch.get_device_module()
    if device_module.is_available():
        device_module.reset_peak_memory_stats(device)


def get_memory_info(device: torch.device | str | None = None) -> dict[str, float]:
    """
    Get basic memory info as a dict (for quick logging).

    Args:
        device: Device to get info for. None uses current device.

    Returns:
        Dict with allocated_mb, reserved_mb, peak_allocated_mb, peak_reserved_mb
    """
    snapshot = capture_memory_snapshot(device)
    return snapshot.to_dict()


# =============================================================================
# Memory Snapshot Registry (Singleton)
# =============================================================================


class _MemorySnapshotRegistry:
    """
    Global registry for memory snapshots at checkpoints.

    This is a singleton that tracks memory snapshots across the inference pipeline.
    """

    _instance: "_MemorySnapshotRegistry | None" = None

    def __init__(self):
        self._snapshots: dict[str, MemorySnapshot] = {}
        self._baseline: MemorySnapshot | None = None

    @classmethod
    def get_instance(cls) -> "_MemorySnapshotRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry for a new request."""
        instance = cls.get_instance()
        instance._snapshots = {}
        instance._baseline = None

    def record(self, checkpoint_name: str, snapshot: MemorySnapshot) -> None:
        """
        Record a memory snapshot at a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint
            snapshot: MemorySnapshot to record
        """
        self._snapshots[checkpoint_name] = snapshot

    def capture_and_record(self, checkpoint_name: str) -> MemorySnapshot:
        """
        Capture and record a memory snapshot at a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint

        Returns:
            The captured MemorySnapshot
        """
        snapshot = capture_memory_snapshot()
        self.record(checkpoint_name, snapshot)
        return snapshot

    def set_baseline(self) -> MemorySnapshot:
        """
        Set the baseline memory snapshot (before inference starts).

        Returns:
            The captured baseline MemorySnapshot
        """
        self._baseline = capture_memory_snapshot()
        return self._baseline

    def get(self, checkpoint_name: str) -> MemorySnapshot | None:
        """Get a recorded snapshot by name."""
        return self._snapshots.get(checkpoint_name)

    def get_all(self) -> dict[str, MemorySnapshot]:
        """Get all recorded snapshots."""
        return self._snapshots.copy()

    def get_analysis(self) -> dict[str, float | dict]:
        """
        Get memory analysis with pool overhead and suggestions.

        Returns:
            Dict with memory analysis data
        """
        if not self._snapshots:
            return {}

        # Find peak snapshot
        peak_snapshot = max(
            self._snapshots.values(),
            key=lambda s: s.peak_reserved_mb,
        )

        # Calculate total device memory
        try:
            total_memory_mb = current_platform.get_device_total_memory() / (1024**2)
        except Exception:
            total_memory_mb = 0.0

        remaining_mb = total_memory_mb - peak_snapshot.peak_reserved_mb if total_memory_mb > 0 else 0.0

        analysis: dict[str, float | dict] = {
            "peak_reserved_mb": peak_snapshot.peak_reserved_mb,
            "peak_allocated_mb": peak_snapshot.peak_allocated_mb,
            "peak_reserved_gb": peak_snapshot.peak_reserved_gb,
            "peak_allocated_gb": peak_snapshot.peak_allocated_gb,
            "pool_overhead_mb": peak_snapshot.pool_overhead_mb,
            "pool_overhead_pct": round(peak_snapshot.pool_overhead_pct, 1),
            "remaining_memory_mb": round(remaining_mb, 2),
            "remaining_memory_gb": round(remaining_mb / 1024, 3),
            "total_device_memory_mb": round(total_memory_mb, 2),
            "total_device_memory_gb": round(total_memory_mb / 1024, 3),
            "snapshots": {name: snapshot.to_dict() for name, snapshot in self._snapshots.items()},
        }

        if self._baseline:
            analysis["baseline"] = self._baseline.to_dict()
            analysis["memory_increase_mb"] = round(peak_snapshot.peak_reserved_mb - self._baseline.reserved_mb, 2)
            analysis["memory_increase_gb"] = round(
                (peak_snapshot.peak_reserved_mb - self._baseline.reserved_mb) / 1024, 3
            )

        return analysis

    def summary(self) -> str:
        """Get human-readable summary string."""
        analysis = self.get_analysis()
        if not analysis:
            return "No memory snapshots recorded."

        lines = [
            "",
            "=" * 60,
            "Memory Analysis",
            "=" * 60,
            f"Peak Reserved: {analysis['peak_reserved_mb']:.0f} MB ({analysis['peak_reserved_gb']:.2f} GB)",
            f"Peak Allocated: {analysis['peak_allocated_mb']:.0f} MB ({analysis['peak_allocated_gb']:.2f} GB)",
            f"Pool Overhead: {analysis['pool_overhead_mb']:.0f} MB ({analysis['pool_overhead_pct']:.1f}%)",
            f"Remaining GPU Memory: {analysis['remaining_memory_mb']:.0f} MB",
        ]

        if "memory_increase_mb" in analysis:
            lines.append(f"Memory Increase: {analysis['memory_increase_mb']:.0f} MB")

        lines.extend(
            [
                "-" * 60,
                "Memory Checkpoints:",
            ]
        )

        for name, snapshot in self.get_all().items():
            snapshot_dict = snapshot.to_dict()
            lines.append(
                f"  {name}: {snapshot_dict['allocated_mb']:.0f}/{snapshot_dict['reserved_mb']:.0f} MB "
                f"(peak: {snapshot_dict['peak_allocated_mb']:.0f}/{snapshot_dict['peak_reserved_mb']:.0f} MB)"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Public API
# =============================================================================


def reset_memory_registry() -> None:
    """Reset the memory snapshot registry for a new request."""
    _MemorySnapshotRegistry.reset()


def record_memory_snapshot(checkpoint_name: str) -> MemorySnapshot:
    """
    Record a memory snapshot at a specific checkpoint.

    Args:
        checkpoint_name: Name of the checkpoint (e.g., "before_forward", "after_denoising")

    Returns:
        The captured MemorySnapshot
    """
    return _MemorySnapshotRegistry.get_instance().capture_and_record(checkpoint_name)


def get_memory_snapshots() -> dict[str, MemorySnapshot]:
    """
    Get all recorded memory snapshots.

    Returns:
        Dict mapping checkpoint names to MemorySnapshots
    """
    return _MemorySnapshotRegistry.get_instance().get_all()


def get_memory_analysis() -> dict[str, float | dict]:
    """
    Get memory analysis with pool overhead and suggestions.

    Returns:
        Dict with memory analysis data including:
        - peak_reserved_mb: Peak reserved memory
        - peak_allocated_mb: Peak allocated memory
        - pool_overhead_mb: Memory pool overhead (reserved - allocated)
        - pool_overhead_pct: Memory pool overhead percentage
        - remaining_memory_mb: Remaining GPU memory
        - snapshots: Dict of recorded snapshots
    """
    return _MemorySnapshotRegistry.get_instance().get_analysis()


def set_memory_baseline() -> MemorySnapshot:
    """
    Set the baseline memory snapshot (before inference starts).

    Returns:
        The captured baseline MemorySnapshot
    """
    return _MemorySnapshotRegistry.get_instance().set_baseline()


def print_memory_analysis() -> None:
    """Print memory analysis summary to logger."""
    summary = _MemorySnapshotRegistry.get_instance().summary()
    logger.info(summary)
