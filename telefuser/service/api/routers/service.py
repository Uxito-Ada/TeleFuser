"""
Service Routes for TeleFuser API

Provides service health, metadata, and metrics endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Response

from telefuser.metrics import get_service_metrics

if TYPE_CHECKING:
    from ..api_server import ApiServer

router = APIRouter(prefix="/v1/service", tags=["service"])


class ServiceRoutes:
    """Service route handlers with dependency injection."""

    def __init__(self, api_server: ApiServer) -> None:
        self.api = api_server

    async def get_status(self) -> dict:
        """Get service status."""
        return self.api.task_manager.get_service_status()

    async def get_metadata(self) -> dict:
        """Get service metadata."""
        assert self.api.inference_service is not None, "Inference service is not initialized"
        metadata = self.api.inference_service.server_metadata()
        metadata["supported_tasks"] = ["t2v", "i2v", "fl2v", "vc", "t2i", "i2i"]
        metadata["supported_media_types"] = ["video", "image"]
        return metadata

    async def health_check(self) -> dict:
        """Health check endpoint for monitoring."""
        from datetime import datetime

        status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
        }

        if self.api.inference_service:
            status["pipeline_ready"] = self.api.inference_service.is_running

        return status


def create_router(api_server: ApiServer) -> APIRouter:
    """Create a new router with fresh routes for the given ApiServer instance."""
    new_router = APIRouter(prefix="/v1/service", tags=["service"])
    routes = ServiceRoutes(api_server)

    @new_router.get("/status", response_model=dict)
    async def get_service_status() -> dict:
        return await routes.get_status()

    @new_router.get("/metadata", response_model=dict)
    async def get_service_metadata() -> dict:
        return await routes.get_metadata()

    @new_router.get("/health")
    async def health_check() -> dict:
        return await routes.health_check()

    @new_router.get("/metrics", summary="Get Prometheus Metrics")
    async def get_metrics_endpoint() -> Response:
        """Get Prometheus-compatible metrics."""
        service_metrics = get_service_metrics()
        return Response(
            content=service_metrics.get_prometheus_format(),
            media_type="text/plain; charset=utf-8",
        )

    @new_router.get("/metrics/json", summary="Get Metrics (JSON)")
    async def get_metrics_json() -> dict:
        """Get metrics in JSON format."""
        service_metrics = get_service_metrics()
        registry = service_metrics.registry
        return {
            "uptime_seconds": service_metrics.service_uptime.value,
            "tasks": {
                "created": service_metrics.tasks_created.value,
                "completed": service_metrics.tasks_completed.value,
                "failed": service_metrics.tasks_failed.value,
                "cancelled": service_metrics.tasks_cancelled.value,
            },
            "queue": {
                "size": service_metrics.queue_size.value,
                "pending": service_metrics.queue_pending.value,
                "processing": service_metrics.queue_processing.value,
            },
            "metrics_count": len(registry.list_metrics()),
            "registered_stages": registry.list_stages(),
        }

    return new_router


def setup_routes(api_server: ApiServer) -> APIRouter:
    """Setup service routes."""
    return create_router(api_server)
