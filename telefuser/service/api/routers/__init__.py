"""
API Routers for TeleFuser Service

This module contains FastAPI route handlers organized by resource type.
Each router can be tested independently.
"""

from __future__ import annotations

from .files import router as files_router
from .service import router as service_router
from .stream import setup_routes as setup_stream_routes
from .tasks import router as tasks_router

try:
    from .webrtc import setup_routes as setup_webrtc_routes
except ImportError:
    setup_webrtc_routes = None  # type: ignore[assignment]

__all__ = ["tasks_router", "files_router", "service_router", "setup_stream_routes", "setup_webrtc_routes"]
