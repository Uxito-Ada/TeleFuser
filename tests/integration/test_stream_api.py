"""Integration tests for stream API endpoints (WebSocket)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


# ---------------------------------------------------------------------------
# Bidirectional (session management) tests
# ---------------------------------------------------------------------------


class TestBidirectionalSessions:
    """Tests for bidirectional session management endpoints."""

    def test_create_session(self, bidirectional_client):
        body = {"task": "s2v", "config": {"fps": 24}}
        resp = bidirectional_client.post("/v1/stream/sessions", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["stream_mode"] == "bidirectional"
        assert data["status"] == "created"

    def test_close_session(self, bidirectional_client):
        create_resp = bidirectional_client.post("/v1/stream/sessions", json={"task": "s2v"})
        session_id = create_resp.json()["session_id"]

        resp = bidirectional_client.delete(f"/v1/stream/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "closed"

    def test_create_session_rejects_server_push_mode(self, server_push_client):
        resp = server_push_client.post("/v1/stream/sessions", json={"task": "t2v"})
        assert resp.status_code == 400
        assert "server_push" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Service status / metadata tests
# ---------------------------------------------------------------------------


class TestStreamServiceStatus:
    """Tests for service health / metadata with stream endpoints."""

    def test_service_status_with_stream(self, server_push_client):
        resp = server_push_client.get("/v1/service/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "service_status" in data

    def test_health_with_stream(self, server_push_client):
        resp = server_push_client.get("/v1/service/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_health_reports_stream_readiness(self, server_push_client):
        resp = server_push_client.get("/v1/service/health")
        data = resp.json()
        assert data["stream_ready"] is True
        assert data["stream_mode"] == "server_push"

    def test_metadata_returns_stream_info_without_inference_service(self, server_push_client):
        resp = server_push_client.get("/v1/service/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service_type"] == "stream"
        assert data["stream_mode"] == "server_push"
        assert data["runner"] == "StreamPipelineService"


# ---------------------------------------------------------------------------
# Session status tests
# ---------------------------------------------------------------------------


class TestSessionStatusLookup:
    """Session status should query stream service, not only TaskManager."""

    def test_active_session_returns_active(self, bidirectional_client):
        create_resp = bidirectional_client.post("/v1/stream/sessions", json={"task": "s2v"})
        session_id = create_resp.json()["session_id"]

        resp = bidirectional_client.get(f"/v1/stream/sessions/{session_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "active"
        assert data["stream_mode"] == "bidirectional"

    def test_unknown_session_returns_unknown(self, bidirectional_client):
        resp = bidirectional_client.get("/v1/stream/sessions/nonexistent/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unknown"

    def test_closed_session_returns_unknown(self, bidirectional_client):
        create_resp = bidirectional_client.post("/v1/stream/sessions", json={"task": "s2v"})
        session_id = create_resp.json()["session_id"]
        bidirectional_client.delete(f"/v1/stream/sessions/{session_id}")

        resp = bidirectional_client.get(f"/v1/stream/sessions/{session_id}/status")
        assert resp.json()["status"] == "unknown"


# ---------------------------------------------------------------------------
# WebSocket mode guard tests
# ---------------------------------------------------------------------------


class TestWebSocketModeGuard:
    """WebSocket endpoint should reject before accept if wrong mode."""

    def test_ws_rejected_on_server_push_mode(self, server_push_client):
        with pytest.raises(Exception):
            with server_push_client.websocket_connect("/v1/stream/ws/demo-session"):
                pass

    def test_ws_rejected_when_stream_not_running(self):
        from telefuser.service.api.api_server import ApiServer
        from telefuser.service.core.task_manager import TaskManager

        server = ApiServer(max_queue_size=10, task_manager=TaskManager(), enable_openai_api=False)
        from fastapi.testclient import TestClient

        with TestClient(server.app) as client:
            with pytest.raises(Exception):
                with client.websocket_connect("/v1/stream/ws/some-session"):
                    pass
