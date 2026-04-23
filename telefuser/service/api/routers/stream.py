"""Stream routes: WebSocket for bidirectional sessions.

Endpoints:
    POST   /v1/stream/sessions               – create session
    WS     /v1/stream/ws/{session_id}        – duplex
    DELETE /v1/stream/sessions/{session_id}  – stop session
    GET    /v1/stream/sessions/{session_id}/status
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from telefuser.utils.logging import logger

from ...core.stream_pipeline_service import STREAM_MODE_BIDIRECTIONAL
from ..stream_schema import (
    StreamChunkMessage,
    StreamDoneMessage,
    StreamSessionRequest,
    StreamSessionResponse,
)

if TYPE_CHECKING:
    from ..api_server import ApiServer


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


class StreamRoutes:
    def __init__(self, api_server: ApiServer) -> None:
        self.api = api_server
        self._ws_connections: int = 0

        from ...core.config import server_config

        self._max_ws_connections: int = server_config.stream_ws_max_connections
        self._chunk_timeout: float = server_config.stream_chunk_timeout

    def _require_service(self):
        svc = self.api.stream_service
        if svc is None or not svc.is_running:
            raise HTTPException(status_code=503, detail="Stream service is not running")
        return svc

    # -- WebSocket ----------------------------------------------------------

    async def create_session(self, request: StreamSessionRequest) -> StreamSessionResponse:
        svc = self._require_service()
        if svc.stream_mode != STREAM_MODE_BIDIRECTIONAL:
            raise HTTPException(
                status_code=400,
                detail=f"Pipeline is {svc.stream_mode}, not bidirectional. Use WebRTC.",
            )

        session_id = svc.create_session(request.model_dump())
        return StreamSessionResponse(
            session_id=session_id,
            stream_mode=STREAM_MODE_BIDIRECTIONAL,
        )

    async def websocket_handler(self, ws: WebSocket, session_id: str) -> None:
        svc = self.api.stream_service
        if svc is None or not svc.is_running:
            await ws.close(code=1013, reason="Stream service not running")
            return

        if svc.stream_mode != STREAM_MODE_BIDIRECTIONAL:
            await ws.close(
                code=1008,
                reason=f"WebSocket requires bidirectional mode, got {svc.stream_mode}",
            )
            return

        max_ws = self._max_ws_connections
        chunk_timeout = self._chunk_timeout
        if self._ws_connections >= max_ws:
            await ws.close(code=1013, reason=f"Max WebSocket connections ({max_ws}) reached")
            return

        await ws.accept()
        self._ws_connections += 1
        logger.info(f"WebSocket connected: session={session_id} (active={self._ws_connections})")

        pull_task: asyncio.Task | None = None
        try:

            async def _push_output():
                async for chunk in svc.pull_chunks(session_id):
                    msg = StreamChunkMessage(
                        session_id=session_id,
                        index=chunk.get("index"),
                        data=self._serialisable(chunk),
                    )
                    await ws.send_text(msg.model_dump_json())
                done = StreamDoneMessage(session_id=session_id)
                await ws.send_text(done.model_dump_json())

            pull_task = asyncio.create_task(_push_output())

            session_closed = False
            while True:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=chunk_timeout)
                data = json.loads(raw)
                if data.get("type") == "stop":
                    svc.close_session(session_id)
                    session_closed = True
                    break
                svc.push_chunk(session_id, data)

        except asyncio.TimeoutError:
            logger.warning(f"WebSocket timeout: session={session_id} (chunk_timeout={chunk_timeout}s)")
            try:
                err = StreamChunkMessage(type="error", session_id=session_id, error="chunk timeout")
                await ws.send_text(err.model_dump_json())
            except Exception:
                pass
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: session={session_id}")
        except Exception as exc:
            logger.exception(f"WebSocket error: session={session_id}")
            try:
                err = StreamChunkMessage(type="error", session_id=session_id, error=str(exc))
                await ws.send_text(err.model_dump_json())
            except Exception:
                pass
        finally:
            self._ws_connections = max(0, self._ws_connections - 1)
            if pull_task is not None and not pull_task.done():
                pull_task.cancel()
                try:
                    await pull_task
                except (asyncio.CancelledError, Exception):
                    pass
            if not session_closed:
                try:
                    svc.close_session(session_id)
                except Exception:
                    pass

    async def close_session(self, session_id: str) -> dict:
        svc = self._require_service()
        try:
            svc.close_session(session_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"session_id": session_id, "status": "closed"}

    async def session_status(self, session_id: str) -> dict:
        task = self.api.task_manager.get_task_status(session_id)
        if task:
            return task

        svc = self.api.stream_service
        if svc is not None and svc.is_running and svc.has_session(session_id):
            return {"session_id": session_id, "status": "active", "stream_mode": "bidirectional"}

        return {"session_id": session_id, "status": "unknown"}

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _serialisable(chunk: dict) -> dict:
        """Strip non-JSON-serialisable values (e.g. tensors) from a chunk."""
        out: dict = {}
        for k, v in chunk.items():
            if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                out[k] = v
            else:
                try:
                    json.dumps(v)
                    out[k] = v
                except (TypeError, ValueError):
                    out[k] = str(v)
        return out


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_router(api_server: ApiServer) -> APIRouter:
    router = APIRouter(prefix="/v1/stream", tags=["stream"])
    routes = StreamRoutes(api_server)

    @router.post("/sessions", response_model=StreamSessionResponse, summary="Create bidirectional session")
    async def create_session(request: StreamSessionRequest):
        return await routes.create_session(request)

    @router.websocket("/ws/{session_id}")
    async def websocket_endpoint(ws: WebSocket, session_id: str):
        await routes.websocket_handler(ws, session_id)

    @router.delete("/sessions/{session_id}", summary="Close session")
    async def close_session(session_id: str):
        return await routes.close_session(session_id)

    @router.get("/sessions/{session_id}/status", summary="Session status")
    async def session_status(session_id: str):
        return await routes.session_status(session_id)

    return router


def setup_routes(api_server: ApiServer) -> APIRouter:
    return create_router(api_server)
