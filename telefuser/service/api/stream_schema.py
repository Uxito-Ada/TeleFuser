"""Request / response schemas for stream endpoints."""

from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# BIDIRECTIONAL  (WebSocket: continuous input ↔ output)
# ---------------------------------------------------------------------------


class StreamSessionRequest(BaseModel):
    """Body for POST /v1/stream/sessions (create session)."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str = Field(description="Task type")
    config: dict = Field(default_factory=dict, description="Session-level configuration")

    model_config = {"extra": "allow"}


class StreamSessionResponse(BaseModel):
    session_id: str
    stream_mode: str
    status: str = "created"


# ---------------------------------------------------------------------------
# Wire messages  (used over WebSocket and internally by WebRTC)
# ---------------------------------------------------------------------------


class StreamChunkMessage(BaseModel):
    """Single chunk pushed to the client."""

    type: str = "chunk"
    session_id: str = ""
    index: int | None = None
    data: dict | None = None
    error: str | None = None
    timestamp: float = Field(default_factory=time.time)


class StreamDoneMessage(BaseModel):
    type: str = "done"
    session_id: str = ""
    total_chunks: int = 0
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# WebRTC signaling
# ---------------------------------------------------------------------------


class WebRTCOfferRequest(BaseModel):
    """Body for POST /v1/stream/webrtc/offer."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sdp: str = Field(description="SDP offer from browser")
    type: str = Field(default="offer", description="SDP type")
    task: str = Field(description="Task type, e.g. t2v, i2v")
    prompt: str | None = None
    fps: int | None = Field(default=24, description="Target video FPS")

    model_config = {"extra": "allow"}


class WebRTCOfferResponse(BaseModel):
    session_id: str
    sdp: str = Field(description="SDP answer from server")
    type: str = Field(default="answer", description="SDP type")
