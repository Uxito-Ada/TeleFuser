"""WebRTC transport for TeleFuser stream server.

Optional — requires ``pip install telefuser[webrtc]`` (aiortc).
"""

from __future__ import annotations

try:
    from .session_manager import WebRTCSessionManager
    from .track import AudioGeneratorTrack, FrameGeneratorTrack
except ImportError:
    AudioGeneratorTrack = None  # type: ignore[assignment,misc]
    FrameGeneratorTrack = None  # type: ignore[assignment,misc]
    WebRTCSessionManager = None  # type: ignore[assignment,misc]

__all__ = ["AudioGeneratorTrack", "FrameGeneratorTrack", "WebRTCSessionManager"]
