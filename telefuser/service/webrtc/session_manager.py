"""WebRTC session manager — tracks RTCPeerConnection lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from aiortc import RTCPeerConnection, RTCSessionDescription

from telefuser.utils.logging import logger

from .track import AudioGeneratorTrack, FrameGeneratorTrack

_SENTINEL = object()


@dataclass
class _Session:
    pc: RTCPeerConnection
    track: FrameGeneratorTrack
    audio_track: AudioGeneratorTrack | None = None


class WebRTCSessionManager:
    """Creates and manages WebRTC peer connections for stream sessions."""

    def __init__(self, max_sessions: int = 10) -> None:
        self._sessions: dict[str, _Session | object] = {}
        self._max_sessions = max_sessions
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        session_id: str,
        offer_sdp: str,
        offer_type: str,
        generator: AsyncGenerator[dict, None],
        fps: int = 24,
    ) -> tuple[str, str]:
        """Process SDP offer and return (answer_sdp, answer_type)."""
        async with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError(f"Max WebRTC sessions ({self._max_sessions}) reached")
            if session_id in self._sessions:
                raise RuntimeError(f"Session {session_id} already exists")
            self._sessions[session_id] = _SENTINEL

        try:
            pc = RTCPeerConnection()

            has_audio_offer = "m=audio" in offer_sdp
            audio_track: AudioGeneratorTrack | None = None
            if has_audio_offer:
                audio_track = AudioGeneratorTrack()

            track = FrameGeneratorTrack(generator, fps=fps, audio_track=audio_track)

            @pc.on("connectionstatechange")
            async def _on_state_change() -> None:
                state = pc.connectionState
                if state in ("failed", "disconnected", "closed"):
                    logger.info(f"WebRTC connection {state}: session={session_id}")
                    await self.close_session(session_id)

            pc.addTrack(track)
            if audio_track is not None:
                pc.addTrack(audio_track)

            offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            async with self._lock:
                self._sessions[session_id] = _Session(pc=pc, track=track, audio_track=audio_track)

            audio_status = " + audio" if audio_track else ""
            logger.info(f"WebRTC session created: session={session_id}{audio_status}")
            return pc.localDescription.sdp, pc.localDescription.type
        except BaseException:
            async with self._lock:
                self._sessions.pop(session_id, None)
            raise

    async def close_session(self, session_id: str) -> bool:
        async with self._lock:
            entry = self._sessions.pop(session_id, None)
        if entry is None or entry is _SENTINEL:
            return False
        assert isinstance(entry, _Session)
        entry.track.stop()
        if entry.track._task is not None and not entry.track._task.done():
            try:
                await entry.track._task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await entry.pc.close()
        except Exception:
            pass
        logger.info(f"WebRTC session closed: session={session_id}")
        return True

    async def close_all(self) -> None:
        async with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            await self.close_session(sid)

    def has_session(self, session_id: str) -> bool:
        entry = self._sessions.get(session_id)
        return entry is not None and entry is not _SENTINEL

    @property
    def active_session_count(self) -> int:
        return sum(1 for v in self._sessions.values() if v is not _SENTINEL)
