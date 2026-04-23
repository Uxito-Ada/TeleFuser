"""WebRTC media tracks — bridges async generators to aiortc.

FrameGeneratorTrack: decodes ``frames_b64`` → ``av.VideoFrame`` at target fps.
AudioGeneratorTrack: receives raw PCM16 bytes → ``av.AudioFrame`` at 20ms pacing.

The video track owns the generator and forwards audio data (``audio_b64``) to
the audio track via ``feed()`` — a demuxer pattern that avoids a separate
fan-out abstraction.
"""

from __future__ import annotations

import asyncio
import base64
import fractions
import time
from collections.abc import AsyncGenerator

import av
import cv2
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from telefuser.utils.logging import logger

_RTP_CLOCK_RATE = 90_000
AUDIO_SAMPLE_RATE = 48_000
_AUDIO_SAMPLES_PER_FRAME = 960  # 20ms at 48kHz — standard Opus frame


class AudioGeneratorTrack(MediaStreamTrack):
    """Audio track fed by raw PCM16 bytes pushed from the video track."""

    kind = "audio"

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = 1,
        samples_per_frame: int = _AUDIO_SAMPLES_PER_FRAME,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._channels = channels
        self._samples_per_frame = samples_per_frame
        self._bytes_per_frame = samples_per_frame * channels * 2  # int16 = 2 bytes
        self._frame_duration = samples_per_frame / sample_rate

        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._buffer = bytearray()
        self._buf_offset = 0
        self._frame_count = 0
        self._start_time: float | None = None
        self._finished = False

    def feed(self, data: bytes) -> None:
        """Push raw PCM16 bytes (called by FrameGeneratorTrack from its consumer task)."""
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            pass

    def signal_done(self) -> None:
        self._finished = True

    async def recv(self) -> av.AudioFrame:
        if self._start_time is None:
            self._start_time = time.time()

        target_time = self._start_time + self._frame_count * self._frame_duration
        wait = target_time - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        buf_avail = len(self._buffer) - self._buf_offset
        while buf_avail < self._bytes_per_frame:
            if self._finished and self._queue.empty():
                if buf_avail > 0:
                    self._buffer.extend(b"\x00" * (self._bytes_per_frame - buf_avail))
                    buf_avail = self._bytes_per_frame
                    break
                raise MediaStreamError("Audio track ended")
            try:
                data = await asyncio.wait_for(self._queue.get(), timeout=2.0)
                self._buffer.extend(data)
                buf_avail = len(self._buffer) - self._buf_offset
            except asyncio.TimeoutError:
                if self._finished:
                    raise MediaStreamError("Audio track ended")
                self._buffer.extend(b"\x00" * self._bytes_per_frame)
                buf_avail = len(self._buffer) - self._buf_offset
                break

        end = self._buf_offset + self._bytes_per_frame
        pcm_bytes = bytes(self._buffer[self._buf_offset : end])
        self._buf_offset = end
        if self._buf_offset > 64_000:
            del self._buffer[: self._buf_offset]
            self._buf_offset = 0

        frame = av.AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
        frame.planes[0].update(pcm_bytes)
        frame.sample_rate = self._sample_rate
        frame.pts = self._frame_count * self._samples_per_frame
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._frame_count += 1
        return frame


class FrameGeneratorTrack(MediaStreamTrack):
    """Video track fed by a ``ServerPushService.serve()`` async generator."""

    kind = "video"

    def __init__(
        self,
        generator: AsyncGenerator[dict, None],
        fps: int = 24,
        audio_track: AudioGeneratorTrack | None = None,
    ) -> None:
        super().__init__()
        self._generator = generator
        self._fps = fps
        self._frame_interval = 1.0 / fps
        self._pts_per_frame = _RTP_CLOCK_RATE // fps
        self._audio_track = audio_track
        self._queue: asyncio.Queue[av.VideoFrame] = asyncio.Queue(maxsize=200)
        self._frame_count = 0
        self._task: asyncio.Task | None = None
        self._finished = False
        self._start_time: float | None = None
        self._last_frame: av.VideoFrame | None = None

    async def _consume_generator(self) -> None:
        try:
            async for chunk in self._generator:
                data = chunk if "frames_b64" in chunk else chunk.get("data", {})
                frames_b64: list[str] = data.get("frames_b64", [])
                for fb64 in frames_b64:
                    raw = base64.b64decode(fb64)
                    np_arr = np.frombuffer(raw, dtype=np.uint8)
                    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    await self._queue.put(frame)

                if self._audio_track is not None:
                    audio_b64 = data.get("audio_b64")
                    if audio_b64:
                        self._audio_track.feed(base64.b64decode(audio_b64))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"FrameGeneratorTrack generator error: {exc}")
        finally:
            self._finished = True
            if self._audio_track is not None:
                self._audio_track.signal_done()

    async def recv(self) -> av.VideoFrame:
        if self._task is None:
            self._task = asyncio.create_task(self._consume_generator())
            self._start_time = time.time()

        target_time = self._start_time + self._frame_count * self._frame_interval
        wait = target_time - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        try:
            frame = self._queue.get_nowait()
            self._last_frame = frame
        except asyncio.QueueEmpty:
            if self._finished and self._last_frame is None:
                raise MediaStreamError("Track ended")
            if self._last_frame is not None:
                frame = self._last_frame
            else:
                try:
                    frame = await asyncio.wait_for(self._queue.get(), timeout=10.0)
                    self._last_frame = frame
                except asyncio.TimeoutError:
                    raise MediaStreamError("Track ended — no frames received")

        frame.pts = self._frame_count * self._pts_per_frame
        frame.time_base = fractions.Fraction(1, _RTP_CLOCK_RATE)
        self._frame_count += 1
        return frame

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        if self._audio_track is not None:
            self._audio_track.stop()
        super().stop()
