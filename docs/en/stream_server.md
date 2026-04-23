# TeleFuser Stream Server Guide

This guide covers TeleFuser's real-time stream server, which delivers continuous video (and optional audio) over **WebRTC** or **WebSocket** connections — as opposed to the batch request-response mode of `telefuser serve`.

---

## Quick Start

```bash
# 1. Install with WebRTC support
pip install -e ".[webrtc]"

# 2. Start the stream server
telefuser stream-serve examples/stream_server/stream_video_replay.py -p 8088 --skip-validation

# 3. Open the WebRTC client demo
python examples/stream_server/webrtc_client_demo.py --server-url http://localhost:8088
```

Open `http://localhost:8090` in a browser, type a prompt, and click **Connect** to see real-time video.

---

## Stream Modes

TeleFuser stream server supports two interaction modes:

| Mode | Transport | Direction | Use Case |
|------|-----------|-----------|----------|
| **Server Push** | WebRTC | Server → Client | Real-time preview, text-to-video streaming |
| **Bidirectional** | WebSocket | Client ↔ Server | Interactive generation, speech-to-video |

### Server Push (WebRTC)

```
Client                          Server
  │                               │
  │  POST /v1/stream/webrtc/offer │
  │  (SDP offer + prompt)         │
  │──────────────────────────────►│
  │                               │  stream_task() → serve()
  │  SDP answer                   │  on background thread
  │◄──────────────────────────────│
  │                               │
  │  ◄──── RTP video frames ────  │
  │  ◄──── RTP audio frames ────  │  (optional)
  │                               │
  │  DELETE /v1/stream/webrtc/{id}│
  │──────────────────────────────►│  cleanup
```

The pipeline's `serve()` method yields chunks containing JPEG-encoded frames. The WebRTC layer decodes them into `av.VideoFrame` objects and streams them to the browser via RTP at the target frame rate.

### Bidirectional (WebSocket)

```
Client                          Server
  │                               │
  │  POST /v1/stream/sessions     │
  │  (task + config)              │
  │──────────────────────────────►│
  │  { session_id, status }       │
  │◄──────────────────────────────│
  │                               │
  │  WS /v1/stream/ws/{session_id}│
  │◄════════════════════════════►│
  │  send: input chunks           │
  │  recv: output chunks          │
  │                               │
  │  DELETE /v1/stream/sessions/{}│
  │──────────────────────────────►│  cleanup
```

The client creates a session, connects a WebSocket, and pushes input chunks (e.g., audio frames for speech-to-video). The server pushes output chunks back over the same WebSocket.

---

## CLI Usage

```bash
telefuser stream-serve <pipeline_file> [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port`, `-p` | `8088` | Server port |
| `--host` | `0.0.0.0` | Bind address |
| `--security-level` | `strict` | Pipeline validation level (`strict`, `standard`, `permissive`) |
| `--skip-validation` | `false` | Skip pipeline file security checks |

### Examples

```bash
# Start with default settings
telefuser stream-serve examples/stream_server/stream_video_replay.py

# Custom port and host
telefuser stream-serve my_pipeline.py -p 9000 --host 127.0.0.1

# Skip validation (development only)
telefuser stream-serve my_pipeline.py -p 8088 --skip-validation
```

---

## Creating a Stream Pipeline

A stream pipeline is a Python file that defines a `get_service()` function returning a service object. The service must implement one of two protocols.

### ServerPushService Protocol

For pipelines that accept a single request and stream output continuously.

```python
from __future__ import annotations
from collections.abc import AsyncGenerator

class MyService:
    def start(self) -> None:
        """Called once at server startup. Load models, open files, etc."""
        ...

    def stop(self) -> None:
        """Called at server shutdown. Release resources."""
        ...

    async def serve(self, request: dict) -> AsyncGenerator[dict, None]:
        """Yield output chunks for a single request.

        The request dict contains all fields from the WebRTC offer
        (prompt, task, fps, duration_s, etc.).
        """
        for i in range(10):
            yield {
                "type": "chunk",
                "index": i,
                "frames_b64": [encode_frame(frame)],
                "fps": 24,
            }

def get_service() -> MyService:
    return MyService()
```

### BidirectionalService Protocol

For pipelines that accept continuous input and produce continuous output.

```python
from __future__ import annotations
from collections.abc import AsyncGenerator

class MyBidirectionalService:
    def start(self) -> None: ...
    def stop(self) -> None: ...

    def create_session(self, config: dict) -> str:
        """Create a new session. Return session_id."""
        ...

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        """Push an input chunk into the session."""
        ...

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        """Yield output chunks for the session."""
        ...

    def close_session(self, session_id: str) -> None:
        """Close the session and free resources."""
        ...

def get_service() -> MyBidirectionalService:
    return MyBidirectionalService()
```

### Chunk Format

Server-push chunks should contain the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `str` | Yes | Always `"chunk"` |
| `index` | `int` | Yes | Chunk sequence number |
| `frames_b64` | `list[str]` | Yes | Base64-encoded JPEG frames |
| `fps` | `int` | Yes | Target frame rate |
| `num_frames` | `int` | No | Number of frames in this chunk |
| `resolution` | `str` | No | Frame resolution (e.g., `"1920x1080"`) |
| `prompt` | `str` | No | Echo of input prompt |
| `timestamp` | `float` | No | Chunk generation timestamp |
| `audio_b64` | `str` | No | Base64-encoded PCM16 audio data |
| `audio_sample_rate` | `int` | No | Audio sample rate (default: 48000) |
| `audio_channels` | `int` | No | Number of audio channels (default: 1) |

> Audio fields are optional. When present, the WebRTC layer creates an audio track alongside the video track.

---

## API Reference

### Endpoints Overview

| Endpoint | Method | Mode | Description |
|----------|--------|------|-------------|
| `/v1/stream/webrtc/offer` | POST | Server Push | WebRTC SDP offer/answer exchange |
| `/v1/stream/webrtc/{session_id}` | DELETE | Server Push | Close WebRTC session |
| `/v1/stream/sessions` | POST | Bidirectional | Create WebSocket session |
| `/v1/stream/ws/{session_id}` | WS | Bidirectional | WebSocket duplex connection |
| `/v1/stream/sessions/{session_id}` | DELETE | Bidirectional | Close WebSocket session |
| `/v1/stream/sessions/{session_id}/status` | GET | Bidirectional | Get session status |

### WebRTC: SDP Offer

**POST** `/v1/stream/webrtc/offer`

Request:

```json
{
  "sdp": "<SDP offer string>",
  "type": "offer",
  "task": "t2v",
  "prompt": "A sunset over the ocean",
  "fps": 24,
  "duration_s": 10
}
```

Response (`200 OK`):

```json
{
  "session_id": "abc123",
  "sdp": "<SDP answer string>",
  "type": "answer"
}
```

Error responses:

| Status | Condition |
|--------|-----------|
| `400` | Invalid SDP or negotiation failure |
| `503` | Stream service not running or max sessions reached |

### WebRTC: Close Session

**DELETE** `/v1/stream/webrtc/{session_id}`

Response (`200 OK`):

```json
{
  "session_id": "abc123",
  "status": "closed"
}
```

### WebSocket: Create Session

**POST** `/v1/stream/sessions`

Request:

```json
{
  "task": "s2v",
  "config": {"fps": 24}
}
```

Response (`200 OK`):

```json
{
  "session_id": "def456",
  "stream_mode": "bidirectional",
  "status": "created"
}
```

### WebSocket: Duplex Connection

**WS** `/v1/stream/ws/{session_id}`

After creating a session, connect a WebSocket. Send JSON messages as input chunks, receive JSON output chunks:

```json
// Received output chunk
{
  "type": "chunk",
  "session_id": "def456",
  "index": 0,
  "data": { ... },
  "timestamp": 1714000000.0
}

// Final message
{
  "type": "done",
  "session_id": "def456",
  "total_chunks": 10,
  "timestamp": 1714000010.0
}
```

### Stream-Specific Health Fields

**GET** `/v1/service/health` returns additional fields when stream service is active:

```json
{
  "status": "healthy",
  "stream_ready": true,
  "stream_mode": "server_push",
  "webrtc_active_sessions": 1,
  "webrtc_max_sessions": 10
}
```

### Stream-Specific Metadata Fields

**GET** `/v1/service/metadata`:

```json
{
  "service_type": "stream",
  "stream_mode": "server_push",
  "pipeline_file": "examples/stream_server/stream_video_replay.py",
  "security_level": "STRICT",
  "runner": "StreamPipelineService",
  "webrtc_active_sessions": 0,
  "webrtc_max_sessions": 10
}
```

---

## Client Integration

### WebRTC (JavaScript)

Minimal browser client for server-push mode:

```javascript
const pc = new RTCPeerConnection();
pc.addTransceiver("video", { direction: "recvonly" });
pc.addTransceiver("audio", { direction: "recvonly" });

const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const resp = await fetch("http://localhost:8088/v1/stream/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    task: "t2v",
    prompt: "A cat playing piano",
  }),
});

const answer = await resp.json();
await pc.setRemoteDescription(answer);

pc.ontrack = (event) => {
  document.getElementById("video").srcObject = event.streams[0];
};
```

### WebSocket (Python)

```python
import asyncio
import json
import websockets
import httpx

async def stream_bidirectional():
    # 1. Create session
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8088/v1/stream/sessions",
            json={"task": "s2v", "config": {"fps": 24}},
        )
        session_id = resp.json()["session_id"]

    # 2. Connect WebSocket
    async with websockets.connect(
        f"ws://localhost:8088/v1/stream/ws/{session_id}"
    ) as ws:
        # Send input
        await ws.send(json.dumps({"type": "audio", "data": "..."}))

        # Receive output
        async for msg in ws:
            chunk = json.loads(msg)
            if chunk["type"] == "done":
                break
            print(f"Chunk {chunk['index']}")

asyncio.run(stream_bidirectional())
```

### Gradio UI

TeleFuser includes a Gradio-based web UI for WebRTC streaming:

```bash
# Start the stream server
telefuser stream-serve examples/stream_server/stream_video_replay.py -p 8088

# Launch the Gradio UI (separate terminal)
python webui/stream_app.py --server-url http://localhost:8088 --port 7860
```

The Gradio UI provides a prompt input, duration slider, and embedded WebRTC video player.

---

## Examples

### Video Replay Service

`examples/stream_server/stream_video_replay.py` — A server-push pipeline that loads a local video file and streams it as frame chunks:

- Loads video frames and audio with PyAV at `start()` time
- Pre-encodes all frames as JPEG (cached, not re-encoded per request)
- Streams chunks at configurable pacing with optional audio
- Supports `duration_s`, `realtime`, and `prompt` request parameters

```bash
telefuser stream-serve examples/stream_server/stream_video_replay.py -p 8088 --skip-validation
```

### WebRTC Client Demo

`examples/stream_server/webrtc_client_demo.py` — A standalone HTML page served via Python's HTTP server:

```bash
python examples/stream_server/webrtc_client_demo.py --server-url http://localhost:8088 --port 8090
```

Opens a browser page with video player, prompt input, and connect/stop/unmute buttons.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEFUSER_WEBRTC_MAX_SESSIONS` | `10` | Maximum concurrent WebRTC sessions (1-100) |
| `TELEFUSER_STREAM_WS_MAX_CONNECTIONS` | `10` | Maximum concurrent WebSocket connections (1-1000) |

### CORS

CORS middleware is automatically added when stream service is initialized (required for browser-based WebRTC clients). All origins are allowed by default.

### Security Levels

Pipeline files are validated before loading. Use `--skip-validation` only for development:

| Level | Description |
|-------|-------------|
| `strict` | Full validation — no dangerous imports, no file system access |
| `standard` | Moderate — allows common libraries |
| `permissive` | Minimal checks |

---

## Troubleshooting

### WebRTC Connection Fails (ICE Error)

The browser and server must be able to reach each other directly (no symmetric NAT). For local development this is not an issue. For remote servers, ensure:

- The server port is accessible from the browser
- No firewall blocks UDP traffic on ephemeral ports
- Consider using a TURN server for NAT traversal in production

### No Audio in Browser

Click the **Unmute** button. Browsers require a user gesture before playing audio. The video element starts muted by default.

### Port Already in Use

```bash
# Find and kill the process using the port
lsof -ti:8088 | xargs kill -9
```

### "Stream service is not running" (503)

The pipeline's `start()` method failed. Check server logs for errors (e.g., missing video file, import errors).

### WebSocket Closes Immediately

- Verify the session was created first via `POST /v1/stream/sessions`
- Verify the stream mode is `bidirectional` (WebSocket requires it)
- Check that the session ID in the URL matches the created session

### High Memory Usage

Each WebRTC session holds video frames in a queue. Limit `webrtc_max_sessions` and ensure pipelines don't buffer excessive frames. The server automatically cleans up sessions when clients disconnect.
