"""WebRTC client demo: serves a minimal HTML page that streams video via WebRTC.

Usage:
    # 1. Start the stream server:
    telefuser stream-serve examples/stream_video_replay.py -p 8088 --skip-validation

    # 2. Start this client (opens browser):
    python examples/webrtc_client_demo.py --server-url http://localhost:8088

    # 3. Enter a prompt and click Connect — video plays in real-time via WebRTC.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import threading
import webbrowser

DEFAULT_SERVER_URL = "http://localhost:8088"
DEFAULT_PORT = 8090

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TeleFuser WebRTC Demo</title>
<style>
  body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
  video {{ width: 100%; max-height: 480px; background: #000; border-radius: 8px; }}
  .controls {{ display: flex; gap: 10px; margin: 16px 0; align-items: center; }}
  input[type=text] {{ flex: 1; padding: 8px; font-size: 14px; border: 1px solid #ccc; border-radius: 4px; }}
  button {{ padding: 8px 20px; font-size: 14px; border: none; border-radius: 4px; cursor: pointer; }}
  #connect {{ background: #2563eb; color: #fff; }}
  #connect:disabled {{ background: #94a3b8; cursor: default; }}
  #stop {{ background: #dc2626; color: #fff; }}
  #unmute {{ background: #16a34a; color: #fff; }}
  #status {{ color: #666; font-size: 13px; margin: 8px 0; }}
</style>
</head>
<body>
<h2>TeleFuser WebRTC Demo</h2>
<video id="video" autoplay playsinline muted></video>
<div class="controls">
  <input id="prompt" type="text" placeholder="Enter a prompt..." value="a dog running">
  <button id="connect">Connect</button>
  <button id="stop" style="display:none">Stop</button>
  <button id="unmute" style="display:none">Unmute</button>
</div>
<div id="status">Ready.</div>

<script>
const SERVER_URL = "{server_url}";
const RTC_CONFIG = {rtc_config};
let pc = null;
let sessionId = null;

document.getElementById("connect").onclick = async () => {{
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) return;

  document.getElementById("status").textContent = "Connecting...";
  document.getElementById("connect").disabled = true;

  try {{
    pc = new RTCPeerConnection(RTC_CONFIG);
    pc.addTransceiver("video", {{ direction: "recvonly" }});
    pc.addTransceiver("audio", {{ direction: "recvonly" }});

    pc.ontrack = (evt) => {{
      if (evt.track.kind === "video") {{
        document.getElementById("video").srcObject = evt.streams[0];
        document.getElementById("status").textContent = "Streaming...";
        document.getElementById("stop").style.display = "inline-block";
        document.getElementById("unmute").style.display = "inline-block";
      }}
    }};

    pc.onconnectionstatechange = () => {{
      if (pc.connectionState === "failed" || pc.connectionState === "closed") {{
        document.getElementById("status").textContent = "Connection " + pc.connectionState;
        cleanup();
      }}
    }};

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const resp = await fetch(SERVER_URL + "/v1/stream/webrtc/offer", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
        task: "t2v",
        prompt: prompt,
        duration_s: 30,
      }}),
    }});

    if (!resp.ok) {{
      const err = await resp.json();
      throw new Error(err.detail || resp.statusText);
    }}

    const answer = await resp.json();
    sessionId = answer.session_id;
    await pc.setRemoteDescription(new RTCSessionDescription({{
      sdp: answer.sdp,
      type: answer.type,
    }}));
  }} catch (e) {{
    document.getElementById("status").textContent = "Error: " + e.message;
    cleanup();
  }}
}};

document.getElementById("stop").onclick = () => {{
  document.getElementById("status").textContent = "Stopped.";
  cleanup();
}};

document.getElementById("unmute").onclick = () => {{
  const video = document.getElementById("video");
  video.muted = !video.muted;
  document.getElementById("unmute").textContent = video.muted ? "Unmute" : "Mute";
}};

function cleanup() {{
  if (pc) {{
    pc.close();
    pc = null;
  }}
  if (sessionId) {{
    fetch(SERVER_URL + "/v1/stream/webrtc/" + sessionId, {{ method: "DELETE" }}).catch(() => {{}});
    sessionId = null;
  }}
  document.getElementById("video").srcObject = null;
  document.getElementById("connect").disabled = false;
  document.getElementById("stop").style.display = "none";
  document.getElementById("unmute").style.display = "none";
  document.getElementById("unmute").textContent = "Unmute";
}}
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="TeleFuser WebRTC client demo")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Stream server base URL")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local HTTP server port")
    parser.add_argument(
        "--turn-url",
        default=os.environ.get("TELEFUSER_TURN_SERVER", ""),
        help="TURN server URL for browser WebRTC ICE, e.g. turn:localhost:3478?transport=tcp",
    )
    parser.add_argument(
        "--turn-username",
        default=os.environ.get("TELEFUSER_TURN_USERNAME", ""),
        help="TURN username; defaults to TELEFUSER_TURN_USERNAME",
    )
    parser.add_argument(
        "--turn-credential",
        default=os.environ.get("TELEFUSER_TURN_CREDENTIAL", ""),
        help="TURN credential; defaults to TELEFUSER_TURN_CREDENTIAL",
    )
    parser.add_argument(
        "--force-turn-relay",
        action="store_true",
        help="Force WebRTC to use relay candidates only. Useful when testing through SSH port forwarding.",
    )
    parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    rtc_config: dict[str, object] = {}
    if args.turn_url:
        turn_server: dict[str, str] = {"urls": args.turn_url}
        if args.turn_username:
            turn_server["username"] = args.turn_username
        if args.turn_credential:
            turn_server["credential"] = args.turn_credential
        rtc_config["iceServers"] = [turn_server]
    if args.force_turn_relay:
        rtc_config["iceTransportPolicy"] = "relay"

    html = HTML_TEMPLATE.format(server_url=args.server_url, rtc_config=json.dumps(rtc_config))

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        def log_message(self, format: str, *_args: object) -> None:
            pass

    server = http.server.HTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Serving WebRTC demo at {url}")
    print(f"Stream server: {args.server_url}")
    if rtc_config:
        print(f"WebRTC config: {json.dumps(rtc_config)}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_open:
        threading.Timer(0.5, functools.partial(webbrowser.open, url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
