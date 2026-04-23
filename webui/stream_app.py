"""Gradio web UI for TeleFuser stream video generation.

Connects to the stream server via WebRTC for real-time playback.
Requires ``pip install telefuser[webrtc]`` on the server side.

Usage:
    # 1. Start the stream server
    telefuser stream-serve examples/stream_video_replay.py -p 8088

    # 2. Launch the Gradio UI
    python webui/stream_app.py --server-url http://localhost:8088
"""

from __future__ import annotations

import argparse

import gradio as gr
import requests

DEFAULT_SERVER_URL = "http://localhost:8088"
DURATION_S = 30


# ---------------------------------------------------------------------------
# WebRTC transport
# ---------------------------------------------------------------------------

_WEBRTC_BTN_JS = """
(prompt, server_url, duration) => {
  // Clean up previous connection
  if (window._telefuserPC) {
    window._telefuserPC.close();
    window._telefuserPC = null;
  }

  const container = document.querySelector('#webrtc-container');
  if (!container) return [prompt, server_url, duration];

  container.innerHTML = `
    <div style="position:relative;">
      <video id="webrtc-video" autoplay playsinline muted
             style="width:100%;max-height:480px;background:#000;border-radius:8px;"></video>
      <div id="webrtc-overlay" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
           color:#fff;font-size:16px;text-shadow:0 1px 4px rgba(0,0,0,0.8);pointer-events:none;">
        Connecting...
      </div>
    </div>
  `;

  const video = document.getElementById("webrtc-video");
  const overlay = document.getElementById("webrtc-overlay");

  const unmuteBtn = document.createElement("button");
  unmuteBtn.textContent = "Unmute";
  unmuteBtn.style.cssText = "margin-top:8px;padding:6px 16px;border:none;" +
    "border-radius:4px;background:#16a34a;color:#fff;cursor:pointer;font-size:14px;";
  unmuteBtn.onclick = () => {
    video.muted = !video.muted;
    unmuteBtn.textContent = video.muted ? "Unmute" : "Mute";
  };
  container.appendChild(unmuteBtn);

  const pc = new RTCPeerConnection();
  window._telefuserPC = pc;

  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });
  pc.ontrack = (evt) => {
    if (evt.track.kind === "video") {
      video.srcObject = evt.streams[0];
      overlay.textContent = "";
    }
  };
  pc.onconnectionstatechange = () => {
    if (pc.connectionState === "failed") overlay.textContent = "Connection failed";
  };

  (async () => {
    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const r = await fetch(server_url.replace(/\\/$/, "") + "/v1/stream/webrtc/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
          task: "t2v",
          prompt: prompt,
          duration_s: duration,
          fps: 24,
        }),
      });
      if (!r.ok) {
        const e = await r.json();
        overlay.textContent = "Error: " + (e.detail || r.statusText);
        return;
      }
      const ans = await r.json();
      window._telefuserSessionId = ans.session_id;
      await pc.setRemoteDescription(new RTCSessionDescription({ sdp: ans.sdp, type: ans.type }));
    } catch(e) {
      overlay.textContent = "Error: " + e.message;
    }
  })();

  return [prompt, server_url, duration];
}
"""


def _generate_webrtc(prompt: str, server_url: str, duration: float):
    """WebRTC mode: JS handles the connection, Python just updates status."""
    if not prompt.strip():
        yield gr.skip(), "Please enter a prompt."
        return

    server_url_clean = server_url.rstrip("/")

    try:
        health = requests.get(f"{server_url_clean}/v1/service/health", timeout=3)
        if health.status_code != 200:
            yield gr.skip(), f"Server not healthy: {health.status_code}"
            return
    except requests.ConnectionError:
        yield gr.skip(), f"Cannot reach server at {server_url_clean}"
        return

    yield gr.skip(), (f"**WebRTC** stream started — prompt: *{prompt}*, duration: {duration:.0f}s")


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def build_app(server_url: str = DEFAULT_SERVER_URL) -> gr.Blocks:
    with gr.Blocks(title="TeleFuser Stream Video", theme=gr.themes.Soft()) as app:
        gr.Markdown("# TeleFuser Stream Video Generator")
        gr.Markdown("Enter a prompt and click **Generate** to stream video from the server via WebRTC.")

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=2,
                )
                duration_input = gr.Slider(
                    minimum=5,
                    maximum=60,
                    value=DURATION_S,
                    step=5,
                    label="Duration (seconds)",
                )
                server_url_input = gr.Textbox(
                    label="Server URL",
                    value=server_url,
                )
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                status_text = gr.Markdown("Ready.")
                webrtc_html = gr.HTML(
                    value='<div id="webrtc-container"></div>',
                )

        generate_btn.click(
            fn=_generate_webrtc,
            inputs=[prompt_input, server_url_input, duration_input],
            outputs=[webrtc_html, status_text],
            js=_WEBRTC_BTN_JS,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="TeleFuser Stream Video Web UI")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Stream server base URL")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()

    app = build_app(server_url=args.server_url)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
