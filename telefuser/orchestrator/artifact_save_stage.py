"""Artifact saving stage for pipeline outputs.

Handles saving video/image outputs to disk with proper metadata.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

from telefuser.utils.logging import logger
from telefuser.utils.video import save_video

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_component(s: str, max_len: int = 48) -> str:
    """Make a string safe for filenames and keep it short."""
    s = (s or "").strip()
    if not s:
        return "req"
    s = _SAFE_NAME_RE.sub("_", s)
    return s[:max_len] if len(s) > max_len else s


@dataclass(frozen=True)
class ArtifactSaveConfig:
    """Configuration for saving artifacts to local disk."""

    # Root directory for outputs; should be absolute at runtime.
    output_root: Path
    # Directory name under output_root; typically pipeline run/config name.
    run_name: str
    # Video encoding defaults.
    fps: int = 16
    quality: int = 6


class ArtifactSaveStage:
    """Pipeline stage for saving generated artifacts."""

    def __init__(self, name: str = "artifact_save"):
        self.name = name

    def process(
        self,
        frames: Sequence[Any],
        *,
        request_id: str,
        config: ArtifactSaveConfig,
        file_ext: str = ".mp4",
        make_date_subdir: bool = True,
    ) -> Dict[str, Any]:
        """Save frames as video file with metadata.

        Args:
            frames: Sequence of PIL Images or numpy arrays
            request_id: Unique request identifier
            config: Save configuration
            file_ext: Output file extension
            make_date_subdir: Whether to create date-based subdirectory

        Returns:
            Artifact metadata dictionary
        """
        if not frames:
            raise ValueError("frames is empty; cannot save video artifact")

        output_root = Path(config.output_root).expanduser().resolve()
        run_name = _safe_component(config.run_name, max_len=64)
        date_dir = datetime.now().strftime("%Y-%m-%d")

        out_dir = output_root / run_name
        if make_date_subdir:
            out_dir = out_dir / date_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_rid = _safe_component(request_id)
        ts = datetime.now().strftime("%H%M%S")
        rand8 = uuid.uuid4().hex[:8]
        file_ext = file_ext if file_ext.startswith(".") else f".{file_ext}"
        out_path = (out_dir / f"{safe_rid}_{ts}_{rand8}{file_ext}").resolve()

        # save_video appends ".temp.mp4" internally; keep save_path unique.
        logger.info(f"[{request_id}] Saving video to: {out_path}")
        save_video(frames, str(out_path), fps=config.fps, quality=config.quality)

        st = os.stat(out_path)

        # Best-effort width/height from the first frame if it looks like a PIL.Image.
        width = height = None
        try:
            first = frames[0]
            if hasattr(first, "size"):
                width, height = first.size  # PIL.Image: (w, h)
        except Exception:
            pass

        artifact: Dict[str, Any] = {
            "kind": "video",
            "format": "mp4",
            "mime": "video/mp4",
            "uri": str(out_path),
            "fps": int(config.fps),
            "frame_count": int(len(frames)),
            "size_bytes": int(st.st_size),
            "run_name": config.run_name,
            "request_id": request_id,
        }
        if width is not None and height is not None:
            artifact["width"] = int(width)
            artifact["height"] = int(height)

        return artifact
