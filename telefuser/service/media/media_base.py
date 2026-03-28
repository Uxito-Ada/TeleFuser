"""
Base classes and utilities for media handling (images, videos, audio).

Provides common functionality for base64 encoding/decoding
and file handling across different media types.
"""

from __future__ import annotations

import base64
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from telefuser.utils.logging import logger


class MediaHandler(ABC):
    """Abstract base class for media file handlers."""

    # File signatures for format detection (magic numbers)
    SIGNATURES: dict[bytes, str] = {}

    # MIME type prefix for data URLs
    MIME_PREFIX: str = "data:"

    # Default extension
    DEFAULT_EXT: str = "bin"

    @abstractmethod
    def is_base64(self, data: str) -> bool:
        """Check if string is base64 encoded media."""
        pass

    @abstractmethod
    def extract_data(self, data: str) -> tuple[str, str | None]:
        """Extract base64 data and format from data URL or plain string."""
        pass

    @abstractmethod
    def detect_format(self, binary_data: bytes) -> str:
        """Detect media format from binary signature."""
        pass

    def save(self, base64_data: str, output_dir: str) -> str:
        """Save base64-encoded media to file.

        Args:
            base64_data: Base64 encoded media (with or without data URL prefix)
            output_dir: Directory to save file

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        data, format_type = self.extract_data(base64_data)
        file_id = str(uuid.uuid4())

        try:
            binary_data = base64.b64decode(data)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")

        if format_type:
            ext = format_type
        else:
            ext = self.detect_format(binary_data)

        file_path = output_path / f"{file_id}.{ext}"
        with open(file_path, "wb") as f:
            f.write(binary_data)

        return str(file_path)


class ImageHandler(MediaHandler):
    """Handler for image files."""

    SIGNATURES = {
        b"\x89PNG\r\n\x1a\n": "png",
        b"\xff\xd8\xff": "jpg",
        b"GIF87a": "gif",
        b"GIF89a": "gif",
    }
    DEFAULT_EXT = "png"

    def is_base64(self, data: str) -> bool:
        """Check if string is a base64-encoded image."""
        if data.startswith("data:image/"):
            return True

        try:
            if len(data) % 4 == 0:
                base64.b64decode(data, validate=True)
                decoded = base64.b64decode(data[:100])
                for sig in self.SIGNATURES:
                    if decoded.startswith(sig):
                        return True
                # WebP signature check
                if len(decoded) > 12 and decoded[8:12] == b"WEBP":
                    return True
        except Exception as e:
            logger.warning(f"Error checking base64 image: {e}")
            return False

        return False

    def extract_data(self, data: str) -> tuple[str, str | None]:
        """Extract base64 data and format from data URL."""
        if data.startswith("data:"):
            match = re.match(r"data:image/(\w+);base64,(.+)", data)
            if match:
                return match.group(2), match.group(1)
        return data, None

    def detect_format(self, binary_data: bytes) -> str:
        """Detect image format from binary signature."""
        for sig, ext in self.SIGNATURES.items():
            if binary_data.startswith(sig):
                return ext
        # WebP check
        if len(binary_data) > 12 and binary_data[8:12] == b"WEBP":
            return "webp"
        return self.DEFAULT_EXT


class VideoHandler(MediaHandler):
    """Handler for video files."""

    SIGNATURES = {
        b"ftyp": "mp4",
        b"moov": "mp4",
        b"mdat": "mp4",
    }
    DEFAULT_EXT = "mp4"

    def is_base64(self, data: str) -> bool:
        """Check if string is a base64-encoded video."""
        if data.startswith("data:video/"):
            return True

        try:
            if len(data) % 4 == 0:
                base64.b64decode(data, validate=True)
                decoded = base64.b64decode(data[:100])

                # Check MP4/MOV signatures at offset 4
                if len(decoded) > 8 and decoded[4:8] in self.SIGNATURES:
                    return True

                # AVI signature
                if decoded.startswith(b"RIFF") and len(decoded) > 12 and decoded[8:12] == b"AVI ":
                    return True

                # WebM/MKV signature
                if decoded.startswith(b"\x1a\x45\xdf\xa3"):
                    return True

                # FLV signature
                if decoded.startswith(b"FLV"):
                    return True
        except Exception as e:
            logger.warning(f"Error checking base64 video: {e}")
            return False

        return False

    def extract_data(self, data: str) -> tuple[str, str | None]:
        """Extract base64 data and format from data URL."""
        if data.startswith("data:"):
            match = re.match(r"data:video/(\w+);base64,(.+)", data)
            if match:
                return match.group(2), match.group(1)
        return data, None

    def detect_format(self, binary_data: bytes) -> str:
        """Detect video format from binary signature."""
        if len(binary_data) < 12:
            return self.DEFAULT_EXT

        # Check MP4/MOV signatures at offset 4
        sig_at_4 = binary_data[4:8]
        if sig_at_4 in self.SIGNATURES:
            return self.SIGNATURES[sig_at_4]

        # AVI
        if binary_data.startswith(b"RIFF") and binary_data[8:12] == b"AVI ":
            return "avi"

        # WebM/MKV - default to mp4 for compatibility
        if binary_data.startswith(b"\x1a\x45\xdf\xa3"):
            return "mp4"

        # FLV
        if binary_data.startswith(b"FLV"):
            return "flv"

        return self.DEFAULT_EXT


class AudioHandler(MediaHandler):
    """Handler for audio files."""

    SIGNATURES = {
        b"ID3": "mp3",
        b"\xff\xfb": "mp3",
        b"\xff\xf3": "mp3",
        b"\xff\xf2": "mp3",
        b"RIFF": "wav",
        b"fLaC": "flac",
        b"OggS": "ogg",
    }
    DEFAULT_EXT = "mp3"

    def is_base64(self, data: str) -> bool:
        """Check if string is a base64-encoded audio."""
        if data.startswith("data:audio/"):
            return True

        try:
            if len(data) % 4 == 0:
                base64.b64decode(data, validate=True)
                decoded = base64.b64decode(data[:100])

                for sig in self.SIGNATURES:
                    if decoded.startswith(sig):
                        return True

                # MPEG audio layer III
                if len(decoded) > 2 and decoded[:2] in [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"]:
                    return True
        except Exception as e:
            logger.warning(f"Error checking base64 audio: {e}")
            return False

        return False

    def extract_data(self, data: str) -> tuple[str, str | None]:
        """Extract base64 data and format from data URL."""
        if data.startswith("data:"):
            match = re.match(r"data:audio/(\w+);base64,(.+)", data)
            if match:
                return match.group(2), match.group(1)
        return data, None

    def detect_format(self, binary_data: bytes) -> str:
        """Detect audio format from binary signature."""
        for sig, ext in self.SIGNATURES.items():
            if binary_data.startswith(sig):
                # For WAV, verify it's actually a WAV file
                if sig == b"RIFF" and len(binary_data) > 12:
                    if binary_data[8:12] == b"WAVE":
                        return "wav"
                    else:
                        continue
                return ext

        # MPEG audio check
        if len(binary_data) > 2 and binary_data[:2] in [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"]:
            return "mp3"

        return self.DEFAULT_EXT


# Convenience functions and backward compatibility


def is_base64_image(data: str) -> bool:
    """Check if string is base64 encoded image."""
    return ImageHandler().is_base64(data)


def is_base64_video(data: str) -> bool:
    """Check if string is base64 encoded video."""
    return VideoHandler().is_base64(data)


def is_base64_audio(data: str) -> bool:
    """Check if string is base64 encoded audio."""
    return AudioHandler().is_base64(data)


def save_base64_image(base64_data: str, output_dir: str) -> str:
    """Save base64 encoded image to file."""
    return ImageHandler().save(base64_data, output_dir)


def save_base64_video(base64_data: str, output_dir: str) -> str:
    """Save base64 encoded video to file."""
    return VideoHandler().save(base64_data, output_dir)


def save_base64_audio(base64_data: str, output_dir: str) -> str:
    """Save base64 encoded audio to file."""
    return AudioHandler().save(base64_data, output_dir)
