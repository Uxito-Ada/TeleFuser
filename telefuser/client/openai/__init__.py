"""OpenAI Compatible Client for TeleFuser

This module provides an OpenAI-compatible Python client for interacting
with the TeleFuser API server.

Usage:
    from telefuser.client.openai import OpenAICompatibleClient

    client = OpenAICompatibleClient(base_url="http://localhost:8000")

    # Generate image
    response = client.images.generate(
        prompt="a beautiful sunset",
        size="1024x1024"
    )

    # Create video
    response = client.videos.create(
        prompt="a cat playing piano",
        seconds=5
    )
"""

from __future__ import annotations

from .client import OpenAICompatibleClient
from .images import ImagesAPI
from .videos import VideosAPI

__all__ = [
    "OpenAICompatibleClient",
    "ImagesAPI",
    "VideosAPI",
]
