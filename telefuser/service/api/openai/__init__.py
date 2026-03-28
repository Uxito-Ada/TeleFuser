"""
OpenAI Compatible API for TeleFuser

This module provides OpenAI-compatible REST API endpoints for image and video generation.
"""

from __future__ import annotations

from .adapter import (
    OpenAIRequestAdapter,
    OpenAIResponseAdapter,
    calculate_num_frames,
    calculate_video_duration,
    decode_base64_to_image,
    encode_image_to_base64,
    infer_aspect_ratio,
)
from .protocol import (
    ErrorDetail,
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
    VideoGenerationsRequest,
    VideoListResponse,
    VideoRemixRequest,
    VideoResponse,
    generate_request_id,
    validate_image_size,
)

__all__ = [
    # Image API
    "ImageGenerationsRequest",
    "ImageEditRequest",
    "ImageResponse",
    "ImageResponseData",
    # Video API
    "VideoGenerationsRequest",
    "VideoResponse",
    "VideoListResponse",
    "VideoRemixRequest",
    # Common
    "ErrorResponse",
    "ErrorDetail",
    # Utilities
    "generate_request_id",
    "validate_image_size",
    # Adapters
    "OpenAIRequestAdapter",
    "OpenAIResponseAdapter",
    "encode_image_to_base64",
    "decode_base64_to_image",
    "infer_aspect_ratio",
    "calculate_num_frames",
    "calculate_video_duration",
]
