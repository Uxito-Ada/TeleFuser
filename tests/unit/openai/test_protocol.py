"""
Unit tests for OpenAI protocol models.

Tests Pydantic models with validation logic. Trivial field access tests
are skipped as they're covered by type hints and Pydantic.
"""

import pytest
from pydantic import ValidationError

from telefuser.service.api.openai.protocol import (
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
    VideoGenerationsRequest,
    VideoListResponse,
    VideoResponse,
    generate_request_id,
    validate_image_size,
)


class TestImageGenerationsRequest:
    """Tests for ImageGenerationsRequest validation."""

    def test_minimal_valid_request(self):
        """Minimal request with only required fields."""
        req = ImageGenerationsRequest(prompt="a cat")
        assert req.prompt == "a cat"
        assert req.n == 1  # default

    def test_prompt_required(self):
        """Prompt is required."""
        with pytest.raises(ValidationError):
            ImageGenerationsRequest()

    @pytest.mark.parametrize(
        "size,valid",
        [
            ("1024x1024", True),
            ("512x768", True),
            ("invalid", False),
            ("-100x100", False),
            ("0x100", False),
        ],
    )
    def test_size_validation(self, size, valid):
        """Size format validation."""
        if valid:
            req = ImageGenerationsRequest(prompt="test", size=size)
            assert req.size == size
        else:
            with pytest.raises(ValidationError):
                ImageGenerationsRequest(prompt="test", size=size)

    @pytest.mark.parametrize("n,valid", [(1, True), (10, True), (0, False), (11, False)])
    def test_n_validation(self, n, valid):
        """n must be between 1-10."""
        if valid:
            req = ImageGenerationsRequest(prompt="test", n=n)
            assert req.n == n
        else:
            with pytest.raises(ValidationError):
                ImageGenerationsRequest(prompt="test", n=n)


class TestVideoGenerationsRequest:
    """Tests for VideoGenerationsRequest validation."""

    def test_minimal_valid_request(self):
        """Minimal valid request."""
        req = VideoGenerationsRequest(prompt="a cat")
        assert req.prompt == "a cat"
        assert req.seconds == 4  # default

    def test_seconds_validation(self):
        """Seconds must be 1-60."""
        with pytest.raises(ValidationError):
            VideoGenerationsRequest(prompt="test", seconds=0)
        with pytest.raises(ValidationError):
            VideoGenerationsRequest(prompt="test", seconds=61)

    def test_image_to_video_detection(self):
        """Task type detected from input_reference."""
        req = VideoGenerationsRequest(prompt="animate", input_reference="/path/to/image.png")
        assert req.input_reference == "/path/to/image.png"


class TestImageResponse:
    """Tests for ImageResponse."""

    def test_response_creation(self):
        """Basic response creation."""
        response = ImageResponse(data=[ImageResponseData(url="http://example.com/image.png")])
        assert len(response.data) == 1
        assert response.created > 0  # auto-generated


class TestVideoResponse:
    """Tests for VideoResponse."""

    def test_response_defaults(self):
        """Default values for video response."""
        response = VideoResponse(id="vid_123")
        assert response.id == "vid_123"
        assert response.status == "queued"
        assert response.object == "video"


class TestErrorResponse:
    """Tests for ErrorResponse."""

    def test_from_exception(self):
        """Factory method creates proper error."""
        error = ErrorResponse.from_exception(
            message="Invalid prompt", type="invalid_request_error", code="invalid_prompt"
        )
        assert error.error["message"] == "Invalid prompt"
        assert error.error["type"] == "invalid_request_error"
        assert error.error["code"] == "invalid_prompt"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_request_id_format(self):
        """Request ID has expected format."""
        req_id = generate_request_id()
        assert req_id.startswith("req_")
        assert len(req_id) == 28  # "req_" + 24 hex chars

    @pytest.mark.parametrize(
        "size,expected",
        [
            ("1024x1024", (1024, 1024)),
            ("512x768", (512, 768)),
        ],
    )
    def test_validate_image_size_valid(self, size, expected):
        """Valid size formats."""
        result = validate_image_size(size)
        assert result == expected

    @pytest.mark.parametrize("size", ["invalid", "-100x100", "0x100", "100"])
    def test_validate_image_size_invalid(self, size):
        """Invalid size formats raise ValueError."""
        with pytest.raises(ValueError):
            validate_image_size(size)
