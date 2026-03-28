"""
Tests for OpenAI compatible client.

Tests actual client behavior without over-mocking internal implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from telefuser.client.openai import OpenAICompatibleClient
from telefuser.client.openai.images import ImagesResponse
from telefuser.client.openai.videos import Video


class TestClient:
    """Tests for OpenAICompatibleClient."""

    def test_client_initialization(self):
        """Client stores configuration."""
        client = OpenAICompatibleClient(
            base_url="http://localhost:8080",
            api_key="test_key",
        )

        assert client.base_url == "http://localhost:8080"
        assert client.api_key == "test_key"

    def test_client_context_manager(self):
        """Client works as context manager."""
        with OpenAICompatibleClient() as client:
            assert client._session is not None


class TestImagesAPI:
    """Tests for Images API client."""

    @pytest.fixture
    def mock_response(self):
        """Create mock response with image data."""
        response = MagicMock()
        response.json.return_value = {
            "created": 1234567890,
            "data": [{"url": "http://example.com/image.png"}],
        }
        return response

    def test_generate_image(self, mock_response):
        """Generate image calls correct endpoint."""
        client = OpenAICompatibleClient()

        with patch.object(client._session, "request", return_value=mock_response):
            result = client.images.generate(prompt="a cat")

        assert isinstance(result, ImagesResponse)
        assert len(result.data) == 1


class TestVideosAPI:
    """Tests for Videos API client."""

    @pytest.fixture
    def mock_response(self):
        """Create mock response with video data."""
        response = MagicMock()
        response.json.return_value = {
            "id": "vid_123",
            "status": "queued",
            "model": "wan-video",
        }
        return response

    def test_create_video(self, mock_response):
        """Create video returns Video object."""
        client = OpenAICompatibleClient()

        with patch.object(client._session, "request", return_value=mock_response):
            result = client.videos.create(prompt="a cat", seconds=5)

        assert isinstance(result, Video)
        assert result.id == "vid_123"
        assert result.status == "queued"

    def test_video_wait_success(self):
        """Wait polls until completion."""
        client = OpenAICompatibleClient()

        # Mock responses: first processing, then completed
        responses = [
            MagicMock(json=lambda: {"id": "vid_123", "status": "generating"}),
            MagicMock(json=lambda: {"id": "vid_123", "status": "completed"}),
        ]

        with patch.object(client._session, "request", side_effect=responses):
            video = Video({"id": "vid_123", "status": "generating"}, client)
            video.wait(timeout=1.0, poll_interval=0.1)

        assert video.status == "completed"

    def test_video_wait_timeout(self):
        """Wait raises TimeoutError if not completed."""
        client = OpenAICompatibleClient()

        with patch.object(
            client._session, "request", return_value=MagicMock(json=lambda: {"id": "vid_123", "status": "generating"})
        ):
            video = Video({"id": "vid_123", "status": "generating"}, client)

            with pytest.raises(TimeoutError):
                video.wait(timeout=0.1, poll_interval=0.05)
