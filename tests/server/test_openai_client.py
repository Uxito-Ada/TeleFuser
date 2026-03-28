"""
End-to-End Tests for OpenAI Compatible Client

Tests the OpenAI client SDK against a real server with fake pipeline.
"""

import pytest

from telefuser.client.openai import OpenAICompatibleClient


@pytest.mark.integration
class TestOpenAIClientImages:
    """Test OpenAI client for image generation."""

    def test_client_initialization(self):
        """Test client initialization with custom parameters."""
        client = OpenAICompatibleClient(
            base_url="http://localhost:8080",
            api_key="test_key",
            timeout=60.0,
        )

        assert client.base_url == "http://localhost:8080"
        assert client.api_key == "test_key"
        assert client.timeout == 60.0

    def test_client_context_manager(self):
        """Test client as context manager."""
        with OpenAICompatibleClient("http://localhost:8000") as client:
            assert client.base_url == "http://localhost:8000"
            assert client._session is not None

    def test_images_api_access(self):
        """Test accessing images API."""
        client = OpenAICompatibleClient()

        images = client.images
        assert images is not None

        # Should return same instance on second access
        assert client.images is images

    def test_generate_image_with_client(self, running_server, tmp_path):
        """Test generating image using client SDK."""
        base_url = running_server["base_url"]

        client = OpenAICompatibleClient(base_url=base_url, timeout=60.0)

        try:
            # Generate image
            response = client.images.generate(
                prompt="a beautiful sunset over mountains",
                size="512x512",
                response_format="url",
                seed=42,
            )

            # Verify response
            assert response is not None
            assert len(response.data) >= 1

            # Check image data
            image = response.data[0]
            assert image.url is not None or image.b64_json is not None

        finally:
            client.close()

    def test_generate_image_b64_format(self, running_server, tmp_path):
        """Test generating image with base64 format."""
        base_url = running_server["base_url"]

        client = OpenAICompatibleClient(base_url=base_url, timeout=60.0)

        try:
            from requests.exceptions import HTTPError

            try:
                response = client.images.generate(
                    prompt="a cute cat sitting",
                    size="512x512",
                    response_format="b64_json",
                    seed=123,
                )

                assert response is not None
                assert len(response.data) == 1

                image = response.data[0]
                assert image.b64_json is not None
                assert len(image.b64_json) > 0

                # Save the image
                output_path = tmp_path / "generated_image.png"
                image.save(output_path)

                assert output_path.exists()
                assert output_path.stat().st_size > 0

            except HTTPError as e:
                # Fake pipeline may not support b64 format
                if e.response.status_code == 500:
                    pytest.skip("Fake pipeline does not support b64 format")
                raise

        finally:
            client.close()

    def test_generate_image_with_all_params(self, running_server):
        """Test generating image with all parameters."""
        base_url = running_server["base_url"]

        client = OpenAICompatibleClient(base_url=base_url, timeout=60.0)

        try:
            response = client.images.generate(
                prompt="a futuristic city",
                model="wan-video",
                n=1,
                quality="hd",
                response_format="url",
                size="1024x1024",
                style="vivid",
                seed=999,
                negative_prompt="blurry",
            )

            assert response is not None
            assert len(response.data) >= 1

        finally:
            client.close()


@pytest.mark.integration
class TestOpenAIClientVideos:
    """Test OpenAI client for video generation."""

    def test_videos_api_access(self):
        """Test accessing videos API."""
        client = OpenAICompatibleClient()

        videos = client.videos
        assert videos is not None
        assert client.videos is videos

    def test_create_video_with_client(self, running_server):
        """Test creating video using client SDK."""
        base_url = running_server["base_url"]

        client = OpenAICompatibleClient(base_url=base_url, timeout=30.0)

        try:
            # Create video
            video = client.videos.create(
                prompt="a cat playing with a ball",
                seconds=2,
                size="480p",
                seed=42,
            )

            # Verify response
            assert video is not None
            assert video.id is not None
            assert video.status == "queued"
            assert video.object == "video"

        finally:
            client.close()

    def test_create_and_wait_for_video(self, running_server):
        """Test creating video and waiting for completion."""
        base_url = running_server["base_url"]

        client = OpenAICompatibleClient(base_url=base_url, timeout=60.0)

        try:
            # Create video with wait
            video = client.videos.create(
                prompt="a dog running",
                seconds=2,
                size="480p",
                seed=42,
                wait=True,
                wait_timeout=30.0,
            )

            # Should be completed after wait
            assert video.is_done()

            if video.is_success():
                assert video.status == "completed"
                assert video.progress == 100

        finally:
            client.close()

    def test_create_video_with_all_params(self, running_server):
        """Test creating video with all parameters."""
        base_url = running_server["base_url"]

        client = OpenAICompatibleClient(base_url=base_url, timeout=30.0)

        try:
            video = client.videos.create(
                prompt="a robot walking",
                model="wan-video",
                seconds=3,
                size="512x512",
                seed=42,
                negative_prompt="blurry",
            )

            assert video is not None
            assert video.id is not None

        finally:
            client.close()

    def test_retrieve_video(self, running_server):
        """Test retrieving video status."""
        base_url = running_server["base_url"]

        client = OpenAICompatibleClient(base_url=base_url, timeout=30.0)

        try:
            # Create video first
            video = client.videos.create(
                prompt="test retrieval",
                seconds=2,
            )

            video_id = video.id

            # Retrieve video
            retrieved = client.videos.retrieve(video_id)

            assert retrieved.id == video_id
            assert retrieved.status is not None

        finally:
            client.close()

    def test_list_videos(self, running_server):
        """Test listing videos."""
        base_url = running_server["base_url"]

        client = OpenAICompatibleClient(base_url=base_url, timeout=30.0)

        try:
            # Create a video first
            client.videos.create(prompt="test list", seconds=2)

            # List videos
            videos = client.videos.list(limit=10)

            assert videos is not None
            assert videos.object == "list"
            assert isinstance(videos.data, list)

        finally:
            client.close()

    def test_cancel_video(self, running_server):
        """Test canceling a video."""
        base_url = running_server["base_url"]

        client = OpenAICompatibleClient(base_url=base_url, timeout=30.0)

        try:
            # Create video
            video = client.videos.create(
                prompt="test cancel",
                seconds=5,  # Longer to allow cancellation
            )

            # Cancel video
            result = video.cancel()

            # Should succeed or be already done
            assert result is True or video.is_done()

        finally:
            client.close()


@pytest.mark.integration
class TestOpenAIClientWorkflows:
    """Test complete client workflows."""

    def test_image_generation_workflow(self, running_server, tmp_path):
        """Complete image generation workflow."""
        base_url = running_server["base_url"]

        with OpenAICompatibleClient(base_url=base_url, timeout=60.0) as client:
            from requests.exceptions import HTTPError

            try:
                # Generate image
                response = client.images.generate(
                    prompt="a beautiful landscape",
                    size="512x512",
                    response_format="b64_json",
                )

                # Save all generated images
                for i, image in enumerate(response.data):
                    output_path = tmp_path / f"image_{i}.png"
                    image.save(output_path)
                    assert output_path.exists()

            except HTTPError as e:
                # Fake pipeline may not support b64 format
                if e.response.status_code == 500:
                    pytest.skip("Fake pipeline does not support b64 format")
                raise

    def test_video_generation_workflow(self, running_server, tmp_path):
        """Complete video generation workflow."""
        base_url = running_server["base_url"]

        with OpenAICompatibleClient(base_url=base_url, timeout=60.0) as client:
            from requests.exceptions import HTTPError

            # Create video
            video = client.videos.create(
                prompt="a cat playing piano",
                seconds=2,
                size="480p",
            )

            # Wait for completion
            video.wait(timeout=30.0)

            if video.is_success():
                try:
                    # Download video
                    output_path = tmp_path / "video.mp4"
                    video.download(output_path)
                    assert output_path.exists()
                except HTTPError as e:
                    # File service may not be available in fake pipeline
                    if e.response.status_code == 503:
                        pytest.skip("File service not available")
                    raise

    def test_batch_video_creation(self, running_server):
        """Test creating multiple videos."""
        base_url = running_server["base_url"]

        prompts = [
            "a dog running",
            "a cat jumping",
            "a bird flying",
        ]

        with OpenAICompatibleClient(base_url=base_url, timeout=60.0) as client:
            videos = []

            # Create all videos
            for prompt in prompts:
                video = client.videos.create(
                    prompt=prompt,
                    seconds=2,
                )
                videos.append(video)

            # Verify all were created
            assert len(videos) == len(prompts)
            for video in videos:
                assert video.id is not None


@pytest.mark.integration
class TestOpenAIClientErrorHandling:
    """Test client error handling."""

    def test_invalid_server_url(self):
        """Test connection to invalid server."""
        client = OpenAICompatibleClient(
            base_url="http://localhost:59999",  # Invalid port
            timeout=1.0,
        )

        try:
            with pytest.raises(Exception):
                client.images.generate(prompt="test")
        finally:
            client.close()

    def test_invalid_parameters(self, running_server):
        """Test handling of invalid parameters."""
        base_url = running_server["base_url"]

        client = OpenAICompatibleClient(base_url=base_url, timeout=10.0)

        try:
            # Missing required parameter
            with pytest.raises(Exception):
                client.videos.create(
                    seconds=5,  # Missing prompt
                )
        finally:
            client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
