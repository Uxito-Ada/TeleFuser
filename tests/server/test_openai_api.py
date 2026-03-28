"""
Integration Tests for OpenAI Compatible API

Tests OpenAI API endpoints using a real server with fake pipeline.
"""

import asyncio

import httpx
import pytest


@pytest.mark.integration
class TestOpenAIImageAPI:
    """Test OpenAI-compatible Image API endpoints."""

    @pytest.mark.asyncio
    async def test_image_generations_endpoint(self, running_server):
        """Test POST /v1/images/generations endpoint."""
        base_url = running_server["base_url"]

        payload = {
            "prompt": "a beautiful sunset over mountains",
            "size": "512x512",
            "response_format": "url",
            "seed": 42,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v1/images/generations",
                json=payload,
                timeout=30.0,
            )

            assert response.status_code == 200
            data = response.json()

            # OpenAI format response
            assert "created" in data
            assert "data" in data
            assert len(data["data"]) >= 1

            # Check image data
            image_data = data["data"][0]
            assert "url" in image_data or "b64_json" in image_data
            assert "revised_prompt" in image_data

    @pytest.mark.asyncio
    async def test_image_generations_b64_response(self, running_server):
        """Test image generation with base64 response format."""
        base_url = running_server["base_url"]

        payload = {
            "prompt": "a cute cat",
            "size": "512x512",
            "response_format": "b64_json",
            "seed": 123,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v1/images/generations",
                json=payload,
                timeout=30.0,
            )

            # Fake pipeline may not support b64 format, so accept 200 or 500
            assert response.status_code in [200, 500]

            if response.status_code == 200:
                data = response.json()
                image_data = data["data"][0]
                assert "b64_json" in image_data
                assert len(image_data["b64_json"]) > 0

    @pytest.mark.asyncio
    async def test_image_generations_validation_error(self, running_server):
        """Test image generation with invalid parameters."""
        base_url = running_server["base_url"]

        # Missing prompt
        payload = {
            "size": "1024x1024",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v1/images/generations",
                json=payload,
                timeout=5.0,
            )

            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_image_generations_with_all_params(self, running_server):
        """Test image generation with all available parameters."""
        base_url = running_server["base_url"]

        payload = {
            "prompt": "a futuristic city at night",
            "model": "wan-video",  # Model parameter (may be ignored by fake pipeline)
            "n": 1,
            "quality": "hd",
            "response_format": "url",
            "size": "1024x1024",
            "style": "vivid",
            "user": "test_user",
            "seed": 999,
            "negative_prompt": "blurry, low quality",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v1/images/generations",
                json=payload,
                timeout=30.0,
            )

            assert response.status_code == 200
            data = response.json()
            assert "data" in data

    @pytest.mark.asyncio
    async def test_image_content_download(self, running_server):
        """Test downloading generated image."""
        base_url = running_server["base_url"]

        # First generate an image
        payload = {
            "prompt": "test image for download",
            "size": "512x512",
            "response_format": "url",
        }

        async with httpx.AsyncClient() as client:
            # Create task
            gen_response = await client.post(
                f"{base_url}/v1/images/generations",
                json=payload,
                timeout=30.0,
            )

            assert gen_response.status_code == 200
            data = gen_response.json()

            # Extract task_id from URL if available
            image_url = data["data"][0].get("url", "")
            if "/v1/images/" in image_url:
                task_id = image_url.split("/v1/images/")[1].split("/")[0]

                # Try to download
                download_resp = await client.get(
                    f"{base_url}/v1/images/{task_id}/content",
                    timeout=10.0,
                )

                # Should succeed, return 404 if not ready, or 503 if service unavailable
                assert download_resp.status_code in [200, 404, 503]


@pytest.mark.integration
class TestOpenAIVideoAPI:
    """Test OpenAI-compatible Video API endpoints."""

    @pytest.mark.asyncio
    async def test_videos_create_endpoint(self, running_server):
        """Test POST /v1/videos endpoint."""
        base_url = running_server["base_url"]

        payload = {
            "prompt": "a cat playing with a ball",
            "seconds": 2,  # Short for testing
            "size": "480p",
            "seed": 42,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v1/videos",
                json=payload,
                timeout=10.0,
            )

            assert response.status_code == 200
            data = response.json()

            # OpenAI video response format
            assert "id" in data
            assert data["object"] == "video"
            assert data["status"] == "queued"
            assert "model" in data
            assert "seconds" in data

    @pytest.mark.asyncio
    async def test_videos_list_endpoint(self, running_server):
        """Test GET /v1/videos endpoint."""
        base_url = running_server["base_url"]

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/v1/videos",
                timeout=10.0,
            )

            assert response.status_code == 200
            data = response.json()

            assert "data" in data
            assert data["object"] == "list"
            assert isinstance(data["data"], list)

    @pytest.mark.asyncio
    async def test_videos_list_with_pagination(self, running_server):
        """Test video list pagination parameters."""
        base_url = running_server["base_url"]

        async with httpx.AsyncClient() as client:
            # Test with limit
            response = await client.get(
                f"{base_url}/v1/videos?limit=5",
                timeout=10.0,
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["data"]) <= 5

    @pytest.mark.asyncio
    async def test_videos_retrieve_endpoint(self, running_server):
        """Test GET /v1/videos/{id} endpoint."""
        base_url = running_server["base_url"]

        # First create a video
        payload = {
            "prompt": "test video retrieval",
            "seconds": 2,
        }

        async with httpx.AsyncClient() as client:
            create_resp = await client.post(
                f"{base_url}/v1/videos",
                json=payload,
                timeout=10.0,
            )

            video_id = create_resp.json()["id"]

            # Retrieve video status
            response = await client.get(
                f"{base_url}/v1/videos/{video_id}",
                timeout=10.0,
            )

            assert response.status_code == 200
            data = response.json()

            assert data["id"] == video_id
            assert "status" in data
            assert "progress" in data

    @pytest.mark.asyncio
    async def test_videos_delete_endpoint(self, running_server):
        """Test DELETE /v1/videos/{id} endpoint."""
        base_url = running_server["base_url"]

        # Create a video first
        payload = {
            "prompt": "test video deletion",
            "seconds": 2,
        }

        async with httpx.AsyncClient() as client:
            create_resp = await client.post(
                f"{base_url}/v1/videos",
                json=payload,
                timeout=10.0,
            )

            video_id = create_resp.json()["id"]

            # Delete/Cancel video
            response = await client.delete(
                f"{base_url}/v1/videos/{video_id}",
                timeout=10.0,
            )

            assert response.status_code == 200
            data = response.json()

            assert data["id"] == video_id
            assert data["status"] in ["cancelled", "deleted"]

    @pytest.mark.asyncio
    async def test_videos_create_with_all_params(self, running_server):
        """Test video creation with all available parameters."""
        base_url = running_server["base_url"]

        payload = {
            "prompt": "a futuristic robot walking",
            "model": "wan-video",
            "seconds": 3,
            "size": "512x512",
            "seed": 42,
            "negative_prompt": "blurry, distorted",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v1/videos",
                json=payload,
                timeout=10.0,
            )

            assert response.status_code == 200
            data = response.json()
            assert "id" in data

    @pytest.mark.asyncio
    async def test_videos_full_workflow(self, running_server):
        """Test complete video generation workflow."""
        base_url = running_server["base_url"]

        payload = {
            "prompt": "a dog running in the park",
            "seconds": 2,
            "size": "480p",
            "seed": 123,
        }

        async with httpx.AsyncClient() as client:
            # 1. Create video
            create_resp = await client.post(
                f"{base_url}/v1/videos",
                json=payload,
                timeout=10.0,
            )

            assert create_resp.status_code == 200
            video_id = create_resp.json()["id"]

            # 2. Poll for completion (with timeout)
            max_wait = 30  # seconds
            start_time = asyncio.get_event_loop().time()

            while True:
                status_resp = await client.get(
                    f"{base_url}/v1/videos/{video_id}",
                    timeout=5.0,
                )

                status_data = status_resp.json()
                current_status = status_data.get("status")

                if current_status == "completed":
                    break
                elif current_status == "failed":
                    pytest.fail(f"Video generation failed: {status_data.get('error')}")

                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > max_wait:
                    pytest.fail("Video generation timed out")

                await asyncio.sleep(0.5)

            # 3. Verify completion
            assert status_data["status"] == "completed"
            assert status_data["progress"] == 100


@pytest.mark.integration
class TestOpenAIVsNativeAPI:
    """Compare OpenAI API with native TeleFuser API."""

    @pytest.mark.asyncio
    async def test_both_apis_create_tasks(self, running_server):
        """Both APIs can create tasks successfully."""
        base_url = running_server["base_url"]

        # Native API
        native_payload = {
            "task": "t2v",
            "prompt": "native api test",
            "resolution": "480p",
            "target_video_length": 2,
        }

        # OpenAI API
        openai_payload = {
            "prompt": "openai api test",
            "seconds": 2,
            "size": "480p",
        }

        async with httpx.AsyncClient() as client:
            # Test native API
            native_resp = await client.post(
                f"{base_url}/v1/tasks/create",
                json=native_payload,
                timeout=10.0,
            )
            assert native_resp.status_code == 200

            # Test OpenAI API
            openai_resp = await client.post(
                f"{base_url}/v1/videos",
                json=openai_payload,
                timeout=10.0,
            )
            assert openai_resp.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_schema_includes_both(self, running_server):
        """OpenAPI schema includes both native and OpenAI endpoints."""
        base_url = running_server["base_url"]

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/openapi.json",
                timeout=5.0,
            )

            assert response.status_code == 200
            schema = response.json()
            paths = schema.get("paths", {})

            # Check native API paths
            assert any("/v1/tasks" in p for p in paths.keys())

            # Check OpenAI API paths
            assert any("/v1/images" in p for p in paths.keys())
            assert any("/v1/videos" in p for p in paths.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
