"""Videos API Client

Provides OpenAI-compatible video generation methods.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union

import requests

from telefuser.utils.logging import logger

if TYPE_CHECKING:
    from .client import OpenAICompatibleClient


class Video:
    """Represents a video generation task or result."""

    def __init__(self, data: dict, client: "OpenAICompatibleClient"):
        self._data = data
        self._client = client

        self.id = data.get("id")
        self.object = data.get("object", "video")
        self.model = data.get("model", "wan-video")
        self.status = data.get("status", "queued")
        self.progress = data.get("progress", 0)
        self.created_at = data.get("created_at")
        self.size = data.get("size", "")
        self.seconds = data.get("seconds", "4")
        self.quality = data.get("quality", "standard")
        self.url = data.get("url")
        self.file_path = data.get("file_path")
        self.completed_at = data.get("completed_at")
        self.error = data.get("error")

    def is_done(self) -> bool:
        """Check if video generation is complete or failed."""
        return self.status in ["completed", "failed", "cancelled"]

    def is_success(self) -> bool:
        """Check if video generation completed successfully."""
        return self.status == "completed"

    def refresh(self) -> "Video":
        """Refresh the video status from the server.

        Returns:
            Updated Video object
        """
        response = self._client.get(f"/v1/videos/{self.id}")
        self.__init__(response.json(), self._client)
        return self

    def wait(
        self,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> "Video":
        """Wait for video generation to complete.

        Args:
            timeout: Maximum time to wait (seconds)
            poll_interval: Time between status checks (seconds)

        Returns:
            Updated Video object

        Raises:
            TimeoutError: If generation times out
            RuntimeError: If generation fails
        """
        start_time = time.time()

        while not self.is_done():
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Video generation timed out after {timeout}s")

            time.sleep(poll_interval)
            self.refresh()
            logger.info(f"Video {self.id} status: {self.status}, progress: {self.progress}%")

        if self.status == "failed":
            error_msg = self.error.get("message", "Unknown error") if self.error else "Unknown error"
            raise RuntimeError(f"Video generation failed: {error_msg}")

        return self

    def download(self, path: Optional[Union[str, Path]] = None) -> bytes:
        """Download the generated video.

        Args:
            path: Optional path to save the video

        Returns:
            Video data as bytes

        Raises:
            RuntimeError: If video is not ready
        """
        if not self.is_success():
            raise RuntimeError(f"Video is not ready (status: {self.status})")

        response = self._client.get(f"/v1/videos/{self.id}/content")
        data = response.content

        if path:
            Path(path).write_bytes(data)
            logger.info(f"Video saved to: {path}")

        return data

    def cancel(self) -> bool:
        """Cancel the video generation.

        Returns:
            True if cancelled successfully
        """
        response = self._client.delete(f"/v1/videos/{self.id}")
        self.__init__(response.json(), self._client)
        return self.status == "cancelled"


class VideosList:
    """List of videos response."""

    def __init__(self, data: dict, client: "OpenAICompatibleClient"):
        self._client = client
        self._data = data
        self.object = data.get("object", "list")
        self.data: List[Video] = [Video(v, client) for v in data.get("data", [])]
        self.has_more = data.get("has_more", False)
        self.first_id = data.get("first_id")
        self.last_id = data.get("last_id")

    def __getitem__(self, index: int) -> Video:
        """Get video by index."""
        return self.data[index]

    def __len__(self) -> int:
        """Get number of videos."""
        return len(self.data)


class VideosAPI:
    """Videos API for generating videos."""

    def __init__(self, client: "OpenAICompatibleClient"):
        self._client = client

    def create(
        self,
        prompt: str,
        input_reference: Optional[Union[str, Path, BinaryIO]] = None,
        reference_url: Optional[str] = None,
        model: Optional[str] = None,
        seconds: Optional[int] = 4,
        size: Optional[str] = None,
        seed: Optional[int] = 1024,
        negative_prompt: Optional[str] = None,
        output_path: Optional[str] = None,
        wait: bool = False,
        wait_timeout: float = 300.0,
    ) -> Video:
        """Create a video generation task.

        Args:
            prompt: Text description of the desired video
            input_reference: Input image/video file (for I2V)
            reference_url: URL of input reference
            model: Model to use
            seconds: Video duration in seconds
            size: Video size (e.g., "1024x576", "720p")
            seed: Random seed
            negative_prompt: What to avoid
            output_path: Custom output path
            wait: Whether to wait for completion
            wait_timeout: Timeout for waiting

        Returns:
            Video object representing the generation task

        Example:
            >>> video = client.videos.create(
            ...     prompt="a cat playing piano",
            ...     seconds=5
            ... )
            >>> video.wait()
            >>> video.download("output.mp4")
        """
        # Check if using file upload or JSON
        if input_reference is not None:
            # Use form-data for file upload
            data = {
                "prompt": prompt,
                "reference_url": reference_url,
                "model": model,
                "seconds": seconds,
                "size": size or "1024x576",
                "seed": seed,
                "negative_prompt": negative_prompt,
                "output_path": output_path,
            }

            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}

            # Prepare files
            files = {}
            if isinstance(input_reference, (str, Path)):
                files["input_reference"] = open(input_reference, "rb")
            else:
                files["input_reference"] = input_reference

            try:
                logger.info(f"Creating video with prompt: {prompt[:50]}...")

                response = self._client.post(
                    "/v1/videos",
                    data=data,
                    files=files,
                )
            finally:
                # Close file handle if we opened it
                if isinstance(input_reference, (str, Path)):
                    files["input_reference"].close()
        else:
            # Use JSON
            payload = {
                "prompt": prompt,
                "reference_url": reference_url,
                "model": model,
                "seconds": seconds,
                "size": size or "1024x576",
                "seed": seed,
                "negative_prompt": negative_prompt,
                "output_path": output_path,
            }

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            logger.info(f"Creating video with prompt: {prompt[:50]}...")

            response = self._client.post("/v1/videos", json=payload)

        video = Video(response.json(), self._client)

        if wait:
            video.wait(timeout=wait_timeout)

        return video

    def retrieve(self, video_id: str) -> Video:
        """Retrieve a video by ID.

        Args:
            video_id: The video ID

        Returns:
            Video object
        """
        response = self._client.get(f"/v1/videos/{video_id}")
        return Video(response.json(), self._client)

    def list(
        self,
        after: Optional[str] = None,
        limit: int = 20,
        order: str = "desc",
    ) -> VideosList:
        """List video generation tasks.

        Args:
            after: Cursor for pagination
            limit: Number of results (1-100)
            order: Sort order ("asc" or "desc")

        Returns:
            VideosList containing video tasks
        """
        params = {}
        if after:
            params["after"] = after
        if limit:
            params["limit"] = limit
        if order:
            params["order"] = order

        response = self._client.get("/v1/videos", params=params)
        return VideosList(response.json(), self._client)

    def delete(self, video_id: str) -> Video:
        """Delete/cancel a video generation task.

        Args:
            video_id: The video ID

        Returns:
            Updated Video object
        """
        response = self._client.delete(f"/v1/videos/{video_id}")
        return Video(response.json(), self._client)
