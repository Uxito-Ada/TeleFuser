"""OpenAI Compatible Client

A Python client compatible with OpenAI's API for TeleFuser.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import requests

from telefuser.utils.logging import logger

from .images import ImagesAPI
from .videos import VideosAPI


class OpenAICompatibleClient:
    """OpenAI-compatible client for TeleFuser API.

    This client provides an interface similar to OpenAI's Python SDK
    for interacting with TeleFuser's image and video generation APIs.

    Args:
        base_url: The base URL of the TeleFuser API server
        api_key: Optional API key for authentication
        timeout: Default timeout for requests (seconds)

    Example:
        >>> client = OpenAICompatibleClient("http://localhost:8000")
        >>>
        >>> # Generate image
        >>> image = client.images.generate(prompt="a cat")
        >>>
        >>> # Create video
        >>> video = client.videos.create(prompt="a dog running", seconds=5)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()

        # Set default headers
        self._session.headers.update(
            {
                "Content-Type": "application/json",
            }
        )
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

        # Initialize API sub-clients
        self._images: ImagesAPI | None = None
        self._videos: VideosAPI | None = None

        logger.debug(f"Initialized OpenAICompatibleClient with base_url: {base_url}")

    @property
    def images(self) -> ImagesAPI:
        """Access the Images API.

        Returns:
            ImagesAPI client for image generation operations
        """
        if self._images is None:
            self._images = ImagesAPI(self)
        return self._images

    @property
    def videos(self) -> VideosAPI:
        """Access the Videos API.

        Returns:
            VideosAPI client for video generation operations
        """
        if self._videos is None:
            self._videos = VideosAPI(self)
        return self._videos

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            path: API path (without base_url)
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}{path}"

        # Set timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout

        try:
            response = self._session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get(self, path: str, **kwargs) -> requests.Response:
        """Make a GET request."""
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        """Make a POST request."""
        return self._request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        """Make a DELETE request."""
        return self._request("DELETE", path, **kwargs)

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self) -> OpenAICompatibleClient:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()
