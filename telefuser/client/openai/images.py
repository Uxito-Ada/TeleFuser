"""Images API Client

Provides OpenAI-compatible image generation methods.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union

import requests

from telefuser.utils.logging import logger

if TYPE_CHECKING:
    from .client import OpenAICompatibleClient


class Image:
    """Represents a generated image."""

    def __init__(self, data: dict):
        self._data = data
        self.b64_json = data.get("b64_json")
        self.url = data.get("url")
        self.revised_prompt = data.get("revised_prompt")

    def save(self, path: Union[str, Path]) -> None:
        """Save the image to a file.

        Args:
            path: Path to save the image
        """
        path = Path(path)

        if self.b64_json:
            # Decode base64 and save
            image_data = base64.b64decode(self.b64_json)
            path.write_bytes(image_data)
        elif self.url and self.url.startswith("http"):
            # Download from URL
            response = requests.get(self.url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        else:
            raise ValueError("No image data available to save")

        logger.info(f"Image saved to: {path}")


class ImagesResponse:
    """Response from image generation."""

    def __init__(self, data: dict):
        self._data = data
        self.created = data.get("created")
        self.data: List[Image] = [Image(img) for img in data.get("data", [])]
        self.peak_memory_mb = data.get("peak_memory_mb")
        self.inference_time_s = data.get("inference_time_s")

    def __getitem__(self, index: int) -> Image:
        """Get image by index."""
        return self.data[index]

    def __len__(self) -> int:
        """Get number of images."""
        return len(self.data)


class ImagesAPI:
    """Images API for generating and editing images."""

    def __init__(self, client: "OpenAICompatibleClient"):
        self._client = client

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        n: Optional[int] = 1,
        quality: Optional[str] = "auto",
        response_format: Optional[str] = "url",
        size: Optional[str] = "1024x1024",
        style: Optional[str] = "vivid",
        user: Optional[str] = None,
        seed: Optional[int] = 42,
        negative_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ImagesResponse:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image
            model: Model to use for generation
            n: Number of images to generate (1-10)
            quality: Image quality ("standard", "hd", "auto")
            response_format: Response format ("url" or "b64_json")
            size: Image size (e.g., "1024x1024", "1024x768")
            style: Image style ("vivid" or "natural")
            user: User identifier
            seed: Random seed for reproducibility
            negative_prompt: What to avoid in the image
            timeout: Request timeout (overrides client default)

        Returns:
            ImagesResponse containing the generated images

        Example:
            >>> response = client.images.generate(
            ...     prompt="a beautiful sunset",
            ...     size="1024x1024"
            ... )
            >>> response.data[0].save("sunset.png")
        """
        payload = {
            "prompt": prompt,
            "model": model,
            "n": n,
            "quality": quality,
            "response_format": response_format,
            "size": size,
            "style": style,
            "user": user,
            "seed": seed,
            "negative_prompt": negative_prompt,
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        logger.info(f"Generating image with prompt: {prompt[:50]}...")

        response = self._client.post(
            "/v1/images/generations",
            json=payload,
            timeout=timeout or self._client.timeout,
        )

        return ImagesResponse(response.json())

    def edit(
        self,
        prompt: str,
        image: Optional[Union[str, Path, BinaryIO]] = None,
        image_url: Optional[str] = None,
        mask: Optional[Union[str, Path, BinaryIO]] = None,
        model: Optional[str] = None,
        n: Optional[int] = 1,
        size: Optional[str] = "1024x1024",
        response_format: Optional[str] = "url",
        seed: Optional[int] = 42,
        negative_prompt: Optional[str] = None,
    ) -> ImagesResponse:
        """Edit an image based on a prompt.

        Args:
            prompt: Text description of the desired edit
            image: Path to image file or file-like object
            image_url: URL of image to edit
            mask: Path to mask file or file-like object
            model: Model to use
            n: Number of images to generate
            size: Output image size
            response_format: Response format ("url" or "b64_json")
            seed: Random seed
            negative_prompt: What to avoid

        Returns:
            ImagesResponse containing the edited images

        Example:
            >>> response = client.images.edit(
            ...     prompt="make it blue",
            ...     image="input.png"
            ... )
        """
        # Prepare form data
        data = {
            "prompt": prompt,
            "image_url": image_url,
            "model": model,
            "n": n,
            "size": size,
            "response_format": response_format,
            "seed": seed,
            "negative_prompt": negative_prompt,
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        # Prepare files
        files = {}
        if image is not None:
            if isinstance(image, (str, Path)):
                files["image"] = open(image, "rb")
            else:
                files["image"] = image

        if mask is not None:
            if isinstance(mask, (str, Path)):
                files["mask"] = open(mask, "rb")
            else:
                files["mask"] = mask

        try:
            logger.info(f"Editing image with prompt: {prompt[:50]}...")

            response = self._client.post(
                "/v1/images/edits",
                data=data,
                files=files if files else None,
            )

            return ImagesResponse(response.json())
        finally:
            # Close file handles
            for f in files.values():
                if hasattr(f, "close"):
                    f.close()
