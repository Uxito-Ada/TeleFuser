"""Base pipeline for multimodal generation."""

from __future__ import annotations

import json
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

from telefuser.utils.logging import logger

if TYPE_CHECKING:
    from telefuser.metrics import StageMetricsManager


class BasePipeline(ABC):
    """Base pipeline for generation tasks."""

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype) -> None:
        self.device = device
        self.torch_dtype = torch_dtype
        # VAE typically requires dimensions divisible by 16
        self.height_division_factor = 16
        self.width_division_factor = 16
        self._metrics_manager: StageMetricsManager | None = None

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection.

        Subclasses should override this method to return their stages.

        Returns:
            List of stage instances that support metrics collection.
        """
        return []

    def enable_metrics(self) -> None:
        """Enable metrics collection for all pipeline stages.

        This method enables metrics tracking for stages returned by _get_stages().

        Example:
            >>> pipeline = MyPipeline.from_pretrained(...)
            >>> pipeline.enable_metrics()
            >>> # Run inference with metrics enabled
            >>> result = pipeline(...)
            >>> # Get metrics
            >>> print(pipeline.get_prometheus_metrics())
        """
        if self._metrics_manager is not None:
            logger.warning("Metrics already enabled, skipping")
            return

        from telefuser.metrics import StageMetricsManager

        self._metrics_manager = StageMetricsManager()
        stages = self._get_stages()

        if stages:
            self._metrics_manager.enable_all_stages(stages)
            logger.info(f"Enabled metrics for stages: {self._metrics_manager.enabled_stages}")
        else:
            logger.warning("No stages available for metrics collection")

    def disable_metrics(self) -> None:
        """Disable metrics collection for all pipeline stages.

        Example:
            >>> pipeline.disable_metrics()
        """
        if self._metrics_manager is None:
            return

        self._metrics_manager.disable_all_stages()
        self._metrics_manager = None
        logger.info("Disabled metrics for all stages")

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus text exposition format.

        Returns:
            String containing all metrics in Prometheus format.

        Example:
            >>> metrics = pipeline.get_prometheus_metrics()
            >>> print(metrics)
            # HELP stage_vae_duration_seconds Execution duration
            # TYPE stage_vae_duration_seconds histogram
            ...

        Raises:
            RuntimeError: If metrics are not enabled.
        """
        if self._metrics_manager is None:
            raise RuntimeError("Metrics not enabled. Call enable_metrics() first.")
        return self._metrics_manager.registry.get_prometheus_format()

    @property
    def metrics_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._metrics_manager is not None

    def check_resize_height_width(self, height: int, width: int) -> tuple[int, int]:
        """Ensure dimensions are divisible by division factors."""
        if height % self.height_division_factor != 0:
            factor = self.height_division_factor
            height = ((height + factor - 1) // factor) * factor
            print(f"Height rounded up to {height}")
        if width % self.width_division_factor != 0:
            factor = self.width_division_factor
            width = ((width + factor - 1) // factor) * factor
            print(f"Width rounded up to {width}")
        return height, width

    def preprocess_image(self, image: Image.Image, height: int | None = None, width: int | None = None) -> torch.Tensor:
        """Preprocess PIL image to tensor."""
        if height is not None and width is not None:
            if height != image.size[1] or width != image.size[0]:
                image = image.resize((width, height), Image.LANCZOS)
        return torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)

    def preprocess_images(
        self, images: Sequence[Image.Image], height: int | None = None, width: int | None = None
    ) -> list[torch.Tensor]:
        """Preprocess multiple images."""
        return [self.preprocess_image(image, height, width) for image in images]

    def tensor2video(
        self, frames: torch.Tensor, height: int | None = None, width: int | None = None
    ) -> list[Image.Image]:
        """Convert tensor to list of PIL images."""
        if height is not None and width is not None:
            if height != frames.shape[2] or width != frames.shape[3]:
                logger.info(f"Resizing video to {width}x{height}")
                frames = F.interpolate(
                    frames, size=(height, width), mode="bicubic", align_corners=False, antialias=True
                )
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    def generate_noise(
        self, shape: Sequence[int], seed: int | None = None, device: str = "cpu", dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """Generate random noise."""
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def dump_config(self, path: str | Path | None = None) -> dict[str, Any]:
        """Dump pipeline configuration to dict or file.

        Captures model definition (Layer 1) and inference algorithm
        parameters (Layer 2) for reproducibility and debugging.

        Args:
            path: Optional file path to save config (JSON format)

        Returns:
            Configuration dictionary with model and inference settings

        Example:
            >>> pipeline = MyPipeline.from_pretrained(...)
            >>> config = pipeline.dump_config()
            >>> pipeline.dump_config("output/config.json")
        """
        from .config_serializer import serialize_config

        config: dict[str, Any] = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "pipeline_type": self.__class__.__name__,
            "device": str(self.device),
            "torch_dtype": str(self.torch_dtype).replace("torch.", ""),
            "layer1_model_definition": {
                "models": self._model_info if hasattr(self, "_model_info") else [],
            },
            "layer2_inference_config": serialize_config(self.config) if hasattr(self, "config") else {},
        }

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Config dumped to {path}")

        return config
