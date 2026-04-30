"""Latent data parsing and validation for cross-request cache."""

from __future__ import annotations

from typing import Optional

import torch

from telefuser.utils.logging import logger


def parse_latent_data(
    latent_data: dict | None,
    expected_shape: tuple,
    total_steps: int,
) -> tuple[Optional[torch.Tensor], int, list[int]]:
    """Parse and validate latent_data, return (cached_latent, skip_step, saved_steps).

    Safety checks:
    1. Shape mismatch -> discard cache
    2. skip_step <= 0 -> discard cache
    3. skip_step >= total_steps -> discard cache
    4. saved_steps < skip_step -> filter out (already done)
    """
    if not latent_data:
        return None, 0, []

    cached = latent_data.get("cached_latent")
    skip = int(latent_data.get("skip_step") or 0)
    saved = [int(s) for s in (latent_data.get("saved_steps") or [])]

    if cached is not None and not isinstance(cached, torch.Tensor):
        logger.warning(
            f"[latent_cache] cached_latent is not a torch.Tensor (got {type(cached).__name__}) -> discard cache"
        )
        cached = None

    if cached is not None and cached.shape != expected_shape:
        logger.warning(
            f"[latent_cache] shape mismatch: cached {cached.shape} vs expected {expected_shape} -> discard cache"
        )
        cached = None

    if skip >= total_steps:
        logger.warning(f"[latent_cache] skip_step {skip} >= total_steps {total_steps} -> discard")
        cached = None

    if cached is None or skip <= 0:
        cached, skip = None, 0

    saved = sorted({s for s in saved if skip <= s < total_steps})

    return cached, skip, saved
