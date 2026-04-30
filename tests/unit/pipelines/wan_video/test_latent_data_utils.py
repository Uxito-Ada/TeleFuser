"""Unit tests for parse_latent_data validation logic.

Validates the cross-request latent cache parsing helper. Pure CPU, no
GPU required.
"""

from __future__ import annotations

import torch

from telefuser.pipelines.wan_video.latent_data_utils import parse_latent_data

SHAPE = (1, 16, 5, 32, 32)


class TestParseLatentData:
    """Tests for parse_latent_data validation logic."""

    def test_none_input(self):
        cached, skip, saved = parse_latent_data(None, SHAPE, 10)
        assert cached is None and skip == 0 and saved == []

    def test_empty_dict(self):
        cached, skip, saved = parse_latent_data({}, SHAPE, 10)
        assert cached is None and skip == 0 and saved == []

    def test_shape_mismatch(self):
        bad = torch.randn(1, 8, 5, 32, 32)  # C=8 != 16
        data = {"cached_latent": bad, "skip_step": 2, "saved_steps": [3]}
        cached, skip, saved = parse_latent_data(data, SHAPE, 10)
        assert cached is None and skip == 0

    def test_valid_hit(self):
        t = torch.randn(*SHAPE)
        data = {"cached_latent": t, "skip_step": 3, "saved_steps": [5, 7]}
        cached, skip, saved = parse_latent_data(data, SHAPE, 10)
        assert cached is not None and skip == 3
        assert saved == [5, 7]

    def test_saved_before_skip_filtered(self):
        t = torch.randn(*SHAPE)
        data = {"cached_latent": t, "skip_step": 5, "saved_steps": [3, 7]}
        cached, skip, saved = parse_latent_data(data, SHAPE, 10)
        assert saved == [7]

    def test_skip_out_of_range(self):
        t = torch.randn(*SHAPE)
        data = {"cached_latent": t, "skip_step": 10, "saved_steps": []}
        cached, skip, saved = parse_latent_data(data, SHAPE, 10)
        assert cached is None and skip == 0

    def test_skip_zero_with_saved(self):
        """skip=0 with saved_steps -> save only, no restore."""
        data = {"saved_steps": [2, 5]}
        cached, skip, saved = parse_latent_data(data, SHAPE, 10)
        assert cached is None and skip == 0
        assert saved == [2, 5]
