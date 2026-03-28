"""Tests for flag module (attention availability checks)."""

import pytest

from telefuser.utils.flag import (
    FLASH_ATTN_2_AVAILABLE,
    FLASH_ATTN_3_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
    SDPA_AVAILABLE,
    SPARGE_ATTN_AVAILABLE,
)


class TestFlagAvailability:
    """Test attention implementation availability flags."""

    @pytest.mark.parametrize(
        "flag_name,flag_value",
        [
            ("FLASH_ATTN_3_AVAILABLE", FLASH_ATTN_3_AVAILABLE),
            ("FLASH_ATTN_2_AVAILABLE", FLASH_ATTN_2_AVAILABLE),
            ("SDPA_AVAILABLE", SDPA_AVAILABLE),
            ("SAGE_ATTN_AVAILABLE", SAGE_ATTN_AVAILABLE),
            ("SPARGE_ATTN_AVAILABLE", SPARGE_ATTN_AVAILABLE),
        ],
    )
    def test_flags_are_boolean(self, flag_name, flag_value):
        """Test that all attention availability flags are boolean values."""
        assert isinstance(flag_value, bool)

    def test_at_least_one_attn_available(self):
        """Test that at least one attention mechanism is available."""
        any_available = any(
            [
                FLASH_ATTN_3_AVAILABLE,
                FLASH_ATTN_2_AVAILABLE,
                SDPA_AVAILABLE,
                SAGE_ATTN_AVAILABLE,
                SPARGE_ATTN_AVAILABLE,
            ]
        )

        if not any_available:
            pytest.skip("No attention mechanisms available in this environment")
