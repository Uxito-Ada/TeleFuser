"""Tests for Ulysses All-to-All communication module."""

from unittest.mock import MagicMock, patch

import pytest
import torch

# Skip if distributed not available
try:
    import torch.distributed as dist

    HAS_DISTRIBUTED = dist.is_available()
except ImportError:
    HAS_DISTRIBUTED = False

pytestmark = [
    pytest.mark.skipif(not HAS_DISTRIBUTED, reason="Distributed not available"),
    pytest.mark.distributed,
]


class TestUlyssesScatterHeads:
    """Test ulysses_scatter_heads function."""

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size")
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank")
    def test_scatter_heads_basic(self, mock_get_rank, mock_get_world_size):
        """Test basic scatter_heads operation."""
        from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 0

        # Input: (B, S_local, H_global, D) = (2, 10, 32, 64)
        tensor = torch.randn(2, 10, 32, 64)

        with patch("telefuser.distributed.ulysses_comm.fc.all_to_all_single") as mock_a2a:
            mock_a2a.return_value = tensor.flatten()

            wait_fn = ulysses_scatter_heads(tensor, MagicMock())
            assert callable(wait_fn)

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size")
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank")
    def test_scatter_heads_with_padding(self, mock_get_rank, mock_get_world_size):
        """Test scatter_heads with head padding."""
        from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

        # 30 heads, 4 ranks -> need padding
        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 0

        tensor = torch.randn(2, 10, 30, 64)

        with patch("telefuser.distributed.ulysses_comm.fc.all_to_all_single") as mock_a2a:
            # Simulate padded output
            padded_tensor = torch.randn(2, 10, 32, 64)
            mock_a2a.return_value = padded_tensor.flatten()

            wait_fn = ulysses_scatter_heads(tensor, MagicMock())
            assert callable(wait_fn)

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size")
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank")
    def test_scatter_heads_invalid_padding(self, mock_get_rank, mock_get_world_size):
        """Test scatter_heads raises error for invalid padding."""
        from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

        # 30 heads, 16 ranks -> invalid padding
        mock_get_world_size.return_value = 16
        mock_get_rank.return_value = 0

        tensor = torch.randn(2, 10, 30, 64)

        with pytest.raises(ValueError, match="Cannot pad"):
            ulysses_scatter_heads(tensor, MagicMock())


class TestUlyssesGatherHeads:
    """Test ulysses_gather_heads function."""

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size")
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank")
    def test_gather_heads_basic(self, mock_get_rank, mock_get_world_size):
        """Test basic gather_heads operation."""
        from telefuser.distributed.ulysses_comm import ulysses_gather_heads

        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 0

        # Input: (B, S_global, H_local, D) = (2, 40, 8, 64)
        tensor = torch.randn(2, 40, 8, 64)

        with patch("telefuser.distributed.ulysses_comm.fc.all_to_all_single") as mock_a2a:
            mock_a2a.return_value = tensor.flatten()

            wait_fn = ulysses_gather_heads(tensor, MagicMock(), num_heads=32)
            assert callable(wait_fn)

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size")
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank")
    def test_gather_heads_divisible_heads(self, mock_get_rank, mock_get_world_size):
        """Test gather_heads with evenly divisible heads (no padding needed)."""
        from telefuser.distributed.ulysses_comm import ulysses_gather_heads

        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 0

        # 32 heads, 4 ranks -> 8 local heads per rank, no padding
        tensor = torch.randn(2, 40, 8, 64)

        with patch("telefuser.distributed.ulysses_comm.fc.all_to_all_single") as mock_a2a:
            mock_a2a.return_value = tensor.flatten()

            wait_fn = ulysses_gather_heads(tensor, MagicMock(), num_heads=32)
            assert callable(wait_fn)


class TestInternalHelpers:
    """Test internal helper functions."""

    def test_compute_head_distribution(self):
        """Test head distribution computation."""
        from telefuser.distributed.ulysses_comm import _compute_head_distribution

        # Even split
        result = _compute_head_distribution(32, 4)
        assert result == [8, 8, 8, 8]
        assert sum(result) == 32

        # Uneven split
        result = _compute_head_distribution(30, 4)
        assert result == [8, 8, 7, 7]
        assert sum(result) == 30

        # Small numbers
        result = _compute_head_distribution(7, 2)
        assert result == [4, 3]
        assert sum(result) == 7


class TestAsyncComm:
    """Test async communication behavior."""

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size")
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank")
    def test_async_comm_default(self, mock_get_rank, mock_get_world_size):
        """Test async_comm=True is default."""
        from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 0

        tensor = torch.randn(2, 10, 32, 64)

        with patch("telefuser.distributed.ulysses_comm.fc.all_to_all_single") as mock_a2a:
            mock_a2a.return_value = tensor.flatten()
            # Default should use async
            ulysses_scatter_heads(tensor, MagicMock())
            mock_a2a.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
