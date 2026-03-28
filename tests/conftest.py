"""Pytest shared fixtures and configuration."""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
from PIL import Image

# ============================================================================
# Collection Ignore - Skip modules that require unavailable dependencies
# ============================================================================

# Check if triton is available (required by telefuser.distributed and telefuser.kernel.triton)
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Skip tests that require triton if it's not available
if not HAS_TRITON:
    collect_ignore = [
        # Distributed tests require triton
        "unit/distributed/test_a2a.py",
        "unit/distributed/test_device_mesh.py",
        "unit/distributed/test_parallel_shard.py",
        "unit/distributed/test_pp_comm.py",
        # Kernel tests require triton
        "unit/kernel/test_rmsnorm.py",
        "unit/kernel/test_rotary.py",
        "unit/kernel/test_scale_shift.py",
        # Ops tests that require triton
        "unit/ops/test_long_context_attention.py",
        "unit/ops/test_parallel_shard_attention.py",
    ]


# ============================================================================
# Hardware Detection Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(has_cuda):
    """Return the appropriate device."""
    return torch.device("cuda" if has_cuda else "cpu")


@pytest.fixture(scope="session")
def gpu_count():
    """Return the number of available GPUs."""
    return torch.cuda.device_count()


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "multi_gpu: marks tests that require multiple GPUs")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "distributed: marks tests that require distributed setup")


# ============================================================================
# Common Fixtures
# ============================================================================


@pytest.fixture
def sample_image_pil():
    """Create a sample PIL image for testing."""
    return Image.new("RGB", (512, 512), color="red")


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    return torch.randn(1, 3, 512, 512)


@pytest.fixture
def sample_noise_tensor():
    """Create a sample noise tensor for testing."""
    return torch.randn(1, 4, 64, 64)


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = Mock()
    manager.load_model = Mock(return_value=None)
    manager.get_model = Mock(return_value=Mock())
    return manager


@pytest.fixture
def mock_pipeline_config():
    """Create a mock pipeline configuration."""
    return {
        "name": "test_pipeline",
        "model_root": "/tmp/test_model",
        "num_inference_steps": 10,
        "height": 512,
        "width": 512,
    }


@pytest.fixture
def clear_cuda_cache():
    """Clear CUDA cache after test if CUDA is available."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def set_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Reset to random state
    torch.seed()
    np.random.seed()


# ============================================================================
# Skip Decorators Helpers
# ============================================================================

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA but CUDA is not available")

requires_multi_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason=f"Test requires multiple GPUs but only {torch.cuda.device_count()} available"
)


# Make these available as fixtures too
@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_single_gpu():
    """Skip test if less than 2 GPUs available."""
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Need 2+ GPUs, found {torch.cuda.device_count()}")
