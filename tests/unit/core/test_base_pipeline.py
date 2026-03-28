"""Tests for BasePipeline class."""

import numpy as np
import pytest
import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline


class MockPipeline(BasePipeline):
    """Mock pipeline for testing."""

    pass


@pytest.fixture
def pipeline():
    """Create a MockPipeline fixture."""
    return MockPipeline(device="cpu", torch_dtype=torch.float32)


class TestBasePipelineInitialization:
    """Test BasePipeline initialization."""

    def test_init_with_cpu(self):
        """Test initialization with CPU device."""
        pipeline = MockPipeline(device="cpu", torch_dtype=torch.float32)
        assert pipeline.device == "cpu"
        assert pipeline.torch_dtype == torch.float32
        assert pipeline.height_division_factor == 16
        assert pipeline.width_division_factor == 16

    @pytest.mark.gpu
    def test_init_with_cuda(self):
        """Test initialization with CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        pipeline = MockPipeline(device="cuda", torch_dtype=torch.float32)
        assert pipeline.device == "cuda"
        assert pipeline.torch_dtype == torch.float32

    def test_init_with_bfloat16(self):
        """Test initialization with bfloat16."""
        pipeline = MockPipeline(device="cpu", torch_dtype=torch.bfloat16)
        assert pipeline.torch_dtype == torch.bfloat16


class TestCheckResizeHeightWidth:
    """Test check_resize_height_width method."""

    @pytest.mark.parametrize(
        "height,width,expected_height,expected_width",
        [
            (512, 512, 512, 512),  # Already valid
            (256, 768, 256, 768),  # Different valid dimensions
            (500, 512, 512, 512),  # Height rounds up
            (512, 500, 512, 512),  # Width rounds up
            (505, 507, 512, 512),  # Both round up
            (16, 16, 16, 16),  # Small dimensions
            (511, 1023, 512, 1024),  # Just below multiple
        ],
    )
    def test_dimension_rounding(self, pipeline, height, width, expected_height, expected_width):
        """Test dimension rounding to multiples of 16."""
        result_height, result_width = pipeline.check_resize_height_width(height, width)
        assert result_height == expected_height
        assert result_width == expected_width


class TestPreprocessImage:
    """Test preprocess_image method."""

    def test_preprocess_shape_and_dtype(self, pipeline):
        """Test preprocessing produces correct shape and dtype."""
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)

        tensor = pipeline.preprocess_image(image)

        assert tensor.shape == (1, 3, 256, 256)
        assert tensor.dtype == torch.float32

    @pytest.mark.parametrize(
        "color_value,expected_range",
        [
            (0, (-1.0, -1.0)),  # Black
            (128, (0.0, 0.05)),  # Mid-gray (approximately 0)
            (255, (1.0, 1.0)),  # White
        ],
    )
    def test_preprocess_color_normalization(self, pipeline, color_value, expected_range):
        """Test that colors are normalized to [-1, 1] range."""
        img_array = np.full((64, 64, 3), color_value, dtype=np.uint8)
        image = Image.fromarray(img_array)

        tensor = pipeline.preprocess_image(image)

        min_val, max_val = expected_range
        assert min_val - 0.01 <= tensor.min() <= max_val + 0.01
        assert min_val - 0.01 <= tensor.max() <= max_val + 0.01

    def test_preprocess_with_resize(self, pipeline):
        """Test preprocessing with resize."""
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)

        tensor = pipeline.preprocess_image(image, height=128, width=128)

        assert tensor.shape == (1, 3, 128, 128)


class TestPreprocessImages:
    """Test preprocess_images method."""

    def test_preprocess_multiple_images(self, pipeline):
        """Test preprocessing multiple images."""
        images = [
            Image.new("RGB", (64, 64), color="red"),
            Image.new("RGB", (64, 64), color="green"),
        ]

        tensors = pipeline.preprocess_images(images)

        assert len(tensors) == 2
        for tensor in tensors:
            assert tensor.shape == (1, 3, 64, 64)

    def test_preprocess_empty_list(self, pipeline):
        """Test preprocessing empty list."""
        tensors = pipeline.preprocess_images([])
        assert tensors == []


class TestTensor2Video:
    """Test tensor2video method."""

    def test_tensor2video_basic(self, pipeline):
        """Test basic tensor to video conversion."""
        tensor = torch.randn(3, 4, 64, 64)

        frames = pipeline.tensor2video(tensor)

        assert len(frames) == 4
        for frame in frames:
            assert isinstance(frame, Image.Image)
            assert frame.size == (64, 64)

    def test_tensor2video_denormalization(self, pipeline):
        """Test that values are correctly denormalized from [-1, 1] to [0, 255]."""
        tensor = torch.zeros(3, 1, 32, 32)  # All zeros -> mid gray

        frames = pipeline.tensor2video(tensor)

        frame_array = np.array(frames[0])
        # 0 -> (0 + 1) * 127.5 = 127.5
        assert 125 <= frame_array.min() <= 130
        assert 125 <= frame_array.max() <= 130

    def test_tensor2video_clipping(self, pipeline):
        """Test that values outside [-1, 1] are clipped."""
        tensor = torch.tensor([[[[2.0]]], [[[-2.0]]], [[[0.5]]]])

        frames = pipeline.tensor2video(tensor)

        frame_array = np.array(frames[0])
        assert frame_array.min() >= 0
        assert frame_array.max() <= 255

    def test_tensor2video_with_resize(self, pipeline):
        """Test tensor to video with resize."""
        tensor = torch.randn(3, 2, 64, 64)

        frames = pipeline.tensor2video(tensor, height=128, width=128)

        assert len(frames) == 2
        assert frames[0].size == (128, 128)


class TestGenerateNoise:
    """Test generate_noise method."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_generate_noise_shape_and_dtype(self, pipeline, dtype):
        """Test noise shape and dtype."""
        noise = pipeline.generate_noise((1, 4, 64, 64), dtype=dtype)
        assert noise.shape == (1, 4, 64, 64)
        assert noise.dtype == dtype

    def test_generate_noise_reproducibility(self, pipeline):
        """Test that same seed produces same noise."""
        noise1 = pipeline.generate_noise((1, 4, 64, 64), seed=42)
        noise2 = pipeline.generate_noise((1, 4, 64, 64), seed=42)
        assert torch.allclose(noise1, noise2)

    def test_generate_noise_different_seeds(self, pipeline):
        """Test that different seeds produce different noise."""
        noise1 = pipeline.generate_noise((1, 4, 64, 64), seed=42)
        noise2 = pipeline.generate_noise((1, 4, 64, 64), seed=43)
        assert not torch.allclose(noise1, noise2)

    def test_generate_noise_distribution(self, pipeline):
        """Test that noise follows normal distribution N(0, 1)."""
        noise = pipeline.generate_noise((1, 4, 256, 256), seed=42)

        mean = noise.mean().item()
        std = noise.std().item()

        # More strict assertions for normal distribution
        assert abs(mean) < 0.15, f"Mean should be close to 0, got {mean}"
        assert 0.85 < std < 1.15, f"Std should be close to 1, got {std}"

    def test_generate_noise_device(self, pipeline):
        """Test noise generation on CPU."""
        noise = pipeline.generate_noise((1, 4, 32, 32), device="cpu")
        assert noise.device.type == "cpu"
