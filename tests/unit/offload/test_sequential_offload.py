"""Tests for offload layers module."""

import pytest
import torch
import torch.nn as nn

from telefuser.offload.sequential_offload import (
    AutoTorchModule,
    AutoWrappedLinear,
    AutoWrappedModule,
    WanAutoCastLayerNorm,
    cast_to,
    enable_sequential_cpu_offload,
)


class TestCastTo:
    """Test cast_to function."""

    @pytest.mark.parametrize(
        "device,dtype",
        [
            ("cpu", torch.float32),
            ("cpu", torch.float16),
        ],
    )
    def test_cast_to(self, device, dtype):
        """Test casting to different device and dtype combinations."""
        weight = torch.randn(10, 10)

        result = cast_to(weight, dtype, device)

        assert result.device.type == device
        assert result.dtype == dtype

    @pytest.mark.gpu
    def test_cast_to_cuda(self):
        """Test casting to CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        weight = torch.randn(10, 10)
        result = cast_to(weight, torch.float32, "cuda")
        assert result.device.type == "cuda"


class TestAutoTorchModule:
    """Test AutoTorchModule base class."""

    @pytest.fixture
    def auto_module(self):
        """Create a minimal AutoTorchModule for testing."""

        class TestModule(AutoTorchModule):
            def __init__(self):
                super().__init__()
                self.offload_dtype = torch.float32
                self.offload_device = "cpu"
                self.onload_dtype = torch.float32
                self.onload_device = "cpu"
                self.computation_dtype = torch.float32
                self.computation_device = "cpu"
                self.vram_limit = None
                self.state = 1

        return TestModule()

    @pytest.mark.parametrize(
        "operation,expected_state",
        [
            ("offload", 0),
            ("onload", 1),
            ("keep", 2),
        ],
    )
    def test_state_transitions(self, auto_module, operation, expected_state):
        """Test state transitions."""
        getattr(auto_module, operation)()
        assert auto_module.state == expected_state


class TestAutoWrappedModule:
    """Test AutoWrappedModule."""

    @pytest.fixture
    def wrapped_module(self):
        """Create an AutoWrappedModule for testing."""
        inner = nn.Linear(10, 20)
        return AutoWrappedModule(
            module=inner,
            offload_dtype=torch.float32,
            offload_device="cpu",
            onload_dtype=torch.float32,
            onload_device="cpu",
            computation_dtype=torch.float32,
            computation_device="cpu",
            vram_limit=None,
        )

    def test_initialization(self, wrapped_module):
        """Test module initialization."""
        assert wrapped_module.state == 0
        assert wrapped_module.module is not None

    @pytest.mark.parametrize("state_op", ["keep", "offload"])
    def test_forward_in_different_states(self, wrapped_module, state_op):
        """Test forward in different states."""
        getattr(wrapped_module, state_op)()

        x = torch.randn(5, 10)
        output = wrapped_module(x)

        assert output.shape == (5, 20)


class TestAutoWrappedLinear:
    """Test AutoWrappedLinear."""

    @pytest.fixture
    def wrapped_linear(self):
        """Create an AutoWrappedLinear for testing."""
        inner = nn.Linear(10, 20)
        return AutoWrappedLinear(
            module=inner,
            offload_dtype=torch.float32,
            offload_device="cpu",
            onload_dtype=torch.float32,
            onload_device="cpu",
            computation_dtype=torch.float32,
            computation_device="cpu",
            vram_limit=None,
            name="test_linear",
        )

    def test_initialization(self, wrapped_linear):
        """Test linear layer initialization."""
        assert wrapped_linear.state == 0
        assert wrapped_linear.name == "test_linear"
        assert wrapped_linear.in_features == 10
        assert wrapped_linear.out_features == 20

    def test_forward_basic(self, wrapped_linear):
        """Test basic forward pass."""
        x = torch.randn(5, 10)
        output = wrapped_linear(x)

        assert output.shape == (5, 20)

    def test_forward_with_bias(self):
        """Test forward with bias."""
        inner = nn.Linear(10, 20, bias=True)
        wrapped = AutoWrappedLinear(
            module=inner,
            offload_dtype=torch.float32,
            offload_device="cpu",
            onload_dtype=torch.float32,
            onload_device="cpu",
            computation_dtype=torch.float32,
            computation_device="cpu",
            vram_limit=None,
        )

        x = torch.randn(5, 10)
        output = wrapped(x)

        assert output.shape == (5, 20)


class TestWanAutoCastLayerNorm:
    """Test WanAutoCastLayerNorm."""

    @pytest.fixture
    def wrapped_layernorm(self):
        """Create a WanAutoCastLayerNorm for testing."""
        inner = nn.LayerNorm(64)
        return WanAutoCastLayerNorm(
            module=inner,
            offload_dtype=torch.float32,
            offload_device="cpu",
            onload_dtype=torch.float32,
            onload_device="cpu",
            computation_dtype=torch.float32,
            computation_device="cpu",
            vram_limit=None,
        )

    def test_initialization(self, wrapped_layernorm):
        """Test LayerNorm initialization."""
        assert wrapped_layernorm.state == 0
        assert wrapped_layernorm.normalized_shape == (64,)

    def test_forward(self, wrapped_layernorm):
        """Test forward pass."""
        x = torch.randn(2, 10, 64)
        output = wrapped_layernorm(x)

        assert output.shape == x.shape

    def test_forward_preserves_dtype(self, wrapped_layernorm):
        """Test that forward preserves input dtype."""
        x = torch.randn(2, 10, 64, dtype=torch.float16)
        output = wrapped_layernorm(x)

        assert output.dtype == torch.float16


class TestEnableVramManagement:
    """Test enable_sequential_cpu_offload function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
        )

    @pytest.fixture
    def module_config(self):
        """Create module config for testing."""
        return {
            "offload_dtype": torch.float32,
            "offload_device": "cpu",
            "onload_dtype": torch.float32,
            "onload_device": "cpu",
            "computation_dtype": torch.float32,
            "computation_device": "cpu",
        }

    def test_enables_sequential_cpu_offload(self, simple_model, module_config):
        """Test that sequential CPU offload is enabled on model."""
        module_map = {nn.Linear: AutoWrappedLinear}

        enable_sequential_cpu_offload(simple_model, module_map, module_config)

        assert hasattr(simple_model, "sequential_cpu_offload_enabled")
        assert simple_model.sequential_cpu_offload_enabled is True

    def test_replaces_modules(self, simple_model, module_config):
        """Test that modules are replaced correctly."""
        module_map = {nn.Linear: AutoWrappedLinear}

        enable_sequential_cpu_offload(simple_model, module_map, module_config)

        # Linear layers should be replaced
        assert isinstance(simple_model[0], AutoWrappedLinear)
        assert isinstance(simple_model[2], AutoWrappedLinear)

        # ReLU should not be replaced
        assert isinstance(simple_model[1], nn.ReLU)
