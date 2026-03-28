"""Tests for feature_cache module."""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from telefuser.feature_cache import (
    AdaTaylorCache,
    AdaTaylorCacheCalibrator,
    AdaTaylorCacheConfig,
    AdaTaylorCacheHook,
    AdaTaylorCacheState,
    FeatureCacheHook,
    FeatureCacheHookManager,
    NoOpHook,
)
from telefuser.feature_cache.ada_taylor_cache import load_cache_params, nearest_interp


class TestAdaTaylorCacheConfig:
    """Test AdaTaylorCacheConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AdaTaylorCacheConfig()
        assert config.enabled is True
        assert config.model_type == ""
        assert config.n_derivatives == 1
        assert config.num_inference_steps == 50
        assert config.taylor_threshold == 2
        assert config.init_step == 0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AdaTaylorCacheConfig(
            enabled=False,
            model_type="test-model",
            n_derivatives=2,
            num_inference_steps=100,
            taylor_threshold=5,
            init_step=10,
        )
        assert config.enabled is False
        assert config.model_type == "test-model"
        assert config.n_derivatives == 2
        assert config.num_inference_steps == 100
        assert config.taylor_threshold == 5
        assert config.init_step == 10

    def test_invalid_n_derivatives(self):
        """Test that negative n_derivatives raises ValueError."""
        with pytest.raises(ValueError, match="n_derivatives must be non-negative"):
            AdaTaylorCacheConfig(n_derivatives=-1)

    def test_invalid_num_inference_steps(self):
        """Test that invalid num_inference_steps raises ValueError."""
        with pytest.raises(ValueError, match="num_inference_steps must be at least 1"):
            AdaTaylorCacheConfig(num_inference_steps=0)

    def test_invalid_taylor_threshold(self):
        """Test that invalid taylor_threshold raises ValueError."""
        with pytest.raises(ValueError, match="taylor_threshold must be at least 1"):
            AdaTaylorCacheConfig(taylor_threshold=0)


class TestNearestInterp:
    """Test nearest_interp utility function."""

    def test_basic_interpolation(self):
        """Test basic nearest neighbor interpolation."""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = nearest_interp(src, target_length=5)
        np.testing.assert_array_almost_equal(result, src)

    def test_downsample(self):
        """Test downsampling with nearest interpolation."""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = nearest_interp(src, target_length=3)
        assert len(result) == 3
        # Nearest should pick indices 0, 2, 4
        np.testing.assert_array_almost_equal(result, np.array([1.0, 3.0, 5.0]))

    def test_upsample(self):
        """Test upsampling with nearest interpolation."""
        src = np.array([1.0, 3.0, 5.0])
        result = nearest_interp(src, target_length=5)
        assert len(result) == 5

    def test_single_element(self):
        """Test with single element target."""
        src = np.array([1.0, 2.0, 3.0])
        result = nearest_interp(src, target_length=1)
        assert len(result) == 1
        assert result[0] == src[-1]

    def test_list_input(self):
        """Test with list input."""
        src = [1.0, 2.0, 3.0]
        result = nearest_interp(src, target_length=3)
        assert len(result) == 3


class TestAdaTaylorCacheState:
    """Test AdaTaylorCacheState class."""

    @pytest.fixture
    def state(self):
        """Create a basic state for testing."""
        mag_ratios = np.ones(10)
        return AdaTaylorCacheState(
            num_inference_steps=10,
            thresh=0.1,
            K=2,
            mag_ratios=mag_ratios,
            retention_ratio=0.2,
            n_derivatives=1,
            taylor_threshold=2,
            init_step=0,
        )

    def test_initialization(self, state):
        """Test state initialization."""
        assert state.num_inference_steps == 10
        assert state.thresh == 0.1
        assert state.K == 2
        assert state.n_derivatives == 1
        assert state.order == 2
        assert state.current_step == -1

    def test_reset(self, state):
        """Test state reset."""
        state.current_step = 5
        state.accumulated_err = 0.5
        state.reset()
        assert state.current_step == -1
        assert state.accumulated_err == 0.0
        assert state.last_residual is None

    def test_set_current_step(self, state):
        """Test setting current step."""
        state.set_current_step(5)
        assert state.current_step == 5

    def test_precompute_compute_steps_includes_first_steps(self, state):
        """Test that pre-computed steps include warmup period."""
        # retention_ratio=0.2 means first 2 steps should always compute
        compute_steps = state.compute_steps
        assert 0 in compute_steps
        assert 1 in compute_steps

    def test_precompute_compute_steps_includes_last_step(self, state):
        """Test that pre-computed steps include last step."""
        compute_steps = state.compute_steps
        assert (state.num_inference_steps - 1) in compute_steps

    def test_should_compute_at_step(self, state):
        """Test should_compute decision at various steps."""
        # Step 0 should always compute
        state.set_current_step(0)
        assert state.should_compute() is True

        # Step 9 (last) should always compute
        state.set_current_step(9)
        assert state.should_compute() is True

    def test_compute_derivatives_first_step(self, state):
        """Test derivative computation at first step."""
        residual = torch.randn(2, 4, 8, 8)
        derivatives = state.compute_derivatives(residual)

        # First derivative should be the residual itself
        assert torch.allclose(derivatives[0], residual)
        # Higher derivatives should be None (no previous data)
        assert derivatives[1] is None

    def test_compute_derivatives_second_step(self, state):
        """Test derivative computation at second step."""
        residual1 = torch.randn(2, 4, 8, 8)
        residual2 = torch.randn(2, 4, 8, 8)

        # First step
        state.last_compute_step = 0
        state.derivatives["dR_current"] = [residual1, None]

        # Second step
        state.current_step = 1
        state.derivatives["dR_prev"] = [residual1, None]
        derivatives = state.compute_derivatives(residual2)

        assert torch.allclose(derivatives[0], residual2)
        # First derivative should be computed
        assert derivatives[1] is not None

    def test_update(self, state):
        """Test state update with output and input."""
        output = torch.randn(2, 4, 8, 8)
        ori_input = torch.randn(2, 4, 8, 8)

        state.set_current_step(0)
        state.update(output, ori_input)

        assert state.last_residual is not None
        assert state.last_compute_step == 0

    def test_approximate_with_no_derivatives(self, state):
        """Test approximation when no derivatives available."""
        current_input = torch.randn(2, 4, 8, 8)
        result = state.approximate(current_input)

        # Should return input unchanged when no derivatives
        assert torch.allclose(result, current_input)

    def test_approximate_with_derivatives(self, state):
        """Test approximation with computed derivatives."""
        output = torch.randn(2, 4, 8, 8)
        ori_input = torch.randn(2, 4, 8, 8)

        state.set_current_step(0)
        state.update(output, ori_input)

        state.set_current_step(1)
        approx = state.approximate(ori_input)

        # Approximation should have correct shape
        assert approx.shape == output.shape


class TestLoadCacheParams:
    """Test load_cache_params function."""

    def test_load_existing_params(self):
        """Test loading existing model parameters."""
        # Wan2.1-T2V-1.3B should exist
        params = load_cache_params("Wan2.1-T2V-1.3B")
        assert "K" in params
        assert "thresh" in params
        assert "retention_ratio" in params
        assert "cond_mag_ratios" in params
        assert "uncond_mag_ratios" in params

    def test_load_nonexistent_params(self):
        """Test loading non-existent model parameters raises error."""
        with pytest.raises(FileNotFoundError):
            load_cache_params("NonExistentModel")

    def test_param_structure(self):
        """Test that loaded parameters have correct structure."""
        params = load_cache_params("Wan2.1-T2V-1.3B")

        assert isinstance(params["K"], int)
        assert isinstance(params["thresh"], (int, float))
        assert isinstance(params["retention_ratio"], (int, float))
        assert isinstance(params["cond_mag_ratios"], list)
        assert isinstance(params["uncond_mag_ratios"], list)


class TestAdaTaylorCache:
    """Test AdaTaylorCache class."""

    @pytest.fixture
    def cache(self):
        """Create a cache instance for testing."""
        return AdaTaylorCache(
            model_type="Wan2.1-T2V-1.3B",
            num_inference_steps=10,
            n_derivatives=1,
            taylor_threshold=2,
            init_step=0,
        )

    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache.num_inference_steps == 10
        assert cache.n_derivatives == 1
        assert cache.taylor_threshold == 2
        assert cache.cond_state is not None
        assert cache.uncond_state is not None

    def test_mark_step_begin(self, cache):
        """Test step counter increment."""
        assert cache.current_step == -1

        cache.mark_step_begin()
        assert cache.current_step == 0
        assert cache.cond_state.current_step == 0
        assert cache.uncond_state.current_step == 0

        cache.mark_step_begin()
        assert cache.current_step == 1

    def test_reset(self, cache):
        """Test cache reset."""
        cache.mark_step_begin()
        cache.reset()

        assert cache.current_step == -1
        assert cache.cond_state.current_step == -1
        assert cache.uncond_state.current_step == -1

    def test_store_cond(self, cache):
        """Test storing for conditional path."""
        output = torch.randn(2, 4, 8, 8)
        ori_input = torch.randn(2, 4, 8, 8)

        cache.mark_step_begin()
        cache.store(output, ori_input, is_cond=True)

        assert cache.cond_state.last_residual is not None

    def test_store_uncond(self, cache):
        """Test storing for unconditional path."""
        output = torch.randn(2, 4, 8, 8)
        ori_input = torch.randn(2, 4, 8, 8)

        cache.mark_step_begin()
        cache.store(output, ori_input, is_cond=False)

        assert cache.uncond_state.last_residual is not None

    def test_approximate_cond(self, cache):
        """Test approximation for conditional path."""
        output = torch.randn(2, 4, 8, 8)
        ori_input = torch.randn(2, 4, 8, 8)

        cache.mark_step_begin()
        cache.store(output, ori_input, is_cond=True)

        cache.mark_step_begin()
        approx = cache.approximate(ori_input, is_cond=True)

        assert approx.shape == output.shape

    def test_should_skip(self, cache):
        """Test skip decision logic."""
        cache.mark_step_begin()
        # First step should not skip (retention period)
        assert cache.should_skip(is_cond=True) is False

    def test_should_compute(self, cache):
        """Test compute decision logic."""
        cache.mark_step_begin()
        # First step should compute
        assert cache.should_compute(is_cond=True) is True

    def test_get_compute_steps(self, cache):
        """Test getting compute steps list."""
        steps = cache.get_compute_steps()
        assert isinstance(steps, list)
        assert 0 in steps  # First step should always be in compute steps


class TestFeatureCacheHookManager:
    """Test FeatureCacheHookManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = FeatureCacheHookManager()
        assert manager.has_hook() is False
        assert manager.get_hook() is None

    def test_set_hook(self):
        """Test setting a hook."""
        manager = FeatureCacheHookManager()
        hook = NoOpHook()
        manager.set_hook(hook)

        assert manager.has_hook() is True
        assert manager.get_hook() is hook

    def test_clear_hook(self):
        """Test clearing the hook."""
        manager = FeatureCacheHookManager()
        manager.set_hook(NoOpHook())
        manager.clear_hook()

        assert manager.has_hook() is False

    def test_pre_forward_without_hook(self):
        """Test pre_forward returns None when no hook set."""
        manager = FeatureCacheHookManager()
        x = torch.randn(2, 4, 8, 8)

        result = manager.pre_forward(x, cond_flag=True)
        assert result is None

    def test_post_forward_without_hook(self):
        """Test post_forward does nothing when no hook set."""
        manager = FeatureCacheHookManager()
        output = torch.randn(2, 4, 8, 8)
        ori_input = torch.randn(2, 4, 8, 8)

        # Should not raise
        manager.post_forward(output, ori_input, cond_flag=True)

    def test_delegation_to_hook(self):
        """Test that manager delegates to hook correctly."""
        manager = FeatureCacheHookManager()
        mock_hook = MagicMock(spec=FeatureCacheHook)
        manager.set_hook(mock_hook)

        x = torch.randn(2, 4, 8, 8)
        manager.pre_forward(x, cond_flag=True)
        mock_hook.pre_forward.assert_called_once_with(x, True)

        output = torch.randn(2, 4, 8, 8)
        manager.post_forward(output, x, cond_flag=False)
        mock_hook.post_forward.assert_called_once_with(output, x, False)


class TestNoOpHook:
    """Test NoOpHook class."""

    def test_pre_forward_returns_none(self):
        """Test that pre_forward always returns None."""
        hook = NoOpHook()
        x = torch.randn(2, 4, 8, 8)

        result = hook.pre_forward(x, cond_flag=True)
        assert result is None

    def test_post_forward_does_nothing(self):
        """Test that post_forward does nothing."""
        hook = NoOpHook()
        output = torch.randn(2, 4, 8, 8)
        ori_input = torch.randn(2, 4, 8, 8)

        # Should not raise
        hook.post_forward(output, ori_input, cond_flag=True)


class TestAdaTaylorCacheHook:
    """Test AdaTaylorCacheHook class."""

    def test_initialization_with_params(self):
        """Test hook initialization with parameters."""
        hook = AdaTaylorCacheHook(
            model_type="Wan2.1-T2V-1.3B",
            num_inference_steps=10,
            n_derivatives=1,
            taylor_threshold=2,
        )

        assert hook.ada_taylor_cache is not None
        assert hook.ada_taylor_cache.num_inference_steps == 10

    def test_initialization_with_cache_instance(self):
        """Test hook initialization with AdaTaylorCache instance."""
        cache = AdaTaylorCache(
            model_type="Wan2.1-T2V-1.3B",
            num_inference_steps=10,
        )
        hook = AdaTaylorCacheHook(ada_taylor_cache=cache)

        assert hook.ada_taylor_cache is cache

    def test_initialization_without_required_params(self):
        """Test that initialization fails without required params."""
        with pytest.raises(ValueError):
            AdaTaylorCacheHook(model_type="test")

    def test_mark_step_begin_increments_step(self):
        """Test that mark_step_begin increments step counter."""
        hook = AdaTaylorCacheHook(
            model_type="Wan2.1-T2V-1.3B",
            num_inference_steps=10,
        )

        assert hook.ada_taylor_cache.current_step == -1

        hook.mark_step_begin(cond_flag=True)
        assert hook.ada_taylor_cache.current_step == 0

        # Should not increment on uncond pass
        hook.mark_step_begin(cond_flag=False)
        assert hook.ada_taylor_cache.current_step == 0

    def test_pre_forward_returns_none_for_compute_step(self):
        """Test that pre_forward returns None when should compute."""
        hook = AdaTaylorCacheHook(
            model_type="Wan2.1-T2V-1.3B",
            num_inference_steps=10,
        )

        hook.mark_step_begin(cond_flag=True)
        x = torch.randn(2, 4, 8, 8)

        result = hook.pre_forward(x, cond_flag=True)
        # First step should compute (returns None)
        assert result is None

    def test_post_forward_stores_data(self):
        """Test that post_forward stores residual data."""
        hook = AdaTaylorCacheHook(
            model_type="Wan2.1-T2V-1.3B",
            num_inference_steps=10,
        )

        hook.mark_step_begin(cond_flag=True)
        output = torch.randn(2, 4, 8, 8)
        ori_input = torch.randn(2, 4, 8, 8)

        hook.post_forward(output, ori_input, cond_flag=True)

        assert hook.ada_taylor_cache.cond_state.last_residual is not None


class TestAdaTaylorCacheCalibrator:
    """Test AdaTaylorCacheCalibrator class."""

    @pytest.fixture
    def temp_output_path(self):
        """Create a temporary output path for testing."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_initialization(self, temp_output_path):
        """Test calibrator initialization."""
        calibrator = AdaTaylorCacheCalibrator(
            num_inference_steps=5,
            sigma_shift=8.0,
            model_name="TestModel",
            output_path=temp_output_path,
        )

        assert calibrator.num_inference_steps == 5
        assert calibrator.sigma_shift == 8.0
        assert calibrator.model_name == "TestModel"

    def test_store_cond(self, temp_output_path):
        """Test storing for cond path."""
        calibrator = AdaTaylorCacheCalibrator(
            num_inference_steps=5,
            sigma_shift=8.0,
            model_name="TestModel",
            output_path=temp_output_path,
        )

        x = torch.randn(2, 4, 8, 8)
        ori_x = torch.randn(2, 4, 8, 8)

        calibrator.store(x, ori_x, is_cond=True)
        assert calibrator.cond_calibrator.cnt == 1

    def test_store_uncond(self, temp_output_path):
        """Test storing for uncond path."""
        calibrator = AdaTaylorCacheCalibrator(
            num_inference_steps=5,
            sigma_shift=8.0,
            model_name="TestModel",
            output_path=temp_output_path,
        )

        x = torch.randn(2, 4, 8, 8)
        ori_x = torch.randn(2, 4, 8, 8)

        calibrator.store(x, ori_x, is_cond=False)
        assert calibrator.uncond_calibrator.cnt == 1

    def test_save_produces_valid_json(self, temp_output_path):
        """Test that save produces valid JSON output."""
        calibrator = AdaTaylorCacheCalibrator(
            num_inference_steps=5,
            sigma_shift=8.0,
            model_name="TestModel",
            output_path=temp_output_path,
        )

        # Simulate running through all steps
        for _ in range(5):
            x = torch.randn(2, 4, 8, 8)
            ori_x = torch.randn(2, 4, 8, 8)
            calibrator.store(x, ori_x, is_cond=True)
            calibrator.store(x, ori_x, is_cond=False)

        # Check file exists
        assert Path(temp_output_path).exists()

        # Check JSON is valid
        with open(temp_output_path) as f:
            data = json.load(f)

        assert "K" in data
        assert "thresh" in data
        assert "retention_ratio" in data
        assert "cond_mag_ratios" in data
        assert "uncond_mag_ratios" in data

    def test_default_parameters_are_smart(self, temp_output_path):
        """Test that default parameters are calculated smartly."""
        calibrator = AdaTaylorCacheCalibrator(
            num_inference_steps=50,
            sigma_shift=8.0,
            model_name="TestModel",
            output_path=temp_output_path,
        )

        # Simulate running through all steps
        for _ in range(50):
            x = torch.randn(2, 4, 8, 8)
            ori_x = torch.randn(2, 4, 8, 8)
            calibrator.store(x, ori_x, is_cond=True)
            calibrator.store(x, ori_x, is_cond=False)

        with open(temp_output_path) as f:
            data = json.load(f)

        # K should be num_steps // 10 (but at least 1, at most 4)
        assert data["K"] >= 1
        assert data["K"] <= 4

        # retention_ratio should be 0.2
        assert data["retention_ratio"] == 0.2

        # thresh should be 0.12
        assert data["thresh"] == 0.12


class TestHybridStrategy:
    """Test the hybrid Taylor/residual reuse strategy."""

    def test_taylor_for_small_elapsed(self):
        """Test Taylor expansion is used for small elapsed."""
        mag_ratios = np.ones(10)
        state = AdaTaylorCacheState(
            num_inference_steps=10,
            thresh=0.1,
            K=2,
            mag_ratios=mag_ratios,
            retention_ratio=0.1,
            n_derivatives=1,
            taylor_threshold=3,
        )

        # Store initial residual
        output = torch.randn(2, 4, 8, 8)
        ori_input = torch.randn(2, 4, 8, 8)
        state.set_current_step(0)
        state.update(output, ori_input)

        # Move to step 2 (elapsed=2, within threshold)
        state.set_current_step(2)
        approx = state.approximate_residual()

        # Should use Taylor (not just last_residual)
        assert approx is not None

    def test_residual_reuse_for_large_elapsed(self):
        """Test residual reuse for large elapsed."""
        mag_ratios = np.ones(10)
        state = AdaTaylorCacheState(
            num_inference_steps=10,
            thresh=0.1,
            K=2,
            mag_ratios=mag_ratios,
            retention_ratio=0.1,
            n_derivatives=1,
            taylor_threshold=1,  # Low threshold
        )

        # Store initial residual
        output = torch.randn(2, 4, 8, 8)
        ori_input = torch.randn(2, 4, 8, 8)
        state.set_current_step(0)
        state.update(output, ori_input)

        # Move to step 2 (elapsed=2, above threshold)
        state.set_current_step(2)
        approx = state.approximate_residual()

        # Should use cached residual
        assert torch.allclose(approx, state.last_residual)


class TestIntegration:
    """Integration tests for feature_cache module."""

    @pytest.fixture
    def mock_params_file(self, tmp_path):
        """Create a mock params file for testing."""
        params = {
            "K": 2,
            "thresh": 0.1,
            "retention_ratio": 0.2,
            "cond_mag_ratios": [1.0] * 10,
            "uncond_mag_ratios": [1.0] * 10,
        }
        params_file = tmp_path / "TestModel.json"
        with open(params_file, "w") as f:
            json.dump(params, f)
        return params_file

    def test_full_workflow(self, mock_params_file):
        """Test complete workflow from initialization to approximation."""
        # Initialize with mock model type
        with patch("telefuser.feature_cache.ada_taylor_cache.ada_taylor_cache.load_cache_params") as mock_load:
            mock_load.return_value = {
                "K": 2,
                "thresh": 0.1,
                "retention_ratio": 0.2,
                "cond_mag_ratios": [1.0] * 10,
                "uncond_mag_ratios": [1.0] * 10,
            }

            cache = AdaTaylorCache(
                model_type="TestModel",
                num_inference_steps=10,
                n_derivatives=1,
                taylor_threshold=2,
            )

            # Simulate CFG inference
            for step in range(10):
                cache.mark_step_begin()

                # Cond path
                x_cond = torch.randn(2, 4, 8, 8)
                if not cache.should_skip(is_cond=True):
                    output_cond = x_cond + torch.randn(2, 4, 8, 8) * 0.1
                    cache.store(output_cond, x_cond, is_cond=True)

                # Uncond path
                x_uncond = torch.randn(2, 4, 8, 8)
                if not cache.should_skip(is_cond=False):
                    output_uncond = x_uncond + torch.randn(2, 4, 8, 8) * 0.1
                    cache.store(output_uncond, x_uncond, is_cond=False)

            # Verify cache worked correctly
            compute_steps = cache.get_compute_steps()
            assert len(compute_steps) > 0
            assert len(compute_steps) <= 10
