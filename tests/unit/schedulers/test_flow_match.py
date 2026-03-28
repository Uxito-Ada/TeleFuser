"""Tests for FlowMatchScheduler."""

import math

import pytest
import torch

from telefuser.schedulers.flow_match import FlowMatchScheduler


class TestFlowMatchSchedulerInitialization:
    """Test FlowMatchScheduler initialization."""

    @pytest.mark.parametrize(
        "template,expected_fn",
        [
            ("FLUX.1", FlowMatchScheduler.set_timesteps_flux),
            ("Wan", FlowMatchScheduler.set_timesteps_wan),
            ("Qwen-Image", FlowMatchScheduler.set_timesteps_qwen_image),
            ("FLUX.2", FlowMatchScheduler.set_timesteps_flux2),
            ("Z-Image", FlowMatchScheduler.set_timesteps_z_image),
            ("LTX.2", FlowMatchScheduler.set_timesteps_ltx2),
            ("Unknown", FlowMatchScheduler.set_timesteps_flux),  # Default fallback
        ],
    )
    def test_template_initialization(self, template, expected_fn):
        """Test scheduler initialization with various templates."""
        scheduler = FlowMatchScheduler(template=template)

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.set_timesteps_fn == expected_fn

    def test_default_template_is_flux(self):
        """Test that default template is FLUX.1."""
        scheduler = FlowMatchScheduler()
        assert scheduler.set_timesteps_fn == FlowMatchScheduler.set_timesteps_flux


class TestSetTimestepsFlux:
    """Test FLUX.1 timestep schedule."""

    def test_default_parameters(self):
        """Test with default parameters."""
        sigmas, timesteps = FlowMatchScheduler.set_timesteps_flux()

        assert len(sigmas) == 100
        assert len(timesteps) == 100
        assert sigmas[-1] < sigmas[0]  # Should be decreasing

    @pytest.mark.parametrize("num_steps", [50, 25, 10])
    def test_custom_num_inference_steps(self, num_steps):
        """Test with various numbers of inference steps."""
        sigmas, timesteps = FlowMatchScheduler.set_timesteps_flux(num_inference_steps=num_steps)

        assert len(sigmas) == num_steps
        assert len(timesteps) == num_steps

    def test_denoising_strength(self):
        """Test with different denoising strengths."""
        sigmas_full, _ = FlowMatchScheduler.set_timesteps_flux(denoising_strength=1.0)
        sigmas_half, _ = FlowMatchScheduler.set_timesteps_flux(denoising_strength=0.5)

        # Full denoising should start with higher sigma
        assert sigmas_full[0] > sigmas_half[0]

    def test_custom_shift(self):
        """Test with custom shift value."""
        sigmas_default, _ = FlowMatchScheduler.set_timesteps_flux(shift=3)
        sigmas_custom, _ = FlowMatchScheduler.set_timesteps_flux(shift=5)

        # Different shifts should produce different schedules
        assert not torch.allclose(sigmas_default, sigmas_custom)

    def test_sigma_range(self):
        """Test sigma values are in expected range."""
        sigmas, _ = FlowMatchScheduler.set_timesteps_flux()

        sigma_min = 0.003 / 1.002
        sigma_max = 1.0

        assert sigmas[-1] >= sigma_min
        assert sigmas[0] <= sigma_max

    def test_timesteps_calculation(self):
        """Test timesteps are calculated from sigmas."""
        sigmas, timesteps = FlowMatchScheduler.set_timesteps_flux()

        expected_timesteps = sigmas * 1000
        assert torch.allclose(timesteps, expected_timesteps)


class TestSetTimestepsWan:
    """Test Wan timestep schedule."""

    def test_default_parameters(self):
        """Test with default parameters."""
        sigmas, timesteps = FlowMatchScheduler.set_timesteps_wan()

        assert len(sigmas) == 100
        assert len(timesteps) == 100

    def test_sigma_range(self):
        """Test sigma starts at 1.0 and ends near 0."""
        sigmas, _ = FlowMatchScheduler.set_timesteps_wan(num_inference_steps=100)

        assert sigmas[0] == 1.0
        assert sigmas[-1] < 0.1  # Should end close to 0


class TestSetTimestepsQwenImage:
    """Test Qwen-Image timestep schedule."""

    def test_default_parameters(self):
        """Test with default parameters."""
        sigmas, timesteps = FlowMatchScheduler.set_timesteps_qwen_image()

        assert len(sigmas) == 100
        assert len(timesteps) == 100

    def test_exponential_shift_mu(self):
        """Test with explicit exponential shift mu."""
        sigmas_custom, _ = FlowMatchScheduler.set_timesteps_qwen_image(exponential_shift_mu=1.0)
        sigmas_default, _ = FlowMatchScheduler.set_timesteps_qwen_image(exponential_shift_mu=0.8)

        assert not torch.allclose(sigmas_custom, sigmas_default)

    def test_dynamic_shift_len(self):
        """Test with dynamic shift length."""
        sigmas, _ = FlowMatchScheduler.set_timesteps_qwen_image(
            dynamic_shift_len=1024,  # 32x32 image
            base_shift=0.5,
            max_shift=0.9,
        )

        assert len(sigmas) == 100

    def test_calculate_shift_qwen_image(self):
        """Test _calculate_shift_qwen_image method."""
        mu = FlowMatchScheduler._calculate_shift_qwen_image(
            image_seq_len=256, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9
        )

        # For base_seq_len, should return base_shift
        assert mu == 0.5

    def test_shift_terminal(self):
        """Test shift_terminal parameter."""
        sigmas_no_shift, _ = FlowMatchScheduler.set_timesteps_qwen_image(shift_terminal=None)
        sigmas_with_shift, _ = FlowMatchScheduler.set_timesteps_qwen_image(shift_terminal=0.02)

        # With terminal shift, last sigma should be different
        assert not torch.allclose(sigmas_no_shift[-1], sigmas_with_shift[-1])


class TestComputeEmpiricalMu:
    """Test compute_empirical_mu method."""

    @pytest.mark.parametrize("seq_len", [256, 1024, 5000])
    def test_empirical_mu_computation(self, seq_len):
        """Test mu computation for various sequence lengths."""
        mu = FlowMatchScheduler.compute_empirical_mu(image_seq_len=seq_len, num_steps=50)

        assert isinstance(mu, float)
        assert mu > 0

    def test_boundary_seq_len(self):
        """Test at boundary (4300)."""
        mu_4299 = FlowMatchScheduler.compute_empirical_mu(4299, 50)
        mu_4300 = FlowMatchScheduler.compute_empirical_mu(4300, 50)
        mu_4301 = FlowMatchScheduler.compute_empirical_mu(4301, 50)

        # 4300 and 4301 should use different formulas
        assert mu_4299 != mu_4300 or mu_4300 != mu_4301


class TestSetTimestepsZImage:
    """Test Z-Image timestep schedule."""

    def test_default_parameters(self):
        """Test with default parameters."""
        sigmas, timesteps = FlowMatchScheduler.set_timesteps_z_image()

        assert len(sigmas) == 100
        assert len(timesteps) == 100

    def test_target_timesteps(self):
        """Test with target timesteps."""
        target = torch.tensor([500, 300, 100], dtype=torch.float32)
        sigmas, timesteps = FlowMatchScheduler.set_timesteps_z_image(target_timesteps=target)

        # Check that target timesteps are in the result
        for t in target:
            assert any(torch.isclose(timesteps, t, atol=1.0))


class TestSetTimestepsLTX:
    """Test LTX timestep schedule."""

    def test_default_parameters(self):
        """Test default LTX schedule shape."""
        sigmas, timesteps = FlowMatchScheduler.set_timesteps_ltx()

        assert len(sigmas) == 101
        assert len(timesteps) == 100
        assert sigmas[0] == 1.0
        assert sigmas[-1] == 0.0

    def test_latent_token_count_changes_schedule(self):
        """Test that token-aware shift changes the sigma schedule."""
        latent_small = torch.zeros(1, 128, 4, 16, 16)
        latent_large = torch.zeros(1, 128, 8, 32, 32)

        sigmas_small, _ = FlowMatchScheduler.set_timesteps_ltx(latent=latent_small)
        sigmas_large, _ = FlowMatchScheduler.set_timesteps_ltx(latent=latent_large)

        assert not torch.allclose(sigmas_small, sigmas_large)


class TestSchedulerMethods:
    """Test scheduler methods."""

    @pytest.fixture
    def scheduler(self):
        return FlowMatchScheduler("FLUX.1")

    def test_set_timesteps(self, scheduler):
        """Test set_timesteps method."""
        scheduler.set_timesteps(num_inference_steps=50)

        assert hasattr(scheduler, "sigmas")
        assert hasattr(scheduler, "timesteps")
        assert len(scheduler.sigmas) == 50
        assert len(scheduler.timesteps) == 50

    def test_set_timesteps_training(self, scheduler):
        """Test set_timesteps with training mode."""
        scheduler.set_timesteps(num_inference_steps=50, training=True)

        assert scheduler.training is True
        assert hasattr(scheduler, "linear_timesteps_weights")

    def test_set_training_weight(self, scheduler):
        """Test set_training_weight method."""
        scheduler.set_timesteps(num_inference_steps=50)
        scheduler.set_training_weight()

        assert len(scheduler.linear_timesteps_weights) == 50
        assert scheduler.linear_timesteps_weights.sum() > 0

    @pytest.mark.parametrize("to_final", [False, True])
    def test_step(self, scheduler, to_final):
        """Test step method with and without to_final."""
        scheduler.set_timesteps(num_inference_steps=10)

        model_output = torch.randn(2, 4, 8, 8)
        sample = torch.randn(2, 4, 8, 8)
        timestep = scheduler.timesteps[0] if not to_final else scheduler.timesteps[5]

        prev_sample = scheduler.step(model_output, timestep, sample, to_final=to_final)

        assert prev_sample.shape == sample.shape

    def test_return_to_timestep(self, scheduler):
        """Test return_to_timestep method."""
        scheduler.set_timesteps(num_inference_steps=10)

        sample = torch.randn(2, 4, 8, 8)
        sample_stablized = torch.randn(2, 4, 8, 8)
        timestep = scheduler.timesteps[0]

        model_output = scheduler.return_to_timestep(timestep, sample, sample_stablized)

        assert model_output.shape == sample.shape

    def test_add_noise(self, scheduler):
        """Test add_noise method."""
        scheduler.set_timesteps(num_inference_steps=10)

        original = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        timestep = scheduler.timesteps[0]

        noisy = scheduler.add_noise(original, noise, timestep)

        assert noisy.shape == original.shape

    def test_add_noise_at_zero(self, scheduler):
        """Test add_noise at timestep 0 returns original."""
        scheduler.set_timesteps(num_inference_steps=10)

        original = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)

        # Find timestep closest to 0
        timestep = scheduler.timesteps[-1]

        noisy = scheduler.add_noise(original, noise, timestep)

        # Should be close to original
        assert torch.allclose(noisy, original, atol=0.1)

    def test_training_target(self, scheduler):
        """Test training_target method."""
        sample = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        timestep = torch.tensor(500.0)

        target = scheduler.training_target(sample, noise, timestep)

        assert target.shape == sample.shape
        expected = noise - sample
        assert torch.allclose(target, expected)

    def test_training_weight(self, scheduler):
        """Test training_weight method."""
        scheduler.set_timesteps(num_inference_steps=10, training=True)

        timestep = scheduler.timesteps[0]
        weight = scheduler.training_weight(timestep)

        assert isinstance(weight, torch.Tensor)
        assert weight.numel() == 1

    def test_ltx_step_matches_euler_diffusion_step(self):
        """Test LTX step uses denoised-sample Euler update semantics."""
        scheduler = FlowMatchScheduler("LTX.2")
        scheduler.set_timesteps(num_inference_steps=4)

        sample = torch.tensor([2.0], dtype=torch.float32)
        denoised_sample = torch.tensor([0.5], dtype=torch.float32)
        timestep = scheduler.timesteps[0]
        sigma = scheduler.sigmas[0]
        sigma_next = scheduler.sigmas[1]
        expected = sample + ((sample - denoised_sample) / sigma) * (sigma_next - sigma)

        prev_sample = scheduler.step(denoised_sample, timestep, sample)

        assert torch.allclose(prev_sample, expected)
