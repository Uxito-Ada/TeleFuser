"""Feature Cache Hooks for DiT models.

This module provides a hook-based interface for feature caching in Diffusion Transformers.
The hook pattern decouples caching logic from the model's forward method, making it
easier to add new caching strategies without modifying the model code.

Architecture:
- FeatureCacheHook: Abstract base class defining the hook interface
- FeatureCacheHookManager: Manages a single hook instance per model
- Concrete implementations: AdaTaylorCacheHook, AdaTaylorCacheCalibratorHook, etc.

Usage:
    hook = AdaTaylorCacheHook(model_type="Wan2.1-T2V-1.3B", num_steps=50)
    model.feature_cache_hook.set_hook(hook)

    # In forward:
    cached_output = model.feature_cache_hook.pre_forward(x, cond_flag)
    if cached_output is None:
        x = model.forward_blocks(...)
        model.feature_cache_hook.post_forward(x, ori_x, cond_flag)
    else:
        x = cached_output
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from .ada_taylor_cache import AdaTaylorCache, AdaTaylorCacheCalibrator


class FeatureCacheHook(ABC):
    """Abstract base class for feature cache hooks.

    Feature cache hooks intercept the forward pass of DiT blocks to enable
    caching and approximation of computed features. This allows skipping
    expensive computations when possible.

    The hook pattern consists of three methods:
    1. mark_step_begin: Called once at the start of each diffusion step
    2. pre_forward: Called before blocks forward to check if computation can be skipped
    3. post_forward: Called after blocks forward to store/update cache

    The hook operates in the context of classifier-free guidance (CFG), where
    the model processes both conditional (cond) and unconditional (uncond) paths.
    The cond_flag parameter indicates which path is being processed.
    """

    @abstractmethod
    def pre_forward(self, x: torch.Tensor, cond_flag: bool) -> Optional[torch.Tensor]:
        """Called before blocks forward.

        Args:
            x: Input tensor to DiT blocks
            cond_flag: True for conditional path, False for unconditional path

        Returns:
            - None: Should run blocks forward normally
            - Tensor: Skip blocks forward, use returned tensor as output
        """
        pass

    @abstractmethod
    def post_forward(self, output: torch.Tensor, ori_input: torch.Tensor, cond_flag: bool) -> None:
        """Called after blocks forward to store/update cache.

        Args:
            output: Output tensor from DiT blocks
            ori_input: Original input tensor (before blocks forward)
            cond_flag: True for conditional path, False for unconditional path
        """
        pass

    def mark_step_begin(self, cond_flag: bool) -> None:
        """Mark the beginning of a new diffusion step.

        Called once per step, typically on the cond pass (first pass in CFG).
        Subclasses can override this to reset step-level state.

        Args:
            cond_flag: True if this is the cond pass (used to identify first pass)
        """
        pass


class NoOpHook(FeatureCacheHook):
    """No-operation hook that always computes.

    This is the default hook that does no caching - every forward pass
    computes normally. Used when feature caching is disabled.
    """

    def pre_forward(self, x: torch.Tensor, cond_flag: bool) -> Optional[torch.Tensor]:
        """Always returns None to compute normally."""
        return None

    def post_forward(self, output: torch.Tensor, ori_input: torch.Tensor, cond_flag: bool) -> None:
        """Does nothing."""
        pass


class AdaTaylorCacheHook(FeatureCacheHook):
    """Hook wrapper for AdaTaylorCache.

    AdaTaylorCache combines adaptive skip logic with Taylor series
    approximation of residuals for efficient and accurate caching.

    When n_derivatives=0, AdaTaylorCache reduces to simple residual caching
    (equivalent to simple residual caching).

    IMPORTANT: This hook implements `mark_step_begin()` to manage a unified step
    counter that is shared between cond and uncond paths. This ensures correct
    behavior in CFG (Classifier-Free Guidance) mode where each diffusion step
    has two forward passes (cond + uncond).

    The `mark_step_begin()` method is called once at the beginning of each
    diffusion step (before cond forward), incrementing the unified step counter.

    See AdaTaylorCache documentation for algorithm details.

    Can be initialized either with an AdaTaylorCache instance or with parameters
    to create one internally.
    """

    def __init__(
        self,
        ada_taylor_cache: "AdaTaylorCache" = None,
        *,
        model_type: str = None,
        num_inference_steps: int = None,
        n_derivatives: int = 1,
        taylor_threshold: int = 2,
        init_step: int = 0,
    ):
        """Initialize AdaTaylorCache hook.

        Args:
            ada_taylor_cache: AdaTaylorCache instance to wrap (if provided, other args ignored)
            model_type: Model type for loading parameters
            num_inference_steps: Total number of inference steps
            n_derivatives: Order of Taylor series expansion (default: 1).
                Set to 0 for simple residual caching (no Taylor expansion).
            taylor_threshold: Threshold for switching to residual reuse (default: 2).
                When elapsed <= threshold: use Taylor expansion.
                When elapsed > threshold: use residual reuse.
            init_step: Initial step number (default: 0)
        """
        if ada_taylor_cache is not None:
            self.ada_taylor_cache = ada_taylor_cache
        elif model_type is not None and num_inference_steps is not None:
            from .ada_taylor_cache import AdaTaylorCache

            self.ada_taylor_cache = AdaTaylorCache(
                model_type=model_type,
                num_inference_steps=num_inference_steps,
                n_derivatives=n_derivatives,
                taylor_threshold=taylor_threshold,
                init_step=init_step,
            )
        else:
            raise ValueError("Either ada_taylor_cache or (model_type, num_inference_steps) must be provided")

    def mark_step_begin(self, cond_flag: bool) -> None:
        """Mark the beginning of a new diffusion step.

        This is called once per diffusion step, typically on the cond pass.
        It increments the unified step counter shared by cond and uncond states.

        Args:
            cond_flag: True if this is the cond pass. The step counter is only
                incremented on the cond pass to ensure each diffusion step
                increments the counter exactly once.
        """
        # Only increment step counter on cond pass (first pass in CFG)
        # This ensures each diffusion step increments the counter exactly once
        if cond_flag:
            self.ada_taylor_cache.mark_step_begin()

    def pre_forward(self, x: torch.Tensor, cond_flag: bool) -> Optional[torch.Tensor]:
        """Check if should compute or use Taylor approximation of residual.

        Args:
            x: Input tensor (used for residual approximation)
            cond_flag: True for cond path

        Returns:
            - Tensor if should use approximation
            - None if should compute
        """
        if self.ada_taylor_cache.should_skip(is_cond=cond_flag):
            return self.ada_taylor_cache.approximate(x, is_cond=cond_flag)
        return None

    def post_forward(self, output: torch.Tensor, ori_input: torch.Tensor, cond_flag: bool) -> None:
        """Update AdaTaylorCache state with computed residual.

        Args:
            output: Output tensor from blocks
            ori_input: Original input tensor
            cond_flag: True for cond path
        """
        self.ada_taylor_cache.store(output, ori_input, is_cond=cond_flag)


class AdaTaylorCacheCalibratorHook(FeatureCacheHook):
    """Hook wrapper for AdaTaylorCacheCalibrator.

    This hook is used during calibration mode to collect residual data
    for generating AdaTaylorCache parameters. It always computes (no skipping)
    and records the residuals.

    See AdaTaylorCacheCalibrator documentation for details.

    Can be initialized either with an AdaTaylorCacheCalibrator instance or with
    parameters to create one internally.
    """

    def __init__(
        self,
        calibrator: "AdaTaylorCacheCalibrator" = None,
        *,
        num_inference_steps: int = None,
        sigma_shift: float = None,
        model_name: str = None,
        output_path: str = None,
    ):
        """Initialize AdaTaylorCacheCalibrator hook.

        Args:
            calibrator: AdaTaylorCacheCalibrator instance to wrap (if provided, other args ignored)
            num_inference_steps: Number of inference steps
            sigma_shift: Sigma shift value used in the scheduler
            model_name: Model name for the output file (e.g., "Wan2.1-T2V-1.3B")
            output_path: Output path for the JSON file. If None, uses default params directory.
        """
        if calibrator is not None:
            self.calibrator = calibrator
        elif num_inference_steps is not None and sigma_shift is not None and model_name is not None:
            from .ada_taylor_cache import AdaTaylorCacheCalibrator

            self.calibrator = AdaTaylorCacheCalibrator(
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                model_name=model_name,
                output_path=output_path,
            )
        else:
            raise ValueError("Either calibrator or (num_inference_steps, sigma_shift, model_name) must be provided")

    def pre_forward(self, x: torch.Tensor, cond_flag: bool) -> Optional[torch.Tensor]:
        """Always returns None to compute normally during calibration."""
        return None

    def post_forward(self, output: torch.Tensor, ori_input: torch.Tensor, cond_flag: bool) -> None:
        """Store residual for calibration.

        Args:
            output: Output tensor from blocks
            ori_input: Original input tensor
            cond_flag: True for cond path
        """
        self.calibrator.store(output, ori_input, is_cond=cond_flag)


class ResidualAnalyzerHook(FeatureCacheHook):
    """Hook wrapper for residual analysis.

    This hook collects input/output data for analyzing Taylor approximation
    errors. It always computes (no skipping) and records the data for
    offline analysis.
    """

    def __init__(self, analyzer: Any) -> None:
        """Initialize ResidualAnalyzer hook.

        Args:
            analyzer: ResidualAnalyzer instance to wrap
        """
        self.analyzer = analyzer

    def pre_forward(self, x: torch.Tensor, cond_flag: bool) -> Optional[torch.Tensor]:
        """Always returns None to compute normally during analysis."""
        return None

    def post_forward(self, output: torch.Tensor, ori_input: torch.Tensor, cond_flag: bool) -> None:
        """Store data for analysis.

        Args:
            output: Output tensor from blocks
            ori_input: Original input tensor
            cond_flag: True for cond path
        """
        self.analyzer.store(output, ori_input, is_cond=cond_flag)


class FeatureCacheHookManager:
    """Manages feature cache hooks for a DiT model.

    This class provides a simple interface for setting and using feature
    cache hooks. Each model instance has one FeatureCacheHookManager that
    can hold a single active hook.

    The manager handles the case where no hook is set (falls back to
    normal computation) and provides a clean separation between the
    model's forward logic and the caching strategy.

    Usage:
        manager = FeatureCacheHookManager()
        manager.set_hook(AdaTaylorCacheHook(model_type="Wan2.1-T2V-1.3B", num_steps=50))

        # In forward:
        manager.mark_step_begin(cond_flag)
        cached_output = manager.pre_forward(x, cond_flag)
        if cached_output is None:
            x = self.forward_blocks(...)
            manager.post_forward(x, ori_x, cond_flag)
        else:
            x = cached_output
    """

    def __init__(self):
        """Initialize the hook manager with no active hook."""
        self._hook: Optional[FeatureCacheHook] = None

    def set_hook(self, hook: Optional[FeatureCacheHook]) -> None:
        """Set the active feature cache hook.

        Args:
            hook: The hook to use, or None to disable caching
        """
        self._hook = hook

    def get_hook(self) -> Optional[FeatureCacheHook]:
        """Get the currently active hook.

        Returns:
            The active hook, or None if no hook is set
        """
        return self._hook

    def has_hook(self) -> bool:
        """Check if a hook is currently active.

        Returns:
            True if a hook is set, False otherwise
        """
        return self._hook is not None

    def clear_hook(self) -> None:
        """Clear the active hook, disabling caching."""
        self._hook = None

    def mark_step_begin(self, cond_flag: bool) -> None:
        """Mark the beginning of a new diffusion step.

        Delegates to the active hook's mark_step_begin method.

        Args:
            cond_flag: True if this is the cond pass
        """
        if self._hook is not None:
            self._hook.mark_step_begin(cond_flag)

    def pre_forward(self, x: torch.Tensor, cond_flag: bool) -> Optional[torch.Tensor]:
        """Check if computation should be skipped.

        Delegates to the active hook's pre_forward method.

        Args:
            x: Input tensor to DiT blocks
            cond_flag: True for conditional path

        Returns:
            - None: Should compute normally
            - Tensor: Use returned tensor as output (skip computation)
        """
        if self._hook is None:
            return None
        return self._hook.pre_forward(x, cond_flag)

    def post_forward(self, output: torch.Tensor, ori_input: torch.Tensor, cond_flag: bool) -> None:
        """Store/update cache after computation.

        Delegates to the active hook's post_forward method.

        Args:
            output: Output tensor from DiT blocks
            ori_input: Original input tensor
            cond_flag: True for conditional path
        """
        if self._hook is None:
            return
        self._hook.post_forward(output, ori_input, cond_flag)
