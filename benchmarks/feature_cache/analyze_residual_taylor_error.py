# ruff: noqa
"""Analyze residual Taylor approximation errors for AdaTaylorCache.

This script collects input/output data at each diffusion step and analyzes
the approximation error of 0th, 1st, and 2nd order Taylor series on residuals.

Usage:
    python analyze_residual_taylor_error.py --seed 42 --num_inference_steps 40
"""

import json
import click
import torch
import numpy as np
from telefuser.utils.logging import logger

from telefuser.core.config import AttentionConfig, AttnImplType, FeatureCacheConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)
from telefuser.utils.video import get_target_video_size_from_ratio


class ResidualAnalyzer:
    """Collects and analyzes residual data for Taylor approximation analysis."""

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.cond_data = {
            'inputs': [],      # List of input tensors
            'outputs': [],     # List of output tensors
            'residuals': [],   # List of residual tensors (output - input)
        }
        self.uncond_data = {
            'inputs': [],
            'outputs': [],
            'residuals': [],
        }
        self.current_step = 0
        self.is_cond = True  # Track if we're in cond or uncond path

    def reset(self):
        """Reset for new run."""
        self.cond_data = {'inputs': [], 'outputs': [], 'residuals': []}
        self.uncond_data = {'inputs': [], 'outputs': [], 'residuals': []}
        self.current_step = 0
        self.is_cond = True

    def store(self, output: torch.Tensor, ori_input: torch.Tensor, is_cond: bool = True):
        """Store input/output pair for analysis."""
        data = self.cond_data if is_cond else self.uncond_data
        residual = output - ori_input

        # Convert to float32 for stable analysis and move to CPU
        data['inputs'].append(ori_input.detach().float().cpu())
        data['outputs'].append(output.detach().float().cpu())
        data['residuals'].append(residual.detach().float().cpu())

    def compute_taylor_approximation(self, residuals: list, step: int, order: int, window: int) -> torch.Tensor:
        """
        Compute Taylor approximation of residual at given step.

        IMPORTANT: We use residuals from steps [ref_step, ref_step+1, ...] to predict
        residual at step (ref_step + window + 1). This ensures we're doing actual
        prediction, not interpolation.

        For example:
        - window=1: Use steps 0,1 to predict step 2
        - window=2: Use steps 0,1 to predict step 3 (with order 1) or steps 0,1,2 to predict step 3 (with order 2)

        Args:
            residuals: List of residual tensors
            step: The step we want to predict (this is ref_step + window + delta)
            order: Taylor order (0, 1, or 2)
            window: Number of steps we're skipping (distance from last known point)

        Returns:
            Approximated residual tensor, or None if prediction not possible
        """
        # We need at least window+1 data points to make a prediction
        # For step being predicted, we need residuals[0] through residuals[window] available
        # And step must equal window + 1 for proper out-of-sample prediction

        if step < window + 1 or len(residuals) <= step:
            return None

        # Reference step is 0 for simplicity (we're always predicting from the start of a window)
        ref_step = 0

        if ref_step >= len(residuals):
            return None

        R_0 = residuals[ref_step]  # 0th derivative (value itself)

        if order == 0:
            # Zero-order: just use R(0)
            elapsed = step - ref_step
            return R_0.clone()

        # Need at least 2 points for 1st derivative
        if ref_step + 1 >= len(residuals):
            return R_0.clone()

        # Compute 1st derivative using consecutive residuals
        dR = (residuals[ref_step + 1] - residuals[ref_step])  # dR/dt (per-step change)

        if order == 1:
            # R(t) = R(0) + dR/dt * t
            elapsed = step - ref_step
            return R_0 + dR * elapsed

        # Need at least 3 points for 2nd derivative
        if ref_step + 2 >= len(residuals):
            # Fall back to 1st order
            elapsed = step - ref_step
            return R_0 + dR * elapsed

        # Compute 2nd derivative
        dR_1 = (residuals[ref_step + 1] - residuals[ref_step])
        dR_2 = (residuals[ref_step + 2] - residuals[ref_step + 1])
        d2R = dR_2 - dR_1  # d²R/dt² (change in derivative)

        if order == 2:
            # R(t) = R(0) + dR/dt * t + d²R/dt² * t²/2
            elapsed = step - ref_step
            return R_0 + dR * elapsed + d2R * (elapsed ** 2) / 2

        return None

    def compute_approximation_errors(self, window: int = 1):
        """
        Compute approximation errors for different Taylor orders.

        This simulates the actual behavior of AdaTaylorCache:
        - We compute at steps 0, 1, ..., window, then skip step (window+1) and predict it
        - The prediction uses residuals from steps 0, 1, ... to build Taylor approximation

        Args:
            window: Number of consecutive compute steps before skipping

        Returns:
            Dictionary with error statistics for each order
        """
        results = {}

        for path_name, data in [('cond', self.cond_data), ('uncond', self.uncond_data)]:
            residuals = data['residuals']
            if len(residuals) < window + 2:
                continue

            errors = {
                'order_0': [],  # Zero-order (constant)
                'order_1': [],  # First-order (linear)
                'order_2': [],  # Second-order (quadratic)
            }

            # Simulate the caching behavior:
            # - Compute at steps 0, 1, ..., window (warmup + compute)
            # - Skip and predict step (window+1)
            # - Then reset and continue

            # We analyze prediction of step (window+1) using data from steps 0 to window
            step_to_predict = window + 1

            if step_to_predict < len(residuals):
                actual_residual = residuals[step_to_predict]

                # Compute approximations using data from steps 0 to window
                for order in [0, 1, 2]:
                    approx = self.compute_taylor_approximation(residuals, step_to_predict, order, window)

                    if approx is not None:
                        # Compute relative error
                        error = (approx - actual_residual).norm() / (actual_residual.norm() + 1e-8)
                        errors[f'order_{order}'].append(error.item())
                    else:
                        errors[f'order_{order}'].append(None)

            # Also analyze predictions for multiple windows (sliding window analysis)
            # For steps after the first prediction window
            for start_step in range(window + 2, len(residuals) - 1, window + 1):
                step_to_predict = start_step + 1

                if step_to_predict >= len(residuals):
                    break

                actual_residual = residuals[step_to_predict]

                # Reference data is from start_step - window to start_step
                ref_residuals = residuals[start_step - window:start_step + 1]

                for order in [0, 1, 2]:
                    approx = self._predict_from_reference(ref_residuals, order, window + 1)

                    if approx is not None:
                        error = (approx - actual_residual).norm() / (actual_residual.norm() + 1e-8)
                        errors[f'order_{order}'].append(error.item())

            results[path_name] = errors

        return results

    def _predict_from_reference(self, ref_residuals: list, order: int, elapsed: int) -> torch.Tensor:
        """
        Predict residual using Taylor series from reference residuals.

        Args:
            ref_residuals: List of residual tensors from reference steps
            order: Taylor order (0, 1, or 2)
            elapsed: Number of steps to predict ahead

        Returns:
            Predicted residual tensor
        """
        if len(ref_residuals) == 0:
            return None

        R_0 = ref_residuals[0]

        if order == 0:
            return R_0.clone()

        if len(ref_residuals) < 2:
            return R_0.clone()

        dR = (ref_residuals[1] - ref_residuals[0])

        if order == 1:
            return R_0 + dR * elapsed

        if len(ref_residuals) < 3:
            return R_0 + dR * elapsed

        dR_1 = (ref_residuals[1] - ref_residuals[0])
        dR_2 = (ref_residuals[2] - ref_residuals[1])
        d2R = dR_2 - dR_1

        if order == 2:
            return R_0 + dR * elapsed + d2R * (elapsed ** 2) / 2

        return None

    def analyze_and_save(self, output_path: str):
        """Analyze and save results."""
        results = {
            'num_steps': self.num_steps,
            'cond': {},
            'uncond': {},
        }

        # Analyze for different window sizes
        for window in [1, 2, 3]:
            errors = self.compute_approximation_errors(window=window)

            for path_name in ['cond', 'uncond']:
                if path_name not in errors:
                    continue

                key = f'window_{window}'
                if key not in results[path_name]:
                    results[path_name][key] = {}

                for order_name, error_list in errors[path_name].items():
                    valid_errors = [e for e in error_list if e is not None]
                    if valid_errors:
                        results[path_name][key][order_name] = {
                            'mean': float(np.mean(valid_errors)),
                            'std': float(np.std(valid_errors)),
                            'max': float(np.max(valid_errors)),
                            'min': float(np.min(valid_errors)),
                            'count': len(valid_errors),
                        }

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\n" + "=" * 80)
        print("RESIDUAL TAYLOR APPROXIMATION ERROR ANALYSIS")
        print("=" * 80)

        for path_name in ['cond', 'uncond']:
            print(f"\n{path_name.upper()} PATH:")
            print("-" * 80)

            for window in [1, 2, 3]:
                key = f'window_{window}'
                if key not in results[path_name]:
                    continue

                print(f"\nWindow size = {window}:")
                print(f"{'Order':<10} | {'Mean Error':>15} | {'Std':>15} | {'Max':>15} | {'Min':>15}")
                print("-" * 80)

                for order in [0, 1, 2]:
                    order_name = f'order_{order}'
                    if order_name in results[path_name][key]:
                        stats = results[path_name][key][order_name]
                        print(f"Order {order:<5} | {stats['mean']:>15.6f} | {stats['std']:>15.6f} | {stats['max']:>15.6f} | {stats['min']:>15.6f}")

        print("\n" + "=" * 80)
        print(f"Results saved to: {output_path}")

        return results


# Default configuration
PPL_CONFIG = dict(
    negative_prompt="Camera shake, overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    tiled=True,
    target_fps=16,
    sample_solver="euler",
    attn_impl=AttnImplType.SAGE_ATTN_2_8_8,
    model_type="Wan2.1-T2V-1.3B",
    sigma_shift=8.0,
)


def get_pipeline(model_root="/dev/shm/Wan2.1-T2V-1.3B/", num_inference_steps=40):
    """Initialize the video generation pipeline."""
    module_manager = ModuleManager(device="cpu")
    module_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [[f"{model_root}/diffusion_pytorch_model.safetensors"]],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth"],
        torch_dtype=torch.bfloat16,
    )

    pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan21VideoPipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.enable_clip_stage = True

    pipe.init(module_manager, pipe_config)
    return pipe


def run_analysis(
    pipeline,
    prompt,
    seed=42,
    num_inference_steps=40,
    num_frames=81,
    cfg_scale=6.0,
    output_path="residual_analysis.json",
):
    """Run analysis to collect residual data and compute approximation errors."""
    width, height = get_target_video_size_from_ratio(
        "16:9",
        resolution="480p",
        height_division_factor=2,
        width_division_factor=2,
    )

    # Create analyzer
    analyzer = ResidualAnalyzer(num_inference_steps)

    # Set analyzer on the model
    pipeline.denoise_stage.dit.set_residual_analyzer(analyzer)

    logger.info(f"Starting residual analysis with {num_inference_steps} steps")

    # Run pipeline - analyzer will collect data during forward passes
    video = pipeline(
        prompt=prompt,
        negative_prompt=PPL_CONFIG['negative_prompt'],
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        cfg_scale=cfg_scale,
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
    )

    # Analyze and save results
    results = analyzer.analyze_and_save(output_path)

    return video, results


@click.command()
@click.option("--seed", default=42, help="Random seed")
@click.option("--model_root", default="/dev/shm/Wan2.1-T2V-1.3B/", help="Model root directory")
@click.option("--num_inference_steps", default=40, help="Number of inference steps")
@click.option("--num_frames", default=81, help="Number of frames")
@click.option("--cfg_scale", default=6.0, help="CFG scale")
@click.option("--output_path", default=None, help="Output path for analysis results")
@click.option(
    "--prompt",
    default="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets.",
    help="Text prompt",
)
def main(seed, model_root, num_inference_steps, num_frames, cfg_scale, output_path, prompt):
    """Analyze residual Taylor approximation errors."""
    if output_path is None:
        output_path = f"residual_analysis_seed{seed}.json"

    logger.info("=" * 60)
    logger.info("Residual Taylor Approximation Error Analysis")
    logger.info("=" * 60)

    # Create pipeline
    pipe = get_pipeline(model_root, num_inference_steps)

    # Run analysis
    video, results = run_analysis(
        pipe,
        prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        cfg_scale=cfg_scale,
        output_path=output_path,
    )

    logger.info(f"Analysis completed. Generated {len(video)} frames")

    del pipe


if __name__ == "__main__":
    main()