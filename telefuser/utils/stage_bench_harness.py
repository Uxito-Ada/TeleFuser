"""
Stage Bench Harness for Isolated Layer 2 Profiling.

This module provides a framework for profiling individual pipeline stages
in isolation, using I/O signatures captured during Layer 1 profiling.

Usage:
    # From CLI
    python -m telefuser.utils.stage_bench_harness \\
        --signature profiler_output/timing_io_signature.json \\
        --stage denoise \\
        --warmup 1 --profile_steps 1

    # Programmatically
    from telefuser.utils.stage_bench_harness import StageBenchHarness

    harness = StageBenchHarness.from_signature_file(
        "profiler_output/timing_io_signature.json",
        stage_name="denoise",
    )
    harness.setup()
    harness.profile(warmup=1, profile_steps=1)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.utils.profiler import (
    _DEVICE_ACTIVITY_MAP,
    StageIOSignature,
    TensorSignature,
    get_profiler_output_dir,
)


@dataclass
class HarnessConfig:
    """Configuration for stage bench harness."""

    warmup: int = 1
    profile_steps: int = 1
    output_dir: str | None = None  # Default: use get_profiler_output_dir()
    capture_output: bool = False  # Whether to capture and check output tensor

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = str(get_profiler_output_dir())


class StageBenchHarness:
    """
    Harness for isolated stage profiling.

    Takes a stage's I/O signature and provides:
    - Mock input tensor construction
    - Single-step execution for iterative stages
    - torch.profiler integration with warmup
    """

    def __init__(
        self,
        stage_name: str,
        io_signature: StageIOSignature,
        stage_instance: Any = None,
        config: HarnessConfig | None = None,
    ):
        """
        Initialize the harness.

        Args:
            stage_name: Name of the stage to profile
            io_signature: I/O signature from Layer 1 profiling
            stage_instance: Optional pre-loaded stage instance
            config: Harness configuration
        """
        self.stage_name = stage_name
        self.io_signature = io_signature
        self.stage_instance = stage_instance
        self.config = config or HarnessConfig()

        self._mock_inputs: dict[str, Any] = {}
        self._single_step_fn: Callable | None = None

    @classmethod
    def from_signature_file(
        cls,
        signature_path: str,
        stage_name: str,
        stage_instance: Any = None,
        config: HarnessConfig | None = None,
    ) -> "StageBenchHarness":
        """
        Create harness from a signature JSON file.

        Args:
            signature_path: Path to the I/O signature JSON file
            stage_name: Name of the stage to profile
            stage_instance: Optional pre-loaded stage instance
            config: Harness configuration

        Returns:
            StageBenchHarness instance
        """
        with open(signature_path, encoding="utf-8") as f:
            data = json.load(f)

        stages_data = data.get("stages", {})
        if stage_name not in stages_data:
            raise ValueError(f"Stage '{stage_name}' not found in signature file. Available: {list(stages_data.keys())}")

        sig_data = stages_data[stage_name]
        io_signature = cls._parse_signature_data(stage_name, sig_data)

        return cls(
            stage_name=stage_name,
            io_signature=io_signature,
            stage_instance=stage_instance,
            config=config,
        )

    @staticmethod
    def _parse_signature_data(stage_name: str, data: dict) -> StageIOSignature:
        """Parse signature data from JSON."""
        input_signatures = {}
        for key, value in data.get("input_signatures", {}).items():
            if isinstance(value, dict) and "shape" in value:
                input_signatures[key] = TensorSignature(
                    shape=tuple(value["shape"]),
                    dtype=value["dtype"],
                    device=value["device"],
                    requires_grad=value.get("requires_grad", False),
                )
            else:
                input_signatures[key] = value

        output_signature = None
        if data.get("output_signature"):
            out_data = data["output_signature"]
            output_signature = TensorSignature(
                shape=tuple(out_data["shape"]),
                dtype=out_data["dtype"],
                device=out_data["device"],
                requires_grad=out_data.get("requires_grad", False),
            )

        return StageIOSignature(
            stage_name=stage_name,
            input_signatures=input_signatures,
            output_signature=output_signature,
            metadata=data.get("metadata", {}),
        )

    def setup(self) -> None:
        """Setup the harness: create mock inputs and prepare stage."""
        self._mock_inputs = self._create_mock_inputs()

        if self.stage_instance is not None:
            self._setup_single_step_fn()

        logger.info(f"[Harness] Setup complete for stage '{self.stage_name}'")
        logger.info(f"[Harness] Mock inputs: {list(self._mock_inputs.keys())}")

    def _create_mock_inputs(self) -> dict[str, Any]:
        """Create mock input tensors based on I/O signature."""
        inputs = {}

        for name, sig in self.io_signature.input_signatures.items():
            if isinstance(sig, TensorSignature):
                # Create random tensor with matching shape/dtype/device
                dtype = getattr(torch, sig.dtype, torch.float32)
                device = sig.device

                # Use randn for normal distribution (typical for latents/embeddings)
                tensor = torch.randn(sig.shape, dtype=dtype, device=device)
                inputs[name] = tensor

                logger.debug(f"[Harness] Created mock input '{name}': {sig.shape}, {sig.dtype}, {sig.device}")

            elif isinstance(sig, list) and sig and isinstance(sig[0], TensorSignature):
                # List of tensors
                inputs[name] = [torch.randn(s.shape, dtype=getattr(torch, s.dtype), device=s.device) for s in sig]
            else:
                # Non-tensor parameter (int, float, str, None)
                inputs[name] = sig
                logger.debug(f"[Harness] Using non-tensor input '{name}': {sig}")

        return inputs

    def _setup_single_step_fn(self) -> None:
        """Setup single-step function for iterative stages like DiT."""
        # Check for DiT stage pattern: has dit and scheduler
        if hasattr(self.stage_instance, "dit") and hasattr(self.stage_instance, "scheduler"):
            self._single_step_fn = self._create_dit_single_step()
            logger.info("[Harness] Created DiT single-step function")
            return

        # Default: use process method as-is
        if hasattr(self.stage_instance, "process"):
            self._single_step_fn = self.stage_instance.process
            logger.info("[Harness] Using stage.process() as single-step function")

    def _create_dit_single_step(self) -> Callable:
        """Create a single-step function for DiT stages."""
        stage = self.stage_instance

        def dit_single_step(**kwargs):
            """Run a single denoising step."""
            import torch

            # Get inputs
            latents = kwargs.get("latents")
            prompt_emb_posi = kwargs.get("prompt_emb_posi")
            prompt_emb_nega = kwargs.get("prompt_emb_nega")
            clip_feature = kwargs.get("clip_feature")
            cfg_scale = kwargs.get("cfg_scale", 5.0)
            ref_latent = kwargs.get("ref_latent")
            sigma_shift = kwargs.get("sigma_shift", 8.0)

            # Setup scheduler with minimal steps
            num_steps = 2
            stage.scheduler.set_timesteps(num_steps, shift=sigma_shift)

            # Take first timestep
            timestep = stage.scheduler.timesteps[0:1].to(dtype=stage.torch_dtype, device=stage.device)

            # Prepare input
            input_latent = latents
            if ref_latent is not None:
                input_latent = torch.cat([input_latent, ref_latent], dim=1)

            # Create sparse_state if radial attention is enabled
            sparse_state = None
            if hasattr(stage.dit, "sparse_attention_state"):
                sparse_state = stage.dit.create_sparse_state(
                    numeral_timestep=num_steps - 1,
                    layer_idx=0,
                )

            with torch.autocast(device_type=stage.device_type, dtype=stage.torch_dtype):
                input_latent = input_latent.to(stage.torch_dtype)
                noise_pred = stage.predict_noise_with_cfg(
                    latents=input_latent,
                    timestep=timestep,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    clip_feature=clip_feature,
                    cfg_scale=cfg_scale,
                    sparse_state=sparse_state,
                )

                if noise_pred is not None:
                    latents = stage.scheduler.step(noise_pred, stage.scheduler.timesteps[0], latents)

            return latents

        return dit_single_step

    def generate_test_script(self) -> str:
        """Generate a complete, executable test script for reproducing the profiling."""
        lines = [
            "#!/usr/bin/env python",
            '"""Auto-generated Layer 2 profiling script.',
            "",
            f"Stage: {self.stage_name}",
            f"Generated from: {self.io_signature.stage_name} I/O signature",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "import time",
            "import json",
            "from pathlib import Path",
            "",
            "import torch",
            "from torch.profiler import profile, ProfilerActivity",
            "",
            "# Note: You need to load the stage instance before running this script.",
            "# Example:",
            "# from my_pipeline import get_pipeline",
            "# pipeline = get_pipeline()",
            f"# stage = pipeline.{self.stage_name}_stage",
            "",
            "",
            "def create_inputs():",
            '    """Create mock inputs based on I/O signature."""',
        ]

        # Generate tensor creation code inside create_inputs()
        for name, sig in self.io_signature.input_signatures.items():
            if isinstance(sig, TensorSignature):
                dtype = f"torch.{sig.dtype}"
                shape_str = str(list(sig.shape))
                lines.append(f'    {name} = torch.randn({shape_str}, dtype={dtype}, device="{sig.device}")')
            elif sig is None:
                lines.append(f"    {name} = None")
            else:
                lines.append(f"    {name} = {repr(sig)}")

        lines.extend(
            [
                "",
                "    return {",
            ]
        )
        for name in self.io_signature.input_signatures.keys():
            lines.append(f'        "{name}": {name},')
        lines.extend(
            [
                "    }",
                "",
                "",
            ]
        )

        # Generate single-step function for DiT stages
        is_dit_stage = hasattr(self.stage_instance, "dit") and hasattr(self.stage_instance, "scheduler")
        if is_dit_stage:
            lines.extend(
                [
                    "def run_single_step(stage, inputs: dict) -> torch.Tensor:",
                    '    """Run a single denoising step for DiT stage."""',
                    '    latents = inputs["latents"]',
                    '    prompt_emb_posi = inputs["prompt_emb_posi"]',
                    '    prompt_emb_nega = inputs["prompt_emb_nega"]',
                    '    clip_feature = inputs.get("clip_feature")',
                    '    cfg_scale = inputs.get("cfg_scale", 5.0)',
                    '    ref_latent = inputs.get("ref_latent")',
                    '    sigma_shift = inputs.get("sigma_shift", 8.0)',
                    "",
                    "    # Setup scheduler with minimal steps",
                    "    num_steps = 2",
                    "    stage.scheduler.set_timesteps(num_steps, shift=sigma_shift)",
                    "",
                    "    # Take first timestep",
                    "    timestep = stage.scheduler.timesteps[0:1].to(dtype=stage.torch_dtype, device=stage.device)",
                    "",
                    "    # Prepare input",
                    "    input_latent = latents",
                    "    if ref_latent is not None:",
                    "        input_latent = torch.cat([input_latent, ref_latent], dim=1)",
                    "",
                    "    # Create sparse_state if radial attention is enabled",
                    "    sparse_state = None",
                    '    if hasattr(stage.dit, "sparse_attention_state"):',
                    "        sparse_state = stage.dit.create_sparse_state(",
                    "            numeral_timestep=num_steps - 1,",
                    "            layer_idx=0,",
                    "        )",
                    "",
                    "    with torch.autocast(device_type=stage.device_type, dtype=stage.torch_dtype):",
                    "        input_latent = input_latent.to(stage.torch_dtype)",
                    "        noise_pred = stage.predict_noise_with_cfg(",
                    "            latents=input_latent,",
                    "            timestep=timestep,",
                    "            prompt_emb_posi=prompt_emb_posi,",
                    "            prompt_emb_nega=prompt_emb_nega,",
                    "            clip_feature=clip_feature,",
                    "            cfg_scale=cfg_scale,",
                    "            sparse_state=sparse_state,",
                    "        )",
                    "",
                    "        if noise_pred is not None:",
                    "            latents = stage.scheduler.step(noise_pred, stage.scheduler.timesteps[0], latents)",
                    "",
                    "    return latents",
                    "",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "def run_single_step(stage, inputs: dict) -> torch.Tensor:",
                    '    """Run a single step of the stage."""',
                    "    # For non-DiT stages, use the process method directly",
                    "    return stage.process(**inputs)",
                    "",
                    "",
                ]
            )

        # Main profiling function
        lines.extend(
            [
                'def profile_stage(stage, warmup: int = 1, profile_steps: int = 1, output_dir: str = ".") -> dict:',
                '    """Profile the stage with warmup."""',
                "    output_path = Path(output_dir)",
                "    output_path.mkdir(parents=True, exist_ok=True)",
                "",
                "    inputs = create_inputs()",
                "",
                "    # Warmup",
                '    print(f"Running {warmup} warmup iteration(s)...")',
                "    for i in range(warmup):",
                "        _ = run_single_step(stage, inputs)",
                "        torch.cuda.synchronize()",
                '        print(f"  Warmup {i + 1}/{warmup} complete")',
                "",
                "    # Clear cache",
                "    torch.cuda.empty_cache()",
                "",
                "    # Profiling",
                '    print(f"Running {profile_steps} profile iteration(s)...")',
                '    results = {"timing_ms": []}',
                "",
                "    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]",
                "    with profile(",
                "        activities=activities,",
                "        record_shapes=True,",
                "        profile_memory=True,",
                "        with_stack=True,",
                "    ) as prof:",
                "        start_time = time.perf_counter()",
                "",
                "        for i in range(profile_steps):",
                "            _ = run_single_step(stage, inputs)",
                "            torch.cuda.synchronize()",
                "            iter_time = (time.perf_counter() - start_time) * 1000",
                '            results["timing_ms"].append(iter_time)',
                "            start_time = time.perf_counter()",
                '            print(f"  Profile {i + 1}/{profile_steps}: {iter_time:.2f} ms")',
                "",
                "    # Export trace",
                f'    trace_path = output_path / "{self.stage_name}_trace.json.gz"',
                "    prof.export_chrome_trace(str(trace_path))",
                '    print(f"Chrome trace saved to: {trace_path}")',
                "",
                "    # Summary",
                '    avg_time = sum(results["timing_ms"]) / len(results["timing_ms"])',
                '    print(f"Average iteration time: {avg_time:.2f} ms")',
                "",
                "    return results",
                "",
                "",
                'if __name__ == "__main__":',
                "    # TODO: Load your stage instance here",
                "    # stage = ...",
                "",
                "    # Example usage:",
                f"    # results = profile_stage(stage, warmup={self.config.warmup}, "
                f'    #                     profile_steps={self.config.profile_steps}, output_dir="./")',
                "",
                '    print("Please load the stage instance before running profiling.")',
                '    print("See the comments at the top of this file for instructions.")',
                "",
            ]
        )

        return "\n".join(lines)

    def run_single_step(self, inputs: dict[str, Any] | None = None) -> Any:
        """
        Run a single step of the stage.

        Args:
            inputs: Input dict (uses mock inputs if not provided)

        Returns:
            Stage output
        """
        if inputs is None:
            inputs = self._mock_inputs

        if self._single_step_fn is None:
            raise RuntimeError("No single-step function available. Call setup() first.")

        return self._single_step_fn(**inputs)

    def profile(
        self,
        warmup: int | None = None,
        profile_steps: int | None = None,
        output_suffix: str = "",
    ) -> dict[str, Any]:
        """
        Run profiling with warmup.

        Args:
            warmup: Number of warmup iterations (default from config)
            profile_steps: Number of profiling iterations (default from config)
            output_suffix: Optional suffix for output files

        Returns:
            Profiling results dict
        """
        warmup = warmup or self.config.warmup
        profile_steps = profile_steps or self.config.profile_steps

        if not self._mock_inputs:
            self.setup()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Warmup
        logger.info(f"[Harness] Running {warmup} warmup iteration(s)...")
        for i in range(warmup):
            _ = self.run_single_step()
            current_platform.synchronize()
            logger.debug(f"[Harness] Warmup iteration {i + 1}/{warmup} complete")

        # Clear memory before profiling
        current_platform.empty_cache()

        # Profiling
        logger.info(f"[Harness] Running {profile_steps} profile iteration(s)...")

        activities = [torch.profiler.ProfilerActivity.CPU]
        device_activity = _DEVICE_ACTIVITY_MAP.get(current_platform.device_type)
        if device_activity:
            activities.append(device_activity)

        results = {
            "stage_name": self.stage_name,
            "warmup": warmup,
            "profile_steps": profile_steps,
            "timing_ms": [],
        }

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            start_time = time.perf_counter()

            for i in range(profile_steps):
                _ = self.run_single_step()
                current_platform.synchronize()
                iter_time = (time.perf_counter() - start_time) * 1000
                results["timing_ms"].append(iter_time)
                start_time = time.perf_counter()
                logger.debug(f"[Harness] Profile iteration {i + 1}/{profile_steps}: {iter_time:.2f} ms")

        # Export trace
        trace_path = output_dir / f"{self.stage_name}{output_suffix}_trace.json.gz"
        prof.export_chrome_trace(str(trace_path))
        logger.info(f"[Harness] Chrome trace saved to: {trace_path}")

        # Export kernel breakdown
        breakdown = self._export_kernel_breakdown(prof, output_dir, output_suffix)
        results["kernel_breakdown"] = breakdown

        # Generate and save test script
        self._save_test_script(output_dir)

        # Summary
        avg_time = sum(results["timing_ms"]) / len(results["timing_ms"])
        logger.info(f"[Harness] Average iteration time: {avg_time:.2f} ms")

        return results

    def _save_test_script(self, output_dir: Path) -> None:
        """Generate and save the test script to output directory."""
        script_content = self.generate_test_script()
        script_path = output_dir / f"profile_{self.stage_name}.py"

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        logger.info(f"[Harness] Test script saved to: {script_path}")

    def _export_kernel_breakdown(
        self,
        prof: torch.profiler.profile,
        output_dir: Path,
        output_suffix: str = "",
    ) -> dict:
        """Export top kernels directly without categorization."""
        try:
            events = prof.key_averages()
        except Exception:
            return {}

        # Collect all kernels with their timing
        kernels = []
        for event in events:
            kernel_name = event.key
            cuda_time_us = getattr(event, "cuda_time_total", 0)
            cpu_time_us = getattr(event, "cpu_time_total", 0)
            total_time_ms = max(cuda_time_us, cpu_time_us) / 1000

            if total_time_ms < 0.01:
                continue

            kernels.append(
                {
                    "name": kernel_name,
                    "ms": round(total_time_ms, 2),
                    "cuda_ms": round(cuda_time_us / 1000, 2),
                    "cpu_ms": round(cpu_time_us / 1000, 2),
                }
            )

        # Sort by total time and take top 50
        kernels.sort(key=lambda x: -x["ms"])
        top_kernels = kernels[:50]

        total_time = sum(k["ms"] for k in kernels)

        report = {
            "name": self.stage_name,
            "total_kernel_time_ms": round(total_time, 2),
            "num_kernels": len(kernels),
            "top_kernels": top_kernels,
        }

        breakdown_path = output_dir / f"{self.stage_name}{output_suffix}_breakdown.json"
        with open(breakdown_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"[Harness] Kernel breakdown saved to: {breakdown_path}")

        return report


def main():
    """CLI entry point for stage bench harness."""
    parser = argparse.ArgumentParser(
        description="Run isolated stage profiling using I/O signatures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Profile a single stage
    python -m telefuser.utils.stage_bench_harness \\
        --signature profiler_output/timing_io_signature.json \\
        --stage denoise

    # Custom warmup and output directory
    python -m telefuser.utils.stage_bench_harness \\
        --signature timing_io_signature.json \\
        --stage vae_decode \\
        --warmup 2 --profile_steps 3 \\
        --output_dir ./profiler_results
""",
    )

    parser.add_argument(
        "--signature",
        type=str,
        required=True,
        help="Path to the I/O signature JSON file from Layer 1",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Name of the stage to profile",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=1,
        help="Number of profiling iterations (default: 1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for traces and breakdowns (default: work_dirs/profiler_output/{pipeline_name}/{date})",
    )
    parser.add_argument(
        "--list_stages",
        action="store_true",
        help="List available stages in the signature file and exit",
    )

    args = parser.parse_args()

    # List stages mode
    if args.list_stages:
        with open(args.signature, encoding="utf-8") as f:
            data = json.load(f)
        stages = list(data.get("stages", {}).keys())
        print(f"Available stages in {args.signature}:")
        for s in stages:
            print(f"  - {s}")
        return

    # Create and run harness
    config = HarnessConfig(
        warmup=args.warmup,
        profile_steps=args.profile_steps,
        output_dir=args.output_dir,  # None will use default
    )

    harness = StageBenchHarness.from_signature_file(
        signature_path=args.signature,
        stage_name=args.stage,
        config=config,
    )

    # Note: For full functionality, the user should provide a stage instance
    # by loading the model. This CLI is mainly for signature inspection.
    # Full profiling requires programmatic usage with a loaded stage.

    logger.warning(
        "CLI mode: Cannot execute stage without loaded model. "
        "Use programmatically with a stage instance for full profiling."
    )
    logger.info(f"Stage I/O signature loaded for: {args.stage}")

    # Print signature info
    sig = harness.io_signature
    print(f"\nStage: {sig.stage_name}")
    print("\nInput Signatures:")
    for name, s in sig.input_signatures.items():
        if isinstance(s, TensorSignature):
            print(f"  {name}: shape={s.shape}, dtype={s.dtype}, device={s.device}")
        else:
            print(f"  {name}: {s}")

    if sig.output_signature:
        print("\nOutput Signature:")
        print(f"  shape={sig.output_signature.shape}, dtype={sig.output_signature.dtype}")

    if sig.metadata:
        print("\nMetadata:")
        for k, v in sig.metadata.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
