"""TeleFuser pipeline regression test.

Runs configured pipelines in isolated subprocesses, compares outputs against
baselines (PSNR/SSIM for video, pixel diff for image), and prints a results table.

Usage:
    python examples/regression_test/run_regression.py --list
    python examples/regression_test/run_regression.py --pipeline wan21_1_3b_t2v
    python examples/regression_test/run_regression.py --all
    python examples/regression_test/run_regression.py --all --update-baseline
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_CONFIG_PATH = Path(__file__).resolve().parent / "regression_config.yaml"
_RESULT_MARKER = "###RESULT###"


def _pipeline_slug(pipeline_key: str) -> str:
    """Convert pipeline key to filesystem-safe slug."""
    return pipeline_key.replace("/", "__")


def _is_oom(exc_or_text: Exception | str) -> bool:
    """Check if an exception or text indicates an out-of-memory error."""
    text = str(exc_or_text).lower()
    return "out of memory" in text or "cuda out of memory" in text


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline test."""

    script: str  # Relative path from examples/, e.g. "wan_video/wan21_1_3b_text_to_video_h100.py"
    enabled: bool = True
    gpu_count: int = 1
    timeout_seconds: int = 1800
    output_type: str = "video"  # "video" | "image"
    seed: int = 42
    model_root: str | None = None
    prompt: str | None = None
    input_image_path: str | None = None
    input_video_path: str | None = None
    ppl_config_overrides: dict = field(default_factory=dict)
    # Regression thresholds
    psnr_min: float = 25.0
    ssim_min: float = 0.85
    pixel_diff_max: float = 0.02
    # Performance thresholds (None = disabled)
    max_elapsed_seconds: float | None = None
    max_gpu_memory_mb: float | None = None


@dataclass
class Config:
    """Top-level regression config."""

    output_root: str = "examples/regression_test/regression_outputs"
    pipelines: dict[str, PipelineConfig] = field(default_factory=dict)


def load_config(config_path: str | None = None) -> Config:
    """Load regression config from YAML."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        return Config()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.get("defaults", {})
    output_root = raw.get("output_root", "examples/regression_test/regression_outputs")

    pipelines: dict[str, PipelineConfig] = {}
    valid_fields = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
    for name, ppl_data in raw.get("pipelines", {}).items():
        if ppl_data is None:
            ppl_data = {}
        # Merge defaults for missing fields
        merged = {**defaults, **ppl_data}
        # Only pass fields that PipelineConfig accepts
        filtered = {k: v for k, v in merged.items() if k in valid_fields}
        pipelines[name] = PipelineConfig(**filtered)

    return Config(output_root=output_root, pipelines=pipelines)


# =============================================================================
# Subprocess Worker (--run-single mode)
# =============================================================================


def _import_example_module(example_path: str) -> ModuleType:
    """Dynamically import an example script as a module."""
    path = Path(example_path).resolve()
    module_name = f"_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {example_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_ppl_config(module: ModuleType, overrides: dict) -> None:
    """Patch PPL_CONFIG or DEFAULT_CONFIG in the module with overrides."""
    from telefuser.core.config import AttnImplType

    config_attr = None
    for attr_name in ("PPL_CONFIG", "DEFAULT_CONFIG"):
        if hasattr(module, attr_name):
            config_attr = attr_name
            break

    if config_attr is None or not overrides:
        return

    config = getattr(module, config_attr)
    for key, value in overrides.items():
        if value is not None:
            if key == "attn_impl" and isinstance(value, str):
                value = AttnImplType[value]
            config[key] = value


def _extract_click_default(module: ModuleType, param_name: str) -> str | None:
    """Extract a click option's default value from the module's main() command."""
    import click

    main_func = getattr(module, "main", None)
    if main_func is None or not isinstance(main_func, click.BaseCommand):
        return None
    for param in main_func.params:
        if param.human_readable_name == param_name and param.default is not None:
            return str(param.default)
    return None


def _call_get_pipeline(module: ModuleType, config: dict) -> object:
    """Call get_pipeline() with arguments matched by signature inspection."""
    func = module.get_pipeline
    params = list(inspect.signature(func).parameters.keys())
    kwargs: dict = {}

    if "parallelism" in params:
        kwargs["parallelism"] = config.get("gpu_count", 1)
    elif "num_gpus" in params:
        kwargs["num_gpus"] = config.get("gpu_count", 1)

    if "model_root" in params and config.get("model_root"):
        kwargs["model_root"] = config["model_root"]

    # FlashVSR: dit_path/vae_path built from model_root + module constants
    model_root = config.get("model_root", "")
    # If model_root not in config, try to extract default from module's click command
    if not model_root:
        model_root = _extract_click_default(module, "model_root") or ""
    if "dit_path" in params and "dit_path" not in kwargs:
        dit_filename = getattr(module, "DIT_FILENAME", None)
        if dit_filename and model_root:
            kwargs["dit_path"] = os.path.join(model_root, dit_filename)
    if "vae_path" in params and "vae_path" not in kwargs:
        vae_filename = getattr(module, "VAE_FILENAME", None)
        if vae_filename and model_root:
            kwargs["vae_path"] = os.path.join(model_root, vae_filename)

    return func(**kwargs)


def _call_run(module: ModuleType, pipeline: object, config: dict) -> object:
    """Call run() with arguments matched by signature inspection."""
    func = module.run
    params = list(inspect.signature(func).parameters.keys())
    kwargs: dict = {"pipeline": pipeline}

    if "prompt" in params:
        kwargs["prompt"] = config.get("prompt") or "A beautiful sunset over the ocean with gentle waves."

    if "seed" in params:
        kwargs["seed"] = config.get("seed", 42)

    if "height" in params:
        kwargs["height"] = config.get("height", 480)
    if "width" in params:
        kwargs["width"] = config.get("width", 832)

    # Image input
    if ("image" in params or "first_image" in params) and config.get("input_image_path"):
        from PIL import Image

        img = Image.open(config["input_image_path"]).convert("RGB")
        if "image" in params:
            kwargs["image"] = img
        if "first_image" in params:
            kwargs["first_image"] = img

    # Video input
    if "input_video" in params and config.get("input_video_path"):
        from telefuser.utils.video import VideoData

        video_data = VideoData(video_file=config["input_video_path"], height=360, width=640)
        kwargs["input_video"] = video_data

    if "LQ_video" in params and config.get("input_video_path"):
        from telefuser.utils.video import VideoData

        kwargs["LQ_video"] = VideoData(video_file=config["input_video_path"], height=360, width=640).raw_data()

    if "scale" in params:
        kwargs["scale"] = 2

    # FlashVSR chunked calling pattern
    if "input_video" in kwargs and "scale" in kwargs:
        return _call_run_flashvsr_chunked(func, kwargs)

    return func(**kwargs)


def _call_run_flashvsr_chunked(func: callable, kwargs: dict) -> object:
    """Call FlashVSR run() in chunks matching the example's main() pattern."""
    from telefuser.utils.video import VideoData

    video_data = kwargs.pop("input_video")
    pipeline = kwargs["pipeline"]

    frames = video_data.raw_data() if isinstance(video_data, VideoData) else list(video_data)
    final_video = []

    # First chunk: 25 frames
    video = func(**{**kwargs, "input_video": frames[:25]})
    if isinstance(video, list):
        final_video.extend(video)

    # Remaining: 16-frame chunks
    offset = 25
    while offset < len(frames):
        end = min(offset + 16, len(frames))
        video = func(**{**kwargs, "input_video": frames[offset:end]})
        if isinstance(video, list):
            final_video.extend(video)
        offset += 16

    if hasattr(pipeline, "clean_cache"):
        pipeline.clean_cache()

    return final_video


def _save_output(
    output: object, output_dir: str, output_type: str, fps: int = 15
) -> tuple[str | None, int | None, str | None]:
    """Save pipeline output to file. Returns (path, num_frames, resolution)."""
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    if output is None:
        return None, None, None

    if output_type == "video":
        if isinstance(output, (list, tuple)) and len(output) > 0:
            # Unpack tuple from longcat pipelines: (frames, latents)
            frames = output
            if isinstance(output, tuple):
                frames = output[0] if isinstance(output[0], list) else output
            output_path = os.path.join(output_dir, "output.mp4")
            from telefuser.utils.video import save_video

            save_video(frames, output_path, fps=fps, quality=6)
            first = frames[0]
            resolution = f"{first.width}x{first.height}" if isinstance(first, Image.Image) else None
            return output_path, len(frames), resolution

    if output_type == "image":
        output_path = os.path.join(output_dir, "output.png")
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], Image.Image):
            output[0].save(output_path)
            return output_path, 1, f"{output[0].width}x{output[0].height}"
        if isinstance(output, Image.Image):
            output.save(output_path)
            return output_path, 1, f"{output.width}x{output.height}"

    return None, None, None


def _validate_output(output: object) -> list[str]:
    """Check pipeline output for NaN/Inf/None. Returns warning strings."""
    warnings: list[str] = []
    if output is None:
        warnings.append("Output is None")
        return warnings

    if isinstance(output, list) and len(output) == 0:
        warnings.append("Output is an empty list")

    tensor = None
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        tensor = output[0]

    if tensor is not None:
        if torch.isnan(tensor).any():
            warnings.append("Output contains NaN")
        if torch.isinf(tensor).any():
            warnings.append("Output contains Inf")

    return warnings


def _emit_result(data: dict) -> None:
    """Print JSON result for parent process to parse."""
    print(f"{_RESULT_MARKER}{json.dumps(data)}", flush=True)


def _run_single(pipeline_key: str, config_path: str | None) -> None:
    """Subprocess entry point: load, run, save one pipeline."""
    cfg = load_config(config_path)
    ppl_cfg = cfg.pipelines.get(pipeline_key)
    if ppl_cfg is None:
        _emit_result({"status": "ERROR", "error": f"Pipeline '{pipeline_key}' not found in config"})
        sys.exit(1)

    examples_root = os.path.join(_PROJECT_ROOT, "examples")
    script_path = os.path.join(examples_root, ppl_cfg.script)

    output_root = cfg.output_root
    if not os.path.isabs(output_root):
        output_root = os.path.join(_PROJECT_ROOT, output_root)
    slug = _pipeline_slug(pipeline_key)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, slug, timestamp)

    runner_config = {
        "gpu_count": ppl_cfg.gpu_count,
        "seed": ppl_cfg.seed,
        "model_root": ppl_cfg.model_root,
        "prompt": ppl_cfg.prompt,
        "input_image_path": ppl_cfg.input_image_path,
        "input_video_path": ppl_cfg.input_video_path,
        "ppl_config_overrides": ppl_cfg.ppl_config_overrides,
    }
    # Merge ppl_config_overrides into runner_config for height/width access
    runner_config.update(ppl_cfg.ppl_config_overrides)

    pipeline = None
    gpu_mem_peak = 0.0
    start = time.time()

    # Phase 1: Model Loading
    try:
        module = _import_example_module(script_path)
        _patch_ppl_config(module, ppl_cfg.ppl_config_overrides)
        pipeline = _call_get_pipeline(module, runner_config)
    except Exception as e:
        tb = traceback.format_exc()
        category = "OOM_ERROR" if _is_oom(e) else "MODEL_LOAD_ERROR"
        _emit_result(
            {
                "status": "ERROR",
                "error": f"{e}\n{tb}",
                "error_category": category,
                "elapsed": round(time.time() - start, 2),
            }
        )
        sys.exit(1)

    # Phase 2: Inference
    output = None
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        output = _call_run(module, pipeline, runner_config)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        tb = traceback.format_exc()
        category = "OOM_ERROR" if _is_oom(e) else "INFERENCE_ERROR"
        _emit_result(
            {
                "status": "ERROR",
                "error": f"{e}\n{tb}",
                "error_category": category,
                "elapsed": round(time.time() - start, 2),
                "peak_gpu_memory_mb": round(gpu_mem_peak, 2),
            }
        )
        del pipeline
        gc.collect()
        sys.exit(1)

    # Phase 3: Validation & Save
    warnings = _validate_output(output)
    ppl_config = getattr(module, "PPL_CONFIG", {})
    num_steps = ppl_config.get("num_inference_steps")
    if isinstance(num_steps, list):
        num_steps = sum(num_steps)
    output_fps = ppl_config.get("target_fps", 15)

    try:
        output_path, num_frames, resolution = _save_output(output, output_dir, ppl_cfg.output_type, fps=output_fps)
    except Exception as e:
        tb = traceback.format_exc()
        _emit_result(
            {
                "status": "ERROR",
                "error": f"{e}\n{tb}",
                "error_category": "OUTPUT_ERROR",
                "elapsed": round(time.time() - start, 2),
                "peak_gpu_memory_mb": round(gpu_mem_peak, 2),
            }
        )
        pipeline = None  # noqa: F841
        gc.collect()
        sys.exit(1)

    elapsed = time.time() - start
    status = "PASS"
    error_msg = ""
    if warnings:
        severe = [w for w in warnings if "NaN" in w or "Inf" in w or "is None" in w]
        if severe:
            status = "ERROR"
            error_msg = "; ".join(warnings)

    _emit_result(
        {
            "status": status,
            "output_path": output_path,
            "error": error_msg,
            "elapsed": round(elapsed, 2),
            "peak_gpu_memory_mb": round(gpu_mem_peak, 2),
            "num_frames": num_frames,
            "resolution": resolution,
            "num_steps": num_steps,
        }
    )

    del pipeline  # noqa: F821
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Metrics
# =============================================================================


def compute_video_metrics(baseline_path: str, current_path: str) -> dict[str, float]:
    """Compute PSNR and SSIM between two videos using streaming frame comparison."""
    import cv2
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    cap_true = cv2.VideoCapture(baseline_path)
    cap_test = cv2.VideoCapture(current_path)
    if not cap_true.isOpened() or not cap_test.isOpened():
        cap_true.release()
        cap_test.release()
        return {}

    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    while True:
        ret_true, frame_true = cap_true.read()
        ret_test, frame_test = cap_test.read()
        if not ret_true or not ret_test:
            break
        psnr_sum += peak_signal_noise_ratio(frame_true, frame_test)
        ssim_sum += structural_similarity(frame_true, frame_test, channel_axis=2)
        n += 1

    cap_true.release()
    cap_test.release()

    if n == 0:
        return {}
    return {"psnr": psnr_sum / n, "ssim": ssim_sum / n}


def compute_image_diff(baseline_path: str, current_path: str) -> float | None:
    """Compute mean absolute pixel difference (0-1) between two images."""
    from PIL import Image

    img_a = np.array(Image.open(baseline_path).convert("RGB")).astype(np.float32) / 255.0
    img_b = np.array(Image.open(current_path).convert("RGB")).astype(np.float32) / 255.0
    if img_a.shape != img_b.shape:
        return None
    return float(np.mean(np.abs(img_a - img_b)))


# =============================================================================
# Baseline Comparison
# =============================================================================


def _baseline_dir(output_root: str, pipeline_key: str) -> str:
    slug = _pipeline_slug(pipeline_key)
    return os.path.join(output_root, slug, "baseline")


def _get_baseline_path(output_root: str, pipeline_key: str) -> str | None:
    bdir = _baseline_dir(output_root, pipeline_key)
    if not os.path.isdir(bdir):
        return None
    for ext in ("mp4", "png"):
        p = os.path.join(bdir, f"output.{ext}")
        if os.path.exists(p):
            return p
    return None


def _update_baseline(output_root: str, pipeline_key: str, current_path: str) -> str:
    import shutil

    bdir = _baseline_dir(output_root, pipeline_key)
    os.makedirs(bdir, exist_ok=True)
    dst = os.path.join(bdir, os.path.basename(current_path))
    shutil.copy2(current_path, dst)
    return dst


def compare_against_baseline(
    output_root: str,
    pipeline_key: str,
    current_path: str | None,
    output_type: str,
    psnr_min: float,
    ssim_min: float,
    pixel_diff_max: float,
) -> dict:
    """Compare current output against baseline. Returns dict with passed, metrics, message."""
    baseline_path = _get_baseline_path(output_root, pipeline_key)

    if baseline_path is None:
        if current_path and os.path.exists(current_path):
            saved = _update_baseline(output_root, pipeline_key, current_path)
            return {"passed": True, "baseline_exists": False, "metrics": {}, "message": f"Saved as baseline: {saved}"}
        return {"passed": True, "baseline_exists": False, "metrics": {}, "message": "No baseline (first run)"}

    if not current_path or not os.path.exists(current_path):
        return {"passed": False, "baseline_exists": True, "metrics": {}, "message": "No output file produced"}

    try:
        if output_type == "video":
            m = compute_video_metrics(baseline_path, current_path)
            psnr, ssim = m.get("psnr"), m.get("ssim")
            passed = True
            msgs = []
            if psnr is not None and psnr < psnr_min:
                passed = False
                msgs.append(f"PSNR {psnr:.2f} < {psnr_min}")
            if ssim is not None and ssim < ssim_min:
                passed = False
                msgs.append(f"SSIM {ssim:.4f} < {ssim_min}")
            msg = "; ".join(msgs) if msgs else f"PSNR={psnr:.2f}, SSIM={ssim:.4f}"
            return {"passed": passed, "baseline_exists": True, "metrics": m, "message": msg}
        else:
            diff = compute_image_diff(baseline_path, current_path)
            if diff is None:
                return {"passed": False, "baseline_exists": True, "metrics": {}, "message": "Image size mismatch"}
            passed = diff <= pixel_diff_max
            msg = f"pixel_diff={diff:.6f}" + ("" if passed else f" > {pixel_diff_max}")
            return {"passed": passed, "baseline_exists": True, "metrics": {"pixel_diff": diff}, "message": msg}
    except (ImportError, ModuleNotFoundError) as e:
        return {"passed": True, "baseline_exists": True, "metrics": {}, "message": f"Comparison skipped ({e})"}
    except Exception as e:
        return {"passed": False, "baseline_exists": True, "metrics": {}, "message": f"Comparison error: {e}"}


# =============================================================================
# Orchestration
# =============================================================================


@dataclass
class Result:
    """Result for a single pipeline run."""

    name: str
    status: str  # PASS | FAIL | ERROR | TIMEOUT | SKIP
    elapsed_seconds: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    num_frames: int | None = None
    resolution: str | None = None
    num_steps: int | None = None
    error_category: str = ""
    error_message: str = ""
    note: str = ""
    regression_metrics: dict = field(default_factory=dict)


def _parse_runner_output(stdout: str) -> dict:
    """Extract JSON result from subprocess stdout."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith(_RESULT_MARKER):
            try:
                return json.loads(line[len(_RESULT_MARKER) :])
            except json.JSONDecodeError:
                pass
    return {}


def run_pipeline(
    pipeline_key: str, ppl_cfg: PipelineConfig, output_root: str, config_path: str | None, update_baseline: bool
) -> Result:
    """Run a single pipeline in a subprocess and evaluate results."""
    gpu_count = ppl_cfg.gpu_count

    # Assign GPUs
    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_PROJECT_ROOT}{os.pathsep}{existing_pypath}" if existing_pypath else _PROJECT_ROOT
    if not env.get("CUDA_VISIBLE_DEVICES"):
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available > 0:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(min(gpu_count, available)))

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-single",
        pipeline_key,
    ]
    if config_path:
        cmd.extend(["--config", config_path])

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=ppl_cfg.timeout_seconds,
            cwd=_PROJECT_ROOT,
            env=env,
        )
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return Result(
            name=pipeline_key,
            status="TIMEOUT",
            elapsed_seconds=round(elapsed, 2),
            note=f"Timeout after {ppl_cfg.timeout_seconds}s",
        )

    # Parse result
    data = _parse_runner_output(proc.stdout)
    status = data.get("status", "ERROR")
    error_msg = data.get("error", "")
    error_cat = data.get("error_category", "")
    peak_mem = data.get("peak_gpu_memory_mb", 0.0)

    # OOM detection fallback
    if not error_cat and proc.stderr and _is_oom(proc.stderr):
        error_cat = "OOM_ERROR"

    if not data and proc.returncode != 0:
        error_msg = f"Process exited with code {proc.returncode}"

    result = Result(
        name=pipeline_key,
        status=status,
        elapsed_seconds=round(elapsed, 2),
        peak_gpu_memory_mb=round(peak_mem, 2),
        num_frames=data.get("num_frames"),
        resolution=data.get("resolution"),
        num_steps=data.get("num_steps"),
        error_category=error_cat,
        error_message=error_msg,
    )

    # Save log
    slug = _pipeline_slug(pipeline_key)
    log_dir = os.path.join(output_root, slug)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "latest.log"), "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n")
        f.write(proc.stdout or "(empty)\n")
        f.write("\n=== STDERR ===\n")
        f.write(proc.stderr or "(empty)\n")

    # Compare against baseline
    if status == "PASS":
        output_path = data.get("output_path")
        cmp = compare_against_baseline(
            output_root,
            pipeline_key,
            output_path,
            ppl_cfg.output_type,
            ppl_cfg.psnr_min,
            ppl_cfg.ssim_min,
            ppl_cfg.pixel_diff_max,
        )
        result.regression_metrics = cmp.get("metrics", {})
        result.note = cmp["message"]
        if cmp["baseline_exists"] and not cmp["passed"]:
            result.status = "FAIL"

        if update_baseline and output_path and os.path.exists(output_path):
            _update_baseline(output_root, pipeline_key, output_path)
            result.note += " [baseline updated]"

    # Performance/memory threshold checks
    if result.status == "PASS" and ppl_cfg.max_elapsed_seconds and result.elapsed_seconds > ppl_cfg.max_elapsed_seconds:
        result.status = "FAIL"
        result.note += f" [PERF: {result.elapsed_seconds:.1f}s > {ppl_cfg.max_elapsed_seconds:.1f}s]"

    if result.status == "PASS" and ppl_cfg.max_gpu_memory_mb and result.peak_gpu_memory_mb > ppl_cfg.max_gpu_memory_mb:
        result.status = "FAIL"
        result.note += f" [MEM: {result.peak_gpu_memory_mb:.0f}MB > {ppl_cfg.max_gpu_memory_mb:.0f}MB]"

    if result.status != "PASS" and not result.note:
        result.note = error_msg[:60] if error_msg else ""
        if error_cat:
            result.note = f"[{error_cat}] {result.note}"

    return result


# =============================================================================
# Reporting
# =============================================================================


def _fmt(val: object, fmt: str) -> str:
    if val is None:
        return "-"
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


def print_results_table(results: list[Result]) -> None:
    """Print a summary table to console."""
    print()
    print("=" * 130)
    print("REGRESSION TEST RESULTS")
    print("=" * 130)
    header = (
        f"  {'Pipeline':<45} {'Status':<8} {'Steps':>6} {'Frames':>7} {'Resolution':>12}"
        f" {'Time(s)':>8} {'VRAM(GB)':>9}  {'PSNR':>7} {'SSIM':>7}  Note"
    )
    print(header)
    print("-" * 130)

    pass_count = fail_count = skip_count = 0
    for r in results:
        if r.status == "PASS":
            pass_count += 1
        elif r.status == "SKIP":
            skip_count += 1
        else:
            fail_count += 1

        steps = _fmt(r.num_steps, "d")
        frames = _fmt(r.num_frames, "d")
        res = r.resolution or "-"
        t = f"{r.elapsed_seconds:.1f}" if r.elapsed_seconds > 0 else "-"
        vram = f"{r.peak_gpu_memory_mb / 1024:.2f}" if r.peak_gpu_memory_mb > 0 else "-"
        psnr = _fmt(r.regression_metrics.get("psnr"), ".1f")
        ssim = _fmt(r.regression_metrics.get("ssim"), ".4f")
        note = (r.note or "")[:50]
        print(
            f"  {r.name:<45} {r.status:<8} {steps:>6} {frames:>7} {res:>12} {t:>8} {vram:>9}  {psnr:>7} {ssim:>7}  {note}"
        )

    print("-" * 130)
    print(f"Total: {len(results)} | PASS: {pass_count} | FAIL: {fail_count} | SKIP: {skip_count}")
    print("=" * 130)


def save_report_json(output_root: str, results: list[Result]) -> str:
    """Save results to JSON report."""
    os.makedirs(output_root, exist_ok=True)
    report_path = os.path.join(output_root, "regression_report.json")

    env_info = {}
    try:
        env_info = {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "N/A",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception:
        pass

    counts = {"pass": 0, "fail": 0, "skip": 0, "error": 0, "timeout": 0}
    for r in results:
        key = r.status.lower()
        counts[key] = counts.get(key, 0) + 1

    report = {
        "generated_at": datetime.now().isoformat(),
        "environment": env_info,
        "summary": {"total": len(results), **counts},
        "results": {r.name: asdict(r) for r in results},
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_path


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="TeleFuser Pipeline Regression Test")
    parser.add_argument("--list", action="store_true", dest="list_pipelines", help="List configured pipelines")
    parser.add_argument("--pipeline", type=str, help="Run a specific pipeline by name")
    parser.add_argument("--all", action="store_true", help="Run all enabled pipelines")
    parser.add_argument("--update-baseline", action="store_true", help="Update baseline outputs after successful runs")
    parser.add_argument("--config", type=str, help="Path to config YAML")

    # Internal: subprocess self-invocation
    parser.add_argument("--run-single", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Subprocess mode: run a single pipeline and exit
    if args.run_single:
        _run_single(args.run_single, args.config)
        return

    # Load config
    cfg = load_config(args.config)

    if args.list_pipelines:
        print(f"\nConfigured pipelines ({len(cfg.pipelines)}):\n")
        print(f"  {'Name':<40} {'Enabled':<8} {'GPUs':>4} {'Type':<6} {'Script'}")
        print("  " + "-" * 110)
        for name, ppl in cfg.pipelines.items():
            enabled = "ON" if ppl.enabled else "OFF"
            print(f"  {name:<40} {enabled:<8} {ppl.gpu_count:>4} {ppl.output_type:<6} {ppl.script}")
        return

    # Determine which pipelines to run
    if args.pipeline:
        if args.pipeline not in cfg.pipelines:
            print(f"Error: pipeline '{args.pipeline}' not found in config. Use --list to see available pipelines.")
            sys.exit(1)
        to_run = {args.pipeline: cfg.pipelines[args.pipeline]}
    elif args.all:
        to_run = {k: v for k, v in cfg.pipelines.items() if v.enabled}
    else:
        parser.print_help()
        return

    if not to_run:
        print("No pipelines to run.")
        return

    output_root = cfg.output_root
    if not os.path.isabs(output_root):
        output_root = os.path.join(_PROJECT_ROOT, output_root)

    # Run pipelines sequentially
    results: list[Result] = []
    total = len(to_run)
    run_start = time.time()

    for idx, (name, ppl_cfg) in enumerate(to_run.items(), 1):
        elapsed_total = time.time() - run_start
        print(f"\n[{idx}/{total}] ({elapsed_total:.0f}s) Running: {name} ...")

        result = run_pipeline(name, ppl_cfg, output_root, args.config, args.update_baseline)
        results.append(result)

        print(f"  -> {result.status} ({result.elapsed_seconds:.1f}s) {result.note}")

    # Report
    print_results_table(results)
    report_path = save_report_json(output_root, results)
    print(f"\nJSON report: {report_path}")

    fail_count = sum(1 for r in results if r.status in ("FAIL", "ERROR", "TIMEOUT"))
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
