#!/usr/bin/env python3
"""Collect TeleFuser wan22 latents for the W03 dataset.

Reads `output/W03-latents-dataset/dataset_500.jsonl`, runs the TeleFuser
Wan2.2 T2V pipeline once per record, and persists per-step latents as
single-tensor safetensors files plus an explicit `index.jsonl` table.

Output layout:
    output/W03-latents-dataset/
    ├── dataset_500.jsonl          (input, untouched)
    ├── manifest.json              (this script writes; inference params + git hash)
    ├── failures.jsonl             (records that errored)
    ├── latents/
    │   ├── index.jsonl            (one row per (req_id, step), authoritative)
    │   └── <req_id>/
    │       ├── step_005.safetensors
    │       ├── step_015.safetensors
    │       ...
    └── videos/<req_id>.mp4        (one mp4 per record)

Usage (run on a remote H100 with TeleFuser installed):
    export TF_MODEL_ZOO_PATH=/path/to/model_zoo
    python scripts/build_latent_dataset.py --parallelism 2
    python scripts/build_latent_dataset.py --limit 5     # smoke test
    python scripts/build_latent_dataset.py --clean       # destructive, wipes latents/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "output" / "W03-latents-dataset"
DATASET_PATH = EXP_DIR / "dataset_500.jsonl"
LATENTS_DIR = EXP_DIR / "latents"
INDEX_PATH = LATENTS_DIR / "index.jsonl"
MANIFEST_PATH = EXP_DIR / "manifest.json"
FAILURES_PATH = EXP_DIR / "failures.jsonl"
VIDEOS_DIR = EXP_DIR / "videos"

DEFAULT_SAVED_STEPS = [5, 10, 15, 20, 25]
DEFAULT_SEED = 42
DEFAULT_NUM_STEPS = 40
DEFAULT_RESOLUTION = "720p"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_NUM_FRAMES = 81
DEFAULT_BOUNDARY = 0.9
DEFAULT_SIGMA_SHIFT = 5.0
DEFAULT_CFG_SCALE_HIGH = 5.0
DEFAULT_CFG_SCALE_LOW = 5.0
DEFAULT_SAMPLE_SOLVER = "euler"
DEFAULT_TARGET_FPS = 16
DEFAULT_MODEL_NAME = "Wan2.2-T2V-A14B"
DEFAULT_MODEL_TYPE = "Wan2_2-T2V-A14B"

NEGATIVE_PROMPT = (
    "Overly saturated colors, overexposed, static, blurry details, subtitles, "
    "style, artwork, painting, frame, still, overall grayish, worst quality, "
    "low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, "
    "fused fingers, static frames, cluttered background, three legs, crowded "
    "background, walking backwards"
)


def build_pipeline(model_root: str, parallelism: int, enable_feature_cache: bool):
    from telefuser.core.config import (
        AttentionConfig,
        AttnImplType,
        FeatureCacheConfig,
        WeightOffloadType,
    )
    from telefuser.core.module_manager import ModuleManager
    from telefuser.pipelines.wan_video.wan22_video import (
        Wan22VideoPipeline,
        Wan22VideoPipelineConfig,
    )

    mm = ModuleManager(device="cpu")
    mm.load_model(f"{model_root}/Wan2.1_VAE.pth", torch_dtype=torch.bfloat16)
    mm.load_model(
        f"{model_root}/high_noise_model/diffusion_pytorch_model-0000*-of-00006.safetensors",
        torch_dtype=torch.bfloat16,
    )
    mm.load_model(
        f"{model_root}/low_noise_model/diffusion_pytorch_model-0000*-of-00006.safetensors",
        torch_dtype=torch.bfloat16,
    )
    mm.load_model(f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth", torch_dtype=torch.bfloat16)

    pipe = Wan22VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    cfg = Wan22VideoPipelineConfig()
    for c in (cfg.text_encoding_config, cfg.vae_config, cfg.dit_high_config, cfg.dit_low_config):
        c.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    cfg.dit_high_config.attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
    cfg.dit_low_config.attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
    cfg.sample_solver = DEFAULT_SAMPLE_SOLVER
    if enable_feature_cache:
        cfg.dit_high_config.feature_cache_config = FeatureCacheConfig(enabled=True, model_type=DEFAULT_MODEL_TYPE)
        cfg.dit_low_config.feature_cache_config = FeatureCacheConfig(enabled=True, model_type=DEFAULT_MODEL_TYPE)

    if parallelism > 1:
        cfg.dit_high_config.parallel_config.cfg_degree = 2
        cfg.dit_high_config.parallel_config.sp_ulysses_degree = parallelism // 2
        cfg.dit_low_config.parallel_config.cfg_degree = 2
        cfg.dit_low_config.parallel_config.sp_ulysses_degree = parallelism // 2
        cfg.dit_high_config.parallel_config.device_ids = list(range(parallelism))
        cfg.dit_low_config.parallel_config.device_ids = list(range(parallelism))
        cfg.enable_denoising_parallel = True

    pipe.init(mm, cfg)
    return pipe


def get_image_size(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    from telefuser.utils.video import get_target_video_size_from_ratio

    return get_target_video_size_from_ratio(
        aspect_ratio,
        resolution=resolution,
        height_division_factor=16,
        width_division_factor=16,
    )


def get_telefuser_git_hash() -> str:
    try:
        import telefuser

        repo = Path(telefuser.__file__).resolve().parent.parent
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


def write_manifest(args, width: int, height: int) -> None:
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "telefuser_git_hash": get_telefuser_git_hash(),
        "model": {
            "name": args.model_name,
            "model_root": args.model_root,
            "torch_dtype": "bfloat16",
        },
        "inference": {
            "num_inference_steps": args.num_steps,
            "saved_steps": args.saved_steps,
            "seed": args.seed,
            "resolution": args.resolution,
            "aspect_ratio": args.aspect_ratio,
            "width": width,
            "height": height,
            "num_frames": args.num_frames,
            "cfg_scale_high": DEFAULT_CFG_SCALE_HIGH,
            "cfg_scale_low": DEFAULT_CFG_SCALE_LOW,
            "boundary": DEFAULT_BOUNDARY,
            "sigma_shift": DEFAULT_SIGMA_SHIFT,
            "sample_solver": DEFAULT_SAMPLE_SOLVER,
            "feature_cache_enabled": args.feature_cache,
            "negative_prompt": NEGATIVE_PROMPT,
        },
        "dataset_path": str(DATASET_PATH),
        "parallelism": args.parallelism,
        "save_video": args.save_video,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_done_set() -> set[tuple[str, int]]:
    """Build (req_id, step) set from existing index.jsonl rows."""
    if not INDEX_PATH.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with open(INDEX_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((str(rec["req_id"]), int(rec["step"])))
            except Exception:
                continue
    return done


def append_index(record: dict) -> None:
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def append_failure(record: dict) -> None:
    with open(FAILURES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def process_record(pipeline, rec: dict, args, width: int, height: int, done: set) -> str:
    req_id = str(rec["req_id"])
    prompt = rec["prompt"]

    expected = {(req_id, s) for s in args.saved_steps}
    if expected.issubset(done):
        return "skip"

    req_dir = LATENTS_DIR / req_id
    if req_dir.exists():
        shutil.rmtree(req_dir)
    req_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=args.num_steps,
            num_frames=args.num_frames,
            cfg_scale_high=DEFAULT_CFG_SCALE_HIGH,
            cfg_scale_low=DEFAULT_CFG_SCALE_LOW,
            seed=args.seed,
            tiled=False,
            height=height,
            width=width,
            sigma_shift=DEFAULT_SIGMA_SHIFT,
            boundary=DEFAULT_BOUNDARY,
            latent_data={"saved_steps": list(args.saved_steps)},
        )
    except Exception as exc:
        return f"pipeline_error: {exc}"

    if not isinstance(result, tuple):
        return "pipeline_returned_non_tuple"

    frames, payload = result
    latent_states = payload.get("latent_states_dict", {}) if isinstance(payload, dict) else {}

    missing = [s for s in args.saved_steps if s not in latent_states]
    if missing:
        return f"missing_steps_in_payload: {missing}"

    prompt_md5 = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    ts = datetime.now().isoformat(timespec="seconds")

    for step in args.saved_steps:
        tensor = latent_states[step]
        if tensor.ndim == 5 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        out_file = req_dir / f"step_{step:03d}.safetensors"
        save_file({"latent": tensor.contiguous()}, str(out_file))
        append_index(
            {
                "req_id": req_id,
                "step": step,
                "file": str(out_file.relative_to(EXP_DIR)),
                "prompt": prompt,
                "prompt_md5": prompt_md5,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "seed": args.seed,
                "model": args.model_name,
                "resolution": args.resolution,
                "aspect_ratio": args.aspect_ratio,
                "ts": ts,
            }
        )
        done.add((req_id, step))

    if args.save_video and frames:
        from telefuser.utils.video import save_video

        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        save_video(frames, str(VIDEOS_DIR / f"{req_id}.mp4"), fps=DEFAULT_TARGET_FPS, quality=6)

    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Build W03 latents dataset using TeleFuser wan22")
    parser.add_argument("--parallelism", type=int, default=1, help="Number of GPUs (1, 2, 4, 8)")
    parser.add_argument("--saved-steps", type=int, nargs="+", default=DEFAULT_SAVED_STEPS)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--aspect-ratio", default=DEFAULT_ASPECT_RATIO)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-root",
        default=os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo") + "/Wan2.2-T2V-A14B",
    )
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--feature-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable AdaTaylor feature cache (faster but approximated)."
            " Use --no-feature-cache for cleaner reference latents."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="DESTRUCTIVE: wipe latents/ and videos/ before running",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records (smoke test)")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        sys.exit(f"Dataset not found: {DATASET_PATH}")

    if args.clean:
        for d in (LATENTS_DIR, VIDEOS_DIR):
            if d.exists():
                print(f"Removing {d}")
                shutil.rmtree(d)
        for f in (INDEX_PATH, FAILURES_PATH, MANIFEST_PATH):
            if f.exists():
                f.unlink()

    LATENTS_DIR.mkdir(parents=True, exist_ok=True)

    width, height = get_image_size(args.resolution, args.aspect_ratio)
    print(f"Image size: {width}x{height}")

    write_manifest(args, width, height)
    print(f"Manifest: {MANIFEST_PATH}")

    done = load_done_set()
    print(f"Resume: {len(done)} (req_id, step) entries already in index.jsonl")

    records: list[dict] = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit is not None:
        records = records[: args.limit]
    print(f"Records to process: {len(records)}")

    print("Loading pipeline...")
    pipeline = build_pipeline(args.model_root, args.parallelism, args.feature_cache)
    print("Pipeline ready")

    stats = {"ok": 0, "skip": 0, "error": 0}
    pbar = tqdm(records, desc="latents")
    for rec in pbar:
        t0 = time.time()
        try:
            result = process_record(pipeline, rec, args, width, height, done)
        except Exception as exc:
            result = f"unhandled: {exc}"
        elapsed = time.time() - t0

        if result == "ok":
            stats["ok"] += 1
        elif result == "skip":
            stats["skip"] += 1
        else:
            stats["error"] += 1
            append_failure(
                {
                    "req_id": str(rec.get("req_id", "")),
                    "prompt": rec.get("prompt", ""),
                    "error": result,
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
            )
        pbar.set_postfix(ok=stats["ok"], skip=stats["skip"], err=stats["error"], t=f"{elapsed:.1f}s")

    print(f"\nDone: {stats}")
    print(f"  Index:    {INDEX_PATH}")
    print(f"  Latents:  {LATENTS_DIR}")
    if args.save_video:
        print(f"  Videos:   {VIDEOS_DIR}")
    if stats["error"] > 0:
        print(f"  Failures: {FAILURES_PATH}")


if __name__ == "__main__":
    main()
