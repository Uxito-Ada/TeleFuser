"""HuggingFace Model Utilities - Common functions for HF Diffusers format support.

This module provides shared utilities for loading models from HuggingFace format:
- Path resolution for diffusers folder structure
- Model downloading and caching
- File discovery helpers
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from telefuser.utils.logging import logger

# Import the analyzer
from .hf_model_analyzer import HFModelAnalyzer, discover_model_components


def is_hf_model_id(path_or_id: str) -> bool:
    """Check if the input is a HuggingFace model ID (not a local path).

    A model ID typically looks like "organization/model-name" and contains a slash.
    Local paths either don't have a slash, start with /, or point to existing directories.

    Args:
        path_or_id: Input string to check

    Returns:
        True if it looks like a HF model ID
    """
    # If it's an existing local path, it's not a model ID
    if os.path.exists(path_or_id):
        return False

    # If it contains a slash but doesn't exist locally, treat as model ID
    # Model IDs like "Wan-AI/Wan2.1-T2V-1.3B"
    if "/" in path_or_id and not path_or_id.startswith(("/", "./", "../")):
        return True

    return False


def get_safetensors_files(folder: str, prefix: str = "") -> Union[str, list[str]]:
    """Get safetensors files from a folder, handling sharded models.

    Args:
        folder: Path to the folder containing model files
        prefix: Prefix to filter files (e.g., "diffusion_pytorch_model")

    Returns:
        Single file path or list of file paths for sharded models
    """
    if not os.path.isdir(folder):
        raise ValueError(f"Not a directory: {folder}")

    # Look for safetensors files
    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".safetensors") and (not prefix or f.startswith(prefix))]
    )

    if not files:
        # Try .bin files as fallback
        files = sorted([f for f in os.listdir(folder) if f.endswith(".bin") and (not prefix or f.startswith(prefix))])

    if not files:
        raise FileNotFoundError(f"No model files found in {folder}" + (f" with prefix '{prefix}'" if prefix else ""))

    paths = [os.path.join(folder, f) for f in files]
    return paths[0] if len(paths) == 1 else paths


def resolve_hf_path(
    model_id_or_path: str,
    cache_dir: str | None = None,
) -> str:
    """Resolve a model ID or path to a local path.

    If model_id_or_path is a HuggingFace model ID, download/snapshot it to cache.
    If it's a local path, return as-is.

    Args:
        model_id_or_path: HF model ID or local path
        cache_dir: Optional cache directory for downloads

    Returns:
        Local path to the model folder
    """
    if not is_hf_model_id(model_id_or_path):
        # It's a local path
        if not os.path.exists(model_id_or_path):
            raise FileNotFoundError(f"Model path not found: {model_id_or_path}")
        return os.path.abspath(model_id_or_path)

    # It's a HF model ID, try to download
    try:
        from huggingface_hub import snapshot_download

        # Default cache location
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/telefuser/models")

        local_dir = os.path.join(cache_dir, model_id_or_path.replace("/", "--"))

        logger.info(f"Downloading model from HuggingFace: {model_id_or_path}")
        local_path = snapshot_download(
            repo_id=model_id_or_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        return local_path
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading models. Install with: pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download model {model_id_or_path}: {e}")


def get_component_path(
    model_root: str,
    component: str,
    default_subfolder: str | None = None,
    file_prefix: str = "diffusion_pytorch_model",
) -> Union[str, list[str]] | None:
    """Get the path to a model component following HF diffusers conventions.

    Args:
        model_root: Root folder of the model
        component: Component name (e.g., "transformer", "vae", "text_encoder")
        default_subfolder: Default subfolder name (uses component name if None)
        file_prefix: Prefix for model files

    Returns:
        Path to the component file(s), or None if not found
    """
    subfolder = default_subfolder or component
    component_path = os.path.join(model_root, subfolder)

    if not os.path.exists(component_path):
        return None

    if os.path.isdir(component_path):
        try:
            return get_safetensors_files(component_path, file_prefix)
        except FileNotFoundError:
            # Try without prefix for text encoders
            if component == "text_encoder":
                try:
                    return get_safetensors_files(component_path, "model")
                except FileNotFoundError:
                    pass
            return None
    else:
        # It's a file
        return component_path


def detect_model_task(model_root: str) -> str:
    """Try to detect the model task from folder structure and config files.

    Args:
        model_root: Root folder of the model

    Returns:
        Detected task type (t2v, i2v, t2i, etc.) or empty string
    """
    # Check for image encoder (indicates I2V)
    if os.path.exists(os.path.join(model_root, "image_encoder")):
        if os.path.exists(os.path.join(model_root, "transformer")) or os.path.exists(os.path.join(model_root, "unet")):
            # Video model with image encoder = I2V
            return "i2v"

    # Check model_index.json if exists
    model_index_path = os.path.join(model_root, "model_index.json")
    if os.path.exists(model_index_path):
        try:
            import json

            with open(model_index_path, "r") as f:
                config = json.load(f)

            # Check for _class_name
            class_name = config.get("_class_name", "").lower()

            if "video" in class_name:
                if "image" in model_root.lower() or "i2v" in model_root.lower():
                    return "i2v"
                return "t2v"
            elif "image" in class_name or "image" in model_root.lower():
                return "t2i"
            elif "text" in class_name and "image" in class_name:
                return "t2i"
        except Exception:
            pass

    # Infer from folder name
    folder_name = os.path.basename(model_root).lower()
    if "i2v" in folder_name:
        return "i2v"
    elif "t2v" in folder_name or "text-to-video" in folder_name:
        return "t2v"
    elif "t2i" in folder_name or "text-to-image" in folder_name:
        return "t2i"
    elif "image-to-image" in folder_name or "i2i" in folder_name:
        return "i2i"

    return ""
