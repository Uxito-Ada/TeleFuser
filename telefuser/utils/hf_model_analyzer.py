"""HuggingFace Model Analyzer - Generic model component detection.

This module analyzes HuggingFace model folders and auto-discovers model components
without hardcoding pipeline-specific logic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from telefuser.utils.logging import logger


@dataclass
class ModelComponentInfo:
    """Information about a model component."""

    name: str
    path: Union[str, list[str]]
    component_type: str  # "transformer", "vae", "text_encoder", "tokenizer", etc.
    estimated_size: int = 0  # Size in bytes


class HFModelAnalyzer:
    """Generic analyzer for HuggingFace model folders.

    Supports:
    - Standard Diffusers format (subfolders)
    - Flat format (all files in root)
    - Custom formats with model_index.json
    """

    # Common file patterns for different component types
    COMPONENT_PATTERNS = {
        "transformer": [
            ("transformer/", "diffusion_pytorch_model"),
            ("unet/", "diffusion_pytorch_model"),
            ("", "diffusion_pytorch_model"),  # Flat format
            ("", "model"),  # Generic model file
        ],
        "vae": [
            ("vae/", "diffusion_pytorch_model"),
            ("vae/", "model"),
            ("", "vae"),  # Flat format
            ("", "ae"),  # Autoencoder
        ],
        "text_encoder": [
            ("text_encoder/", "model"),
            ("text_encoder/", "pytorch_model"),
            ("", "text_encoder"),
            ("", "t5"),
            ("", "umt5"),
            ("", "clip"),
        ],
        "text_encoder_2": [
            ("text_encoder_2/", "model"),
            ("", "text_encoder_2"),
        ],
        "image_encoder": [
            ("image_encoder/", "model"),
            ("image_encoder/", "pytorch_model"),
            ("", "image_encoder"),
            ("", "vision_encoder"),
        ],
        "tokenizer": [
            ("tokenizer/", None),
            ("tokenizer_2/", None),
            ("", "tokenizer"),  # Flat format
        ],
        "scheduler": [
            ("scheduler/", None),
        ],
    }

    def __init__(self, model_root: str):
        self.model_root = Path(model_root)
        self.model_index: dict | None = None
        self.config: dict | None = None
        self._load_configs()

    def _load_configs(self):
        """Load model_index.json and config.json if present."""
        model_index_path = self.model_root / "model_index.json"
        if model_index_path.exists():
            try:
                with open(model_index_path, "r") as f:
                    self.model_index = json.load(f)
                logger.debug(f"Loaded model_index.json from {self.model_root}")
            except Exception as e:
                logger.warning(f"Failed to load model_index.json: {e}")

        config_path = self.model_root / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                logger.debug(f"Loaded config.json from {self.model_root}")
            except Exception as e:
                logger.warning(f"Failed to load config.json: {e}")

    def analyze(self) -> dict[str, ModelComponentInfo]:
        """Analyze the model folder and discover all components.

        Returns:
            Dict mapping component names to their info
        """
        components = {}

        # Strategy 1: Use model_index.json if available
        if self.model_index:
            components.update(self._analyze_from_model_index())

        # Strategy 2: Use config.json to infer structure
        if self.config and not components:
            components.update(self._analyze_from_config())

        # Strategy 3: Pattern matching on files
        if not components:
            components.update(self._analyze_from_patterns())

        return components

    def _analyze_from_model_index(self) -> dict[str, ModelComponentInfo]:
        """Analyze using model_index.json (standard HF format)."""
        components = {}

        for key, value in self.model_index.items():
            if key.startswith("_"):
                continue

            if isinstance(value, str):
                # Simple path reference
                component_path = self.model_root / value
                if component_path.exists():
                    components[key] = ModelComponentInfo(
                        name=key,
                        path=str(component_path),
                        component_type=key,
                        estimated_size=self._get_path_size(component_path),
                    )
            elif isinstance(value, dict):
                # Detailed config with subfolder
                subfolder = value.get("subfolder", "")
                component_path = self.model_root / subfolder if subfolder else self.model_root

                if component_path.exists():
                    # Find model files in the component folder
                    model_files = self._find_model_files(component_path)
                    if model_files:
                        components[key] = ModelComponentInfo(
                            name=key,
                            path=model_files if len(model_files) > 1 else model_files[0],
                            component_type=key,
                            estimated_size=self._get_path_size(component_path),
                        )

        return components

    def _analyze_from_config(self) -> dict[str, ModelComponentInfo]:
        """Analyze using config.json to infer model structure."""
        components = {}

        # Get model class from config
        class_name = self.config.get("_class_name", "").lower()
        model_type = self.config.get("model_type", "").lower()

        # Infer architecture from class name
        if "autoencoder" in class_name or "vae" in model_type:
            # This is a VAE folder
            model_files = self._find_model_files(self.model_root)
            if model_files:
                components["vae"] = ModelComponentInfo(
                    name="vae",
                    path=model_files[0] if len(model_files) == 1 else model_files,
                    component_type="vae",
                    estimated_size=self._get_path_size(self.model_root),
                )
        elif "transformer" in class_name or "unet" in class_name:
            # This is a Transformer/DiT folder
            model_files = self._find_model_files(self.model_root)
            if model_files:
                components["transformer"] = ModelComponentInfo(
                    name="transformer",
                    path=model_files[0] if len(model_files) == 1 else model_files,
                    component_type="transformer",
                    estimated_size=self._get_path_size(self.model_root),
                )

        return components

    def _analyze_from_patterns(self) -> dict[str, ModelComponentInfo]:
        """Analyze using file patterns (fallback for non-standard formats)."""
        components = {}

        # Get all model files
        all_files = (
            list(self.model_root.rglob("*.safetensors"))
            + list(self.model_root.rglob("*.bin"))
            + list(self.model_root.rglob("*.pth"))
        )

        if not all_files:
            logger.warning(f"No model files found in {self.model_root}")
            return components

        # Sort by size (largest first) for better matching
        all_files.sort(key=lambda p: p.stat().st_size, reverse=True)

        # Try to identify each component
        identified_files = set()

        for component_type, patterns in self.COMPONENT_PATTERNS.items():
            if component_type in components:
                continue

            for subfolder, prefix in patterns:
                search_path = self.model_root / subfolder if subfolder else self.model_root

                if not search_path.exists():
                    continue

                # Find matching files
                if prefix:
                    # Look for files with specific prefix
                    matching_files = [f for f in all_files if f.parent == search_path and prefix in f.name.lower()]
                else:
                    # Look for any model files in the folder
                    if subfolder:
                        matching_files = [f for f in all_files if f.parent == search_path]
                    else:
                        continue  # Skip flat format without prefix

                if matching_files:
                    # Filter out already identified files
                    new_files = [f for f in matching_files if str(f) not in identified_files]
                    if not new_files:
                        continue

                    # Add component
                    if len(new_files) == 1:
                        path = str(new_files[0])
                    else:
                        path = [str(f) for f in sorted(new_files)]

                    components[component_type] = ModelComponentInfo(
                        name=component_type,
                        path=path,
                        component_type=component_type,
                        estimated_size=sum(f.stat().st_size for f in new_files),
                    )

                    # Mark files as identified
                    for f in new_files:
                        identified_files.add(str(f))

                    break  # Found this component, move to next

        # If still no transformer found, use the largest remaining file
        if "transformer" not in components and "unet" not in components:
            remaining = [f for f in all_files if str(f) not in identified_files]
            if remaining:
                largest = remaining[0]
                components["transformer"] = ModelComponentInfo(
                    name="transformer",
                    path=str(largest),
                    component_type="transformer",
                    estimated_size=largest.stat().st_size,
                )
                identified_files.add(str(largest))

        return components

    def _find_model_files(self, path: Path) -> list[str]:
        """Find all model files in a path."""
        files = []
        for ext in ["*.safetensors", "*.bin", "*.pth"]:
            files.extend(path.glob(ext))
        return [str(f) for f in sorted(files)]

    def _get_path_size(self, path: Path) -> int:
        """Get total size of a file or directory."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return 0

    def _find_first_file(
        self,
        patterns: list[str],
        exclude_patterns: tuple[str, ...] = (),
    ) -> str | None:
        """Find the first matching file under the model root."""
        for pattern in patterns:
            for candidate in sorted(self.model_root.rglob(pattern)):
                candidate_str = str(candidate)
                if any(exclude in candidate_str for exclude in exclude_patterns):
                    continue
                if candidate.is_file():
                    return candidate_str
        return None

    def get_gemma_root_path(self) -> str | None:
        """Get Gemma model root containing config and tokenizer files."""
        for candidate in sorted(self.model_root.rglob("config.json")):
            parent = candidate.parent
            parent_str = str(parent).lower()
            if "gemma" not in parent_str:
                continue
            if (parent / "tokenizer.json").exists() or (parent / "tokenizer.model").exists():
                return str(parent)
        return None

    def get_upsampler_path(self) -> str | None:
        """Get LTX spatial upsampler weights."""
        return self._find_first_file(
            ["*spatial-upscaler*.safetensors", "*upscaler*x2*.safetensors", "*upscaler*x1.5*.safetensors"]
        )

    def get_distilled_lora_path(self) -> str | None:
        """Get optional distilled LoRA weights."""
        return self._find_first_file(["*distilled-lora*.safetensors"])

    def get_transformer_path(self) -> Union[str, list[str]] | None:
        """Get transformer/DiT model path."""
        components = self.analyze()
        if "transformer" in components:
            return components["transformer"].path
        if "unet" in components:
            return components["unet"].path
        return self._find_first_file(
            ["*.safetensors"],
            exclude_patterns=("upscaler", "lora", "temporal", "spatial"),
        )

    def get_vae_path(self) -> Union[str, list[str]] | None:
        """Get VAE model path."""
        components = self.analyze()
        if "vae" in components:
            return components["vae"].path
        return None

    def get_text_encoder_path(self) -> Union[str, list[str]] | None:
        """Get text encoder model path."""
        components = self.analyze()
        if "text_encoder" in components:
            return components["text_encoder"].path
        return None

    def get_tokenizer_path(self) -> str | None:
        """Get tokenizer path."""
        components = self.analyze()
        if "tokenizer" in components:
            return components["tokenizer"].path
        return None

    def get_image_encoder_path(self) -> Union[str, list[str]] | None:
        """Get image encoder model path."""
        components = self.analyze()
        if "image_encoder" in components:
            return components["image_encoder"].path
        return None


def discover_model_components(model_root: str) -> dict[str, Union[str, list[str]]]:
    """Convenience function to discover model components.

    Args:
        model_root: Root folder of the model

    Returns:
        Dict mapping component names to their paths
    """
    analyzer = HFModelAnalyzer(model_root)
    components = analyzer.analyze()

    # Convert to simple dict
    return {name: info.path for name, info in components.items()}
