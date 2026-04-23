"""Reusable task and manifest templates for service-compatible example scripts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .pipeline_contract import default_task_contract

_BASE_PARAMETER_CONTRACTS: dict[str, dict[str, Any]] = {
    "prompt": {
        "type": "string",
        "required": True,
        "default": "",
        "description": "Primary generation prompt.",
        "exposed": True,
    },
    "negative_prompt": {
        "type": "string",
        "required": False,
        "default": "",
        "description": "Negative prompt appended to the example defaults.",
        "exposed": True,
    },
    "seed": {
        "type": "integer",
        "required": False,
        "default": 42,
        "description": "Random seed for reproducibility.",
        "exposed": True,
    },
    "output_path": {
        "type": "string",
        "required": False,
        "default": "",
        "description": "Destination path for the generated artifact.",
        "exposed": True,
    },
    "resolution": {
        "type": "string",
        "required": False,
        "default": "720p",
        "description": "Target output resolution.",
        "enum": ["480p", "720p", "1080p"],
        "exposed": True,
    },
    "aspect_ratio": {
        "type": "string",
        "required": False,
        "default": "16:9",
        "description": "Requested aspect ratio when the example supports it.",
        "enum": ["16:9", "9:16", "4:3", "3:4", "1:1", "2:3", "3:2"],
        "exposed": True,
    },
    "target_video_length": {
        "type": "integer",
        "required": False,
        "default": 5,
        "description": "Requested output video length in seconds when supported.",
        "exposed": True,
    },
    "output_format": {
        "type": "string",
        "required": False,
        "default": "png",
        "description": "Requested image output format.",
        "enum": ["png", "jpg", "jpeg", "webp"],
        "exposed": True,
    },
}

_TASK_PARAMETER_KEYS: dict[str, tuple[str, ...]] = {
    "t2v": ("prompt", "negative_prompt", "seed", "resolution", "aspect_ratio", "target_video_length", "output_path"),
    "i2v": ("prompt", "negative_prompt", "seed", "resolution", "aspect_ratio", "target_video_length", "output_path"),
    "fl2v": ("prompt", "negative_prompt", "seed", "resolution", "aspect_ratio", "target_video_length", "output_path"),
    "vc": ("prompt", "negative_prompt", "seed", "resolution", "target_video_length", "output_path"),
    "vsr": ("prompt", "negative_prompt", "seed", "resolution", "output_path"),
    "s2v": ("prompt", "negative_prompt", "seed", "resolution", "target_video_length", "output_path"),
    "t2i": ("prompt", "negative_prompt", "seed", "resolution", "aspect_ratio", "output_format", "output_path"),
    "i2i": ("prompt", "negative_prompt", "seed", "resolution", "aspect_ratio", "output_format", "output_path"),
    "edit": ("prompt", "negative_prompt", "seed", "resolution", "aspect_ratio", "output_format", "output_path"),
}


def build_task_contract_template(
    task: str,
    *,
    parameter_overrides: dict[str, dict[str, Any]] | None = None,
    excluded_parameters: Iterable[str] | None = None,
    required_inputs: Iterable[str] | None = None,
    optional_inputs: Iterable[str] | None = None,
    media_type: str | None = None,
) -> dict[str, Any]:
    """Build a reusable task-contract mapping for an example pipeline."""
    base_contract = default_task_contract(task)
    task_name = base_contract.task
    parameters = {
        name: deepcopy(_BASE_PARAMETER_CONTRACTS[name])
        for name in _TASK_PARAMETER_KEYS.get(task_name, tuple())
        if name in _BASE_PARAMETER_CONTRACTS
    }

    for name, override in (parameter_overrides or {}).items():
        merged = deepcopy(parameters.get(name, {}))
        merged.update(override)
        parameters[name] = merged

    for name in excluded_parameters or ():
        parameters.pop(name, None)

    return {
        "media_type": media_type or base_contract.media_type,
        "required_inputs": list(required_inputs or base_contract.required_inputs),
        "optional_inputs": list(optional_inputs or base_contract.optional_inputs),
        "parameters": parameters,
    }


def build_pipeline_manifest(
    *,
    pipeline_name: str,
    supported_tasks: Iterable[str],
    task_contracts: dict[str, dict[str, Any]] | None = None,
    execution_mode: str = "serial_single_pipeline",
    effective_max_concurrent_tasks: int = 1,
    entrypoints: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a service-facing pipeline manifest from standard task templates."""
    normalized_tasks = [str(task).strip() for task in supported_tasks if str(task).strip()]
    manifest_task_contracts = {
        task: deepcopy((task_contracts or {}).get(task) or build_task_contract_template(task))
        for task in normalized_tasks
    }
    supported_media_types = list(
        dict.fromkeys(contract.get("media_type", "unknown") for contract in manifest_task_contracts.values())
    )

    return {
        "contract_version": "v1",
        "pipeline_name": pipeline_name,
        "supported_tasks": normalized_tasks,
        "supported_media_types": supported_media_types,
        "execution_mode": execution_mode,
        "effective_max_concurrent_tasks": effective_max_concurrent_tasks,
        "entrypoints": entrypoints or {"get_pipeline": "get_pipeline", "run_with_file": "run_with_file"},
        "task_contracts": manifest_task_contracts,
    }
