"""Config serialization utilities for dump operations."""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch


def serialize_value(obj: Any) -> Any:
    """Serialize a value to JSON-compatible type.

    Args:
        obj: Value to serialize

    Returns:
        JSON-compatible representation
    """
    if obj is None:
        return None
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return serialize_config(obj)
    if isinstance(obj, (list, tuple)):
        return [serialize_value(v) for v in obj]
    if isinstance(obj, dict):
        return {k: serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)


def serialize_config(config: Any) -> dict:
    """Serialize dataclass config to dict.

    Args:
        config: Dataclass instance to serialize

    Returns:
        Dictionary with serialized values
    """
    if not hasattr(config, "__dataclass_fields__"):
        return {}
    return {f.name: serialize_value(getattr(config, f.name)) for f in fields(config)}
