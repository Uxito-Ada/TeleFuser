"""General utility functions."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import TypeVar

T = TypeVar("T")


def split_list(lst: list[T], n: int) -> list[list[T]]:
    """Split list into n parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def import_function_from_file(file_path: str, function_name: str):
    """Import function from Python file."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def get_example_name(file_path: str, ext: str = "mp4") -> str:
    """Generate example name from file path."""
    folder = os.path.basename(os.path.dirname(file_path))
    name = os.path.splitext(os.path.basename(file_path))[0]
    return f"{folder}_{name}.{ext}"
