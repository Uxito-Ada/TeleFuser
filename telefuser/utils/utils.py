"""General utility functions."""

from __future__ import annotations

import hashlib
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
    """Import a symbol from a Python file by path.

    The module is registered in ``sys.modules`` under a path-derived unique
    name so files with the same basename do not overwrite each other; on
    load failure the entry is removed to avoid leaking partial state.
    """
    abs_path = os.path.abspath(file_path)
    basename = os.path.splitext(os.path.basename(abs_path))[0]
    path_hash = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:12]
    module_name = f"_telefuser_pipeline_{basename}_{path_hash}"

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for file_path={abs_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return getattr(module, function_name)


def get_example_name(file_path: str, ext: str = "mp4") -> str:
    """Generate example name from file path."""
    folder = os.path.basename(os.path.dirname(file_path))
    name = os.path.splitext(os.path.basename(file_path))[0]
    return f"{folder}_{name}.{ext}"
