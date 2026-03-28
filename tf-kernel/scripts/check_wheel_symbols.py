#!/usr/bin/env python3
"""
Check wheel files for undefined symbols in shared libraries.

This script extracts .so files from a wheel and checks for undefined symbols
that should have been defined (e.g., architecture-specific symbols like FP4
that shouldn't be exported in SM80/SM90 builds).

Usage:
    python scripts/check_wheel_symbols.py path/to/wheel.whl
    python scripts/check_wheel_symbols.py dist/*.whl
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple


def get_undefined_symbols(so_path: str) -> List[Tuple[str, str]]:
    """
    Get undefined symbols from a shared library using nm.
    
    Returns a list of (symbol_name, symbol_type) tuples where type is 'U' for undefined.
    """
    try:
        result = subprocess.run(
            ["nm", "-D", so_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        undefined = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                # Format: "                 U symbol_name" or "address U symbol_name"
                if parts[0] == "U":
                    undefined.append((parts[1], "U"))
                elif len(parts) >= 3 and parts[1] == "U":
                    undefined.append((parts[2], "U"))
        
        return undefined
    except Exception as e:
        print(f"Warning: Failed to run nm on {so_path}: {e}")
        return []


def get_defined_symbols(so_path: str) -> Dict[str, str]:
    """
    Get defined symbols from a shared library using nm.
    
    Returns a dict mapping symbol_name -> symbol_type.
    """
    try:
        result = subprocess.run(
            ["nm", "-D", so_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        defined = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                # Format: "address T symbol_name" (T = defined text section)
                symbol_type = parts[1]
                if symbol_type in "TtWw":  # Text/Data symbols
                    defined[parts[2]] = symbol_type
        
        return defined
    except Exception as e:
        print(f"Warning: Failed to run nm on {so_path}: {e}")
        return {}


def check_wheel_symbols(wheel_path: str) -> Tuple[bool, Dict[str, Dict]]:
    """
    Check a wheel file for undefined symbol issues.
    
    Returns (success, results) where results is a dict with architecture-specific info.
    """
    results = {
        "sm80": {"undefined": [], "fp4_symbols": [], "status": "ok"},
        "sm90": {"undefined": [], "fp4_symbols": [], "status": "ok"},
        "sm100": {"undefined": [], "fp4_symbols": [], "status": "ok"},
    }
    
    # FP4-related symbols that should ONLY appear in SM100
    fp4_patterns = [
        r"sageattn3_",
        r"scaled_fp4_",
        r"cutlass_scaled_fp4",
        r"nvfp4",
    ]
    
    def is_fp4_symbol(symbol: str) -> bool:
        return any(re.search(p, symbol, re.IGNORECASE) for p in fp4_patterns)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract wheel
        with zipfile.ZipFile(wheel_path, 'r') as zf:
            zf.extractall(tmpdir)
        
        # Find all common_ops.so files
        for arch in ["sm80", "sm90", "sm100"]:
            pattern = os.path.join(tmpdir, "tf_kernel", arch, "common_ops*.so")
            so_files = glob.glob(pattern)
            
            if not so_files:
                results[arch]["status"] = "missing"
                continue
            
            so_path = so_files[0]
            print(f"\n{'='*60}")
            print(f"Checking: {arch}/common_ops.so")
            print(f"Path: {so_path}")
            print(f"{'='*60}")
            
            # Check for undefined symbols (excluding system/external deps)
            undefined = get_undefined_symbols(so_path)
            defined = get_defined_symbols(so_path)
            
            # Filter out known external dependencies (Python, CUDA, PyTorch, etc.)
            external_prefixes = [
                "Py",           # Python
                "_Py",          # Python internal
                "_ZTV",         # C++ vtable
                "_ZTI",         # C++ typeinfo
                "_ZTS",         # C++ typeinfo name
                "_Unwind",      # C++ exception handling
                "__gxx_personality",
                "c10::",        # PyTorch C10
                "at::",         # ATen
                "_ZN2at",       # ATen (mangled: at::)
                "_ZNK2at",      # ATen const methods (mangled)
                "_ZN3c10",      # C10 (mangled: c10::)
                "_ZNK3c10",     # C10 const methods (mangled)
                "_ZN6caffe2",   # caffe2 namespace
                "torch::",      # Torch
                "_ZN5torch",    # Torch (mangled: torch::)
                "_ZNR5torch",   # Torch Library methods (mangled)
                "cuda",         # CUDA runtime
                "cublas",       # cuBLAS
                "nvrtc",        # NVRTC
                "libcudart",
                "GLIBC",        # glibc
                "_ZSt",         # std:: (C++ standard library)
                "_ZNSt",        # std:: (C++ standard library)
                "__cxa",        # C++ ABI
                "_ZNSt3__1",    # libc++ std::
                "_Zdl",         # operator delete
                "__udiv",       # GCC division builtins
                "__umod",       # GCC modulo builtins
                "cuTensor",     # CUDA tensor operations
            ]
            
            def is_external(symbol: str) -> bool:
                return any(symbol.startswith(p) or p in symbol for p in external_prefixes)
            
            # Find problematic undefined symbols (not from external deps)
            problematic = []
            for sym, _ in undefined:
                if not is_external(sym):
                    problematic.append(sym)
            
            # Check for FP4 symbols in non-SM100 builds
            fp4_in_build = []
            for sym in defined.keys():
                if is_fp4_symbol(sym):
                    fp4_in_build.append(sym)
            
            results[arch]["undefined"] = problematic
            results[arch]["fp4_symbols"] = fp4_in_build
            
            if problematic:
                print(f"  ❌ Found {len(problematic)} problematic undefined symbols:")
                for sym in problematic[:10]:  # Show first 10
                    print(f"     - {sym}")
                if len(problematic) > 10:
                    print(f"     ... and {len(problematic) - 10} more")
                results[arch]["status"] = "error"
            else:
                print(f"  ✓ No problematic undefined symbols")
            
            if arch != "sm100" and fp4_in_build:
                print(f"  ❌ ERROR: Found {len(fp4_in_build)} FP4 symbols (should only be in SM100):")
                for sym in fp4_in_build[:5]:
                    print(f"     - {sym}")
                if len(fp4_in_build) > 5:
                    print(f"     ... and {len(fp4_in_build) - 5} more")
                results[arch]["status"] = "error"
            elif arch == "sm100" and fp4_in_build:
                print(f"  ✓ Found {len(fp4_in_build)} FP4 symbols (expected for SM100)")
            elif arch != "sm100" and not fp4_in_build:
                print(f"  ✓ No FP4 symbols (correct for {arch})")
    
    # Determine overall success
    success = all(r["status"] in ("ok", "missing") for r in results.values())
    
    return success, results


def main():
    parser = argparse.ArgumentParser(
        description="Check wheel files for undefined symbols"
    )
    parser.add_argument(
        "wheel",
        nargs="?",
        default="dist/*.whl",
        help="Path to wheel file (default: dist/*.whl)"
    )
    
    args = parser.parse_args()
    
    # Expand user home directory and environment variables
    wheel_path = os.path.expanduser(os.path.expandvars(args.wheel))
    
    # Find wheel files
    if os.path.isfile(wheel_path):
        wheels = [wheel_path]
    else:
        wheels = glob.glob(wheel_path)
    
    if not wheels:
        print(f"Error: No wheel files found matching '{args.wheel}'")
        print("Please build the wheel first with 'make build'")
        return 1
    
    # Use the most recent wheel
    wheel_path = max(wheels, key=os.path.getmtime)
    print(f"Checking wheel: {wheel_path}")
    
    success, results = check_wheel_symbols(wheel_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for arch, info in results.items():
        status_icon = {
            "ok": "✓",
            "error": "❌",
            "missing": "⚠"
        }.get(info["status"], "?")
        
        print(f"  {status_icon} {arch}: {info['status']}")
        if info["undefined"]:
            print(f"      - {len(info['undefined'])} undefined symbols")
        if info["fp4_symbols"] and arch != "sm100":
            print(f"      - {len(info['fp4_symbols'])} unexpected FP4 symbols")
    
    if success:
        print(f"\n✓ All checks passed!")
        return 0
    else:
        print(f"\n❌ Symbol check failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())