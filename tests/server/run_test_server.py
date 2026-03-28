#!/usr/bin/env python3
"""
Test Server Runner

This script starts a test server with the fake pipeline for integration testing.
"""

import os
import sys
from pathlib import Path

# Set environment variable BEFORE any imports
os.environ["TELEFUSER_SECURITY_LEVEL"] = "NONE"

import argparse

from telefuser.service.main import run_server


def main():
    parser = argparse.ArgumentParser(description="Run TeleFuser test server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=18000,
        help="Server port (default: 18000)",
    )
    parser.add_argument(
        "--cache-dir",
        "-c",
        default=None,
        help="Cache directory (default: auto-created temp dir)",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    args = parser.parse_args()

    # Get pipeline path
    pipeline_path = Path(__file__).parent / "pipeline" / "fake_t2v_pipeline.py"

    if not pipeline_path.exists():
        print(f"Error: Pipeline not found at {pipeline_path}")
        sys.exit(1)

    # Setup cache directory
    if args.cache_dir is None:
        import tempfile

        cache_dir = tempfile.mkdtemp(prefix="telefuser_test_")
        print(f"Using temp cache dir: {cache_dir}")
    else:
        cache_dir = args.cache_dir

    print("=" * 60)
    print("Starting TeleFuser Test Server")
    print("=" * 60)
    print(f"Pipeline: {pipeline_path}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Cache: {cache_dir}")
    print(f"Parallelism: {args.parallelism}")
    print("=" * 60)
    print()

    try:
        # Reload config to pick up environment variable
        from telefuser.service.core.config import load_config_from_env

        load_config_from_env()

        run_server(
            pipe_path=str(pipeline_path),
            task="t2v",
            port=args.port,
            host=args.host,
            cache_dir=cache_dir,
            parallelism=args.parallelism,
            enable_rate_limit=False,  # Disable rate limiting for testing
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\n\nServer error: {e}")
        raise


if __name__ == "__main__":
    main()
