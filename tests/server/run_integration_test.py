#!/usr/bin/env python3
"""
Integration Test Runner

This script starts the test server, runs tests against it, and then shuts it down.
"""

import subprocess
import sys
import time
from pathlib import Path

import requests


def wait_for_server(url, timeout=30):
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/v1/service/status", timeout=1)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run integration tests")
    parser.add_argument("--port", "-p", type=int, default=18003, help="Server port")
    parser.add_argument("--test-only", action="store_true", help="Only run tests, don't start server")
    args = parser.parse_args()

    host = "127.0.0.1"
    port = args.port
    base_url = f"http://{host}:{port}"

    server_process = None

    if not args.test_only:
        print("=" * 60)
        print("Starting Test Server...")
        print("=" * 60)

        # Start server in subprocess
        server_script = Path(__file__).parent / "run_test_server.py"
        server_process = subprocess.Popen(
            [sys.executable, str(server_script), "--port", str(port), "--host", host],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        print(f"Waiting for server at {base_url}...")
        if not wait_for_server(base_url, timeout=30):
            print("✗ Server failed to start")
            server_process.terminate()
            return 1

        print("✓ Server is ready")
        print()

    # Run tests
    print("=" * 60)
    print("Running Tests...")
    print("=" * 60)

    try:
        # Import and run test client
        from client.test_client import TestClient, run_basic_tests

        success = run_basic_tests(base_url)

        if success:
            # Run load test
            from client.test_client import run_load_test

            run_load_test(base_url, num_tasks=2)

        return 0 if success else 1

    except Exception as e:
        print(f"Test error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        if server_process:
            print("\n" + "=" * 60)
            print("Shutting down server...")
            print("=" * 60)
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("✓ Server stopped")


if __name__ == "__main__":
    sys.exit(main())
