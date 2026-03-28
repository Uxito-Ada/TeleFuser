"""
Pytest fixtures for server tests.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

# Set environment variable at module level
os.environ["TELEFUSER_SECURITY_LEVEL"] = "NONE"


@pytest.fixture(scope="session")
def pipeline_path():
    """Get the path to the fake pipeline."""
    return str(Path(__file__).parent / "pipeline" / "fake_t2v_pipeline.py")


@pytest.fixture(scope="function")
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="function")
def sample_video(temp_cache_dir):
    """Create a sample video file for testing."""
    video_path = Path(temp_cache_dir) / "test_output.mp4"
    video_path.write_bytes(b"FAKE_VIDEO_DATA")
    return video_path


@pytest.fixture(scope="function")
def running_server(pipeline_path, temp_cache_dir):
    """
    Start a test server and yield its URL.

    Uses subprocess for better isolation.
    """
    import socket

    # Find an available port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    server_script = Path(__file__).parent / "run_test_server.py"

    # Start server in subprocess
    env = os.environ.copy()
    env["TELEFUSER_SECURITY_LEVEL"] = "NONE"

    proc = subprocess.Popen(
        [
            sys.executable,
            str(server_script),
            "--port",
            str(port),
            "--host",
            host,
            "--cache-dir",
            temp_cache_dir,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Wait for server to be ready
    server_info = {"base_url": base_url, "ready": False, "proc": proc}
    max_wait = 30
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{base_url}/v1/service/status", timeout=1)
            if response.status_code == 200:
                server_info["ready"] = True
                break
        except Exception:
            # Check if process died
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                print(f"Server process exited early:\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}")
                pytest.skip(f"Server process exited with code {proc.returncode}")
        time.sleep(0.5)

    if not server_info["ready"]:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        pytest.skip("Server failed to start within timeout")

    yield server_info

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
