# TeleFuser Server Test Suite

This directory contains a complete test environment for the TeleFuser server, including:

- **Fake T2V Pipeline**: A mock pipeline that simulates video generation without requiring actual models
- **Test Client**: Lightweight client for testing server APIs
- **Integration Tests**: pytest-based tests for server functionality

## Quick Start

### 1. Start the Test Server

```bash
cd /dev/shm/zuoxin/workspace/TeleFuser
python tests/server/run_test_server.py
```

Options:
```bash
python tests/server/run_test_server.py --port 18000 --host 127.0.0.1
```

### 2. Run Basic Tests

In another terminal:

```bash
cd /dev/shm/zuoxin/workspace/TeleFuser
python tests/server/client/test_client.py http://localhost:18000
```

### 3. Run pytest Tests

```bash
cd /dev/shm/zuoxin/workspace/TeleFuser
pytest tests/server/test_server.py -v
```

## Directory Structure

```
tests/server/
├── README.md                    # This file
├── __init__.py
├── run_test_server.py           # Server launcher script
├── pipeline/
│   ├── __init__.py
│   └── fake_t2v_pipeline.py     # Mock T2V pipeline
├── client/
│   ├── __init__.py
│   └── test_client.py           # Test client
├── fixtures/
│   └── __init__.py
└── test_server.py               # pytest test suite
```

## Fake Pipeline

The `fake_t2v_pipeline.py` provides:

- `get_pipeline(parallelism=1)`: Creates a mock pipeline
- `run(pipeline, prompt, ...)`: Generates fake video frames
- `run_with_file(...)`: Saves output to file
- Compatible interface with real pipelines

The fake pipeline:
- Simulates 2-5 seconds processing time
- Generates colored frames (no actual ML inference)
- Saves mock video files

## Test Client

The `TestClient` class provides:

- `health_check()`: Check server status
- `create_t2v_task()`: Create text-to-video tasks
- `get_task_status()`: Query task status
- `wait_for_task()`: Wait for completion
- `cancel_task()`: Cancel pending tasks

## Integration Tests

The pytest suite tests:

- Server lifecycle (start/stop)
- Task creation and validation
- Task processing (end-to-end)
- Queue management
- File operations
- Service metadata

## Usage Examples

### Using Test Client

```python
from tests.server.client.test_client import TestClient

client = TestClient("http://localhost:18000")

# Create task
result = client.create_t2v_task(
    prompt="A beautiful sunset",
    seed=42
)
task_id = result["task_id"]

# Wait for completion
status = client.wait_for_task(task_id)
print(f"Task completed: {status}")
```

### Using Fake Pipeline Directly

```python
from tests.server.pipeline.fake_t2v_pipeline import get_pipeline, run_with_file

pipe = get_pipeline(parallelism=1)

run_with_file(
    pipe,
    prompt="Test video",
    negative_prompt="",
    seed=42,
    resolution="480p",
    output_path="/tmp/output.mp4",
    aspect_ratio="16:9",
)
```

## Testing Checklist

When refactoring the server, ensure:

- [ ] Server starts without errors
- [ ] Health check endpoint works
- [ ] Can create T2V tasks
- [ ] Task status can be queried
- [ ] Tasks complete successfully
- [ ] Queue status is accurate
- [ ] Cancel task works
- [ ] File download works
- [ ] Service metadata is correct
- [ ] Multiple tasks can run sequentially
- [ ] Client SDK works correctly

## Troubleshooting

### Server won't start

Check:
1. Port is not in use: `lsof -i :18000`
2. Pipeline file exists: `ls tests/server/pipeline/fake_t2v_pipeline.py`
3. Dependencies installed: `pip install -e .`

### Tests fail with connection error

Ensure server is running before running tests:
```bash
# Terminal 1
python tests/server/run_test_server.py

# Terminal 2
python tests/server/client/test_client.py
```

### Import errors

Add project root to Python path:
```bash
export PYTHONPATH=/dev/shm/zuoxin/workspace/TeleFuser:$PYTHONPATH
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
- name: Start Test Server
  run: |
    python tests/server/run_test_server.py &
    sleep 5

- name: Run Tests
  run: |
    python tests/server/client/test_client.py
    pytest tests/server/test_server.py -v
```
