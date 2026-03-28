#!/bin/bash
# One-click launcher for TeleFuser example regression tests.
# See examples/regression_test/README.md for detailed usage.
#
# Usage:
#   bash scripts/run_regression_test.sh --list
#   bash scripts/run_regression_test.sh --family wan_video --max-parallel 4
#   bash scripts/run_regression_test.sh --force --update-baseline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ ! -f "pyproject.toml" ]; then
    echo "Error: Cannot find project root (pyproject.toml missing)"
    exit 1
fi

echo "=========================================="
echo "TeleFuser Example Regression Tests"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

python -m examples.regression_test.run_regression "$@"
