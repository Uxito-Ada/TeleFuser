#!/bin/bash
# Run all Triton kernel benchmarks
#
# Usage:
#   ./benchmarks/kernel/run_benchmarks.sh
#   ./benchmarks/kernel/run_benchmarks.sh --kernel rmsnorm
#   ./benchmarks/kernel/run_benchmarks.sh --quick  # CI mode with fewer configs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default settings
QUICK_MODE=false
KERNEL_FILTER=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            export CI=true
            shift
            ;;
        --kernel)
            KERNEL_FILTER="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick       Run in quick mode (fewer configs, for CI)"
            echo "  --kernel NAME Run only the specified kernel benchmark (rmsnorm, rotary, scale_shift)"
            echo "  --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Run all benchmarks"
            echo "  $0 --quick                  # Quick run for CI"
            echo "  $0 --kernel rmsnorm         # Run only RMSNorm benchmark"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo "========================================================================"
echo " TeleFuser Triton Kernel Benchmarks"
echo "========================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Quick mode: $QUICK_MODE"
echo "Kernel filter: ${KERNEL_FILTER:-all}"
echo ""

# Function to run a benchmark
run_benchmark() {
    local script="$1"
    local name="$2"
    
    if [[ -n "$KERNEL_FILTER" ]] && [[ "$name" != *"$KERNEL_FILTER"* ]]; then
        echo "Skipping $name (filter: $KERNEL_FILTER)"
        return
    fi
    
    echo ""
    echo "----------------------------------------------------------------------"
    echo " Running: $name"
    echo "----------------------------------------------------------------------"
    
    if python "$script" "${EXTRA_ARGS[@]}"; then
        echo "✅ $name completed"
    else
        echo "❌ $name failed"
        return 1
    fi
}

# Track failures
FAILURES=()

# Run benchmarks
if [[ -z "$KERNEL_FILTER" ]] || [[ "$KERNEL_FILTER" == "rmsnorm" ]]; then
    run_benchmark "$SCRIPT_DIR/bench_rmsnorm.py" "rmsnorm" || FAILURES+=("rmsnorm")
fi

if [[ -z "$KERNEL_FILTER" ]] || [[ "$KERNEL_FILTER" == "rotary" ]]; then
    run_benchmark "$SCRIPT_DIR/bench_rotary.py" "rotary" || FAILURES+=("rotary")
fi

if [[ -z "$KERNEL_FILTER" ]] || [[ "$KERNEL_FILTER" == "scale_shift" ]]; then
    run_benchmark "$SCRIPT_DIR/bench_scale_shift.py" "scale_shift" || FAILURES+=("scale_shift")
fi

# Summary
echo ""
echo "========================================================================"
echo " Benchmark Summary"
echo "========================================================================"

if [ ${#FAILURES[@]} -eq 0 ]; then
    echo "✅ All benchmarks passed!"
    exit 0
else
    echo "❌ Failed benchmarks: ${FAILURES[*]}"
    exit 1
fi