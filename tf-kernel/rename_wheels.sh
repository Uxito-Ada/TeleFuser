#!/usr/bin/env bash
set -ex

WHEEL_DIR="dist"

# Get PyTorch version
torch_version=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "unknown")

# Note: Git commit hash is no longer included in wheel names
# Version is managed by setuptools-scm based on git tags

wheel_files=($WHEEL_DIR/*.whl)
for wheel in "${wheel_files[@]}"; do
    # Skip if not a file
    [ -f "$wheel" ] || continue

    # Replace 'linux' with 'manylinux2014' (only once)
    # Handle pattern: -linux_x86_64.whl -> -manylinux2014_x86_64.whl
    intermediate_wheel="${wheel/-linux_/-manylinux2014_}"

    # Extract the current python version from the wheel name
    if [[ $intermediate_wheel =~ -cp([0-9]+)- ]]; then
        cp_version="${BASH_REMATCH[1]}"
    else
        echo "Could not extract Python version from wheel name: $intermediate_wheel"
        continue
    fi

    # Extract full version string (including any dev/rc suffix and local version)
    # Example: tf_kernel-0.1.1.dev4+gbfcfb42cf.d20260310-cp310-abi3-linux_x86_64.whl
    # We need to extract: 0.1.1.dev4+gbfcfb42cf.d20260310
    if [[ $intermediate_wheel =~ tf_kernel-([0-9]+\.[0-9]+\.[0-9]+\.[a-z]+[0-9]*)(\+[a-z0-9.]+)?- ]]; then
        # Version like 0.1.1.dev4 (without local tag)
        base_version="${BASH_REMATCH[1]}"  # e.g., 0.1.1.dev4
        # Local version tag (after +), if any
        local_tag="${BASH_REMATCH[2]}"  # e.g., +gbfcfb42cf.d20260310
        # Full version including local tag
        full_version="${base_version}${local_tag}"
        # Extract existing local version tag (after +), if any
        if [[ -n "$local_tag" ]]; then
            existing_local="${local_tag:1}"  # Remove leading '+'
        else
            existing_local=""
        fi
    elif [[ $intermediate_wheel =~ tf_kernel-([0-9]+\.[0-9]+\.[0-9]+)(\+[a-z0-9.]+)?- ]]; then
        # Version like 0.1.1 (without dev/rc suffix)
        base_version="${BASH_REMATCH[1]}"  # e.g., 0.1.1
        local_tag="${BASH_REMATCH[2]}"  # e.g., +gbfcfb42cf.d20260310
        full_version="${base_version}${local_tag}"
        if [[ -n "$local_tag" ]]; then
            existing_local="${local_tag:1}"
        else
            existing_local=""
        fi
    else
        echo "Could not extract version from wheel name: $intermediate_wheel"
        continue
    fi

    # Detect CUDA version
    cuda_tag=""
    if ls /usr/local/ 2>/dev/null | grep -q "12.4"; then
        cuda_tag="cu124"
    elif ls /usr/local/ 2>/dev/null | grep -q "12.8"; then
        cuda_tag="cu128"
    elif ls /usr/local/ 2>/dev/null | grep -q "13.0"; then
        cuda_tag="cu130"
    fi

    # Build the new local version tag
    # PEP 440: local version label can only have one '+' separator
    # Format: torch{version}.{cuda_tag}.{git_info}
    local_version="torch${torch_version}"
    if [[ -n "$cuda_tag" ]]; then
        local_version="${local_version}.${cuda_tag}"
    fi
    if [[ -n "$existing_local" ]]; then
        local_version="${local_version}.${existing_local}"
    fi

    # Construct new wheel name
    # Replace the full version part with version+new_local_version
    new_wheel="${intermediate_wheel/tf_kernel-${full_version}/tf_kernel-${base_version}+${local_version}}"

    if [[ "$wheel" != "$new_wheel" ]]; then
        echo "Renaming $wheel to $new_wheel"
        mv -- "$wheel" "$new_wheel"
    fi
done
echo "Wheel renaming completed."
echo "Note: Version is managed by setuptools-scm based on git tags"
