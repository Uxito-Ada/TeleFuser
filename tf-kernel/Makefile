.PHONY: help check-deps install-deps tree ln submodule install build build-auto build-sm80 build-sm90 build-sm100 clean rebuild test test-cov test-cpu format format-check docs docs-clean docs-serve update test-wheel

# ---------------------------
# Build resource controls
# ---------------------------
# By default, build uses all available CPU cores, but users can override:
#   make build MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"
NPROC ?= $(shell nproc 2>/dev/null || echo 1)
MAX_JOBS ?= $(NPROC)
CMAKE_BUILD_PARALLEL_LEVEL ?= $(MAX_JOBS)
UV_BUILD_DIR ?= build
CMAKE_POLICY_VERSION_MINIMUM ?= 3.5

# Target SM architecture selection
# Options: AUTO (detect local GPU), ALL (all supported architectures), SM80, SM90, SM100
TF_KERNEL_TARGET_SM ?= ALL

# Show help for each target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

check-deps: ## Check and install required Python formatting dependencies
	@command -v isort >/dev/null 2>&1 || (echo "Installing isort..." && pip install isort)

install-deps: ## Install Python formatting tools (isort)
	pip install scikit-build-core isort

tree: ## Show project directory structure
	@tree --prune -I "__pycache__|*.egg-info|*.so|build|3rdparty|dist"

submodule: ## Initialize and update git submodules
	@git submodule update --init --recursive

ln: submodule ## Create compilation database
	@rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DCMAKE_POLICY_VERSION_MINIMUM=3.5

install: submodule ## Install package in development mode
	@pip install -e ".[dev]" --no-build-isolation

build: install-deps submodule ## Build and install wheel package (default: ALL)
	@rm -rf dist/* || true && \
		CMAKE_POLICY_VERSION_MINIMUM=$(CMAKE_POLICY_VERSION_MINIMUM) \
		MAX_JOBS=$(MAX_JOBS) \
		CMAKE_BUILD_PARALLEL_LEVEL=$(CMAKE_BUILD_PARALLEL_LEVEL) \
		CMAKE_ARGS="$(CMAKE_ARGS) -DTF_KERNEL_TARGET_SM=$(TF_KERNEL_TARGET_SM)" \
		uv build --wheel -Cbuild-dir=$(UV_BUILD_DIR) . --verbose --color=always --no-build-isolation && \
		$(MAKE) rename-wheels && \
		pip3 install dist/*whl --force-reinstall --no-deps

build-auto: ## Build for auto-detected GPU architecture
	@$(MAKE) build TF_KERNEL_TARGET_SM=AUTO

build-sm80: ## Build for SM80 architecture (Ampere, Ada Lovelace)
	@$(MAKE) build TF_KERNEL_TARGET_SM=SM80

build-sm90: ## Build for SM90 architecture (Hopper H100)
	@$(MAKE) build TF_KERNEL_TARGET_SM=SM90

build-sm100: ## Build for SM100 architecture (Blackwell)
	@$(MAKE) build TF_KERNEL_TARGET_SM=SM100

rename-wheels: ## Rename wheel files to include CUDA version and manylinux2014 tag
	@echo "Renaming wheel files..."
	@bash rename_wheels.sh

clean: ## Remove build artifacts
	@rm -rf  dist *.egg-info

rebuild: clean submodule build ## Clean and rebuild the project
	@echo "Succeed to rebuild"

test: ## Run all tests
	@pytest tests/ -v

test-cov: ## Run all tests with coverage report
	@pytest tests/ -v --cov=tf_kernel --cov-report=term-missing --cov-report=html

test-cpu: ## Run tests that don't require GPU
	@pytest tests/ -v -m "not gpu"

test-wheel: ## Check wheel for undefined symbols and architecture-specific issues
	@python3 scripts/check_wheel_symbols.py

format: check-deps ## Format all source files (isort only, black/clang-format disabled)
	@echo "Formatting Python imports with isort..."
	@find tf_kernel tests -name '*.py' | xargs isort
	@echo "Running pre-commit hooks..."
	@pre-commit run --all-files

format-check: check-deps ## Check format without modifying files (for CI)
	@echo "Checking Python imports with isort..."
	@find tf_kernel tests benchmark -name '*.py' | xargs isort --check-only --diff
	@echo "Running pre-commit hooks (check only)..."
	@pre-commit run --all-files --show-diff-on-failure

docs: ## Build documentation
	@cd docs && LC_ALL=C.UTF-8 LANG=C.UTF-8 make html

docs-clean: ## Clean documentation build
	@cd docs && make clean

docs-serve: ## Serve documentation locally (requires sphinx-autobuild)
	@cd docs && make livehtml

FILES_TO_UPDATE = tf_kernel/version.py \
                 pyproject.toml \
                 pyproject_rocm.toml \
                 pyproject_cpu.toml

update: ## Update version numbers across project files. Usage: make update <new_version>
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Version required. Usage: make update <new_version>"; \
		exit 1; \
	fi
	@OLD_VERSION=$$(grep "version" tf_kernel/version.py | cut -d '"' -f2); \
	NEW_VERSION=$(filter-out $@,$(MAKECMDGOALS)); \
	echo "Updating version from $$OLD_VERSION to $$NEW_VERSION"; \
	for file in $(FILES_TO_UPDATE); do \
		if [ "$(shell uname)" = "Darwin" ]; then \
			sed -i '' -e "s/$$OLD_VERSION/$$NEW_VERSION/g" $$file; \
		else \
			sed -i -e "s/$$OLD_VERSION/$$NEW_VERSION/g" $$file; \
		fi \
	done; \
	echo "Version update complete"

%:
	@:
