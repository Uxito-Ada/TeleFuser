"""Pipeline service with integrated security validation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import torch.multiprocessing as mp

from telefuser.utils.logging import logger

from ..security.security_validator import (
    PipelineSecurityValidator,
    SecurityError,
    SecurityLevel,
    validate_with_report,
)
from .config import server_config
from .pipeline_contract import load_pipeline_contract
from .pipeline_runner import PipelineRunner

mp.set_start_method("spawn", force=True)


class PipelineService:
    """Pipeline service with security validation for pipeline configuration files.

    Security features:
    - Static AST analysis for dangerous operations
    - Import restriction and verification
    - Content pattern matching
    - Optional sandboxed execution
    - Configurable security levels
    """

    def __init__(self, security_level: SecurityLevel | None = None) -> None:
        """Initialize PipelineService."""
        self.is_running = False
        self.pipeline = None
        self.task: str | None = None
        self.ppl_file: str | None = None
        self.parallelism: int | None = None

        self._module: ModuleType | None = None
        self._runner: PipelineRunner | None = None
        self._contract = None
        self._declared_contract = False

        self.security_level = security_level or getattr(server_config, "security_level", SecurityLevel.STRICT)
        self.security_validator = PipelineSecurityValidator(
            security_level=self.security_level,
            max_file_size=getattr(server_config, "max_ppl_file_size", 1024 * 1024),
        )

        logger.info(f"PipelineService initialized with security_level={self.security_level.name}")

    def _load_pipeline_module(self, ppl_file: str) -> ModuleType:
        """Load pipeline module from file path with a stable unique module name.

        Using a unique module name avoids collisions when importing different pipeline files with the same basename.
        """
        import importlib.util
        import sys

        resolved = str(Path(ppl_file).expanduser().resolve())
        module_hash = hashlib.md5(resolved.encode("utf-8")).hexdigest()[:12]
        module_name = f"telefuser_ppl_{module_hash}"

        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec for {ppl_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _validate_pipeline_file(self, ppl_file: str) -> bool:
        """Validate pipeline file for security issues."""
        if self.security_level == SecurityLevel.NONE:
            logger.warning("Security validation is disabled (SecurityLevel.NONE)")
            return True

        try:
            logger.info(f"Validating pipeline file: {ppl_file}")
            result = self.security_validator.validate_file(ppl_file)

            if result.is_safe:
                logger.info(f"Pipeline file '{ppl_file}' passed security validation")
                if result.warnings:
                    logger.warning(f"  {len(result.warnings)} warnings found")
                    for w in result.warnings[:3]:
                        logger.warning(f"    Line {w.line_number}: {w.description}")
                return True
            else:
                report = validate_with_report(ppl_file)
                logger.error(f"Security validation failed:\n{report}")

                critical_count = sum(1 for v in result.violations if v.severity == "critical")
                if critical_count > 0:
                    raise SecurityError(
                        f"Pipeline file contains {critical_count} critical security violations. "
                        f"Execution blocked. Run with detailed report for more info."
                    )

                if getattr(server_config, "allow_unsafe_pipelines", False):
                    logger.warning("Allowing unsafe pipeline due to server_config.allow_unsafe_pipelines=True")
                    return True
                else:
                    raise SecurityError(
                        f"Pipeline file failed security validation with {len(result.violations)} violations. "
                        f"Set security_level=SecurityLevel.NONE to bypass, "
                        f"or server_config.allow_unsafe_pipelines=True to allow with warnings."
                    )

        except SecurityError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during security validation: {e}")
            if getattr(server_config, "strict_validation", True):
                raise SecurityError(f"Validation failed with error: {e}")
            return True

    def start_pipeline(self, ppl_file: str, parallelism: int, task: str, skip_validation: bool = False) -> bool:
        """Start the pipeline with security validation."""
        if self.is_running:
            logger.warning("Distributed inference service is already running")
            return True

        try:
            if not skip_validation:
                self._validate_pipeline_file(ppl_file)
            else:
                logger.warning("Skipping security validation for pipeline file")

            self.task = task
            self.ppl_file = ppl_file
            self.parallelism = parallelism

            self._module = self._load_pipeline_module(ppl_file)
            self._contract, self._declared_contract = load_pipeline_contract(
                self._module,
                ppl_file=ppl_file,
                default_task=task,
            )

            if task not in self._contract.supported_tasks:
                raise RuntimeError(
                    f"Pipeline contract for {self._contract.pipeline_name} does not declare support for task '{task}'"
                )

            get_pipeline_name = self._contract.entrypoints.get_pipeline
            run_with_file_name = self._contract.entrypoints.run_with_file
            if not hasattr(self._module, get_pipeline_name):
                raise RuntimeError(f"Pipeline file must define {get_pipeline_name}(parallelism=...)")
            if not hasattr(self._module, run_with_file_name):
                raise RuntimeError(f"Pipeline file must define {run_with_file_name}(...) for service execution")

            get_pipeline = getattr(self._module, get_pipeline_name)
            run_with_file = getattr(self._module, run_with_file_name)

            self.pipeline = get_pipeline(parallelism=parallelism)
            self._runner = PipelineRunner(pipeline=self.pipeline, run_with_file=run_with_file, module=self._module)
            self.is_running = True

            logger.info(f"Pipeline service started with security_level={self.security_level.name}")
            self._log_contract_startup_summary()
            return True

        except SecurityError as e:
            logger.error(f"Security validation failed: {e}")
            return False
        except Exception as e:
            logger.exception(f"Error occurred while starting distributed inference service: {str(e)}")
            asyncio.run(self.aclose())
            return False

    async def aclose(self) -> None:
        """Async close and cleanup resources."""
        if not self.is_running:
            return

        try:
            if self._runner is not None:
                await self._runner.shutdown()
        except Exception as e:
            logger.warning(f"Error during pipeline shutdown: {e}")
        finally:
            self._runner = None
            self._module = None
            self._contract = None
            self._declared_contract = False

            if self.pipeline is not None:
                try:
                    del self.pipeline
                except Exception:
                    pass
                self.pipeline = None

            self.is_running = False

    def stop_pipeline_inference(self) -> None:
        """Stop the pipeline inference service (sync wrapper)."""
        try:
            asyncio.run(self.aclose())
        except RuntimeError:
            # If called from within an event loop, schedule cleanup and return.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.aclose())
            except Exception:
                self.is_running = False

    async def run_task_with_stop_event(
        self,
        task_data: dict[str, Any],
        stop_event: threading.Event,
        timeout_s: float | None = None,
        output_root: str | None = None,
    ) -> dict[str, Any]:
        """Run a task and return a normalized result dict."""
        if not self.is_running or self.pipeline is None or self._runner is None:
            raise RuntimeError("Pipeline service is not started")

        timeout_s = timeout_s if timeout_s is not None else float(getattr(server_config, "task_timeout", 600))

        result = await self._runner.run(
            task_data=task_data,
            stop_event=stop_event,
            timeout_s=timeout_s,
            output_root=output_root,
        )

        return {
            "task_id": task_data.get("task_id", ""),
            "status": result.status,
            "output_path": result.output_path or "",
            "message": result.message,
            "raw": result.raw,
        }

    def server_metadata(self) -> dict:
        """Get server metadata."""
        metadata = {
            "pipeline_file": self.ppl_file,
            "parallelism": self.parallelism,
            "task": self.task,
            "security_level": self.security_level.name if self.security_level else "NONE",
            "runner": "PipelineRunner",
            "declared_pipeline_contract": self._declared_contract,
        }
        if self._contract is not None:
            metadata.update(self._contract.to_metadata())
        return metadata

    def supported_tasks(self) -> tuple[str, ...]:
        """Return tasks declared by the loaded pipeline contract."""
        if self._contract is None:
            return tuple()
        return self._contract.supported_tasks

    def get_task_contract(self, task: str):
        """Return the task-level contract for a declared task, if available."""
        if self._contract is None:
            return None
        return self._contract.get_task_contract(task)

    def _log_contract_startup_summary(self) -> None:
        """Log a concise startup summary of the active pipeline contract."""
        if self._contract is None:
            return

        pipeline_summary = {
            "pipeline_name": self._contract.pipeline_name,
            "declared_pipeline_contract": self._declared_contract,
            "supported_tasks": list(self._contract.supported_tasks),
            "supported_media_types": list(self._contract.supported_media_types),
            "execution_mode": self._contract.execution_mode,
            "effective_max_concurrent_tasks": self._contract.effective_max_concurrent_tasks,
            "entrypoints": {
                "get_pipeline": self._contract.entrypoints.get_pipeline,
                "run_with_file": self._contract.entrypoints.run_with_file,
            },
            "metadata_endpoint": "/v1/service/metadata",
        }
        logger.info(
            "Pipeline contract startup summary: "
            f"{json.dumps(pipeline_summary, ensure_ascii=True, sort_keys=True, default=str)}"
        )

        for task in self._contract.supported_tasks:
            task_contract = self._contract.get_task_contract(task)
            if task_contract is None:
                continue

            task_summary = {
                "task": task,
                "media_type": task_contract.media_type,
                "required_inputs": list(task_contract.required_inputs),
                "optional_inputs": list(task_contract.optional_inputs),
                "parameters": {
                    name: {
                        "type": parameter.type,
                        "required": parameter.required,
                        "default": parameter.default,
                        "enum": list(parameter.enum),
                    }
                    for name, parameter in task_contract.parameters.items()
                    if parameter.exposed
                },
                "metadata_endpoint": "/v1/service/metadata",
            }
            logger.info(
                "Pipeline task startup summary: "
                f"{json.dumps(task_summary, ensure_ascii=True, sort_keys=True, default=str)}"
            )

    def __del__(self) -> None:
        """Destructor - attempts cleanup but should not be relied upon."""
        self._cleanup_pipeline()

    def _cleanup_pipeline(self) -> None:
        """Safely clean up pipeline resources."""
        if getattr(self, "pipeline", None) is not None:
            try:
                del self.pipeline
                self.pipeline = None
            except Exception:
                pass

    def close(self) -> None:
        """Explicitly close and cleanup resources."""
        self.stop_pipeline_inference()

    def __enter__(self) -> PipelineService:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Context manager exit - ensures cleanup."""
        self.close()
        return False  # Don't suppress exceptions


class SecurePipelineService(PipelineService):
    """Pre-configured secure pipeline service with strict validation."""

    def __init__(self) -> None:
        super().__init__(security_level=SecurityLevel.STRICT)

    def start_pipeline(self, ppl_file: str, parallelism: int, task: str, **kwargs) -> bool:
        """Start pipeline with strict validation (skip_validation not allowed)."""
        if kwargs.get("skip_validation"):
            raise SecurityError("SecurePipelineService does not allow skip_validation")
        return super().start_pipeline(ppl_file, parallelism, task, skip_validation=False)
