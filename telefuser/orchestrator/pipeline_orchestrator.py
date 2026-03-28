"""Pipeline orchestrator for multi-stage inference workflows.

Manages execution of pipeline stages with support for parallel groups,
result routing, and progress tracking.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from telefuser.utils.logging import logger

from .stage_wrapper import EnhancedPipelineStageWrapper, StageConfig, StageResult, StageTask


@dataclass
class RequestState:
    """State tracking for a single request through the pipeline."""

    request_id: str
    current_stage_id: int = 0
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    start_time: float = field(default_factory=time.time)
    stage_times: List[float] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    request_inputs: Any = None


class FlexiblePipelineOrchestrator:
    """Orchestrates multi-stage pipeline execution with parallel groups."""

    def __init__(
        self,
        pipeline: Any,
        stage_configs: List[StageConfig],
    ):
        self.pipeline = pipeline
        self.stage_configs = stage_configs

        self.stages: List[EnhancedPipelineStageWrapper] = []

        self.request_states: Dict[str, RequestState] = {}

        self.output_handler_task: Optional[asyncio.Task] = None
        self._running = False

        # Shared result queue for all stages - eliminates polling
        self._result_queue: asyncio.Queue = asyncio.Queue()

        self._init_stages()

    def _init_stages(self):
        """Initialize stage wrappers from configurations."""
        shared_locks: Dict[str, threading.Lock] = {}

        for config in self.stage_configs:
            if not hasattr(self.pipeline, config.pipeline_attr):
                raise ValueError(
                    f"Pipeline does not have attribute '{config.pipeline_attr}'. "
                    f"Available attributes: {[a for a in dir(self.pipeline) if not a.startswith('_')]}"
                )

            stage_callable = getattr(self.pipeline, config.pipeline_attr)

            shared_lock = None
            if config.shared_lock_group:
                if config.shared_lock_group not in shared_locks:
                    shared_locks[config.shared_lock_group] = threading.Lock()
                shared_lock = shared_locks[config.shared_lock_group]
                logger.debug(f"[Stage-{config.stage_id}] Using shared lock group: {config.shared_lock_group}")

            wrapper = EnhancedPipelineStageWrapper(
                stage_id=config.stage_id,
                stage_name=config.stage_name,
                stage_callable=stage_callable,
                stage_config=config,
                result_queue=self._result_queue,
                shared_call_lock=shared_lock,
            )

            self.stages.append(wrapper)

        logger.info(f"Initialized {len(self.stages)} stages from config (single-request mode)")

    async def start(self):
        """Start the orchestrator and all stages."""
        if self._running:
            return

        self._running = True

        for stage in self.stages:
            await stage.start()

        self.output_handler_task = asyncio.create_task(self._output_handler_loop())

        logger.info("FlexiblePipelineOrchestrator started")

    async def stop(self):
        """Stop the orchestrator and all stages."""
        self._running = False

        # Stop OutputHandler
        if self.output_handler_task:
            self.output_handler_task.cancel()
            try:
                await self.output_handler_task
            except asyncio.CancelledError:
                pass

        # Stop all stages
        for stage in self.stages:
            await stage.stop()

        logger.info("FlexiblePipelineOrchestrator stopped")

    async def _output_handler_loop(self):
        """Route stage results to appropriate request queues."""
        logger.info("OutputHandler started")

        while self._running:
            try:
                # Wait for any stage to produce a result (no polling)
                result = await self._result_queue.get()

                req_state = self.request_states.get(result.request_id)
                if req_state is None:
                    logger.warning(f"Request {result.request_id} not found, may have been aborted")
                    continue

                await req_state.queue.put(result)
                req_state.current_stage_id = result.stage_id

                logger.debug(f"OutputHandler routed result from Stage-{result.stage_id} to request {result.request_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"OutputHandler error: {e}")

        logger.info("OutputHandler stopped")

    def _group_stages(self) -> List[List[Any]]:
        """Group stages by parallel_group for concurrent execution."""
        groups = []
        current_group = []
        current_group_id = None

        for stage in self.stages:
            group_id = stage.stage_config.parallel_group

            if group_id is None:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([stage])
                current_group_id = None
            elif group_id == current_group_id:
                current_group.append(stage)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [stage]
                current_group_id = group_id

        if current_group:
            groups.append(current_group)

        return groups

    async def _execute_parallel_group(
        self,
        stages: List[Any],
        request_id: str,
        inputs: Any,
        req_state: RequestState,
    ) -> Dict[int, StageResult]:
        """Execute a group of parallel stages concurrently."""
        logger.info(
            f"[{request_id}] Executing parallel group with {len(stages)} stages: {[s.stage_name for s in stages]}"
        )

        for stage in stages:
            task = StageTask(
                request_id=request_id,
                stage_id=stage.stage_id,
                inputs=inputs,
                context=req_state.context,
            )
            await stage.submit(task)
            logger.info(f"[{request_id}] Submitted to Stage-{stage.stage_id} ({stage.stage_name})")

        results = {}
        pending_stages = {s.stage_id: s for s in stages}

        while pending_stages:
            result = await req_state.queue.get()

            if result.stage_id in pending_stages:
                results[result.stage_id] = result
                del pending_stages[result.stage_id]

                logger.info(
                    f"[{request_id}] Stage-{result.stage_id} completed ({len(pending_stages)} remaining in group)"
                )
            else:
                logger.warning(f"[{request_id}] Unexpected result from Stage-{result.stage_id}")

        return results

    async def generate(
        self,
        request_id: str,
        inputs: Any,
        initial_context: Optional[Dict[str, Any]] = None,
    ):
        """Execute the full pipeline for a request.

        Yields progress updates and final result.
        """
        logger.info(f"[{request_id}] Starting pipeline generation")

        req_state = RequestState(
            request_id=request_id,
            context=initial_context or {},
        )
        req_state.request_inputs = inputs
        self.request_states[request_id] = req_state

        try:
            stage_groups = self._group_stages()
            logger.info(f"[{request_id}] Stage groups: {[[s.stage_name for s in g] for g in stage_groups]}")

            for group_idx, group in enumerate(stage_groups):
                group_start = time.time()

                if len(group) == 1:
                    stage = group[0]
                    result = await self._execute_single_stage(stage, request_id, req_state.request_inputs, req_state)

                    if result.error:
                        yield {
                            "finished": False,
                            "stage_id": stage.stage_id,
                            "stage_name": stage.stage_name,
                            "error": result.error,
                        }
                        return

                    output_key = stage.stage_config.metadata.get("output_key", stage.stage_name)
                    req_state.context[stage.stage_name] = result.outputs
                    if output_key != stage.stage_name:
                        req_state.context[output_key] = result.outputs

                    stage_time = time.time() - group_start
                    req_state.stage_times.append(stage_time)

                    yield {
                        "finished": False,
                        "stage_id": stage.stage_id,
                        "stage_name": stage.stage_name,
                        "stage_time_ms": stage_time * 1000,
                        "metrics": result.metrics,
                    }
                else:
                    results = await self._execute_parallel_group(group, request_id, req_state.request_inputs, req_state)

                    for stage in group:
                        result = results[stage.stage_id]
                        if result.error:
                            logger.error(f"[{request_id}] Stage-{stage.stage_id} failed: {result.error}")
                            yield {
                                "finished": False,
                                "stage_id": stage.stage_id,
                                "stage_name": stage.stage_name,
                                "error": result.error,
                            }
                            return

                    sorted_stages = sorted(group, key=lambda s: s.stage_id)
                    for stage in sorted_stages:
                        result = results[stage.stage_id]
                        output_key = stage.stage_config.metadata.get("output_key", stage.stage_name)
                        req_state.context[stage.stage_name] = result.outputs
                        if output_key != stage.stage_name:
                            req_state.context[output_key] = result.outputs

                    for stage in sorted_stages:
                        result = results[stage.stage_id]
                        req_state.stage_times.append(result.metrics.get("stage_time_ms", 0) / 1000)

                        yield {
                            "finished": False,
                            "stage_id": stage.stage_id,
                            "stage_name": stage.stage_name,
                            "stage_time_ms": result.metrics.get("stage_time_ms", 0),
                            "metrics": result.metrics,
                        }

                logger.info(f"[{request_id}] Group {group_idx} completed in {time.time() - group_start:.3f}s")

            total_time = time.time() - req_state.start_time
            # Final output defaults to the last stage's output_key (or stage_name).
            final_output = None
            if self.stages:
                last = self.stages[-1]
                last_key = last.stage_config.metadata.get("output_key", last.stage_name)
                final_output = req_state.context.get(last_key, req_state.context.get(last.stage_name))

            yield {
                "finished": True,
                "final_output": final_output,
                "context": req_state.context,
                "metrics": {
                    "total_time_s": total_time,
                    "stage_times_ms": [t * 1000 for t in req_state.stage_times],
                },
            }

            logger.info(
                f"[{request_id}] Pipeline completed in {total_time:.3f}s "
                f"(stages: {', '.join(f'{t:.2f}s' for t in req_state.stage_times)})"
            )

        finally:
            self.request_states.pop(request_id, None)

    async def _execute_single_stage(
        self,
        stage: Any,
        request_id: str,
        inputs: Any,
        req_state: RequestState,
    ) -> StageResult:
        """Execute a single stage."""
        task = StageTask(
            request_id=request_id,
            stage_id=stage.stage_id,
            inputs=inputs,
            context=req_state.context,
        )

        await stage.submit(task)
        logger.info(f"[{request_id}] Submitted to Stage-{stage.stage_id} ({stage.stage_name})")

        result = await req_state.queue.get()

        if not result.error:
            logger.info(
                f"[{request_id}] Stage-{stage.stage_id} ({stage.stage_name}) "
                f"completed in {result.metrics.get('stage_time_ms', 0):.2f}ms"
            )

        return result
