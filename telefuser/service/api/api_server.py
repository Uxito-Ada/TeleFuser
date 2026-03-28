"""API Server for TeleFuser Service."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from telefuser.utils.logging import logger

from ..core.file_service import FileService
from ..core.pipeline_service import PipelineService
from ..core.task_manager import TaskManager, TaskStatus
from ..core.task_service import MediaGenerationService
from . import routers


class ApiServer:
    """Main API server coordinating all services.

    Manages FastAPI app lifecycle, route setup, task processing loop,
    and OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        max_queue_size: int = 10,
        app: FastAPI | None = None,
        task_manager: TaskManager | None = None,
        enable_rate_limit: bool = True,
        enable_logging: bool = False,
        enable_openai_api: bool = True,
    ) -> None:
        self.app = app or FastAPI(
            title="TeleFuser API",
            description="API for video and image generation using TeleFuser framework. "
            "OpenAI compatible endpoints available at /v1/images and /v1/videos.",
            version="1.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )
        self.file_service: FileService | None = None
        self.inference_service: PipelineService | None = None
        self.media_service: MediaGenerationService | None = None
        self.max_queue_size = max_queue_size
        self._task_manager = task_manager
        self.enable_openai_api = enable_openai_api

        self.processing_thread: threading.Thread | None = None
        self.stop_processing = threading.Event()

        self._setup_routes()

        from .middleware import setup_middleware

        setup_middleware(self.app, enable_rate_limit=enable_rate_limit, enable_logging=enable_logging)

    @property
    def task_manager(self):
        """Get task manager (supports dependency injection)."""
        if self._task_manager is not None:
            return self._task_manager

    def _setup_routes(self) -> None:
        """Setup all API routes."""
        tasks_router = routers.tasks.setup_routes(self)
        files_router = routers.files.setup_routes(self)
        service_router = routers.service.setup_routes(self)

        self.app.include_router(tasks_router)
        self.app.include_router(files_router)
        self.app.include_router(service_router)

        if self.enable_openai_api:
            try:
                from .openai import image_routes, video_routes

                image_router = image_routes.setup_routes(self)
                video_router = video_routes.setup_routes(self)

                self.app.include_router(image_router)
                self.app.include_router(video_router)

                logger.info("OpenAI compatible API enabled at /v1/images and /v1/videos")
            except Exception as e:
                logger.warning(f"Failed to setup OpenAI routes: {e}")

    def _write_file_sync(self, file_path: Path, content: bytes) -> None:
        with open(file_path, "wb") as buffer:
            buffer.write(content)

    def _stream_file_response(self, file_path: Path, filename: str | None = None) -> StreamingResponse:
        """Stream file response with proper security checks."""
        assert self.file_service is not None, "File service is not initialized"

        try:
            resolved_path = file_path.resolve()

            # Security: ensure file is within allowed directories
            if not str(resolved_path).startswith(str(self.file_service.output_video_dir.resolve())):
                raise HTTPException(status_code=403, detail="Access to this file is not allowed")

            if not resolved_path.exists() or not resolved_path.is_file():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

            file_size = resolved_path.stat().st_size
            actual_filename = filename or resolved_path.name

            mime_type = "application/octet-stream"
            if actual_filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                mime_type = "video/mp4"
            elif actual_filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                mime_type = "image/jpeg"

            headers = {
                "Content-Disposition": f'attachment; filename="{actual_filename}"',
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            }

            def file_stream_generator(file_path: str, chunk_size: int = 1024 * 1024):
                with open(file_path, "rb") as file:
                    while chunk := file.read(chunk_size):
                        yield chunk

            return StreamingResponse(
                file_stream_generator(str(resolved_path)),
                media_type=mime_type,
                headers=headers,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error occurred while processing file stream response: {e}")
            raise HTTPException(status_code=500, detail="File transfer failed")

    async def _validate_image_url(self, image_url: str) -> bool:
        """Validate image URL is accessible."""
        from ..core.config import server_config

        if not image_url or not image_url.startswith("http"):
            return True

        try:
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return False

            timeout = httpx.Timeout(connect=5.0, read=5.0)
            verify = server_config.ssl_cert_path if server_config.ssl_cert_path else server_config.verify_ssl
            async with httpx.AsyncClient(verify=verify, timeout=timeout) as client:
                response = await client.head(image_url, follow_redirects=True)
                return response.status_code < 400
        except Exception as e:
            logger.warning(f"URL validation failed for {image_url}: {str(e)}")
            return False

    def _ensure_processing_thread_running(self) -> None:
        """Ensure the task processing thread is running."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(target=self._task_processing_loop, daemon=True)
            self.processing_thread.start()
            logger.info("Started task processing thread")

    def _task_processing_loop(self) -> None:
        """Main loop that processes tasks from the queue one by one."""
        logger.info("Task processing loop started")

        while not self.stop_processing.is_set():
            task_id = self.task_manager.get_next_pending_task()

            if task_id is None:
                time.sleep(1)
                continue

            task_info = self.task_manager.get_task(task_id)
            if task_info and task_info.status == TaskStatus.PENDING:
                logger.info(f"Processing task {task_id}")
                self._process_single_task(task_info)

        logger.info("Task processing loop stopped")

    def _process_single_task(self, task_info: Any) -> None:
        """Process a single task with proper locking."""
        assert self.media_service is not None, "Media service is not initialized"

        task_id = task_info.task_id
        message = task_info.message

        lock_acquired = self.task_manager.acquire_processing_lock(task_id, timeout=1)
        if not lock_acquired:
            logger.error(f"Task {task_id} failed to acquire processing lock")
            self.task_manager.fail_task(task_id, "Failed to acquire processing lock")
            return

        try:
            self.task_manager.start_task(task_id)

            if task_info.stop_event.is_set():
                logger.info(f"Task {task_id} cancelled before processing")
                self.task_manager.fail_task(task_id, "Task cancelled")
                return

            result = asyncio.run(self.media_service.generate_media_with_stop_event(message, task_info.stop_event))

            if result:
                self.task_manager.complete_task(task_id, result.output_path)
                logger.info(f"Task {task_id} completed successfully")
            else:
                if task_info.stop_event.is_set():
                    self.task_manager.fail_task(task_id, "Task cancelled during processing")
                    logger.info(f"Task {task_id} cancelled during processing")
                else:
                    self.task_manager.fail_task(task_id, "Generation failed")
                    logger.error(f"Task {task_id} generation failed")

        except Exception as e:
            logger.exception(f"Task {task_id} processing failed")
            self.task_manager.fail_task(task_id, str(e))
        finally:
            if lock_acquired:
                self.task_manager.release_processing_lock(task_id)

    def initialize_services(self, cache_dir: Path, inference_service: PipelineService) -> None:
        """Initialize file and media services."""
        self.file_service = FileService(cache_dir)
        self.inference_service = inference_service
        self.media_service = MediaGenerationService(self.file_service, inference_service)

    async def cleanup(self) -> None:
        """Cleanup resources and stop processing thread."""
        self.stop_processing.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        if self.file_service:
            await self.file_service.cleanup()

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app
