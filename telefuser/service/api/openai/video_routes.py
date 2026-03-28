"""
OpenAI Compatible Video API Routes

Provides OpenAI-compatible REST API endpoints for video generation:
- POST /v1/videos - Create video generation task (async)
- GET /v1/videos - List video generation tasks
- GET /v1/videos/{video_id} - Get video generation status
- DELETE /v1/videos/{video_id} - Cancel/delete video generation
- GET /v1/videos/{video_id}/content - Download generated video

Reference: OpenAI Video API (https://platform.openai.com/docs/api-reference/videos)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from telefuser.service.core.task_manager import TaskManager, TaskStatus
from telefuser.utils.logging import logger

from .adapter import OpenAIRequestAdapter, OpenAIResponseAdapter
from .protocol import ErrorResponse, VideoGenerationsRequest, VideoListResponse, VideoResponse

if TYPE_CHECKING:
    from ..api_server import ApiServer


def create_router(api_server: ApiServer) -> APIRouter:
    """Create a new router with fresh routes for the given ApiServer instance."""
    router = APIRouter(prefix="/v1/videos", tags=["videos"])
    routes = VideoRoutes(api_server)

    @router.post("", response_model=VideoResponse)
    async def create_video(
        request: Request,
        prompt: str | None = Form(None),
        input_reference: UploadFile | None = File(None, description="Input image/video file"),
        reference_url: str | None = Form(None, description="URL of input reference"),
        model: str | None = Form(None),
        seconds: int | None = Form(4),
        size: str | None = Form(None),
        seed: int | None = Form(1024),
        negative_prompt: str | None = Form(None),
        output_path: str | None = Form(None),
    ) -> VideoResponse:
        """Create a video generation task (OpenAI compatible, async)."""
        content_type = request.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            body = await request.json()
            try:
                req = VideoGenerationsRequest(**body)
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e))
        else:
            input_path = None
            if input_reference and input_reference.filename:
                input_path = await routes._save_uploaded_file(input_reference, "input")
            elif reference_url:
                input_path = reference_url

            req = VideoGenerationsRequest(
                prompt=prompt or "",
                input_reference=input_path,
                model=model,
                seconds=seconds,
                size=size or "1024x576",
                seed=seed,
                negative_prompt=negative_prompt,
                output_path=output_path,
            )

        return await routes.create_video(req)

    @router.get("", response_model=VideoListResponse)
    async def list_videos(
        after: str | None = Query(None, description="Cursor for pagination"),
        limit: int | None = Query(20, ge=1, le=100, description="Number of results"),
        order: str | None = Query("desc", description="Sort order: asc or desc"),
    ) -> VideoListResponse:
        """List video generation tasks."""
        return await routes.list_videos(after=after, limit=limit, order=order)

    @router.get("/{video_id}", response_model=VideoResponse)
    async def retrieve_video(video_id: str) -> VideoResponse:
        """Get video generation status by ID."""
        return await routes.retrieve_video(video_id)

    @router.delete("/{video_id}", response_model=VideoResponse)
    async def delete_video(video_id: str) -> VideoResponse:
        """Cancel or delete a video generation task."""
        return await routes.delete_video(video_id)

    @router.get("/{video_id}/content")
    async def get_video_content(video_id: str) -> FileResponse:
        """Download a generated video by its ID."""
        return await routes.get_video_content(video_id)

    return router


class VideoRoutes:
    """Video API route handlers with dependency injection support."""

    def __init__(self, api_server: ApiServer) -> None:
        """Initialize with ApiServer instance."""
        self.api = api_server
        self.task_manager: TaskManager | None = getattr(api_server, "task_manager", None)
        self.file_service = getattr(api_server, "file_service", None)

    async def create_video(self, request: VideoGenerationsRequest) -> VideoResponse:
        """Handle video creation request (async - returns queued status immediately)."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        try:
            task_request = OpenAIRequestAdapter.to_task_request(request)
            task_id = self.task_manager.create_task(task_request)
            logger.info(f"Created video generation task: {task_id}")

            self._ensure_processing()

            response = OpenAIResponseAdapter.to_video_response(
                task_id=task_id,
                status=TaskStatus.PENDING.value,
                prompt=request.prompt,
                size=request.size or "1024x576",
                seconds=request.seconds or 4,
                model=request.model or "wan-video",
                progress=0,
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Video creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Video creation failed: {str(e)}")

    async def list_videos(self, after: str | None = None, limit: int = 20, order: str = "desc") -> VideoListResponse:
        """List video generation tasks with pagination."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        try:
            all_tasks = self.task_manager.get_all_tasks()

            # Filter to video tasks only
            video_tasks = []
            for task_id, task_data in all_tasks.items():
                task_type = task_data.get("task", "")
                if task_type in ["t2v", "i2v", "fl2v", "vc"]:
                    video_tasks.append((task_id, task_data))

            # Sort by start_time
            reverse = order.lower() == "desc"
            video_tasks.sort(key=lambda x: x[1].get("start_time") or "", reverse=reverse)

            # Apply cursor pagination
            if after:
                found = False
                filtered_tasks = []
                for task_id, task_data in video_tasks:
                    if found:
                        filtered_tasks.append((task_id, task_data))
                    if task_id == after:
                        found = True
                video_tasks = filtered_tasks

            video_tasks = video_tasks[:limit]

            # Convert to VideoResponse objects
            videos: List[VideoResponse] = []
            for task_id, task_data in video_tasks:
                video = OpenAIResponseAdapter.to_video_response(
                    task_id=task_id,
                    status=task_data.get("status", TaskStatus.PENDING.value),
                    prompt=task_data.get("prompt", ""),
                    size=task_data.get("resolution", ""),
                    seconds=task_data.get("target_video_length", 4),
                    model=task_data.get("model", "wan-video"),
                    output_path=task_data.get("output_path"),
                )
                videos.append(video)

            has_more = len(all_tasks) > len(videos)
            first_id = videos[0].id if videos else None
            last_id = videos[-1].id if videos else None

            return OpenAIResponseAdapter.to_video_list_response(
                videos=videos, has_more=has_more, first_id=first_id, last_id=last_id
            )

        except Exception as e:
            logger.exception(f"List videos failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

    async def retrieve_video(self, video_id: str) -> VideoResponse:
        """Get video generation status."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        task_status = self.task_manager.get_task_status(video_id)
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

        task_info = self.task_manager.get_task(video_id)
        message = task_info.message if task_info else None

        # Calculate progress (simplified)
        status = task_status.get("status", TaskStatus.PENDING.value)
        progress = 0
        if status == TaskStatus.COMPLETED.value:
            progress = 100
        elif status == TaskStatus.PROCESSING.value:
            progress = 50

        response = OpenAIResponseAdapter.to_video_response(
            task_id=video_id,
            status=status,
            prompt=message.prompt if message else "",
            size=message.resolution if message else "",
            seconds=message.target_video_length if message else 4,
            model=getattr(message, "model", None) or "wan-video",
            output_path=task_status.get("output_path"),
            progress=progress,
        )

        if status == TaskStatus.FAILED.value:
            response.error = {"message": task_status.get("error", "Unknown error")}

        return response

    async def delete_video(self, video_id: str) -> VideoResponse:
        """Cancel or delete a video generation task."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        task_status = self.task_manager.get_task_status(video_id)
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

        cancelled = self.task_manager.cancel_task(video_id)

        return VideoResponse(
            id=video_id,
            status="cancelled" if cancelled else "deleted",
            model="wan-video",
        )

    async def get_video_content(self, video_id: str) -> FileResponse:
        """Download generated video."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        if not self.file_service:
            raise HTTPException(status_code=503, detail="File service not initialized")

        task_info = self.task_manager.get_task(video_id)
        if not task_info:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

        task_status = self.task_manager.get_task_status(video_id)
        if task_status.get("status") != TaskStatus.COMPLETED.value:
            raise HTTPException(
                status_code=400,
                detail=f"Video {video_id} is not ready (status: {task_status.get('status')})",
            )

        output_path = task_info.output_path or task_status.get("output_path")
        if not output_path:
            raise HTTPException(status_code=404, detail=f"Video {video_id} has no output path")

        path = Path(output_path)
        if not path.is_absolute():
            output_dir = getattr(self.file_service, "output_video_dir", None)
            if output_dir:
                path = output_dir / path

        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found: {path}")

        suffix = path.suffix.lower()
        media_types = {
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }
        media_type = media_types.get(suffix, "video/mp4")

        return FileResponse(path=str(path), media_type=media_type, filename=path.name)

    def _ensure_processing(self) -> None:
        """Ensure the task processing thread is running."""
        if hasattr(self.api, "_ensure_processing_thread_running"):
            self.api._ensure_processing_thread_running()

    async def _save_uploaded_file(self, file: UploadFile, prefix: str = "upload") -> str:
        """Save an uploaded file to disk."""
        import uuid

        if not self.file_service:
            raise HTTPException(status_code=503, detail="File service not initialized")

        input_dir = getattr(self.file_service, "input_video_dir", None)
        if not input_dir:
            input_dir = getattr(self.file_service, "input_image_dir", None)
        if not input_dir:
            input_dir = Path("/tmp/telefuser/inputs")
            input_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename or "input.mp4").suffix
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
        file_path = input_dir / filename

        content = await file.read()
        await asyncio.to_thread(self._write_file_sync, file_path, content)

        return str(file_path)

    def _write_file_sync(self, file_path: Path, content: bytes) -> None:
        """Write file synchronously."""
        file_path.write_bytes(content)


def setup_routes(api_server: ApiServer) -> APIRouter:
    """Setup video routes."""
    return create_router(api_server)
