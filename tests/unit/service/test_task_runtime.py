from __future__ import annotations

import asyncio

from telefuser.service.api.schema import TaskRequest, TaskResponse
from telefuser.service.core.task_manager import TaskManager, TaskStatus
from telefuser.service.core.task_processor import AsyncTaskProcessor


def test_complete_task_does_not_override_cancelled_status() -> None:
    """Cancelled tasks must remain cancelled even if completion is reported later."""
    task_manager = TaskManager(max_queue_size=10)
    message = TaskRequest(task="t2i")

    task_id = task_manager.create_task(message)

    assert task_manager.cancel_task(task_id) is True

    task_manager.complete_task(task_id, output_path="ignored.png")

    status = task_manager.get_task_status(task_id)
    assert status is not None
    assert status["status"] == TaskStatus.CANCELLED.value
    assert status["error"] == "Task cancelled by user"
    assert status["output_path"] == message.output_path


class _ControlledMediaService:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finish = asyncio.Event()

    async def generate_media_with_stop_event(self, message: TaskRequest, stop_event) -> TaskResponse:
        self.started.set()
        await self.finish.wait()
        return TaskResponse(
            task_id=message.task_id,
            task_status="completed",
            output_path=message.output_path,
        )


def test_async_task_processor_preserves_cancelled_status() -> None:
    """Cancellation should win even if the media service returns after cancellation."""

    async def scenario() -> None:
        task_manager = TaskManager(max_queue_size=10)
        media_service = _ControlledMediaService()
        processor = AsyncTaskProcessor(task_manager=task_manager, media_service=media_service, max_concurrent=1)
        message = TaskRequest(task="t2i")

        task_id = task_manager.create_task(message)

        await processor.start()
        try:
            await asyncio.wait_for(media_service.started.wait(), timeout=1.0)

            assert task_manager.cancel_task(task_id) is True

            media_service.finish.set()

            async def wait_for_cancelled_status() -> None:
                while True:
                    status = task_manager.get_task_status(task_id)
                    if status and status["status"] == TaskStatus.CANCELLED.value:
                        return
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_cancelled_status(), timeout=1.0)

            status = task_manager.get_task_status(task_id)
            assert status is not None
            assert status["status"] == TaskStatus.CANCELLED.value
            assert status["error"] == "Task cancelled by user"
        finally:
            await processor.stop()

    asyncio.run(scenario())
