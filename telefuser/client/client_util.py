"""Client utilities for task management and server communication."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

from telefuser.utils.logging import logger


def send_and_monitor_task(
    url: str,
    message: Dict[str, Any],
    task_index: int,
    complete_bar: Optional[tqdm],
    complete_lock: Optional[threading.Lock],
) -> bool:
    """Send task to server and monitor until completion.

    Args:
        url: Server base URL
        message: Task message payload
        task_index: Index of the task for logging
        complete_bar: Optional progress bar
        complete_lock: Optional lock for thread-safe progress updates

    Returns:
        True if task completed successfully, False otherwise
    """
    try:
        response = requests.post(f"{url}/v1/tasks/create", json=message)
        response_data: Dict[str, Any] = response.json()
        task_id: Optional[str] = response_data.get("task_id")

        if not task_id:
            logger.error(f"No task_id received from {url}")
            return False

        # Step 2: Monitor task status until completion
        while True:
            try:
                status_response = requests.get(f"{url}/v1/tasks/{task_id}/status")
                status_data: Dict[str, Any] = status_response.json()
                task_status: Optional[str] = status_data.get("status")

                if task_status == "completed":
                    logger.info(status_data)
                    logger.info(response_data)
                    # Update completion bar safely
                    if complete_bar and complete_lock:
                        with complete_lock:
                            complete_bar.update(1)
                    return True
                elif task_status == "failed":
                    logger.error(f"Task {task_index + 1} (task_id: {task_id}) failed")
                    if complete_bar and complete_lock:
                        with complete_lock:
                            complete_bar.update(1)  # Still update progress even if failed
                    return False
                else:
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to check status for task_id {task_id}: {e}")
                time.sleep(0.5)

    except Exception as e:
        logger.error(f"Failed to send task to {url}: {e}")
        return False


def get_available_urls(urls: List[str]) -> Optional[List[str]]:
    """Check which URLs are available and return the list.

    Args:
        urls: List of server URLs to check

    Returns:
        List of available URLs, or None if none are available
    """
    available_urls: List[str] = []
    for url in urls:
        try:
            _ = requests.get(f"{url}/v1/service/status").json()
            available_urls.append(url)
        except Exception:
            continue

    if not available_urls:
        logger.error("No available urls.")
        return None

    logger.info(f"available_urls: {available_urls}")
    return available_urls


def find_idle_server(available_urls: List[str]) -> str:
    """Find an idle server from available URLs.

    Args:
        available_urls: List of available server URLs

    Returns:
        URL of an idle server
    """
    while True:
        for url in available_urls:
            try:
                response = requests.get(f"{url}/v1/service/status").json()
                if response.get("service_status") == "idle":
                    return url
            except Exception:
                continue
        time.sleep(3)


def process_tasks_async(
    messages: List[Dict[str, Any]],
    available_urls: List[str],
    show_progress: bool = True,
) -> bool:
    """Process a list of tasks asynchronously across multiple servers.

    Args:
        messages: List of task messages to send
        available_urls: List of available server URLs
        show_progress: Whether to show progress bar

    Returns:
        True if all tasks were processed
    """
    if not available_urls:
        logger.error("No available servers to process tasks.")
        return False

    active_threads: List[threading.Thread] = []

    logger.info(f"Sending {len(messages)} tasks to available servers...")

    complete_bar: Optional[tqdm] = None
    complete_lock: Optional[threading.Lock] = None
    if show_progress:
        complete_bar = tqdm(total=len(messages), desc="Completing tasks")
        complete_lock = threading.Lock()  # Thread-safe updates to completion bar

    for idx, message in enumerate(messages):
        # Find an idle server
        server_url = find_idle_server(available_urls)

        # Create and start thread for sending and monitoring task
        thread = threading.Thread(
            target=send_and_monitor_task,
            args=(server_url, message, idx, complete_bar, complete_lock),
        )
        thread.daemon = False
        thread.start()
        active_threads.append(thread)

        # Small delay to let thread start
        time.sleep(0.5)

    # Wait for all threads to complete
    for thread in active_threads:
        thread.join()

    # Close completion bar
    if complete_bar:
        complete_bar.close()

    logger.info("All tasks processing completed!")
    return True
