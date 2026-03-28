"""TeleFuser Client Module

This module provides client-side functionality for interacting with the TeleFuser API server.

Usage:
    from telefuser.client import TAPClient

    client = TAPClient(base_url="http://localhost:8000")
    response = client.create_t2v_task(prompt="Astronaut walking on the moon")

    # For image generation
    response = client.create_t2i_task(prompt="A beautiful landscape")
"""

from __future__ import annotations

from .client import TAPClient
from .client_util import send_and_monitor_task

__all__ = [
    "TAPClient",
    "send_and_monitor_task",
]
