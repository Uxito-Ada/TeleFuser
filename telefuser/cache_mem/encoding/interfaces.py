from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from PIL import Image


class PromptEncoder(ABC):
    """Prompt 编码器接口。"""

    @abstractmethod
    def encode(self, prompt: str) -> List[float]:
        pass

    @abstractmethod
    def decompose_prompt(self, prompt: str) -> Dict[str, str]:
        pass


class VideoEncoder(ABC):
    """Video 编码器接口。"""

    @abstractmethod
    def encode_video(self, frames: List["Image.Image"]) -> List[float]:
        pass
