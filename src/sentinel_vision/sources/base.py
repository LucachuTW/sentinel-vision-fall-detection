from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_vision.domain import FramePacket


class VideoSource(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @property
    def ready(self) -> bool:
        return True

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> FramePacket | None: ...

    @abstractmethod
    def close(self) -> None: ...
