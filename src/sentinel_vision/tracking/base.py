from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_vision.domain import PoseDetection


class Tracker(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @abstractmethod
    def update(
        self, detections: tuple[PoseDetection, ...], sequence: int
    ) -> tuple[PoseDetection, ...]: ...
