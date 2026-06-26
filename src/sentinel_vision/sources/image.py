from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from sentinel_vision.config import SourceConfig
from sentinel_vision.domain import FramePacket, Image
from sentinel_vision.sources.base import VideoSource


class ImageLoopSource(VideoSource):
    """Loop a local image at a fixed rate for reproducible full-model benchmarks."""

    def __init__(self, config: SourceConfig, realtime: bool = True) -> None:
        self.config = config
        self.realtime = realtime
        self._image: Image | None = None
        self._sequence = 0
        self._next_frame_at = 0.0

    @property
    def backend_name(self) -> str:
        return "image-loop-benchmark"

    @property
    def ready(self) -> bool:
        return self._image is not None

    def open(self) -> None:
        path = Path(self.config.uri).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"benchmark image not found: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not decode benchmark image: {path}")
        if image.shape[1] != self.config.width or image.shape[0] != self.config.height:
            image = cv2.resize(image, (self.config.width, self.config.height))
        self._image = np.asarray(image, dtype=np.uint8)
        self._sequence = 0
        self._next_frame_at = time.monotonic()

    def read(self) -> FramePacket:
        if self._image is None:
            raise RuntimeError("source is not open")
        if self.realtime:
            now = time.monotonic()
            if self._next_frame_at > now:
                time.sleep(self._next_frame_at - now)
            self._next_frame_at = max(
                self._next_frame_at + 1.0 / self.config.fps,
                time.monotonic(),
            )
        self._sequence += 1
        return FramePacket(
            source_id=self.config.source_id,
            sequence=self._sequence,
            captured_at=datetime.now(UTC),
            monotonic_ns=time.monotonic_ns(),
            image=self._image.copy(),
        )

    def close(self) -> None:
        self._image = None
