from __future__ import annotations

import math
import time
from datetime import UTC, datetime

import cv2
import numpy as np

from sentinel_vision.config import SourceConfig
from sentinel_vision.domain import FramePacket
from sentinel_vision.sources.base import VideoSource


class SyntheticSource(VideoSource):
    """Deterministic industrial scene used for demos, tests, and CI."""

    def __init__(self, config: SourceConfig, realtime: bool = True) -> None:
        self.config = config
        self.realtime = realtime
        self._sequence = 0
        self._open = False
        self._next_frame_at = 0.0

    @property
    def backend_name(self) -> str:
        return "synthetic-scene"

    @property
    def ready(self) -> bool:
        return self._open

    def open(self) -> None:
        self._sequence = 0
        self._open = True
        self._next_frame_at = time.monotonic()

    def read(self) -> FramePacket:
        if not self._open:
            raise RuntimeError("source is not open")
        if self.realtime:
            now = time.monotonic()
            if self._next_frame_at > now:
                time.sleep(self._next_frame_at - now)
            self._next_frame_at = max(self._next_frame_at + 1.0 / self.config.fps, time.monotonic())

        self._sequence += 1
        image = self._render(self._sequence)
        return FramePacket(
            source_id=self.config.source_id,
            sequence=self._sequence,
            captured_at=datetime.now(UTC),
            monotonic_ns=time.monotonic_ns(),
            image=image,
        )

    def close(self) -> None:
        self._open = False

    def _render(self, sequence: int) -> np.ndarray:
        width, height = self.config.width, self.config.height
        image = np.full((height, width, 3), (19, 26, 37), dtype=np.uint8)
        for x in range(0, width, 80):
            cv2.line(image, (x, 0), (x, height), (30, 41, 55), 1)
        for y in range(0, height, 80):
            cv2.line(image, (0, y), (width, y), (30, 41, 55), 1)

        cv2.rectangle(image, (20, 18), (260, 52), (12, 18, 28), -1)
        cv2.putText(
            image,
            "SYNTHETIC INDUSTRIAL FLOOR",
            (32, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 200, 255),
            1,
            cv2.LINE_AA,
        )

        phase = sequence / max(self.config.fps, 1.0)
        x_a = int(width * 0.20 + math.sin(phase * 0.7) * width * 0.10)
        x_b = int(width * 0.52 + math.sin(phase * 0.38 + 1.5) * width * 0.07)
        x_c = int(width * 0.80 + math.sin(phase * 0.55 + 3.1) * width * 0.06)
        floor = int(height * 0.86)

        self._draw_person(image, (x_a, floor), 1.0, (255, 179, 71), "walking", phase)
        behavior_b = "feeding" if int(phase / 4) % 2 else "standing"
        self._draw_person(image, (x_b, floor), 0.92, (100, 220, 160), behavior_b, phase)
        behavior_c = "resting" if int(phase / 6) % 2 else "standing"
        self._draw_person(image, (x_c, floor), 1.05, (210, 120, 245), behavior_c, phase)
        return image

    @staticmethod
    def _draw_person(
        image: np.ndarray,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        behavior: str,
        phase: float,
    ) -> None:
        x, floor = origin
        thickness = max(9, int(14 * scale))
        if behavior == "resting":
            y = floor - int(35 * scale)
            cv2.ellipse(image, (x, y), (int(65 * scale), int(22 * scale)), 0, 0, 360, color, -1)
            cv2.circle(image, (x + int(70 * scale), y - 3), int(17 * scale), color, -1)
            cv2.line(image, (x - 45, y + 12), (x - 72, floor), color, thickness)
            return

        head_y = floor - int(155 * scale)
        hip_y = floor - int(62 * scale)
        shoulder_y = floor - int(122 * scale)
        bend = int(45 * scale) if behavior == "feeding" else 0
        head = (x + bend, head_y + bend)
        shoulder = (x + int(bend * 0.35), shoulder_y + int(bend * 0.35))
        hip = (x, hip_y)

        cv2.line(image, shoulder, hip, color, thickness + 4)
        cv2.circle(image, head, int(19 * scale), color, -1)
        cv2.line(image, head, shoulder, color, thickness)
        stride = int(math.sin(phase * 5) * 22 * scale) if behavior == "walking" else 5
        cv2.line(image, hip, (x - stride, floor), color, thickness)
        cv2.line(image, hip, (x + stride, floor), color, thickness)
        hand_y = shoulder[1] + int(58 * scale)
        cv2.line(image, shoulder, (shoulder[0] - int(32 * scale), hand_y), color, thickness)
        cv2.line(image, shoulder, (shoulder[0] + int(32 * scale), hand_y), color, thickness)
