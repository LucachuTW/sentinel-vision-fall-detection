from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
Mask = NDArray[np.bool_]


class MemoryDomain(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: BoundingBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def clipped(self, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            x1=min(max(self.x1, 0.0), float(width - 1)),
            y1=min(max(self.y1, 0.0), float(height - 1)),
            x2=min(max(self.x2, 0.0), float(width - 1)),
            y2=min(max(self.y2, 0.0), float(height - 1)),
        )

    def as_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True, slots=True)
class Keypoint:
    x: float
    y: float
    confidence: float

    def as_list(self) -> list[float]:
        return [round(self.x, 2), round(self.y, 2), round(self.confidence, 4)]


@dataclass(frozen=True, slots=True)
class PoseDetection:
    bbox: BoundingBox
    confidence: float
    keypoints: tuple[Keypoint, ...]
    class_id: int = 0
    label: str = "person"
    track_id: int | None = None

    def with_track(self, track_id: int) -> PoseDetection:
        return replace(self, track_id=track_id)

    @property
    def keypoint_confidence(self) -> float:
        if not self.keypoints:
            return 0.0
        return sum(item.confidence for item in self.keypoints) / len(self.keypoints)

    def as_dict(self, include_keypoints: bool = True) -> dict[str, Any]:
        output: dict[str, Any] = {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": [round(value, 2) for value in self.bbox.as_xyxy()],
            "keypoint_confidence": round(self.keypoint_confidence, 4),
        }
        if include_keypoints:
            output["keypoints"] = [item.as_list() for item in self.keypoints]
        return output


@dataclass(frozen=True, slots=True)
class FramePacket:
    source_id: str
    sequence: int
    captured_at: datetime
    monotonic_ns: int
    image: Image
    memory_domain: MemoryDomain = MemoryDomain.CPU

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.image.shape


@dataclass(frozen=True, slots=True)
class PosePacket:
    frame: FramePacket
    detections: tuple[PoseDetection, ...]
    inference_ms: float


@dataclass(frozen=True, slots=True)
class TrackedPacket:
    frame: FramePacket
    detections: tuple[PoseDetection, ...]
    pose_ms: float
    tracking_ms: float


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    track_id: int
    mask: Mask
    score: float

    @property
    def area_pixels(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    track_id: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_ready: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "probabilities": {key: round(value, 4) for key, value in self.probabilities.items()},
            "model_ready": self.model_ready,
        }


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    event_id: str
    source_id: str
    sequence: int
    track_id: int
    kind: str
    severity: str
    action: str
    confidence: float
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    narrative: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_id": self.source_id,
            "sequence": self.sequence,
            "track_id": self.track_id,
            "kind": self.kind,
            "severity": self.severity,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "occurred_at": self.occurred_at.isoformat(),
            "narrative": self.narrative,
        }


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    frame: FramePacket
    detections: tuple[PoseDetection, ...]
    segmentations: tuple[SegmentationResult, ...]
    actions: tuple[ActionPrediction, ...]
    stage_latency_ms: dict[str, float]
    end_to_end_ms: float
    rendered_jpeg: bytes
