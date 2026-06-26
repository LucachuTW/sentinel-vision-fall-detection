from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np

from sentinel_vision.config import SegmentationConfig
from sentinel_vision.domain import Image, PoseDetection, SegmentationResult


class SegmentationScheduler:
    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config.policy
        self._last_segmented: dict[int, int] = {}

    def select(
        self, detections: tuple[PoseDetection, ...], sequence: int
    ) -> tuple[PoseDetection, ...]:
        prioritized: list[tuple[int, float, PoseDetection]] = []
        for detection in detections:
            if detection.track_id is None:
                continue
            last = self._last_segmented.get(detection.track_id)
            is_new = last is None
            due = last is None or sequence - last >= self.config.every_n_frames
            low_confidence = detection.keypoint_confidence < self.config.low_pose_confidence
            uncertainty_due = low_confidence and (
                last is None or sequence - last >= self.config.uncertain_every_n_frames
            )
            if due or uncertainty_due or (is_new and self.config.segment_new_tracks):
                priority = 0 if is_new else 1 if low_confidence else 2
                prioritized.append((priority, detection.confidence, detection))
        prioritized.sort(key=lambda item: (item[0], -item[1]))
        selected = tuple(item[2] for item in prioritized[: self.config.max_objects_per_frame])
        for detection in selected:
            assert detection.track_id is not None
            self._last_segmented[detection.track_id] = sequence
        active_ids = {item.track_id for item in detections}
        self._last_segmented = {
            key: value for key, value in self._last_segmented.items() if key in active_ids
        }
        return selected


class Segmenter(ABC):
    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        self.scheduler = SegmentationScheduler(config)

    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @property
    @abstractmethod
    def ready(self) -> bool: ...

    def segment_on_demand(
        self,
        image: Image,
        detections: tuple[PoseDetection, ...],
        sequence: int,
    ) -> tuple[SegmentationResult, ...]:
        selected = self.scheduler.select(detections, sequence)
        if not selected:
            return ()
        return self.segment(image, selected)

    @abstractmethod
    def segment(
        self, image: Image, detections: tuple[PoseDetection, ...]
    ) -> tuple[SegmentationResult, ...]: ...


class ContourSegmenter(Segmenter):
    """Foreground-mask fallback for the synthetic demo."""

    @property
    def backend_name(self) -> str:
        return "contour-fallback"

    @property
    def ready(self) -> bool:
        return True

    def segment(
        self, image: Image, detections: tuple[PoseDetection, ...]
    ) -> tuple[SegmentationResult, ...]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        foreground = (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 80)
        output: list[SegmentationResult] = []
        height, width = image.shape[:2]
        for detection in detections:
            if detection.track_id is None:
                continue
            x1 = max(0, int(detection.bbox.x1))
            y1 = max(0, int(detection.bbox.y1))
            x2 = min(width, int(detection.bbox.x2) + 1)
            y2 = min(height, int(detection.bbox.y2) + 1)
            mask = np.zeros((height, width), dtype=np.bool_)
            mask[y1:y2, x1:x2] = foreground[y1:y2, x1:x2]
            output.append(SegmentationResult(detection.track_id, mask, 0.90))
        return tuple(output)


class UltralyticsSam2Segmenter(Segmenter):
    def __init__(self, config: SegmentationConfig) -> None:
        super().__init__(config)
        try:
            from ultralytics import SAM  # type: ignore[import-not-found,attr-defined]
            from ultralytics.models.sam import (  # type: ignore[import-not-found]
                SAM2DynamicInteractivePredictor,
            )
        except ImportError as exc:
            raise RuntimeError("SAM 2 requires the ML dependencies: uv sync --extra ml") from exc
        self._dynamic = config.dynamic_memory
        self._model: Any
        if self._dynamic:
            overrides = {
                "conf": 0.01,
                "task": "segment",
                "mode": "predict",
                "imgsz": config.image_size,
                "model": config.model,
                "save": False,
                "device": None if config.device == "auto" else config.device,
            }
            self._model = SAM2DynamicInteractivePredictor(
                overrides=overrides,
                max_obj_num=config.policy.max_objects_per_frame,
            )
        else:
            self._model = SAM(config.model)
        self._device = None if config.device == "auto" else config.device

    @property
    def backend_name(self) -> str:
        mode = "memory-enabled" if self.config.dynamic_memory else "box-prompted"
        return f"sam2:{self.config.model}:{mode}"

    @property
    def ready(self) -> bool:
        return True

    def segment(
        self, image: Image, detections: tuple[PoseDetection, ...]
    ) -> tuple[SegmentationResult, ...]:
        boxes = [item.bbox.as_xyxy() for item in detections]
        if self._dynamic:
            # ponytail: per-call object slots keep obj_ids < max_obj_num (Ultralytics asserts this);
            # real ByteTrack ids grow unbounded. Stable per-track slot recycling would be needed for
            # true cross-frame memory continuity — until then the demo uses box-prompted mode.
            results = self._model(
                source=image,
                bboxes=boxes,
                obj_ids=list(range(len(detections))),
                update_memory=True,
            )
        else:
            results = self._model.predict(
                source=image,
                bboxes=boxes,
                device=self._device,
                imgsz=self.config.image_size,
                verbose=False,
            )
        if not results or results[0].masks is None:
            return ()
        masks = results[0].masks.data.detach().cpu().numpy()
        output: list[SegmentationResult] = []
        for detection, raw_mask in zip(detections, masks, strict=False):
            if detection.track_id is None:
                continue
            if raw_mask.shape != image.shape[:2]:
                raw_mask = cv2.resize(
                    raw_mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            output.append(SegmentationResult(detection.track_id, raw_mask.astype(bool), 1.0))
        return tuple(output)


def create_segmenter(config: SegmentationConfig) -> Segmenter:
    if config.backend == "sam2":
        return UltralyticsSam2Segmenter(config)
    return ContourSegmenter(config)
