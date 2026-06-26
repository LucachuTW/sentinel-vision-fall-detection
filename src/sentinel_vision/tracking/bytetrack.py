from __future__ import annotations

import warnings

import numpy as np

from sentinel_vision.config import TrackerConfig
from sentinel_vision.domain import BoundingBox, Keypoint, PoseDetection
from sentinel_vision.tracking.base import Tracker


def match_source_detections(
    boxes: list[BoundingBox], detections: tuple[PoseDetection, ...]
) -> list[int | None]:
    """Greedy 1-to-1 assignment of tracked boxes to source detections by IoU.

    Each source detection is consumed at most once, so overlapping tracks never reuse the
    same keypoints. Returns, per tracked box, the source-detection index or None.
    """
    pairs = sorted(
        (
            (boxes[box_index].iou(detections[det_index].bbox), box_index, det_index)
            for box_index in range(len(boxes))
            for det_index in range(len(detections))
        ),
        reverse=True,
    )
    assigned: list[int | None] = [None] * len(boxes)
    used_boxes: set[int] = set()
    used_detections: set[int] = set()
    for iou, box_index, det_index in pairs:
        if iou <= 0.0 or box_index in used_boxes or det_index in used_detections:
            continue
        assigned[box_index] = det_index
        used_boxes.add(box_index)
        used_detections.add(det_index)
    return assigned


class ByteTrackAdapter(Tracker):
    def __init__(self, config: TrackerConfig) -> None:
        try:
            import supervision as sv  # type: ignore[import-not-found]
            from supervision.tracker.byte_tracker.core import (  # type: ignore[import-not-found]
                ByteTrack,
            )
        except ImportError as exc:
            raise RuntimeError(
                "ByteTrack requires the ML dependencies: uv sync --extra ml"
            ) from exc
        self._sv = sv
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = ByteTrack(
                track_activation_threshold=config.track_activation_threshold,
                lost_track_buffer=config.lost_track_buffer,
                minimum_matching_threshold=config.minimum_matching_threshold,
                frame_rate=config.frame_rate,
            )

    @property
    def backend_name(self) -> str:
        return "bytetrack"

    def update(
        self, detections: tuple[PoseDetection, ...], sequence: int
    ) -> tuple[PoseDetection, ...]:
        del sequence
        if not detections:
            self._tracker.update_with_detections(self._sv.Detections.empty())
            return ()
        raw = self._sv.Detections(
            xyxy=np.asarray([item.bbox.as_xyxy() for item in detections], dtype=np.float32),
            confidence=np.asarray([item.confidence for item in detections], dtype=np.float32),
            class_id=np.asarray([item.class_id for item in detections], dtype=np.int32),
        )
        output = self._tracker.update_with_detections(raw)
        if output.tracker_id is None:
            return ()
        confidences = output.confidence
        class_ids = output.class_id
        if confidences is None or class_ids is None:
            raise RuntimeError("ByteTrack returned detections without confidence or class IDs")

        boxes = [BoundingBox(*(float(value) for value in row)) for row in output.xyxy]
        assignment = match_source_detections(boxes, detections)

        tracked: list[PoseDetection] = []
        for index, (box, confidence, class_id, track_id) in enumerate(
            zip(boxes, confidences, class_ids, output.tracker_id, strict=True)
        ):
            source = assignment[index]
            original = detections[source] if source is not None else None
            tracked.append(
                PoseDetection(
                    bbox=box,
                    confidence=float(confidence),
                    class_id=int(class_id),
                    label=original.label if original else "person",
                    keypoints=tuple(
                        Keypoint(point.x, point.y, point.confidence) for point in original.keypoints
                    )
                    if original
                    else (),
                    track_id=int(track_id),
                )
            )
        return tuple(tracked)
