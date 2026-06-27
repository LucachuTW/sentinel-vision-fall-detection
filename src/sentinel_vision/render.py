from __future__ import annotations

import cv2
import numpy as np

from sentinel_vision.domain import (
    ActionPrediction,
    Image,
    PoseDetection,
    SegmentationResult,
)

_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
_PALETTE = (
    (64, 196, 255),
    (95, 224, 143),
    (222, 155, 92),
    (212, 120, 245),
    (84, 210, 210),
    (235, 110, 135),
)


def render_frame(
    image: Image,
    detections: tuple[PoseDetection, ...],
    segmentations: tuple[SegmentationResult, ...],
    actions: tuple[ActionPrediction, ...],
    stage_latency_ms: dict[str, float],
    jpeg_quality: int,
) -> bytes:
    canvas = image.copy()
    action_map = {item.track_id: item for item in actions}
    for segmentation in segmentations:
        mask_color = np.asarray(_color(segmentation.track_id), dtype=np.float32)
        pixels = canvas[segmentation.mask].astype(np.float32)
        canvas[segmentation.mask] = (pixels * 0.65 + mask_color * 0.35).astype(np.uint8)

    for detection in detections:
        track_id = detection.track_id or 0
        color = _color(track_id)
        box = detection.bbox
        cv2.rectangle(
            canvas,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            color,
            2,
        )
        for start, end in _SKELETON:
            if start >= len(detection.keypoints) or end >= len(detection.keypoints):
                continue
            a, b = detection.keypoints[start], detection.keypoints[end]
            if min(a.confidence, b.confidence) < 0.25:
                continue
            cv2.line(canvas, (int(a.x), int(a.y)), (int(b.x), int(b.y)), color, 2)
        for point in detection.keypoints:
            if point.confidence >= 0.25:
                cv2.circle(canvas, (int(point.x), int(point.y)), 3, (245, 248, 252), -1)

        action = action_map.get(track_id)
        action_text = f"{action.label} {action.confidence:.0%}" if action else "warming up"
        label = f"ID {track_id} | {action_text}"
        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        top = max(0, int(box.y1) - 23)
        cv2.rectangle(
            canvas, (int(box.x1), top), (int(box.x1) + text_width + 10, top + 22), color, -1
        )
        cv2.putText(
            canvas,
            label,
            (int(box.x1) + 5, top + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (10, 18, 28),
            1,
            cv2.LINE_AA,
        )

    latency = " | ".join(f"{name} {value:.1f}ms" for name, value in stage_latency_ms.items())
    cv2.rectangle(
        canvas, (0, canvas.shape[0] - 31), (canvas.shape[1], canvas.shape[0]), (10, 15, 23), -1
    )
    cv2.putText(
        canvas,
        latency,
        (15, canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (176, 191, 211),
        1,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("could not encode rendered frame")
    return encoded.tobytes()


def _color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]
