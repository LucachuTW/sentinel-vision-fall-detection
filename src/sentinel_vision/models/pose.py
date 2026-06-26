from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sentinel_vision.config import PoseConfig
from sentinel_vision.domain import BoundingBox, Image, Keypoint, PoseDetection

# COCO-17 keypoint locations inside a canonical upright person box.
_COCO_TEMPLATE = np.asarray(
    [
        (0.50, 0.08),
        (0.46, 0.07),
        (0.54, 0.07),
        (0.42, 0.09),
        (0.58, 0.09),
        (0.35, 0.28),
        (0.65, 0.28),
        (0.27, 0.48),
        (0.73, 0.48),
        (0.22, 0.67),
        (0.78, 0.67),
        (0.42, 0.56),
        (0.58, 0.56),
        (0.39, 0.76),
        (0.61, 0.76),
        (0.36, 0.97),
        (0.64, 0.97),
    ],
    dtype=np.float32,
)


class PoseEstimator(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @property
    @abstractmethod
    def ready(self) -> bool: ...

    @abstractmethod
    def warmup(self, image: Image, iterations: int) -> None: ...

    @abstractmethod
    def infer(self, image: Image) -> tuple[PoseDetection, ...]: ...


class SyntheticPoseEstimator(PoseEstimator):
    """Color-component detector for the deterministic demo scene."""

    def __init__(self, config: PoseConfig) -> None:
        self.config = config

    @property
    def backend_name(self) -> str:
        return "synthetic-pose-fallback"

    @property
    def ready(self) -> bool:
        return True

    def warmup(self, image: Image, iterations: int) -> None:
        for _ in range(iterations):
            self.infer(image)

    def infer(self, image: Image) -> tuple[PoseDetection, ...]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        foreground = np.where((saturation > 65) & (value > 80), 255, 0).astype(np.uint8)
        foreground = np.asarray(
            cv2.morphologyEx(
                foreground,
                cv2.MORPH_CLOSE,
                np.ones((11, 11), dtype=np.uint8),
            ),
            dtype=np.uint8,
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        height, width = image.shape[:2]
        detections: list[PoseDetection] = []
        for component in range(1, count):
            x, y, box_width, box_height, area = (int(v) for v in stats[component])
            if area < 500 or box_height < 35 or box_width < 15 or y < 60:
                continue
            padding = 8
            bbox = BoundingBox(
                x - padding,
                y - padding,
                x + box_width + padding,
                y + box_height + padding,
            ).clipped(width, height)
            keypoints = tuple(
                Keypoint(
                    bbox.x1 + float(nx) * bbox.width,
                    bbox.y1 + float(ny) * bbox.height,
                    0.92,
                )
                for nx, ny in _COCO_TEMPLATE
            )
            detections.append(PoseDetection(bbox=bbox, confidence=0.94, keypoints=keypoints))
        detections.sort(key=lambda item: item.bbox.x1)
        return tuple(detections[: self.config.max_detections])


class UltralyticsYolo26PoseEstimator(PoseEstimator):
    def __init__(self, config: PoseConfig) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found,attr-defined]
        except ImportError as exc:
            raise RuntimeError("YOLO26 requires the ML dependencies: uv sync --extra ml") from exc
        self.config = config
        model_path = config.engine_path or config.model
        if config.engine_path and not Path(config.engine_path).is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {config.engine_path}")
        self._model = YOLO(model_path, task="pose")
        self._device = None if config.device == "auto" else config.device

    @property
    def backend_name(self) -> str:
        artifact = self.config.engine_path or self.config.model
        return f"yolo26-pose:{Path(artifact).suffix.lstrip('.')}:{self.config.precision}"

    @property
    def ready(self) -> bool:
        return True

    def warmup(self, image: Image, iterations: int) -> None:
        for _ in range(iterations):
            self.infer(image)

    def infer(self, image: Image) -> tuple[PoseDetection, ...]:
        results = self._model.predict(
            source=image,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            max_det=self.config.max_detections,
            device=self._device,
            quantize=(
                16 if self.config.precision == "fp16" and self.config.engine_path is None else None
            ),
            verbose=False,
        )
        if not results:
            return ()
        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return ()

        boxes = _as_numpy(result.boxes.xyxy)
        confidences = _as_numpy(result.boxes.conf)
        xy = _as_numpy(result.keypoints.xy)
        keypoint_confidences = result.keypoints.conf
        if keypoint_confidences is None:
            kp_conf = np.ones(xy.shape[:2], dtype=np.float32)
        else:
            kp_conf = _as_numpy(keypoint_confidences)

        output: list[PoseDetection] = []
        for box_values, confidence, points, point_confidences in zip(
            boxes, confidences, xy, kp_conf, strict=True
        ):
            output.append(
                PoseDetection(
                    bbox=BoundingBox(*(float(value) for value in box_values)),
                    confidence=float(confidence),
                    keypoints=tuple(
                        Keypoint(float(point[0]), float(point[1]), float(point_confidence))
                        for point, point_confidence in zip(points, point_confidences, strict=True)
                    ),
                )
            )
        return tuple(output)


def create_pose_estimator(config: PoseConfig) -> PoseEstimator:
    if config.backend == "ultralytics":
        return UltralyticsYolo26PoseEstimator(config)
    return SyntheticPoseEstimator(config)


def _as_numpy(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
