#!/usr/bin/env python3
"""Run every local vision component on one image and emit auditable JSON."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sentinel_vision.config import PoseConfig, SegmentationConfig, TrackerConfig
from sentinel_vision.models.pose import UltralyticsYolo26PoseEstimator
from sentinel_vision.models.segmentation import UltralyticsSam2Segmenter
from sentinel_vision.observability import gpu_snapshot
from sentinel_vision.tracking.bytetrack import ByteTrackAdapter


def timed(function: Callable[..., Any], *args: Any, device: str = "cpu") -> tuple[Any, float]:
    started = time.perf_counter()
    result = function(*args)
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000


def latency(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "min": round(float(array.min()), 3),
        "mean": round(float(array.mean()), 3),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "max": round(float(array.max()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--pose", default="models/yolo26n-pose.pt")
    parser.add_argument("--sam", default="models/sam2.1_t.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dynamic-memory", action="store_true")
    parser.add_argument("--max-objects", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sam-iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.image is None:
        from ultralytics import ASSETS

        args.image = Path(ASSETS) / "bus.jpg"
    for artifact in (args.image, Path(args.pose), Path(args.sam)):
        if not artifact.is_file():
            raise SystemExit(f"missing local artifact: {artifact}")
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"could not decode image: {args.image}")

    precision = "fp16" if args.device.startswith("cuda") else "fp32"
    pose = UltralyticsYolo26PoseEstimator(
        PoseConfig(
            backend="ultralytics",
            model=args.pose,
            device=args.device,
            precision=precision,
        )
    )
    detections_obj, pose_cold_ms = timed(pose.infer, image, device=args.device)
    detections = detections_obj
    if not isinstance(detections, tuple) or not detections:
        raise SystemExit("YOLO26 returned no pose detections for the smoke image")
    pose.warmup(image, args.warmup)
    pose_latencies = [
        timed(pose.infer, image, device=args.device)[1] for _ in range(args.iterations)
    ]

    tracker = ByteTrackAdapter(TrackerConfig(backend="bytetrack"))
    first = tracker.update(detections, 1)
    second = tracker.update(detections, 2)
    stable_ids = [item.track_id for item in first] == [item.track_id for item in second]
    tracker_latencies = [
        timed(tracker.update, detections, sequence)[1] for sequence in range(3, args.iterations + 3)
    ]
    selected = second[: max(1, args.max_objects)]

    segmenter = UltralyticsSam2Segmenter(
        SegmentationConfig(
            backend="sam2",
            model=args.sam,
            device=args.device,
            dynamic_memory=args.dynamic_memory,
            image_size=640,
        )
    )
    masks_obj, sam_cold_ms = timed(segmenter.segment, image, selected, device=args.device)
    masks = masks_obj
    if not isinstance(masks, tuple):
        raise RuntimeError("unexpected SAM 2 result")
    for _ in range(max(0, args.warmup - 1)):
        segmenter.segment(image, selected)
    sam_latencies = [
        timed(segmenter.segment, image, selected, device=args.device)[1]
        for _ in range(args.sam_iterations)
    ]

    gpu = gpu_snapshot()
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "device": args.device,
        "image": str(args.image),
        "warmup_iterations": args.warmup,
        "pose": {
            "backend": pose.backend_name,
            "detections": len(detections),
            "keypoints_per_detection": len(detections[0].keypoints),
            "cold_start_ms": round(pose_cold_ms, 3),
            "steady_state_ms": latency(pose_latencies),
            "steady_state_fps": round(1000 / np.mean(pose_latencies), 3),
        },
        "tracking": {
            "backend": tracker.backend_name,
            "ids": [item.track_id for item in second],
            "stable_across_frames": stable_ids,
            "steady_state_ms": latency(tracker_latencies),
        },
        "segmentation": {
            "backend": segmenter.backend_name,
            "masks": len(masks),
            "areas_pixels": [item.area_pixels for item in masks],
            "cold_start_ms": round(sam_cold_ms, 3),
            "steady_state_ms": latency(sam_latencies),
        },
        "gpu": gpu,
    }
    output = args.output or Path(
        f"artifacts/benchmarks/models-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"json={output}")


if __name__ == "__main__":
    main()
