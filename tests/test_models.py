from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel_vision.config import (
    PoseConfig,
    SegmentationConfig,
    SegmentationPolicyConfig,
    SourceConfig,
    TemporalConfig,
    TrackerConfig,
)
from sentinel_vision.domain import BoundingBox, Keypoint, PoseDetection
from sentinel_vision.models.pose import SyntheticPoseEstimator
from sentinel_vision.models.segmentation import ContourSegmenter, SegmentationScheduler
from sentinel_vision.models.temporal import (
    HeuristicActionClassifier,
    SkeletonHistory,
    normalize_skeleton,
)
from sentinel_vision.sources.image import ImageLoopSource
from sentinel_vision.sources.synthetic import SyntheticSource
from sentinel_vision.tracking.iou import IoUTracker


def _detection(x: float = 10, track_id: int | None = 1) -> PoseDetection:
    box = BoundingBox(x, 10, x + 40, 110)
    points = tuple(Keypoint(x + 20, 15 + index * 5, 0.9) for index in range(17))
    return PoseDetection(box, 0.9, points, track_id=track_id)


def test_synthetic_scene_pose_tracking_and_segmentation() -> None:
    source_config = SourceConfig(width=640, height=360, fps=30)
    source = SyntheticSource(source_config, realtime=False)
    source.open()
    frame = source.read()
    pose = SyntheticPoseEstimator(PoseConfig(max_detections=10))
    detections = pose.infer(frame.image)
    assert 2 <= len(detections) <= 3

    tracked = IoUTracker(TrackerConfig(minimum_iou=0.1)).update(detections, frame.sequence)
    assert all(item.track_id is not None for item in tracked)
    masks = ContourSegmenter(SegmentationConfig()).segment(frame.image, tracked)
    assert len(masks) == len(tracked)
    assert all(item.area_pixels > 0 for item in masks)


def test_image_loop_source_is_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), np.full((240, 320, 3), 127, dtype=np.uint8))
    config = SourceConfig(kind="image", uri=str(path), width=320, height=240, fps=30)
    source = ImageLoopSource(config, realtime=False)
    source.open()
    first, second = source.read(), source.read()
    assert first.sequence == 1
    assert second.sequence == 2
    np.testing.assert_array_equal(first.image, second.image)
    source.close()
    assert source.ready is False


def test_segmentation_scheduler_only_repeats_when_due() -> None:
    scheduler = SegmentationScheduler(
        SegmentationConfig(
            policy=SegmentationPolicyConfig(every_n_frames=5, segment_new_tracks=True)
        )
    )
    detection = _detection()
    assert scheduler.select((detection,), 1)
    assert not scheduler.select((detection,), 2)
    assert scheduler.select((detection,), 6)


def test_segmentation_uncertainty_has_its_own_cooldown() -> None:
    scheduler = SegmentationScheduler(
        SegmentationConfig(
            policy=SegmentationPolicyConfig(
                every_n_frames=20,
                uncertain_every_n_frames=5,
                low_pose_confidence=0.5,
            )
        )
    )
    detection = _detection()
    uncertain = PoseDetection(
        detection.bbox,
        detection.confidence,
        tuple(Keypoint(point.x, point.y, 0.1) for point in detection.keypoints),
        track_id=detection.track_id,
    )
    assert scheduler.select((uncertain,), 1)
    assert not scheduler.select((uncertain,), 2)
    assert scheduler.select((uncertain,), 6)


def test_skeleton_normalization_is_box_relative() -> None:
    normalized = normalize_skeleton(_detection())
    assert normalized.shape == (17, 3)
    assert np.all((normalized[:, :2] >= -0.5) & (normalized[:, :2] <= 0.5))
    assert np.allclose(normalized[:, 2], 0.9)


def test_heuristic_classifier_returns_normalized_probabilities() -> None:
    classifier = HeuristicActionClassifier(TemporalConfig())
    predictions = classifier.classify((_detection(),))
    assert len(predictions) == 1
    assert predictions[0].label in classifier.config.labels
    np.testing.assert_allclose(sum(predictions[0].probabilities.values()), 1.0)


def _moving_detection(center_x: float, top: float, height: float) -> PoseDetection:
    half = height * 0.2
    box = BoundingBox(center_x - half, top, center_x + half, top + height)
    points = tuple(Keypoint(center_x, top + height * index / 17, 0.9) for index in range(17))
    return PoseDetection(box, 0.9, points, track_id=1)


def test_motion_is_resolution_invariant() -> None:
    small = SkeletonHistory(8)
    large = SkeletonHistory(8)
    for step in range(5):
        small.update(_moving_detection(100 + step * 10, 50, 100))  # 10px/frame over 100px body
        large.update(_moving_detection(200 + step * 20, 100, 200))  # 20px/frame over 200px body
    # Both move 0.1 body-heights/frame, so normalized motion must match across scales.
    assert small.motion(1) == pytest.approx(large.motion(1), rel=1e-3)
    assert small.motion(1) == pytest.approx(0.1, abs=1e-6)


def test_heuristic_flags_sudden_horizontal_transition() -> None:
    classifier = HeuristicActionClassifier(TemporalConfig())
    classifier.classify((_detection(),))
    upright = _detection()
    horizontal = PoseDetection(
        BoundingBox(10, 10, 180, 70),
        upright.confidence,
        upright.keypoints,
        track_id=1,
    )
    prediction = classifier.classify((horizontal,))[0]
    assert prediction.label == "incident"
