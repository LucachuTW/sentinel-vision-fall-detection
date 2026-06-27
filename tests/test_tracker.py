from sentinel_vision.config import TrackerConfig
from sentinel_vision.domain import BoundingBox, PoseDetection
from sentinel_vision.tracking.bytetrack import match_source_detections
from sentinel_vision.tracking.iou import IoUTracker


def _box(x: float) -> PoseDetection:
    return PoseDetection(BoundingBox(x, 0, x + 20, 40), 0.9, ())


def test_match_source_detections_is_one_to_one() -> None:
    d0 = PoseDetection(BoundingBox(0, 0, 20, 40), 0.9, ())
    d1 = PoseDetection(BoundingBox(100, 0, 120, 40), 0.9, ())
    boxes = [BoundingBox(1, 0, 21, 40), BoundingBox(101, 0, 121, 40)]
    assert match_source_detections(boxes, (d0, d1)) == [0, 1]

    # Two tracked boxes both overlap the only detection: it is consumed once, never reused.
    overlapping = [BoundingBox(0, 0, 20, 40), BoundingBox(3, 0, 23, 40)]
    assignment = match_source_detections(overlapping, (d0,))
    assert assignment.count(0) == 1
    assert assignment.count(None) == 1


def test_iou_tracker_preserves_ids_and_creates_new_tracks() -> None:
    tracker = IoUTracker(TrackerConfig(minimum_iou=0.2, lost_track_buffer=2))
    first = tracker.update((_box(0), _box(100)), sequence=1)
    second = tracker.update((_box(2), _box(102)), sequence=2)
    assert [item.track_id for item in first] == [1, 2]
    assert [item.track_id for item in second] == [1, 2]
    third = tracker.update((_box(250),), sequence=3)
    assert third[0].track_id == 3
