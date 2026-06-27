import pytest

from sentinel_vision.domain import BoundingBox


def test_bounding_box_iou_and_geometry() -> None:
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 15, 15)
    assert a.area == 100
    assert a.center == (5, 5)
    assert a.iou(b) == pytest.approx(25 / 175)
    assert a.iou(BoundingBox(20, 20, 30, 30)) == 0


def test_bounding_box_clips_to_image() -> None:
    assert BoundingBox(-4, -2, 120, 80).clipped(100, 50) == BoundingBox(0, 0, 99, 49)
