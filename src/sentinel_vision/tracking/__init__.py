from sentinel_vision.config import TrackerConfig
from sentinel_vision.tracking.base import Tracker
from sentinel_vision.tracking.bytetrack import ByteTrackAdapter
from sentinel_vision.tracking.iou import IoUTracker


def create_tracker(config: TrackerConfig) -> Tracker:
    if config.backend == "bytetrack":
        return ByteTrackAdapter(config)
    return IoUTracker(config)


__all__ = ["Tracker", "create_tracker"]
