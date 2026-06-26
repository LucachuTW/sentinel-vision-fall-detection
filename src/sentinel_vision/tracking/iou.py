from __future__ import annotations

from dataclasses import dataclass

from sentinel_vision.config import TrackerConfig
from sentinel_vision.domain import BoundingBox, PoseDetection
from sentinel_vision.tracking.base import Tracker


@dataclass(slots=True)
class _Track:
    track_id: int
    bbox: BoundingBox
    last_sequence: int
    hits: int = 1


class IoUTracker(Tracker):
    """Deterministic fallback tracker; production uses ByteTrackAdapter."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    @property
    def backend_name(self) -> str:
        return "iou-fallback"

    def update(
        self, detections: tuple[PoseDetection, ...], sequence: int
    ) -> tuple[PoseDetection, ...]:
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if sequence - track.last_sequence > self.config.lost_track_buffer
        ]
        for track_id in stale:
            del self._tracks[track_id]

        candidates = sorted(
            (
                (detection.bbox.iou(track.bbox), det_index, track_id)
                for det_index, detection in enumerate(detections)
                for track_id, track in self._tracks.items()
            ),
            reverse=True,
        )
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for iou, det_index, track_id in candidates:
            if iou < self.config.minimum_iou:
                break
            if det_index in assignments or track_id in used_tracks:
                continue
            assignments[det_index] = track_id
            used_tracks.add(track_id)

        tracked: list[PoseDetection] = []
        for det_index, detection in enumerate(detections):
            assigned_id = assignments.get(det_index)
            if assigned_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id, detection.bbox, sequence)
            else:
                track_id = assigned_id
                track = self._tracks[track_id]
                track.bbox = detection.bbox
                track.last_sequence = sequence
                track.hits += 1
            tracked.append(detection.with_track(track_id))
        return tuple(tracked)
