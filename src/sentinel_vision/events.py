from __future__ import annotations

import time
import uuid

from sentinel_vision.config import TemporalConfig
from sentinel_vision.domain import ActionPrediction, FramePacket, PipelineEvent

# Per-label event severity. Covers both the synthetic demo labels and the real fall-detection
# labels; anything unlisted is informational.
_SEVERITY_BY_LABEL = {
    "incident": "critical",
    "fallen": "critical",
    "falling": "warning",
    "feeding": "warning",
}


class ActionEventDetector:
    def __init__(self, config: TemporalConfig) -> None:
        self.config = config
        self._last_label: dict[int, str] = {}
        self._last_emitted: dict[tuple[int, str], float] = {}

    def update(
        self, frame: FramePacket, actions: tuple[ActionPrediction, ...]
    ) -> tuple[PipelineEvent, ...]:
        now = time.monotonic()
        output: list[PipelineEvent] = []
        active_ids = {item.track_id for item in actions}
        self._last_label = {
            track_id: label
            for track_id, label in self._last_label.items()
            if track_id in active_ids
        }
        for action in actions:
            previous = self._last_label.get(action.track_id)
            self._last_label[action.track_id] = action.label
            # Fire on a confident label change; also fire when a track is first seen already in an
            # elevated state (e.g. a fall where the tracker re-IDs mid-motion), but not on a plain
            # first-sight of a normal activity.
            first_sight_normal = previous is None and action.label not in _SEVERITY_BY_LABEL
            if (
                previous == action.label
                or action.confidence < self.config.event_threshold
                or first_sight_normal
            ):
                continue
            key = (action.track_id, action.label)
            if now - self._last_emitted.get(key, -1e9) < self.config.event_cooldown_seconds:
                continue
            self._last_emitted[key] = now
            severity = _SEVERITY_BY_LABEL.get(action.label, "info")
            output.append(
                PipelineEvent(
                    event_id=uuid.uuid4().hex,
                    source_id=frame.source_id,
                    sequence=frame.sequence,
                    track_id=action.track_id,
                    kind="action_transition",
                    severity=severity,
                    action=action.label,
                    confidence=action.confidence,
                )
            )
        return tuple(output)
