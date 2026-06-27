from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentinel_vision.config import AppConfig
from sentinel_vision.domain import PipelineEvent, ProcessedFrame


class StateStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._latest: ProcessedFrame | None = None
        self._events: deque[PipelineEvent] = deque(maxlen=config.runtime.event_history_size)
        self._started_at: datetime | None = None
        self._last_frame_at: datetime | None = None
        self._error: str | None = None
        self._running = False
        self._components: dict[str, str] = {}
        self._degradations: list[str] = []
        self._processed_times: deque[float] = deque(maxlen=120)
        self._telemetry: deque[dict[str, Any]] = deque(maxlen=600)
        self._new_frame = asyncio.Condition()
        self._db: sqlite3.Connection | None = None
        if config.runtime.event_db_path:
            path = Path(config.runtime.event_db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "event_id TEXT PRIMARY KEY, source_id TEXT, sequence INTEGER, track_id INTEGER, "
                "kind TEXT, severity TEXT, action TEXT, confidence REAL, occurred_at TEXT, "
                "narrative TEXT)"
            )
            self._load_events()

    def start(self, components: dict[str, str], degradations: list[str]) -> None:
        with self._lock:
            self._running = True
            self._started_at = datetime.now(UTC)
            self._components = components
            self._degradations = degradations
            self._error = None

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def set_error(self, error: str) -> None:
        with self._lock:
            self._error = error

    async def update_frame(self, frame: ProcessedFrame) -> None:
        with self._lock:
            self._latest = frame
            self._last_frame_at = datetime.now(UTC)
            self._processed_times.append(time.monotonic())
            self._telemetry.append(
                {
                    "timestamp": self._last_frame_at.isoformat(),
                    "sequence": frame.frame.sequence,
                    "fps": round(self._fps(), 3),
                    "end_to_end_ms": round(frame.end_to_end_ms, 3),
                    "tracks": len(frame.detections),
                    "latency_ms": {
                        key: round(value, 3) for key, value in frame.stage_latency_ms.items()
                    },
                }
            )
        async with self._new_frame:
            self._new_frame.notify_all()

    def add_events(self, events: tuple[PipelineEvent, ...]) -> None:
        with self._lock:
            self._events.extend(events)
            if self._db is not None:
                self._db.executemany(
                    "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [self._event_row(event) for event in events],
                )

    async def add_narrative(self, event_id: str, narrative: str) -> None:
        with self._lock:
            updated = [
                PipelineEvent(
                    event_id=item.event_id,
                    source_id=item.source_id,
                    sequence=item.sequence,
                    track_id=item.track_id,
                    kind=item.kind,
                    severity=item.severity,
                    action=item.action,
                    confidence=item.confidence,
                    occurred_at=item.occurred_at,
                    narrative=narrative,
                )
                if item.event_id == event_id
                else item
                for item in self._events
            ]
            self._events = deque(
                updated,
                maxlen=self.config.runtime.event_history_size,
            )
            if self._db is not None:
                self._db.execute(
                    "UPDATE events SET narrative = ? WHERE event_id = ?", (narrative, event_id)
                )

    def health(self) -> dict[str, Any]:
        with self._lock:
            ready = self._running and self._latest is not None and self._error is None
            status = (
                "ready"
                if ready
                else "error"
                if self._error
                else "starting"
                if self._running
                else "stopped"
            )
            return {
                "status": status,
                "ready": ready,
                "running": self._running,
                "environment": self.config.environment,
                "last_frame_at": self._last_frame_at.isoformat() if self._last_frame_at else None,
                "error": self._error,
                "components": dict(self._components),
                "degradations": list(self._degradations),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latest = self._latest
            fps = self._fps()
            if latest is None:
                return {
                    "source_id": self.config.source.source_id,
                    "sequence": 0,
                    "fps": fps,
                    "tracks": [],
                    "actions": [],
                    "latency_ms": {},
                    "events_total": len(self._events),
                }
            return {
                "source_id": latest.frame.source_id,
                "sequence": latest.frame.sequence,
                "captured_at": latest.frame.captured_at.isoformat(),
                "fps": round(fps, 2),
                "tracks": [item.as_dict() for item in latest.detections],
                "actions": [item.as_dict() for item in latest.actions],
                "segmented_track_ids": [item.track_id for item in latest.segmentations],
                "latency_ms": {
                    key: round(value, 3) for key, value in latest.stage_latency_ms.items()
                },
                "end_to_end_ms": round(latest.end_to_end_ms, 3),
                "events_total": len(self._events),
            }

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [item.as_dict() for item in list(self._events)[-limit:]][::-1]

    def telemetry(self, limit: int = 120) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._telemetry)[-limit:]

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest.rendered_jpeg if self._latest else None

    def latest_sequence(self) -> int:
        with self._lock:
            return self._latest.frame.sequence if self._latest else 0

    async def wait_for_frame(self, after_sequence: int, max_wait_seconds: float = 2.0) -> int:
        async with self._new_frame:
            with suppress(TimeoutError):
                async with asyncio.timeout(max_wait_seconds):
                    await self._new_frame.wait_for(lambda: self.latest_sequence() > after_sequence)
        return self.latest_sequence()

    def _fps(self) -> float:
        if len(self._processed_times) < 2:
            return 0.0
        elapsed = self._processed_times[-1] - self._processed_times[0]
        return (len(self._processed_times) - 1) / elapsed if elapsed > 0 else 0.0

    def _load_events(self) -> None:
        assert self._db is not None
        rows = self._db.execute(
            "SELECT event_id, source_id, sequence, track_id, kind, severity, action, confidence,"
            " occurred_at, narrative FROM events ORDER BY rowid DESC LIMIT ?",
            (self.config.runtime.event_history_size,),
        ).fetchall()
        for row in reversed(rows):
            self._events.append(self._row_to_event(row))

    @staticmethod
    def _event_row(event: PipelineEvent) -> tuple[Any, ...]:
        return (
            event.event_id,
            event.source_id,
            event.sequence,
            event.track_id,
            event.kind,
            event.severity,
            event.action,
            event.confidence,
            event.occurred_at.isoformat(),
            event.narrative,
        )

    @staticmethod
    def _row_to_event(row: tuple[Any, ...]) -> PipelineEvent:
        return PipelineEvent(
            event_id=row[0],
            source_id=row[1],
            sequence=row[2],
            track_id=row[3],
            kind=row[4],
            severity=row[5],
            action=row[6],
            confidence=row[7],
            occurred_at=datetime.fromisoformat(row[8]),
            narrative=row[9],
        )
