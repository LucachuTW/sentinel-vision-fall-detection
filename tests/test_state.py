import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sentinel_vision.config import AppConfig, RuntimeConfig
from sentinel_vision.domain import FramePacket, PipelineEvent, ProcessedFrame
from sentinel_vision.state import StateStore


def _frame(sequence: int) -> ProcessedFrame:
    packet = FramePacket(
        "test",
        sequence,
        datetime.now(UTC),
        time.monotonic_ns(),
        np.zeros((32, 32, 3), dtype=np.uint8),
    )
    return ProcessedFrame(packet, (), (), (), {}, 1.0, b"jpeg")


async def test_state_waits_for_new_frame_and_updates_narrative() -> None:
    state = StateStore(AppConfig())
    state.start({}, [])
    waiter = asyncio.create_task(state.wait_for_frame(0, max_wait_seconds=1))
    await state.update_frame(_frame(1))
    assert await waiter == 1
    assert state.telemetry()[0]["sequence"] == 1
    assert state.telemetry()[0]["end_to_end_ms"] == 1.0
    event = PipelineEvent("event-1", "test", 1, 7, "action_transition", "warning", "feeding", 0.8)
    state.add_events((event,))
    await state.add_narrative("event-1", "Observed transition.")
    assert state.events()[0]["narrative"] == "Observed transition."


async def test_events_persist_to_sqlite_across_restart(tmp_path: Path) -> None:
    config = AppConfig(runtime=RuntimeConfig(event_db_path=str(tmp_path / "events.db")))
    first = StateStore(config)
    event = PipelineEvent("e1", "cam", 5, 7, "action_transition", "warning", "feeding", 0.8)
    first.add_events((event,))
    await first.add_narrative("e1", "Persisted note.")

    reopened = StateStore(config)  # simulates a process restart on the same database
    events = reopened.events()
    assert len(events) == 1
    assert events[0]["event_id"] == "e1"
    assert events[0]["narrative"] == "Persisted note."
