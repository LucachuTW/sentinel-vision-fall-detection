import asyncio
import json
import logging
import time
from datetime import UTC, datetime

import httpx
import numpy as np

from sentinel_vision.async_runtime import run_async
from sentinel_vision.config import OllamaConfig, TemporalConfig
from sentinel_vision.domain import ActionPrediction, FramePacket, PipelineEvent
from sentinel_vision.events import ActionEventDetector
from sentinel_vision.models.narrative import LocalNarrativeService
from sentinel_vision.observability import JsonFormatter, PipelineMetrics, configure_logging


def test_run_async_uses_available_event_loop() -> None:
    async def value() -> int:
        await asyncio.sleep(0)
        return 42

    assert run_async(value()) == 42


async def test_local_narrative_generates_bounded_fact_based_text() -> None:
    event = PipelineEvent(
        "event-1", "camera-1", 9, 3, "action_transition", "warning", "feeding", 0.81
    )
    updates: list[tuple[str, str]] = []

    async def update(event_id: str, text: str) -> None:
        updates.append((event_id, text))

    service = LocalNarrativeService(
        OllamaConfig(enabled=True, model="qwen3:1.7b", min_severity="warning"),
        update,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/generate"
        assert payload["model"] == "qwen3:1.7b"
        assert "camera=camera-1" in payload["prompt"]
        return httpx.Response(
            200, json={"response": "  Transición observada sin causa confirmada.  "}
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama"
    ) as client:
        result = await service._generate(client, event)
    assert result == "Transición observada sin causa confirmada."
    for _ in range(35):
        service.submit(event)
    assert service._queue.qsize() == 32
    service.submit(PipelineEvent("info", "camera", 1, 1, "transition", "info", "standing", 0.9))
    assert service._queue.qsize() == 32
    service.stop()


async def test_disabled_narrative_worker_returns_immediately() -> None:
    async def update(_: str, __: str) -> None:
        return None

    service = LocalNarrativeService(OllamaConfig(enabled=False), update)
    service.submit(PipelineEvent("id", "camera", 1, 1, "transition", "critical", "incident", 0.9))
    await service.run()
    assert service.backend_name == "disabled"
    assert service._queue.empty()


def test_json_logs_and_metrics_exposition() -> None:
    record = logging.LogRecord("test", logging.INFO, "", 0, "processed %s", (3,), None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "processed 3"
    metrics = PipelineMetrics()
    metrics.drop("pose")
    metrics.configure_queues({"pose": 2})
    metrics.set_queue_depths({"pose": 1})
    metrics.success("pose")
    metrics.error("pose", "RuntimeError")
    rendered = metrics.render()
    assert b'sentinel_frames_dropped_total{stage="pose"} 1.0' in rendered
    assert b'sentinel_queue_depth{stage="pose"} 1.0' in rendered
    assert b'sentinel_queue_capacity{stage="pose"} 2.0' in rendered
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_event_detector_ignores_initial_state_and_emits_transition() -> None:
    detector = ActionEventDetector(TemporalConfig(event_threshold=0.5))
    frame = FramePacket(
        "camera",
        1,
        datetime.now(UTC),
        time.monotonic_ns(),
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    standing = ActionPrediction(1, "standing", 0.9, {"standing": 0.9}, True)
    incident = ActionPrediction(1, "incident", 0.9, {"incident": 0.9}, True)
    assert detector.update(frame, (standing,)) == ()
    events = detector.update(frame, (incident,))
    assert len(events) == 1
    assert events[0].severity == "critical"
