import asyncio

import httpx
import pytest
from fastapi import HTTPException, WebSocketDisconnect
from fastapi.testclient import TestClient

from sentinel_vision.api import create_app
from sentinel_vision.config import ApiConfig, AppConfig, SourceConfig
from sentinel_vision.pipeline import VideoInferencePipeline


async def test_api_liveness_and_safe_config() -> None:
    config = AppConfig(
        environment="development",
        source=SourceConfig(kind="rtsp", uri="rtsp://alice:secret@camera.local/live"),
    )
    pipeline = VideoInferencePipeline(config)
    app = create_app(config, pipeline, manage_pipeline=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        live = await client.get("/health/live")
        safe_config = await client.get("/v1/config")
    assert live.status_code == 200
    assert safe_config.json()["source"]["uri"] == "rtsp://***:***@camera.local/live"


async def test_readiness_is_503_before_pipeline_start() -> None:
    config = AppConfig()
    pipeline = VideoInferencePipeline(config)
    app = create_app(config, pipeline, manage_pipeline=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


async def test_auth_guard_rejects_anonymous_but_allows_token_and_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SV_API_TOKEN", "s3cret-token")
    config = AppConfig(api=ApiConfig(require_auth=True))
    pipeline = VideoInferencePipeline(config)
    app = create_app(config, pipeline, manage_pipeline=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        anonymous = await client.get("/v1/state")
        authorized = await client.get("/v1/state", headers={"Authorization": "Bearer s3cret-token"})
        wrong = await client.get("/v1/state", headers={"Authorization": "Bearer nope"})
        live = await client.get("/health/live")
    assert anonymous.status_code == 401
    assert wrong.status_code == 401
    assert authorized.status_code == 200
    assert live.status_code == 200  # health probes stay open


def test_build_source_validates_and_rejects_unsafe_inputs() -> None:
    from sentinel_vision.api import SourceSwitch, build_source

    assert build_source(SourceSwitch(kind="camera", uri="0")).kind == "camera"
    assert build_source(SourceSwitch(kind="rtsp", uri="rtsp://cam/live")).kind == "rtsp"
    assert build_source(SourceSwitch(kind="synthetic")).uri == "synthetic://live"
    with pytest.raises(HTTPException):  # camera must be a numeric device index, not a path
        build_source(SourceSwitch(kind="camera", uri="/etc/passwd"))
    with pytest.raises(HTTPException):  # only rtsp:// urls are accepted
        build_source(SourceSwitch(kind="rtsp", uri="http://evil/x"))


async def test_source_switch_changes_active_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SV_API_TOKEN", raising=False)
    config = AppConfig()
    pipeline = VideoInferencePipeline(config)
    await pipeline.start()
    try:
        app = create_app(config, pipeline, manage_pipeline=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/source", json={"kind": "synthetic"})
            current = await client.get("/v1/config")
        assert response.status_code == 200
        assert response.json()["source"]["kind"] == "synthetic"
        assert current.json()["source"]["source_id"] == "synthetic"
    finally:
        await app.state.pipeline.stop()


def test_auth_fails_closed_when_token_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SV_API_TOKEN", raising=False)
    config = AppConfig(api=ApiConfig(require_auth=True))
    pipeline = VideoInferencePipeline(config)
    with pytest.raises(RuntimeError):
        create_app(config, pipeline, manage_pipeline=False)


async def test_mjpeg_is_guarded_and_responses_carry_security_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SV_API_TOKEN", "stream-token")
    config = AppConfig(api=ApiConfig(require_auth=True))
    app = create_app(config, VideoInferencePipeline(config), manage_pipeline=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # The guard rejects before streaming starts; we never enter the infinite generator.
        denied = await client.get("/v1/stream.mjpeg")
        live = await client.get("/health/live")
    assert denied.status_code == 401  # query-param accept path is covered by the WebSocket test
    assert live.headers["x-content-type-options"] == "nosniff"


def test_websocket_requires_token_via_query_param(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SV_API_TOKEN", "ws-token")
    config = AppConfig(api=ApiConfig(require_auth=True))
    with TestClient(
        create_app(config, VideoInferencePipeline(config), manage_pipeline=False)
    ) as client:
        # Closed before accept when the token is absent.
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/v1/ws"):
            pass
        with client.websocket_connect("/v1/ws?token=ws-token"):  # accepted with a valid token
            pass


async def test_live_pipeline_state_events_and_metrics_endpoints() -> None:
    config = AppConfig()
    pipeline = VideoInferencePipeline(config)
    await pipeline.start()
    try:
        for _ in range(50):
            if pipeline.state.latest_sequence() > 0:
                break
            await asyncio.sleep(0.02)
        app = create_app(config, pipeline, manage_pipeline=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            ready = await client.get("/health/ready")
            state = await client.get("/v1/state")
            events = await client.get("/v1/events?limit=5")
            telemetry = await client.get("/v1/telemetry?limit=10")
            system = await client.get("/v1/system")
            metrics = await client.get("/metrics")
        assert ready.status_code == 200
        assert state.json()["sequence"] > 0
        assert isinstance(events.json(), list)
        assert telemetry.json()[0]["sequence"] > 0
        assert "gpu" in system.json()
        assert "sentinel_frames_processed_total" in metrics.text
        assert "sentinel_queue_depth" in metrics.text
    finally:
        await pipeline.stop()
