from __future__ import annotations

import asyncio
import hmac
import os
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sentinel_vision.config import AppConfig
from sentinel_vision.observability import gpu_snapshot
from sentinel_vision.pipeline import VideoInferencePipeline

WEB_ROOT = Path(__file__).parent / "web"


def _token_ok(required_token: str | None, provided: str | None) -> bool:
    """True when auth is disabled, or the provided token matches in constant time."""
    if required_token is None:
        return True
    return provided is not None and hmac.compare_digest(provided, required_token)


def _make_token_guard(required_token: str | None) -> Callable[..., Coroutine[Any, Any, None]]:
    """Token check for the data/metrics surface. No-op when auth is disabled.

    Accepts an ``Authorization: Bearer <token>`` header or a ``?token=`` query parameter,
    because browsers cannot set headers on WebSocket or ``<img>`` requests.
    """

    async def guard(
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> None:
        provided = token
        if provided is None and authorization is not None and authorization.startswith("Bearer "):
            provided = authorization.removeprefix("Bearer ")
        if not _token_ok(required_token, provided):
            raise HTTPException(status_code=401, detail="invalid or missing API token")

    return guard


def create_app(
    config: AppConfig,
    pipeline: VideoInferencePipeline | None = None,
    *,
    manage_pipeline: bool = True,
) -> FastAPI:
    runtime = pipeline or VideoInferencePipeline(config)

    required_token = (os.environ.get("SV_API_TOKEN") or "").strip() or None
    if config.api.require_auth and required_token is None:
        raise RuntimeError("api.require_auth is enabled but SV_API_TOKEN is empty")
    active_token = required_token if config.api.require_auth else None
    guard = Depends(_make_token_guard(active_token))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if manage_pipeline:
            await runtime.start()
        yield
        if manage_pipeline:
            await runtime.stop()

    app = FastAPI(
        title=config.project_name,
        version="0.1.0",
        description="Low-latency local video analytics API",
        lifespan=lifespan,
    )
    app.state.pipeline = runtime

    @app.middleware("http")
    async def security_headers(request: Any, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Server"] = "sentinel-vision"
        return response

    if config.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.cors_origins,
            allow_credentials=False,
            allow_methods=["GET"],
            allow_headers=["*"],
        )
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    async def readiness() -> JSONResponse:
        health = runtime.state.health()
        return JSONResponse(health, status_code=200 if health["ready"] else 503)

    @app.get("/v1/state", tags=["inference"], dependencies=[guard])
    async def state() -> dict[str, Any]:
        return runtime.state.snapshot()

    @app.get("/v1/events", tags=["inference"], dependencies=[guard])
    async def events(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
        return runtime.state.events(limit)

    @app.get("/v1/telemetry", tags=["inference"], dependencies=[guard])
    async def telemetry(
        limit: int = Query(default=120, ge=2, le=600),
    ) -> list[dict[str, Any]]:
        return runtime.state.telemetry(limit)

    @app.get("/v1/config", tags=["system"], dependencies=[guard])
    async def safe_config() -> dict[str, object]:
        return config.safe_dict()

    @app.get("/v1/system", tags=["system"], dependencies=[guard])
    async def system_metrics() -> dict[str, object]:
        return {"gpu": gpu_snapshot()}

    @app.get("/metrics", include_in_schema=False, dependencies=[guard])
    async def metrics() -> Response:
        return Response(runtime.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/stream.mjpeg", tags=["inference"], dependencies=[guard])
    async def video_stream() -> StreamingResponse:
        async def frames() -> AsyncIterator[bytes]:
            sequence = 0
            while True:
                sequence = await runtime.state.wait_for_frame(sequence)
                jpeg = runtime.state.jpeg()
                if jpeg is None:
                    await asyncio.sleep(0.05)
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/v1/ws")
    async def websocket_state(websocket: WebSocket) -> None:
        if not _token_ok(active_token, websocket.query_params.get("token")):
            await websocket.close(code=1008)  # policy violation
            return
        await websocket.accept()
        sequence = 0
        try:
            while True:
                sequence = await runtime.state.wait_for_frame(sequence)
                await websocket.send_json(runtime.state.snapshot())
        except WebSocketDisconnect:
            return

    return app
