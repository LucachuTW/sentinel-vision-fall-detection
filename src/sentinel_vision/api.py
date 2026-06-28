from __future__ import annotations

import asyncio
import hmac
import os
import re
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sentinel_vision.config import AppConfig, SourceConfig, redact_uri
from sentinel_vision.observability import gpu_snapshot
from sentinel_vision.pipeline import VideoInferencePipeline

WEB_ROOT = Path(__file__).parent / "web"
UPLOADS = Path("uploads")
_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}


class SourceSwitch(BaseModel):
    # "demo" restores the server's original startup source (handled in the endpoint).
    kind: Literal["demo", "synthetic", "camera", "rtsp"]
    uri: str | None = None
    width: int = 1280
    height: int = 720
    fps: float = 30.0


def build_source(switch: SourceSwitch) -> SourceConfig:
    """Validate a source-switch request into a SourceConfig, rejecting unsafe inputs.

    `file` is intentionally not reachable here — uploaded videos go through the upload
    endpoint, which controls the path, so the API never opens an arbitrary local path.
    """
    if switch.kind == "camera":
        device = (switch.uri or "0").strip()
        if not device.isdigit():
            raise HTTPException(status_code=400, detail="camera uri must be a device index")
        return SourceConfig(
            kind="camera",
            uri=device,
            source_id=f"webcam-{device}",
            hardware_decode=False,
            width=switch.width,
            height=switch.height,
            fps=switch.fps,
        )
    if switch.kind == "rtsp":
        if not switch.uri or not switch.uri.startswith("rtsp://"):
            raise HTTPException(status_code=400, detail="rtsp uri must start with rtsp://")
        return SourceConfig(
            kind="rtsp",
            uri=switch.uri,
            source_id="rtsp-camera",
            width=switch.width,
            height=switch.height,
            fps=switch.fps,
        )
    if switch.kind == "synthetic":
        return SourceConfig(
            kind="synthetic",
            uri="synthetic://live",
            source_id="synthetic",
            hardware_decode=False,
            width=switch.width,
            height=switch.height,
            fps=switch.fps,
        )
    raise HTTPException(status_code=400, detail=f"unsupported source kind: {switch.kind}")


def _safe_filename(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name))
    extension = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    if not base or extension not in _VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"unsupported video file: {name}")
    return base


def _source_view(source: SourceConfig) -> dict[str, object]:
    return {"kind": source.kind, "uri": redact_uri(source.uri), "source_id": source.source_id}


def _persist_upload(name: str, body: bytes) -> SourceConfig:
    """Blocking disk write + decode probe; run off the event loop via asyncio.to_thread."""
    import cv2

    UPLOADS.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS / name
    destination.write_bytes(body)
    capture = cv2.VideoCapture(str(destination))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="could not decode uploaded video")
    return SourceConfig(
        kind="file",
        uri=str(destination),
        source_id=f"upload-{name}",
        width=min(7680, max(320, width)),
        height=min(4320, max(240, height)),
        fps=min(240.0, max(1.0, fps or 30.0)),
        loop=True,
    )


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
    async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
        if manage_pipeline:
            await app_.state.pipeline.start()
        yield
        if manage_pipeline:
            await app_.state.pipeline.stop()

    app = FastAPI(
        title=config.project_name,
        version="0.1.0",
        description="Low-latency local video analytics API",
        lifespan=lifespan,
    )
    app.state.pipeline = runtime
    app.state.config = config
    switch_lock = asyncio.Lock()

    async def switch_source(new_source: SourceConfig) -> None:
        """Hot-swap the input source, reusing the loaded models, metrics, and read model.

        Reusing StateStore/PipelineMetrics keeps the dashboard connected across the switch;
        reusing the model adapters avoids reloading YOLO26/SAM2. The tracker and classifier
        carry a little state from the previous scene but self-heal as stale ids age out.
        """
        async with switch_lock:
            old = app.state.pipeline
            new_config = config.model_copy(update={"source": new_source})
            new = VideoInferencePipeline(
                new_config,
                pose=old.pose,
                tracker=old.tracker,
                segmenter=old.segmenter,
                classifier=old.classifier,
                metrics=old.metrics,
                state=old.state,
            )
            await old.stop()
            await new.start()
            app.state.pipeline = new
            app.state.config = new_config

    @app.middleware("http")
    async def security_headers(request: Any, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Server"] = "sentinel-vision"
        # Live dashboard + assets must never be served stale from browser cache.
        response.headers.setdefault("Cache-Control", "no-cache")
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
        current: AppConfig = app.state.config
        return current.safe_dict()

    @app.get("/v1/system", tags=["system"], dependencies=[guard])
    async def system_metrics() -> dict[str, object]:
        return {"gpu": gpu_snapshot()}

    @app.post("/v1/source", tags=["system"], dependencies=[guard])
    async def set_source(switch: SourceSwitch) -> dict[str, object]:
        # "demo" restores the original startup source (a known, server-controlled path).
        source = config.source if switch.kind == "demo" else build_source(switch)
        await switch_source(source)
        return {"status": "switching", "source": _source_view(source)}

    @app.post("/v1/source/upload", tags=["system"], dependencies=[guard])
    async def upload_source(
        request: Request, filename: str = Query(min_length=1)
    ) -> dict[str, object]:
        name = _safe_filename(filename)
        body = await request.body()
        if len(body) > 300 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="uploaded video exceeds 300 MB")
        source = await asyncio.to_thread(_persist_upload, name, body)
        await switch_source(source)
        return {"status": "switching", "source": _source_view(source), "fps": source.fps}

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
