from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(StrictModel):
    kind: Literal["synthetic", "image", "file", "rtsp", "camera"] = "synthetic"
    uri: str = "synthetic://industrial-floor"
    source_id: str = "camera-01"
    width: int = Field(default=1280, ge=320, le=7680)
    height: int = Field(default=720, ge=240, le=4320)
    fps: float = Field(default=30.0, gt=0, le=240)
    codec: Literal["h264", "h265", "auto"] = "auto"
    hardware_decode: bool = True
    latency_ms: int = Field(default=100, ge=0, le=5000)
    loop: bool = True


class PoseConfig(StrictModel):
    backend: Literal["synthetic", "ultralytics"] = "synthetic"
    model: str = "yolo26n-pose.pt"
    device: str = "auto"
    precision: Literal["fp32", "fp16", "int8"] = "fp16"
    engine_path: str | None = None
    confidence: float = Field(default=0.35, ge=0, le=1)
    image_size: int = Field(default=640, ge=320, le=1920)
    max_detections: int = Field(default=30, ge=1, le=500)

    @model_validator(mode="after")
    def validate_yolo26(self) -> PoseConfig:
        name = Path(self.engine_path or self.model).name.lower()
        if self.backend == "ultralytics" and "yolo26" not in name:
            raise ValueError("pose.model must be a YOLO26 pose checkpoint or engine")
        if self.backend == "ultralytics" and self.engine_path is None and "pose" not in name:
            raise ValueError("pose.model must use a YOLO26 pose checkpoint")
        if self.precision == "int8" and not self.engine_path:
            raise ValueError(
                "INT8 inference requires pose.engine_path built with a calibration dataset"
            )
        return self


class TrackerConfig(StrictModel):
    backend: Literal["iou", "bytetrack"] = "iou"
    track_activation_threshold: float = Field(default=0.25, ge=0, le=1)
    lost_track_buffer: int = Field(default=30, ge=1, le=1000)
    minimum_matching_threshold: float = Field(default=0.72, ge=0, le=1)
    minimum_iou: float = Field(default=0.25, ge=0, le=1)
    frame_rate: int = Field(default=30, ge=1, le=240)


class SegmentationPolicyConfig(StrictModel):
    every_n_frames: int = Field(default=12, ge=1, le=10000)
    uncertain_every_n_frames: int = Field(default=5, ge=1, le=10000)
    max_objects_per_frame: int = Field(default=4, ge=1, le=100)
    low_pose_confidence: float = Field(default=0.45, ge=0, le=1)
    segment_new_tracks: bool = True


class SegmentationConfig(StrictModel):
    backend: Literal["contour", "sam2"] = "contour"
    model: str = "sam2.1_t.pt"
    device: str = "auto"
    image_size: int = Field(default=1024, ge=256, le=2048)
    dynamic_memory: bool = True
    policy: SegmentationPolicyConfig = Field(default_factory=SegmentationPolicyConfig)


class TemporalConfig(StrictModel):
    backend: Literal["heuristic", "transformer"] = "heuristic"
    checkpoint: str = "models/temporal-transformer.pt"
    labels: list[str] = Field(
        default_factory=lambda: ["standing", "active", "feeding", "resting", "incident"]
    )
    sequence_length: int = Field(default=32, ge=4, le=512)
    event_threshold: float = Field(default=0.72, ge=0, le=1)
    event_cooldown_seconds: float = Field(default=5.0, ge=0, le=3600)
    device: str = "auto"


class OllamaConfig(StrictModel):
    enabled: bool = False
    endpoint: str = "http://127.0.0.1:11434"
    model: str = "qwen3:1.7b"
    timeout_seconds: float = Field(default=8.0, gt=0, le=60)
    min_severity: Literal["info", "warning", "critical"] = "warning"


class RuntimeConfig(StrictModel):
    queue_size: int = Field(default=2, ge=1, le=128)
    target_fps: float = Field(default=30.0, gt=0, le=240)
    jpeg_quality: int = Field(default=82, ge=20, le=100)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    event_history_size: int = Field(default=500, ge=10, le=100000)
    warmup_iterations: int = Field(default=2, ge=0, le=100)
    # When set, events are persisted to SQLite and reloaded on restart. None keeps them in memory only.
    event_db_path: str | None = None


class ApiConfig(StrictModel):
    # Safe default: bind loopback. Configs that must be reachable (Docker, production)
    # set 0.0.0.0 explicitly and should pair it with require_auth.
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=list)
    # When true the JSON/metrics surface requires a bearer token from SV_API_TOKEN.
    # Fail-closed: the app refuses to start if the token is missing.
    require_auth: bool = False


class AppConfig(StrictModel):
    project_name: str = "Sentinel Vision"
    environment: Literal["demo", "development", "production"] = "demo"
    source: SourceConfig = Field(default_factory=SourceConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    @model_validator(mode="after")
    def validate_production_backends(self) -> AppConfig:
        if self.environment == "production":
            fallback = {
                "pose": self.pose.backend == "synthetic",
                "tracker": self.tracker.backend == "iou",
                "segmentation": self.segmentation.backend == "contour",
                "temporal": self.temporal.backend == "heuristic",
            }
            active = [name for name, enabled in fallback.items() if enabled]
            if active:
                raise ValueError(f"production cannot use fallback backends: {', '.join(active)}")
        return self

    def safe_dict(self) -> dict[str, object]:
        data = self.model_dump(mode="json")
        source = data["source"]
        assert isinstance(source, dict)
        source["uri"] = redact_uri(str(source["uri"]))
        return data


def redact_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if parsed.username is None and parsed.password is None:
        return uri
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    netloc = f"***:***@{hostname}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    source_uri = os.getenv("SV_SOURCE_URI")
    if source_uri:
        raw.setdefault("source", {})["uri"] = source_uri
    ollama_endpoint = os.getenv("SV_OLLAMA_ENDPOINT")
    if ollama_endpoint:
        raw.setdefault("ollama", {})["endpoint"] = ollama_endpoint
    return AppConfig.model_validate(raw)
