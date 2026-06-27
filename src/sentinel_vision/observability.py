from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info, generate_latest


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


class PipelineMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.build = Info("sentinel_build", "Build and runtime metadata", registry=self.registry)
        self.build.info({"version": "0.1.0", "runtime": "python"})
        self.frames_ingested = Counter(
            "sentinel_frames_ingested_total",
            "Frames accepted from video sources",
            ["source"],
            registry=self.registry,
        )
        self.frames_processed = Counter(
            "sentinel_frames_processed_total",
            "Frames emitted by the complete pipeline",
            ["source"],
            registry=self.registry,
        )
        self.frames_dropped = Counter(
            "sentinel_frames_dropped_total",
            "Frames dropped to preserve live-stream latency",
            ["stage"],
            registry=self.registry,
        )
        self.stage_latency = Histogram(
            "sentinel_stage_latency_seconds",
            "Inference stage latency",
            ["stage"],
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.2),
            registry=self.registry,
        )
        self.end_to_end_latency = Histogram(
            "sentinel_end_to_end_latency_seconds",
            "Capture-to-output latency",
            buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.2, 2.5),
            registry=self.registry,
        )
        self.active_tracks = Gauge(
            "sentinel_active_tracks",
            "Currently visible tracked objects",
            registry=self.registry,
        )
        self.events = Counter(
            "sentinel_events_total",
            "Detected temporal events",
            ["kind", "severity"],
            registry=self.registry,
        )
        self.pipeline_ready = Gauge(
            "sentinel_pipeline_ready",
            "Whether the pipeline is ready to process frames",
            registry=self.registry,
        )
        self.source_connected = Gauge(
            "sentinel_source_connected",
            "Whether the configured source is connected",
            ["source"],
            registry=self.registry,
        )
        self.source_reconnects = Counter(
            "sentinel_source_reconnects_total",
            "Video source reconnect attempts",
            ["source"],
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "sentinel_queue_depth",
            "Current number of items waiting at a stage boundary",
            ["stage"],
            registry=self.registry,
        )
        self.queue_capacity = Gauge(
            "sentinel_queue_capacity",
            "Configured queue capacity at a stage boundary",
            ["stage"],
            registry=self.registry,
        )
        self.inferences = Counter(
            "sentinel_inferences_total",
            "Completed inference operations",
            ["stage", "status"],
            registry=self.registry,
        )
        self.stage_errors = Counter(
            "sentinel_stage_errors_total",
            "Unhandled errors by pipeline stage",
            ["stage", "exception"],
            registry=self.registry,
        )
        self.detections = Counter(
            "sentinel_detections_total",
            "Pose detections produced by the model",
            registry=self.registry,
        )
        self.segmentations = Counter(
            "sentinel_segmentations_total",
            "Segmentation outcomes",
            ["outcome"],
            registry=self.registry,
        )
        self.actions = Counter(
            "sentinel_actions_total",
            "Temporal action predictions",
            ["label"],
            registry=self.registry,
        )
        self.frame_age = Gauge(
            "sentinel_frame_age_seconds",
            "Age of the latest processed frame at publication time",
            registry=self.registry,
        )
        self.last_success = Gauge(
            "sentinel_stage_last_success_timestamp_seconds",
            "Unix timestamp of the latest successful stage execution",
            ["stage"],
            registry=self.registry,
        )
        self.gpu_available = Gauge(
            "sentinel_gpu_available", "Whether NVML can observe a GPU", registry=self.registry
        )
        self.gpu_utilization = Gauge(
            "sentinel_gpu_utilization_ratio",
            "GPU compute utilization from zero to one",
            registry=self.registry,
        )
        self.gpu_memory_used = Gauge(
            "sentinel_gpu_memory_used_bytes", "GPU memory currently used", registry=self.registry
        )
        self.gpu_memory_total = Gauge(
            "sentinel_gpu_memory_total_bytes", "Total GPU memory", registry=self.registry
        )
        self.gpu_temperature = Gauge(
            "sentinel_gpu_temperature_celsius", "GPU temperature", registry=self.registry
        )
        self.gpu_power = Gauge(
            "sentinel_gpu_power_watts", "GPU board power draw", registry=self.registry
        )
        self._last_gpu_sample = 0.0

    def drop(self, stage: str) -> None:
        self.frames_dropped.labels(stage=stage).inc()

    def configure_queues(self, capacities: dict[str, int]) -> None:
        for stage, capacity in capacities.items():
            self.queue_capacity.labels(stage=stage).set(capacity)

    def set_queue_depths(self, depths: dict[str, int]) -> None:
        for stage, depth in depths.items():
            self.queue_depth.labels(stage=stage).set(depth)

    def success(self, stage: str) -> None:
        self.inferences.labels(stage=stage, status="success").inc()
        self.last_success.labels(stage=stage).set_to_current_time()

    def error(self, stage: str, exception: str) -> None:
        self.inferences.labels(stage=stage, status="error").inc()
        self.stage_errors.labels(stage=stage, exception=exception).inc()

    def sample_gpu(self, minimum_interval_seconds: float = 1.0) -> dict[str, float] | None:
        now = time.monotonic()
        if now - self._last_gpu_sample < minimum_interval_seconds:
            return None
        self._last_gpu_sample = now
        snapshot = gpu_snapshot()
        if snapshot is None:
            self.gpu_available.set(0)
            return None
        self.gpu_available.set(1)
        self.gpu_utilization.set(snapshot["utilization_ratio"])
        self.gpu_memory_used.set(snapshot["memory_used_bytes"])
        self.gpu_memory_total.set(snapshot["memory_total_bytes"])
        self.gpu_temperature.set(snapshot["temperature_celsius"])
        self.gpu_power.set(snapshot["power_watts"])
        return snapshot

    def render(self) -> bytes:
        return generate_latest(self.registry)


def gpu_snapshot() -> dict[str, float] | None:
    try:
        import pynvml  # type: ignore[import-not-found,import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
        return {
            "utilization_ratio": float(utilization.gpu) / 100.0,
            "memory_used_bytes": float(memory.used),
            "memory_total_bytes": float(memory.total),
            "temperature_celsius": float(temperature),
            "power_watts": float(power_milliwatts) / 1000.0,
        }
    except Exception:
        return None
