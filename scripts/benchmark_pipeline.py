#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from sentinel_vision.async_runtime import run_async
from sentinel_vision.config import load_config
from sentinel_vision.observability import gpu_snapshot
from sentinel_vision.pipeline import VideoInferencePipeline


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"samples": 0, "min": 0, "mean": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "min": round(float(array.min()), 3),
        "mean": round(float(array.mean()), 3),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p90": round(float(np.percentile(array, 90)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "p99": round(float(np.percentile(array, 99)), 3),
        "max": round(float(array.max()), 3),
    }


def package_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for package in ("torch", "ultralytics", "opencv-python-headless", "supervision"):
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not-installed"
    return output


def nvidia_identity() -> dict[str, str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=4, check=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    fields = [item.strip() for item in result.stdout.strip().split(",")]
    if len(fields) != 3:
        return None
    return {"name": fields[0], "driver": fields[1], "memory_mib": fields[2]}


async def benchmark(config_path: str, seconds: float, warmup_seconds: float) -> dict[str, Any]:
    config = load_config(config_path)
    pipeline = VideoInferencePipeline(config)
    await pipeline.start()
    last_sequence = 0
    warmup_deadline = time.monotonic() + warmup_seconds
    while time.monotonic() < warmup_deadline and pipeline.running:
        last_sequence = await pipeline.state.wait_for_frame(last_sequence, 1.0)

    samples: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, float]] = []
    dropped_at_start = pipeline.dropped_frames
    started = time.monotonic()
    deadline = started + seconds
    next_gpu_sample = started
    first_sequence = last_sequence
    while time.monotonic() < deadline and pipeline.running:
        sequence = await pipeline.state.wait_for_frame(last_sequence, 1.0)
        if sequence > last_sequence:
            samples.append(pipeline.state.snapshot())
            last_sequence = sequence
        if time.monotonic() >= next_gpu_sample:
            gpu = gpu_snapshot()
            if gpu:
                gpu_samples.append(gpu)
            next_gpu_sample = time.monotonic() + 0.5
    measured_seconds = time.monotonic() - started
    health = pipeline.state.health()
    dropped = {
        stage: count - dropped_at_start[stage] for stage, count in pipeline.dropped_frames.items()
    }
    await pipeline.stop()

    latencies = [float(item.get("end_to_end_ms", 0)) for item in samples]
    stage_names = ("pose", "track", "sam2", "action")
    stages = {
        stage: distribution([float(item.get("latency_ms", {}).get(stage, 0)) for item in samples])
        for stage in stage_names
    }
    observed_frames = max(0, last_sequence - first_sequence)
    sequences = [int(item.get("sequence", 0)) for item in samples]
    sequence_gaps = sum(max(0, current - previous - 1) for previous, current in pairwise(sequences))
    total_dropped = sum(dropped.values())
    drop_ratio = sequence_gaps / max(len(samples) + sequence_gaps, 1)
    effective_fps = len(samples) / measured_seconds if measured_seconds > 0 else 0.0
    latency_stats = distribution(latencies)
    target_fps = config.runtime.target_fps
    slo = {
        "throughput_at_least_90pct_target": effective_fps >= target_fps * 0.9,
        "p95_latency_below_200ms": float(latency_stats["p95"]) < 200,
        "drop_ratio_below_5pct": drop_ratio < 0.05,
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": config_path,
        "config": config.safe_dict(),
        "components": health["components"],
        "degradations": health["degradations"],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": package_versions(),
            "nvidia": nvidia_identity(),
        },
        "measurement": {
            "warmup_seconds": warmup_seconds,
            "duration_seconds": round(measured_seconds, 3),
            "samples": len(samples),
            "observed_source_frames": observed_frames,
            "target_fps": target_fps,
            "effective_fps": round(effective_fps, 3),
            "drop_ratio": round(drop_ratio, 6),
            "processed_sequence_gaps": sequence_gaps,
            "total_stage_drops": total_dropped,
            "dropped_frames": dropped,
        },
        "latency_ms": {"end_to_end": latency_stats, "stages": stages},
        "tracks": {
            "mean": round(statistics.mean(len(item.get("tracks", [])) for item in samples), 3)
            if samples
            else 0,
            "max": max((len(item.get("tracks", [])) for item in samples), default=0),
        },
        "gpu": {
            "samples": len(gpu_samples),
            "utilization_pct": distribution(
                [item["utilization_ratio"] * 100 for item in gpu_samples]
            ),
            "memory_used_mib": distribution(
                [item["memory_used_bytes"] / 2**20 for item in gpu_samples]
            ),
            "temperature_celsius": distribution(
                [item["temperature_celsius"] for item in gpu_samples]
            ),
            "power_watts": distribution([item["power_watts"] for item in gpu_samples]),
        },
        "slo": {**slo, "passed": all(slo.values())},
    }


def markdown_report(result: dict[str, Any]) -> str:
    measurement = result["measurement"]
    latency = result["latency_ms"]["end_to_end"]
    gpu = result["gpu"]
    rows = [
        "# Sentinel Vision benchmark",
        "",
        f"Generated: `{result['generated_at']}`  ",
        f"Config: `{result['config_path']}`",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Effective throughput | {measurement['effective_fps']} FPS |",
        f"| Frame drop ratio | {measurement['drop_ratio']:.2%} |",
        f"| E2E latency p50 | {latency['p50']} ms |",
        f"| E2E latency p95 | {latency['p95']} ms |",
        f"| E2E latency p99 | {latency['p99']} ms |",
        f"| Peak GPU memory | {gpu['memory_used_mib']['max']} MiB |",
        f"| Peak GPU utilization | {gpu['utilization_pct']['max']}% |",
        f"| SLO result | {'PASS' if result['slo']['passed'] else 'FAIL'} |",
        "",
        "## Active components",
        "",
    ]
    rows.extend(f"- **{name}:** `{value}`" for name, value in result["components"].items())
    if result["degradations"]:
        rows.extend(["", "## Declared degradations", ""])
        rows.extend(f"- {item}" for item in result["degradations"])
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--warmup-seconds", type=float, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = run_async(benchmark(args.config, args.seconds, args.warmup_seconds))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(f"artifacts/benchmarks/pipeline-{timestamp}.json")
    markdown = args.markdown or output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"json={output}\nmarkdown={markdown}")


if __name__ == "__main__":
    main()
