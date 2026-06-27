from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Annotated

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from sentinel_vision.api import create_app
from sentinel_vision.async_runtime import run_async
from sentinel_vision.config import load_config
from sentinel_vision.observability import configure_logging
from sentinel_vision.pipeline import VideoInferencePipeline
from sentinel_vision.sources.opencv import opencv_has_gstreamer

app = typer.Typer(no_args_is_help=True, help="Sentinel Vision local inference platform")
console = Console()


@app.command()
def run(
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("config/demo.yaml"),
) -> None:
    """Run the inference API and dashboard."""
    config = load_config(config_path)
    configure_logging(config.runtime.log_level)
    uvicorn.run(
        create_app(config),
        host=config.api.host,
        port=config.api.port,
        log_config=None,
    )


@app.command()
def demo(
    seconds: Annotated[float, typer.Option(min=1, max=3600)] = 8.0,
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("config/demo.yaml"),
) -> None:
    """Exercise the complete pipeline headlessly and print its final state."""
    config = load_config(config_path)
    configure_logging(config.runtime.log_level)

    async def execute() -> None:
        pipeline = VideoInferencePipeline(config)
        await pipeline.start()
        await asyncio.sleep(seconds)
        snapshot = pipeline.state.snapshot()
        health = pipeline.state.health()
        await pipeline.stop()
        console.print_json(json.dumps({"health": health, "state": snapshot}))

    run_async(execute())


@app.command()
def doctor(
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("config/demo.yaml"),
) -> None:
    """Audit runtime capabilities without starting inference."""
    config = load_config(config_path)
    rows: list[tuple[str, str, str]] = []
    for module in ("cv2", "torch", "ultralytics", "supervision"):
        present = importlib.util.find_spec(module) is not None
        rows.append((f"python:{module}", "ready" if present else "missing", ""))
    rows.append(
        (
            "opencv:gstreamer",
            "ready" if opencv_has_gstreamer() else "unavailable",
            "FFmpeg fallback is used when unavailable",
        )
    )
    rows.append(
        (
            "model:pose",
            "ready"
            if Path(config.pose.engine_path or config.pose.model).is_file()
            else "on-demand",
            config.pose.engine_path or config.pose.model,
        )
    )
    rows.append(
        (
            "model:sam2",
            "ready" if Path(config.segmentation.model).is_file() else "on-demand",
            config.segmentation.model,
        )
    )
    rows.append(
        (
            "model:temporal",
            "ready" if Path(config.temporal.checkpoint).is_file() else "missing",
            config.temporal.checkpoint,
        )
    )
    rows.append(
        (
            "nvidia",
            *_command_status(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        )
    )
    rows.append(("gstreamer:nvdec", *_command_status(["gst-inspect-1.0", "nvh264dec"])))
    if config.ollama.enabled:
        try:
            response = httpx.get(f"{config.ollama.endpoint}/api/tags", timeout=2)
            response.raise_for_status()
            names = [item["name"] for item in response.json().get("models", [])]
            rows.append(
                (
                    "ollama",
                    "ready" if config.ollama.model in names else "model-missing",
                    ", ".join(names),
                )
            )
        except Exception as exc:
            rows.append(("ollama", "unavailable", str(exc)))

    table = Table("Capability", "Status", "Details", title="Sentinel Vision Doctor")
    for row in rows:
        table.add_row(*row)
    console.print(table)


def _command_status(command: list[str]) -> tuple[str, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=4, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return "missing", str(exc)
    details = (result.stdout or result.stderr).strip().splitlines()
    return ("ready" if result.returncode == 0 else "unavailable", details[0] if details else "")
