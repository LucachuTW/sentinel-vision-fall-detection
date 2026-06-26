from __future__ import annotations

from pathlib import Path

from sentinel_vision.config import SourceConfig


def _quoted(value: str) -> str:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError("GStreamer property contains a forbidden control character")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_rtsp_pipeline(config: SourceConfig) -> str:
    if config.kind != "rtsp":
        raise ValueError("RTSP pipeline requires source.kind=rtsp")
    codec = config.codec if config.codec != "auto" else "h264"
    depay = "rtph264depay" if codec == "h264" else "rtph265depay"
    parser = "h264parse" if codec == "h264" else "h265parse"
    decoder = f"nv{codec}dec" if config.hardware_decode else "decodebin"

    stages = [
        f"rtspsrc location={_quoted(config.uri)} latency={config.latency_ms} protocols=tcp",
        depay,
        parser,
        decoder,
    ]
    if config.hardware_decode:
        # OpenCV cannot consume GstCudaMemory directly. The explicit download is the
        # compatibility boundary; DeepStream deployment keeps NVMM through inference.
        stages.extend(["cudaconvertscale", "cudadownload"])
    stages.extend(
        [
            "videoconvert",
            f"video/x-raw,format=BGR,width={config.width},height={config.height}",
            "appsink sync=false max-buffers=1 drop=true",
        ]
    )
    return " ! ".join(stages)


def build_file_pipeline(config: SourceConfig) -> str:
    path = Path(config.uri).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"video file not found: {path}")
    return " ! ".join(
        [
            f"filesrc location={_quoted(str(path))}",
            "decodebin",
            "videoconvert",
            f"videoscale ! video/x-raw,format=BGR,width={config.width},height={config.height}",
            "appsink sync=false max-buffers=1 drop=true",
        ]
    )
