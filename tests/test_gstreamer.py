import numpy as np
import pytest

from sentinel_vision.config import SourceConfig
from sentinel_vision.sources.gstreamer import build_rtsp_pipeline
from sentinel_vision.sources.opencv import OpenCvSource


class _FakeCapture:
    def __init__(self, *_: object, **__: object) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, np.zeros((240, 320, 3), dtype=np.uint8)

    def set(self, *_: object) -> bool:
        return True

    def release(self) -> None:
        self.released = True


def test_hardware_rtsp_pipeline_has_explicit_cpu_boundary() -> None:
    config = SourceConfig(
        kind="rtsp",
        uri="rtsp://camera/live",
        codec="h264",
        hardware_decode=True,
        width=1920,
        height=1080,
    )
    pipeline = build_rtsp_pipeline(config)
    assert "nvh264dec" in pipeline
    assert "cudadownload" in pipeline
    assert "max-buffers=1 drop=true" in pipeline


def test_rtsp_pipeline_rejects_control_characters() -> None:
    config = SourceConfig(kind="rtsp", uri="rtsp://camera/live\nmalicious")
    with pytest.raises(ValueError, match="control"):
        build_rtsp_pipeline(config)


def test_source_reports_ffmpeg_when_opencv_lacks_gstreamer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sentinel_vision.sources.opencv.opencv_has_gstreamer", lambda: False)
    source = OpenCvSource(SourceConfig(kind="rtsp", uri="rtsp://camera/live"))
    assert source.backend_name == "opencv-ffmpeg-cpu-fallback"


def test_ffmpeg_fallback_opens_reads_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sentinel_vision.sources.opencv.opencv_has_gstreamer", lambda: False)
    monkeypatch.setattr("sentinel_vision.sources.opencv.cv2.VideoCapture", _FakeCapture)
    source = OpenCvSource(SourceConfig(kind="rtsp", uri="rtsp://alice:secret@camera/live"))
    source.open()
    frame = source.read()
    assert frame is not None
    assert frame.image.shape == (240, 320, 3)
    assert source.ready is True
    source.close()
    assert source.ready is False
