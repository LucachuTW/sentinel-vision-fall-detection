from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import cv2
import numpy as np

from sentinel_vision.config import SourceConfig, redact_uri
from sentinel_vision.domain import FramePacket
from sentinel_vision.sources.base import VideoSource
from sentinel_vision.sources.gstreamer import build_file_pipeline, build_rtsp_pipeline

LOGGER = logging.getLogger(__name__)


def opencv_has_gstreamer() -> bool:
    return any(
        "GStreamer:" in line and "YES" in line for line in cv2.getBuildInformation().splitlines()
    )


class OpenCvSource(VideoSource):
    """GStreamer/OpenCV source with low-latency appsink backpressure."""

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._sequence = 0
        self._ready = False
        self._has_gstreamer = opencv_has_gstreamer()
        self._next_frame_at = 0.0

    @property
    def backend_name(self) -> str:
        if self.config.kind == "rtsp":
            if not self._has_gstreamer:
                return "opencv-ffmpeg-cpu-fallback"
            decode = "nvdec" if self.config.hardware_decode else "software"
            return f"gstreamer-{decode}-cpu-handoff"
        return "opencv-gstreamer" if self._has_gstreamer else "opencv-ffmpeg"

    @property
    def ready(self) -> bool:
        return self._ready

    def open(self) -> None:
        self.close()
        if self.config.kind == "rtsp":
            LOGGER.info("opening RTSP source %s", redact_uri(self.config.uri))
            if self._has_gstreamer:
                pipeline = build_rtsp_pipeline(self.config)
                self._capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                LOGGER.warning(
                    "OpenCV has no GStreamer support; using FFmpeg CPU fallback for %s",
                    redact_uri(self.config.uri),
                )
                self._capture = self._ffmpeg_capture(self.config.uri)
        elif self.config.kind == "file":
            LOGGER.info("opening video file %s", self.config.uri)
            if self._has_gstreamer:
                pipeline = build_file_pipeline(self.config)
                self._capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                self._capture = self._ffmpeg_capture(self.config.uri)
        elif self.config.kind == "camera":
            camera_id = int(self.config.uri) if self.config.uri.isdigit() else self.config.uri
            self._capture = cv2.VideoCapture(camera_id)
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            raise ValueError(f"unsupported source kind: {self.config.kind}")
        if self._capture is None or not self._capture.isOpened():
            self.close()
            raise RuntimeError(f"could not open source {redact_uri(self.config.uri)}")
        self._ready = True

    @staticmethod
    def _ffmpeg_capture(uri: str) -> cv2.VideoCapture:
        return cv2.VideoCapture(
            uri,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                2000,
            ],
        )

    def read(self) -> FramePacket | None:
        if self._capture is None:
            raise RuntimeError("source is not open")
        # A file plays as fast as it decodes; pace it to its real fps so it behaves like a live
        # camera and the temporal model sees continuous windows instead of dropped frames.
        if self.config.kind == "file":
            now = time.monotonic()
            if self._next_frame_at > now:
                time.sleep(self._next_frame_at - now)
            self._next_frame_at = max(self._next_frame_at + 1.0 / self.config.fps, time.monotonic())
        ok, image = self._capture.read()
        if not ok:
            if self.config.kind == "file" and self.config.loop:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, image = self._capture.read()
            if not ok:
                self._ready = False
                return None
        self._sequence += 1
        return FramePacket(
            source_id=self.config.source_id,
            sequence=self._sequence,
            captured_at=datetime.now(UTC),
            monotonic_ns=time.monotonic_ns(),
            image=np.asarray(image, dtype=np.uint8),
        )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._ready = False
