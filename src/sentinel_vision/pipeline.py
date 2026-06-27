from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sentinel_vision.config import AppConfig
from sentinel_vision.domain import FramePacket, PosePacket, ProcessedFrame, TrackedPacket
from sentinel_vision.events import ActionEventDetector
from sentinel_vision.models.narrative import LocalNarrativeService
from sentinel_vision.models.pose import PoseEstimator, create_pose_estimator
from sentinel_vision.models.segmentation import Segmenter, create_segmenter
from sentinel_vision.models.temporal import ActionClassifier, create_action_classifier
from sentinel_vision.observability import PipelineMetrics
from sentinel_vision.queues import LatestValueQueue
from sentinel_vision.render import render_frame
from sentinel_vision.sources import VideoSource, create_source
from sentinel_vision.state import StateStore
from sentinel_vision.tracking import Tracker, create_tracker

LOGGER = logging.getLogger(__name__)


class VideoInferencePipeline:
    def __init__(
        self,
        config: AppConfig,
        *,
        source: VideoSource | None = None,
        pose: PoseEstimator | None = None,
        tracker: Tracker | None = None,
        segmenter: Segmenter | None = None,
        classifier: ActionClassifier | None = None,
        metrics: PipelineMetrics | None = None,
        state: StateStore | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics or PipelineMetrics()
        self.state = state or StateStore(config)
        self.source = source or create_source(config.source)
        self.pose = pose or create_pose_estimator(config.pose)
        self.tracker = tracker or create_tracker(config.tracker)
        self.segmenter = segmenter or create_segmenter(config.segmentation)
        self.classifier = classifier or create_action_classifier(config.temporal)
        self.event_detector = ActionEventDetector(config.temporal)
        self.narratives = LocalNarrativeService(config.ollama, self.state.add_narrative)
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._warmed_up = False
        self._executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="sentinel-stage")

        size = config.runtime.queue_size
        self._pose_queue: LatestValueQueue[FramePacket] = LatestValueQueue(
            size, on_drop=lambda: self.metrics.drop("before_pose")
        )
        self._track_queue: LatestValueQueue[PosePacket] = LatestValueQueue(
            size, on_drop=lambda: self.metrics.drop("before_tracking")
        )
        self._enrich_queue: LatestValueQueue[TrackedPacket] = LatestValueQueue(
            size, on_drop=lambda: self.metrics.drop("before_enrichment")
        )
        self.metrics.configure_queues(
            {"before_pose": size, "before_tracking": size, "before_enrichment": size}
        )
        self._update_queue_metrics()

    @property
    def running(self) -> bool:
        return bool(self._tasks) and not self._stop.is_set()

    @property
    def dropped_frames(self) -> dict[str, int]:
        return {
            "before_pose": self._pose_queue.dropped,
            "before_tracking": self._track_queue.dropped,
            "before_enrichment": self._enrich_queue.dropped,
        }

    async def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        components = {
            "source": self.source.backend_name,
            "pose": self.pose.backend_name,
            "tracker": self.tracker.backend_name,
            "segmentation": self.segmenter.backend_name,
            "temporal": self.classifier.backend_name,
            "narrative": self.narratives.backend_name,
        }
        self.state.start(components, self._degradations())
        self.metrics.source_connected.labels(source=self.config.source.source_id).set(0)
        loops: list[tuple[str, Coroutine[Any, Any, None]]] = [
            ("ingest", self._ingest_loop()),
            ("pose", self._pose_loop()),
            ("tracking", self._tracking_loop()),
            ("enrichment", self._enrichment_loop()),
        ]
        if self.config.ollama.enabled:
            loops.append(("narrative", self.narratives.run()))
        self._tasks = [
            asyncio.create_task(self._guard(name, loop), name=f"sentinel-{name}")
            for name, loop in loops
        ]
        LOGGER.info("pipeline started with components=%s", components)

    async def stop(self) -> None:
        if not self._tasks:
            return
        self._stop.set()
        self.narratives.stop()
        # Close synchronously before cancelling the task that may be blocked in read().
        # OpenCV release is the unblock mechanism; queueing close behind that read can deadlock.
        self.source.close()
        self.metrics.source_connected.labels(source=self.config.source.source_id).set(0)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.state.stop()
        self.metrics.pipeline_ready.set(0)
        LOGGER.info("pipeline stopped")

    async def _guard(self, name: str, loop: Coroutine[Any, Any, None]) -> None:
        try:
            await loop
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = f"{name} stage failed: {type(exc).__name__}: {exc}"
            LOGGER.exception(message)
            self.metrics.error(name, type(exc).__name__)
            self.state.set_error(message)
            self.metrics.pipeline_ready.set(0)
            self._stop.set()

    async def _ingest_loop(self) -> None:
        await self._offload(self.source.open)
        self.metrics.source_connected.labels(source=self.config.source.source_id).set(1)
        reconnect_delay = 0.25
        while not self._stop.is_set():
            frame = await self._offload(self.source.read)
            if frame is None:
                if self.config.source.kind == "file" and not self.config.source.loop:
                    self._stop.set()
                    return
                self.metrics.source_connected.labels(source=self.config.source.source_id).set(0)
                self.metrics.source_reconnects.labels(source=self.config.source.source_id).inc()
                await self._offload(self.source.close)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(5.0, reconnect_delay * 2)
                await self._offload(self.source.open)
                self.metrics.source_connected.labels(source=self.config.source.source_id).set(1)
                continue
            reconnect_delay = 0.25
            self.metrics.frames_ingested.labels(source=frame.source_id).inc()
            self._pose_queue.put_latest(frame)
            self._update_queue_metrics()

    async def _pose_loop(self) -> None:
        while not self._stop.is_set():
            frame = await self._get(self._pose_queue)
            if frame is None:
                continue
            started = time.perf_counter()
            if not self._warmed_up:
                await self._offload(
                    self.pose.warmup, frame.image, self.config.runtime.warmup_iterations
                )
                self._warmed_up = True
            detections = await self._offload(self.pose.infer, frame.image)
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.metrics.stage_latency.labels(stage="pose").observe(elapsed_ms / 1000)
            self.metrics.success("pose")
            self.metrics.detections.inc(len(detections))
            self._track_queue.put_latest(PosePacket(frame, detections, elapsed_ms))
            self._pose_queue.task_done()
            self._update_queue_metrics()

    async def _tracking_loop(self) -> None:
        while not self._stop.is_set():
            packet = await self._get(self._track_queue)
            if packet is None:
                continue
            started = time.perf_counter()
            detections = self.tracker.update(packet.detections, packet.frame.sequence)
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.metrics.stage_latency.labels(stage="tracking").observe(elapsed_ms / 1000)
            self.metrics.success("tracking")
            self._enrich_queue.put_latest(
                TrackedPacket(packet.frame, detections, packet.inference_ms, elapsed_ms)
            )
            self._track_queue.task_done()
            self._update_queue_metrics()

    async def _enrichment_loop(self) -> None:
        while not self._stop.is_set():
            packet = await self._get(self._enrich_queue)
            if packet is None:
                continue
            segmentation_started = time.perf_counter()
            segmentations = await self._offload(
                self.segmenter.segment_on_demand,
                packet.frame.image,
                packet.detections,
                packet.frame.sequence,
            )
            segmentation_ms = (time.perf_counter() - segmentation_started) * 1000
            self.metrics.stage_latency.labels(stage="segmentation").observe(segmentation_ms / 1000)
            self.metrics.success("segmentation")
            self.metrics.segmentations.labels(
                outcome="produced" if segmentations else "skipped_or_empty"
            ).inc(len(segmentations) if segmentations else 1)

            temporal_started = time.perf_counter()
            actions = await self._offload(self.classifier.classify, packet.detections)
            temporal_ms = (time.perf_counter() - temporal_started) * 1000
            self.metrics.stage_latency.labels(stage="temporal").observe(temporal_ms / 1000)
            self.metrics.success("temporal")
            for action in actions:
                self.metrics.actions.labels(label=action.label).inc()
            stages = {
                "pose": packet.pose_ms,
                "track": packet.tracking_ms,
                "sam2": segmentation_ms,
                "action": temporal_ms,
            }
            jpeg = await self._offload(
                render_frame,
                packet.frame.image,
                packet.detections,
                segmentations,
                actions,
                stages,
                self.config.runtime.jpeg_quality,
            )
            end_to_end_ms = (time.monotonic_ns() - packet.frame.monotonic_ns) / 1e6
            processed = ProcessedFrame(
                frame=packet.frame,
                detections=packet.detections,
                segmentations=segmentations,
                actions=actions,
                stage_latency_ms=stages,
                end_to_end_ms=end_to_end_ms,
                rendered_jpeg=jpeg,
            )
            events = self.event_detector.update(packet.frame, actions)
            self.state.add_events(events)
            for event in events:
                self.metrics.events.labels(kind=event.kind, severity=event.severity).inc()
                self.narratives.submit(event)
            await self.state.update_frame(processed)
            self.metrics.frames_processed.labels(source=packet.frame.source_id).inc()
            self.metrics.active_tracks.set(len(packet.detections))
            self.metrics.end_to_end_latency.observe(end_to_end_ms / 1000)
            self.metrics.frame_age.set(end_to_end_ms / 1000)
            self.metrics.sample_gpu()
            self.metrics.pipeline_ready.set(1)
            self._enrich_queue.task_done()
            self._update_queue_metrics()

    async def _get(self, queue: asyncio.Queue[Any]) -> Any | None:
        try:
            return await asyncio.wait_for(queue.get(), timeout=0.25)
        except TimeoutError:
            return None

    async def _offload(self, function: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(function, *args),
        )

    def _update_queue_metrics(self) -> None:
        self.metrics.set_queue_depths(
            {
                "before_pose": self._pose_queue.qsize(),
                "before_tracking": self._track_queue.qsize(),
                "before_enrichment": self._enrich_queue.qsize(),
            }
        )

    def _degradations(self) -> list[str]:
        output: list[str] = []
        if self.config.pose.backend == "synthetic":
            output.append("pose uses the deterministic demo backend, not YOLO26 inference")
        if self.config.tracker.backend == "iou":
            output.append("tracking uses IoU fallback, not ByteTrack")
        if self.config.segmentation.backend == "contour":
            output.append("segmentation uses contour fallback, not SAM 2")
        if self.config.temporal.backend == "heuristic":
            output.append("actions use auditable kinematics until a trained checkpoint is supplied")
        if self.config.source.kind == "rtsp" and self.config.source.hardware_decode:
            if "ffmpeg" in self.source.backend_name:
                output.append("OpenCV lacks GStreamer support; RTSP uses the FFmpeg CPU fallback")
            else:
                output.append(
                    "OpenCV appsink downloads decoded frames to CPU; select the DeepStream deployment for GPU zero-copy"
                )
        return output
