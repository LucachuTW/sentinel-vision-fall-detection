import asyncio

from sentinel_vision.config import AppConfig, RuntimeConfig, SourceConfig
from sentinel_vision.pipeline import VideoInferencePipeline
from sentinel_vision.sources.synthetic import SyntheticSource


async def test_pipeline_processes_frames_end_to_end() -> None:
    config = AppConfig(
        source=SourceConfig(width=640, height=360, fps=120),
        runtime=RuntimeConfig(queue_size=2, warmup_iterations=0),
    )
    pipeline = VideoInferencePipeline(
        config,
        source=SyntheticSource(config.source, realtime=False),
    )
    await pipeline.start()
    try:
        for _ in range(50):
            if pipeline.state.latest_sequence() > 0:
                break
            await asyncio.sleep(0.05)
        snapshot = pipeline.state.snapshot()
        assert snapshot["sequence"] > 0
        assert snapshot["tracks"]
        assert snapshot["actions"]
        assert pipeline.state.jpeg().startswith(b"\xff\xd8")
        assert pipeline.state.health()["ready"] is True
    finally:
        await pipeline.stop()
