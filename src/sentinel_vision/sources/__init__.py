from sentinel_vision.config import SourceConfig
from sentinel_vision.sources.base import VideoSource
from sentinel_vision.sources.image import ImageLoopSource
from sentinel_vision.sources.opencv import OpenCvSource
from sentinel_vision.sources.synthetic import SyntheticSource


def create_source(config: SourceConfig) -> VideoSource:
    if config.kind == "synthetic":
        return SyntheticSource(config)
    if config.kind == "image":
        return ImageLoopSource(config)
    return OpenCvSource(config)


__all__ = ["VideoSource", "create_source"]
