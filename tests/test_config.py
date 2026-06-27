from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel_vision.config import AppConfig, PoseConfig, load_config, redact_uri


@pytest.mark.parametrize(
    "path",
    [
        "config/demo.yaml",
        "config/benchmark-gpu.yaml",
        "config/local-gpu.yaml",
        "config/production.yaml",
    ],
)
def test_checked_in_configs_are_valid(path: str) -> None:
    assert load_config(path).project_name


def test_pose_backend_rejects_non_yolo26_artifact() -> None:
    with pytest.raises(ValidationError, match="YOLO26"):
        PoseConfig(backend="ultralytics", model="legacy-pose.pt")


def test_int8_requires_calibrated_engine() -> None:
    with pytest.raises(ValidationError, match="engine"):
        PoseConfig(backend="ultralytics", model="yolo26n-pose.pt", precision="int8")


def test_production_rejects_fallbacks() -> None:
    with pytest.raises(ValidationError, match="fallback"):
        AppConfig(environment="production")


def test_uri_redaction_preserves_location_without_credentials() -> None:
    assert redact_uri("rtsp://alice:secret@camera.local:8554/live?quality=1") == (
        "rtsp://***:***@camera.local:8554/live?quality=1"
    )
    assert redact_uri("synthetic://floor") == "synthetic://floor"


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_default_api_host_is_loopback() -> None:
    assert AppConfig().api.host == "127.0.0.1"
