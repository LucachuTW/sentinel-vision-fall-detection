"""End-to-end smoke for the real ML temporal path. Skipped unless torch is installed."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from sentinel_vision.config import TemporalConfig  # noqa: E402
from sentinel_vision.domain import BoundingBox, Keypoint, PoseDetection  # noqa: E402
from sentinel_vision.models.temporal import (  # noqa: E402
    build_temporal_model,
    create_action_classifier,
)

_GEN = Path(__file__).resolve().parent.parent / "scripts" / "make_synthetic_windows.py"
_spec = importlib.util.spec_from_file_location("make_synthetic_windows", _GEN)
assert _spec is not None and _spec.loader is not None
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_transformer_trains_on_synthetic_data_and_classifies(tmp_path: Path) -> None:
    torch.manual_seed(0)
    labels = gen.DEFAULT_LABELS
    length = 16
    x, y = gen.build_dataset(samples_per_class=12, length=length, labels=labels, seed=1)
    features = torch.from_numpy(x.reshape(len(x), length, -1))
    targets = torch.from_numpy(y)

    model = build_temporal_model(
        nn, input_dim=51, class_count=len(labels), max_sequence_length=length
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        criterion(model(features), targets).backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        accuracy = (model(features).argmax(dim=-1) == targets).float().mean().item()
    assert accuracy > 0.7  # the synthetic classes are separable

    checkpoint = tmp_path / "temporal.pt"
    torch.save({"state_dict": model.state_dict(), "labels": labels}, checkpoint)

    # Drive the production classifier: checkpoint load + label validation + masked inference.
    classifier = create_action_classifier(
        TemporalConfig(
            backend="transformer",
            checkpoint=str(checkpoint),
            labels=labels,
            sequence_length=length,
            device="cpu",
        )
    )
    box = BoundingBox(100, 50, 140, 150)
    points = tuple(Keypoint(120, 50 + index * 5, 0.9) for index in range(17))
    prediction = classifier.classify((PoseDetection(box, 0.9, points, track_id=1),))[0]
    assert prediction.label in labels
    assert prediction.model_ready is True
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-4
