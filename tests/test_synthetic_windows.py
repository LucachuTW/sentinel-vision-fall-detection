import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "make_synthetic_windows.py"
_spec = importlib.util.spec_from_file_location("make_synthetic_windows", _PATH)
assert _spec is not None and _spec.loader is not None
make_synthetic_windows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_synthetic_windows)


def test_synthetic_windows_have_expected_shape_and_separable_classes() -> None:
    labels = make_synthetic_windows.DEFAULT_LABELS
    x, y = make_synthetic_windows.build_dataset(
        samples_per_class=8, length=24, labels=labels, seed=3
    )
    assert x.shape == (8 * len(labels), 24, 17, 3)
    assert set(y.tolist()) == set(range(len(labels)))
    assert (x[..., :2] >= -1.0).all() and (x[..., :2] <= 1.0).all()
    assert (x[..., 2] >= 0.0).all() and (x[..., 2] <= 1.0).all()
    # The training script needs the classes to be distinguishable.
    assert make_synthetic_windows.class_separation(x, y) > 0.5
