import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "prepare_urfd_dataset.py"
_spec = importlib.util.spec_from_file_location("prepare_urfd_dataset", _PATH)
assert _spec is not None and _spec.loader is not None
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def test_urfd_labels_map_to_three_classes() -> None:
    assert prep.map_urfd_label(-1) == 0  # upright
    assert prep.map_urfd_label(0) == 1  # falling
    assert prep.map_urfd_label(1) == 2  # fallen


def test_windows_are_labelled_by_their_last_frame() -> None:
    skeletons = [np.full((17, 3), i, dtype=np.float32) for i in range(10)]
    labels = [0, 0, 1, 1, 2, 2, 2, 0, 0, 0]
    windows = prep.window_labeled_frames(skeletons, labels, seq_length=4, stride=2)
    assert [label for _, label in windows] == [labels[3], labels[5], labels[7], labels[9]]
    assert all(window.shape == (4, 17, 3) for window, _ in windows)


def test_split_by_sequence_is_disjoint_and_covers_everything() -> None:
    sequences = [f"fall-{i:02d}" for i in range(1, 6)]
    train, val = prep.split_by_sequence(sequences, val_fraction=0.4, seed=3)
    assert len(val) == 2
    assert train.isdisjoint(val)  # no sequence (and thus no window) leaks across the split
    assert train | val == set(sequences)
