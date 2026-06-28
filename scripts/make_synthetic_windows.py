#!/usr/bin/env python3
"""Generate a separable synthetic skeleton-window dataset for the temporal transformer.

Windows live in the same normalized space as
``sentinel_vision.models.temporal.normalize_skeleton`` (x/y in [-0.5, 0.5] plus a
confidence channel), so the flagship transformer path becomes runnable end-to-end
without a labelled site corpus. This is a smoke/demo dataset, not real behavioural data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# COCO-17 upright template (matches models/pose.py), centered to [-0.5, 0.5].
_TEMPLATE = (
    np.asarray(
        [
            (0.50, 0.08),
            (0.46, 0.07),
            (0.54, 0.07),
            (0.42, 0.09),
            (0.58, 0.09),
            (0.35, 0.28),
            (0.65, 0.28),
            (0.27, 0.48),
            (0.73, 0.48),
            (0.22, 0.67),
            (0.78, 0.67),
            (0.42, 0.56),
            (0.58, 0.56),
            (0.39, 0.76),
            (0.61, 0.76),
            (0.36, 0.97),
            (0.64, 0.97),
        ],
        dtype=np.float32,
    )
    - 0.5
)

_HEAD = (0, 1, 2, 3, 4)
_WRISTS = (9, 10)
_LIMBS = (5, 6, 7, 8, 9, 10, 13, 14, 15, 16)

DEFAULT_LABELS = ["standing", "active", "feeding", "resting", "incident"]


def _window(label: str, length: int, rng: np.random.Generator) -> np.ndarray:
    """Build one class-conditional [T, 17, 3] normalized skeleton window."""
    base = np.repeat(_TEMPLATE[None, :, :], length, axis=0).copy()  # [T, 17, 2]
    phase = np.linspace(0.0, 2.0 * np.pi, length, dtype=np.float32)
    lying = np.stack([_TEMPLATE[:, 1], -_TEMPLATE[:, 0]], axis=-1)  # 90deg rotation

    if label == "standing":
        pass
    elif label == "active":
        base[:, _LIMBS, 0] += 0.06 * np.sin(phase)[:, None]
        base[:, _LIMBS, 1] += 0.04 * np.cos(phase)[:, None]
        base[:, :, 0] += (0.03 * phase / (2.0 * np.pi))[:, None]  # bodily drift
    elif label == "feeding":
        base[:, _HEAD, 1] += 0.18  # head lowered toward the floor
        base[:, _WRISTS, 1] -= 0.20  # wrists raised toward the head
        base[:, _WRISTS, 0] *= 0.5
    elif label == "resting":
        base[:] = lying  # lying down for the whole window
    elif label == "incident":
        half = length // 2
        for t in range(length):
            alpha = 0.0 if t < half else min(1.0, (t - half) / max(half // 3, 1))
            base[t] = (1.0 - alpha) * _TEMPLATE + alpha * lying  # sudden fall mid-window
    else:
        raise ValueError(f"unknown label: {label}")

    base += rng.normal(0.0, 0.01, base.shape).astype(np.float32)
    conf = np.clip(rng.normal(0.9, 0.03, (length, 17, 1)), 0.0, 1.0).astype(np.float32)
    return np.concatenate([base, conf], axis=-1)  # [T, 17, 3]


def build_dataset(
    samples_per_class: int, length: int, labels: list[str], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    targets: list[int] = []
    for index, label in enumerate(labels):
        for _ in range(samples_per_class):
            windows.append(_window(label, length, rng))
            targets.append(index)
    return np.stack(windows).astype(np.float32), np.asarray(targets, dtype=np.int64)


def class_separation(x: np.ndarray, y: np.ndarray) -> float:
    """Minimum L2 distance between per-class mean windows; >0 means separable."""
    means = np.stack(
        [x[y == c].reshape(int((y == c).sum()), -1).mean(axis=0) for c in np.unique(y)]
    )
    return min(
        float(np.linalg.norm(means[i] - means[j]))
        for i in range(len(means))
        for j in range(i + 1, len(means))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthetic-windows.npz"))
    parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    x, y = build_dataset(args.samples_per_class, args.sequence_length, args.labels, args.seed)
    separation = class_separation(x, y)
    assert separation > 0.5, (
        f"classes are not separable enough (min mean distance {separation:.3f})"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y)
    print(f"saved={args.output} windows={len(x)} shape={x.shape} class_separation={separation:.3f}")


if __name__ == "__main__":
    main()
