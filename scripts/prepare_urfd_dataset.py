#!/usr/bin/env python3
"""Build a real skeleton-action dataset from the UR Fall Detection Dataset (URFD).

Downloads URFD cam0 RGB sequences, runs YOLO26-pose to extract COCO-17 skeletons, windows them,
and labels each window from the URFD per-frame annotation (-1 upright, 0 falling, 1 fallen). Splits
by *sequence* (no window leakage) into train/val NPZs and encodes one held-out fall to an MP4 for the
live demo. This is the real-data backbone of the fall/incident classifier.

URFD: M. Kepski, B. Kwolek, University of Rzeszów — http://fenix.ur.edu.pl/~mkepski/ds/uf.html
Research use; the dataset is downloaded locally and never redistributed by this repository.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import cv2
import httpx
import numpy as np

from sentinel_vision.config import PoseConfig
from sentinel_vision.models.pose import create_pose_estimator
from sentinel_vision.models.temporal import normalize_skeleton

BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"
LABELS = ["upright", "falling", "fallen"]
_URFD_TO_CLASS = {-1: 0, 0: 1, 1: 2}  # not-lying / transition / lying


def map_urfd_label(raw: int) -> int:
    """URFD per-frame annotation → class index in LABELS."""
    return _URFD_TO_CLASS[raw]


def window_labeled_frames(
    skeletons: list[np.ndarray], labels: list[int], seq_length: int, stride: int
) -> list[tuple[np.ndarray, int]]:
    """Sliding windows over consecutive frames; each window is labelled by its last frame."""
    out: list[tuple[np.ndarray, int]] = []
    for end in range(seq_length, len(skeletons) + 1, stride):
        window = np.stack(skeletons[end - seq_length : end]).astype(np.float32)
        out.append((window, labels[end - 1]))
    return out


def split_by_sequence(
    sequences: list[str], val_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Assign whole sequences to train/val so no window leaks across the split."""
    rng = np.random.default_rng(seed)
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    val = set(shuffled[:val_count])
    return set(shuffled[val_count:]), val


def _download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)
    temporary.replace(destination)


def _load_labels(csv_path: Path) -> dict[tuple[str, int], int]:
    labels: dict[tuple[str, int], int] = {}
    for line in csv_path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 3:
            continue
        labels[(fields[0], int(fields[1]))] = int(fields[2])
    return labels


def _sequence_frames(workdir: Path, sequence: str) -> list[Path]:
    folder = workdir / "frames" / f"{sequence}-cam0-rgb"
    return sorted(folder.glob(f"{sequence}-cam0-rgb-*.png"))


def _ensure_sequence(workdir: Path, sequence: str) -> None:
    folder = workdir / "frames" / f"{sequence}-cam0-rgb"
    if folder.is_dir() and any(folder.glob("*.png")):
        return
    archive = workdir / "zips" / f"{sequence}.zip"
    _download(f"{BASE_URL}/{sequence}-cam0-rgb.zip", archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(workdir / "frames")


def _frame_index(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[1])


def _extract_sequence(
    workdir: Path, sequence: str, labels: dict[tuple[str, int], int], pose: object
) -> tuple[list[np.ndarray], list[int]]:
    skeletons: list[np.ndarray] = []
    frame_labels: list[int] = []
    for png in _sequence_frames(workdir, sequence):
        key = (sequence, _frame_index(png))
        if key not in labels:
            continue
        image = cv2.imread(str(png))
        if image is None:
            continue
        detections = pose.infer(image)  # type: ignore[attr-defined]
        if not detections:
            continue
        best = max(detections, key=lambda item: item.confidence)
        skeletons.append(normalize_skeleton(best))
        frame_labels.append(map_urfd_label(labels[key]))
    return skeletons, frame_labels


def _encode_demo_video(workdir: Path, sequence: str, output: Path, fps: int) -> None:
    frames = _sequence_frames(workdir, sequence)
    if not frames:
        return
    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for png in frames:
        image = cv2.imread(str(png))
        if image is not None:
            writer.write(image)
    writer.release()


def _save(path: Path, samples: list[tuple[np.ndarray, int]]) -> None:
    x = np.stack([item[0] for item in samples]).astype(np.float32)
    y = np.asarray([item[1] for item in samples], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y)
    counts = {LABELS[i]: int((y == i).sum()) for i in range(len(LABELS))}
    print(f"saved={path} windows={len(y)} class_counts={counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--falls", type=int, default=30)
    parser.add_argument("--adls", type=int, default=15)
    parser.add_argument("--seq-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workdir", type=Path, default=Path("artifacts/urfd"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--pose-model", default="models/yolo26n-pose.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--demo-fps", type=int, default=30)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    falls_csv = args.workdir / "falls.csv"
    adls_csv = args.workdir / "adls.csv"
    _download(f"{BASE_URL}/urfall-cam0-falls.csv", falls_csv)
    _download(f"{BASE_URL}/urfall-cam0-adls.csv", adls_csv)
    labels = _load_labels(falls_csv) | _load_labels(adls_csv)

    sequences = [f"fall-{i:02d}" for i in range(1, args.falls + 1)]
    sequences += [f"adl-{i:02d}" for i in range(1, args.adls + 1)]

    pose = create_pose_estimator(
        PoseConfig(
            backend="ultralytics",
            model=args.pose_model,
            device=args.device,
            precision="fp16" if args.device.startswith("cuda") else "fp32",
            confidence=args.confidence,
        )
    )

    per_sequence: dict[str, list[tuple[np.ndarray, int]]] = {}
    for sequence in sequences:
        _ensure_sequence(args.workdir, sequence)
        skeletons, frame_labels = _extract_sequence(args.workdir, sequence, labels, pose)
        windows = window_labeled_frames(skeletons, frame_labels, args.seq_length, args.stride)
        per_sequence[sequence] = windows
        print(f"{sequence}: frames={len(skeletons)} windows={len(windows)}")

    fall_sequences = [s for s in sequences if s.startswith("fall")]
    train_seqs, val_seqs = split_by_sequence(fall_sequences, args.val_fraction, args.seed)
    # ADL sequences (no falls) always train, so val stays a clean held-out fall set.
    train_seqs |= {s for s in sequences if s.startswith("adl")}

    train = [item for seq in train_seqs for item in per_sequence[seq]]
    val = [item for seq in val_seqs for item in per_sequence[seq]]
    _save(args.out_dir / "urfd-train.npz", train)
    _save(args.out_dir / "urfd-val.npz", val)
    print(f"train_sequences={sorted(train_seqs)}")
    print(f"val_sequences={sorted(val_seqs)}")

    demo_sequence = sorted(val_seqs)[0]
    _encode_demo_video(args.workdir, demo_sequence, args.out_dir / "demo-fall.mp4", args.demo_fps)
    print(f"demo_video={args.out_dir / 'demo-fall.mp4'} (sequence={demo_sequence})")


if __name__ == "__main__":
    main()
