#!/usr/bin/env python3
"""Train the compact skeleton transformer from a labelled NPZ dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, required=True, help="NPZ with x [N,T,17,3] and y [N]"
    )
    parser.add_argument(
        "--val-dataset",
        type=Path,
        default=None,
        help="optional held-out NPZ; when given, --dataset is used fully for training "
        "(use this for an honest subject/sequence split)",
    )
    parser.add_argument("--output", type=Path, default=Path("models/temporal-transformer.pt"))
    parser.add_argument(
        "--labels", nargs="+", default=["standing", "active", "feeding", "resting", "incident"]
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from sentinel_vision.models.temporal import build_temporal_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(path)
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        if x.ndim != 4 or x.shape[2:] != (17, 3):
            raise SystemExit(f"expected x shape [N,T,17,3] in {path}, got {x.shape}")
        if y.shape != (x.shape[0],):
            raise SystemExit(f"expected y shape [{x.shape[0]}] in {path}, got {y.shape}")
        if y.min() < 0 or y.max() >= len(args.labels):
            raise SystemExit(f"class indexes in {path} fall outside --labels")
        return x, y

    x, y = load_npz(args.dataset)
    if args.val_dataset is not None:
        x_train, y_train = x, y
        x_val, y_val = load_npz(args.val_dataset)
        if x_val.shape[1] != x_train.shape[1]:
            raise SystemExit("train and validation windows have different sequence lengths")
    else:
        if x.shape[0] < 20:
            raise SystemExit("at least 20 labelled windows are required")
        indexes = np.random.permutation(len(x))
        validation_count = max(1, int(len(x) * args.validation_fraction))
        val_idx, train_idx = indexes[:validation_count], indexes[validation_count:]
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

    sequence_length = x_train.shape[1]
    train = TensorDataset(
        torch.from_numpy(x_train.reshape(len(x_train), sequence_length, -1)),
        torch.from_numpy(y_train),
    )
    validation = TensorDataset(
        torch.from_numpy(x_val.reshape(len(x_val), sequence_length, -1)),
        torch.from_numpy(y_val),
    )
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size)

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build_temporal_model(
        nn,
        input_dim=51,
        class_count=len(args.labels),
        max_sequence_length=sequence_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.02)
    # Inverse-frequency class weights: URFD falling/fallen frames are rarer than upright.
    counts = np.bincount(y_train, minlength=len(args.labels)).astype(np.float64)
    weights = counts.sum() / (len(args.labels) * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05,
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
    )
    print(f"train_class_counts={dict(zip(args.labels, counts.astype(int), strict=True))}")

    def evaluate() -> tuple[float, float, np.ndarray]:
        model.eval()
        confusion = np.zeros((len(args.labels), len(args.labels)), dtype=np.int64)
        with torch.inference_mode():
            for features, labels in validation_loader:
                predictions = model(features.to(device)).argmax(dim=-1).cpu().numpy()
                for true, predicted in zip(labels.numpy(), predictions, strict=True):
                    confusion[true, predicted] += 1
        accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
        f1s = []
        for c in range(len(args.labels)):
            tp = confusion[c, c]
            precision = tp / max(confusion[:, c].sum(), 1)
            recall = tp / max(confusion[c, :].sum(), 1)
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return accuracy, float(np.mean(f1s)), confusion

    best_macro_f1 = -1.0
    best_confusion = np.zeros((len(args.labels), len(args.labels)), dtype=np.int64)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to(device)), labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        accuracy, macro_f1, confusion = evaluate()
        print(f"epoch={epoch:03d} val_accuracy={accuracy:.4f} val_macro_f1={macro_f1:.4f}")
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_confusion = confusion
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": args.labels,
                    "sequence_length": sequence_length,
                    "validation_accuracy": accuracy,
                    "validation_macro_f1": macro_f1,
                    "seed": args.seed,
                },
                args.output,
            )
    print(f"saved={args.output} best_val_macro_f1={best_macro_f1:.4f}")
    print(f"confusion_matrix (rows=true {args.labels}):\n{best_confusion}")


if __name__ == "__main__":
    main()
