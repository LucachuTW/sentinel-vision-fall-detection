#!/usr/bin/env python3
"""Export a YOLO26 pose checkpoint to a calibrated TensorRT engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26s-pose.pt")
    parser.add_argument(
        "--data", required=True, type=Path, help="Ultralytics dataset YAML for INT8 calibration"
    )
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--workspace", default=4.0, type=float, help="TensorRT workspace in GiB")
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true", help="Build FP16 instead of INT8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = Path(args.model).name.lower()
    if "yolo26" not in name or "pose" not in name:
        raise SystemExit("--model must be a YOLO26 pose checkpoint")
    if not args.data.is_file():
        raise SystemExit(f"calibration dataset YAML does not exist: {args.data}")
    dataset = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or "train" not in dataset:
        raise SystemExit("calibration dataset must define a representative train split")

    from ultralytics import YOLO

    model = YOLO(args.model, task="pose")
    artifact = model.export(
        format="engine",
        quantize=16 if args.half else 8,
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        dynamic=True,
        workspace=args.workspace,
        device=args.device,
        simplify=True,
    )
    print(f"TensorRT artifact: {artifact}")


if __name__ == "__main__":
    main()
