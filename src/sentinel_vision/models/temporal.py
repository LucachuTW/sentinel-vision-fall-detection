from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentinel_vision.config import TemporalConfig
from sentinel_vision.domain import ActionPrediction, PoseDetection


def normalize_skeleton(detection: PoseDetection) -> NDArray[np.float32]:
    """Normalize keypoints to bbox-relative coordinates while retaining confidence."""
    output = np.zeros((len(detection.keypoints), 3), dtype=np.float32)
    width = max(detection.bbox.width, 1.0)
    height = max(detection.bbox.height, 1.0)
    for index, point in enumerate(detection.keypoints):
        output[index] = (
            (point.x - detection.bbox.x1) / width - 0.5,
            (point.y - detection.bbox.y1) / height - 0.5,
            point.confidence,
        )
    return output


class SkeletonHistory:
    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self._items: dict[int, deque[NDArray[np.float32]]] = defaultdict(
            lambda: deque(maxlen=sequence_length)
        )
        self._centers: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=sequence_length)
        )
        self._aspects: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=sequence_length))
        self._heights: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=sequence_length))

    def update(self, detection: PoseDetection) -> None:
        if detection.track_id is None:
            return
        self._items[detection.track_id].append(normalize_skeleton(detection))
        self._centers[detection.track_id].append(detection.bbox.center)
        self._aspects[detection.track_id].append(
            detection.bbox.width / max(detection.bbox.height, 1.0)
        )
        self._heights[detection.track_id].append(detection.bbox.height)

    def sequence(self, track_id: int) -> NDArray[np.float32]:
        items = self._items[track_id]
        if not items:
            return np.empty((0, 17, 3), dtype=np.float32)
        return np.stack(items)

    def motion(self, track_id: int) -> float:
        """Mean center displacement per frame, in body-heights (resolution-independent)."""
        centers = list(self._centers[track_id])
        heights = list(self._heights[track_id])
        if len(centers) < 2:
            return 0.0
        distances = [
            math.dist(previous, current) / max((h_prev + h_cur) / 2.0, 1.0)
            for previous, current, h_prev, h_cur in zip(
                centers, centers[1:], heights, heights[1:], strict=False
            )
        ]
        return float(np.mean(distances[-8:]))

    def sudden_horizontal_transition(self, track_id: int) -> bool:
        aspects = self._aspects[track_id]
        if len(aspects) < 2 or aspects[-1] <= 1.55:
            return False
        return any(value < 1.0 for value in list(aspects)[-5:-1])

    def prune(self, active_ids: set[int]) -> None:
        for mapping in (self._items, self._centers, self._aspects, self._heights):
            stale = [track_id for track_id in mapping if track_id not in active_ids]
            for track_id in stale:
                del mapping[track_id]


class ActionClassifier(ABC):
    def __init__(self, config: TemporalConfig) -> None:
        self.config = config
        self.history = SkeletonHistory(config.sequence_length)

    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @property
    @abstractmethod
    def ready(self) -> bool: ...

    def classify(self, detections: tuple[PoseDetection, ...]) -> tuple[ActionPrediction, ...]:
        active_ids = {item.track_id for item in detections if item.track_id is not None}
        self.history.prune({int(item) for item in active_ids})
        output: list[ActionPrediction] = []
        for detection in detections:
            if detection.track_id is None:
                continue
            self.history.update(detection)
            output.append(self.predict(detection, self.history.sequence(detection.track_id)))
        return tuple(output)

    @abstractmethod
    def predict(
        self, detection: PoseDetection, sequence: NDArray[np.float32]
    ) -> ActionPrediction: ...


class HeuristicActionClassifier(ActionClassifier):
    """Auditable fallback that prevents random untrained neural outputs in the demo."""

    @property
    def backend_name(self) -> str:
        return "kinematic-heuristic-fallback"

    @property
    def ready(self) -> bool:
        return True

    def predict(self, detection: PoseDetection, sequence: NDArray[np.float32]) -> ActionPrediction:
        assert detection.track_id is not None
        aspect = detection.bbox.width / max(detection.bbox.height, 1.0)
        # ponytail: motion is in body-heights/frame, so thresholds are resolution-independent;
        # this stays a fallback heuristic — a trained transformer is the real classifier.
        motion = self.history.motion(detection.track_id)
        nose_y = detection.keypoints[0].y if detection.keypoints else detection.bbox.y1
        hip_y = (
            (detection.keypoints[11].y + detection.keypoints[12].y) / 2
            if len(detection.keypoints) >= 13
            else detection.bbox.center[1]
        )
        feeding_signal = (nose_y - detection.bbox.y1) / max(detection.bbox.height, 1) > 0.36

        scores = {label: 0.03 for label in self.config.labels}
        if self.history.sudden_horizontal_transition(detection.track_id):
            scores["incident"] = 0.90
        elif aspect > 1.55:
            scores["resting"] = 0.86
            scores["incident"] = 0.06
        elif feeding_signal or nose_y > hip_y - detection.bbox.height * 0.18:
            scores["feeding"] = 0.78
        elif motion > 0.015:
            scores["active"] = min(0.92, 0.68 + motion * 1.5)
        else:
            scores["standing"] = 0.82
        total = sum(scores.values())
        probabilities = {label: value / total for label, value in scores.items()}
        label = max(probabilities, key=probabilities.__getitem__)
        return ActionPrediction(
            track_id=detection.track_id,
            label=label,
            confidence=probabilities[label],
            probabilities=probabilities,
            model_ready=True,
        )


class TransformerActionClassifier(ActionClassifier):
    def __init__(self, config: TemporalConfig) -> None:
        super().__init__(config)
        try:
            import torch  # type: ignore[import-not-found]
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "the temporal transformer requires the ML dependencies: uv sync --extra ml"
            ) from exc
        checkpoint_path = Path(config.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"trained temporal checkpoint not found: {checkpoint_path}; "
                "run scripts/train_temporal.py with a labelled dataset"
            )
        self._torch = torch
        self._device = self._resolve_device(config.device)
        self._model = build_temporal_model(
            nn,
            input_dim=51,
            class_count=len(config.labels),
            max_sequence_length=config.sequence_length,
        )
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location=self._device, weights_only=True
        )
        checkpoint_labels = checkpoint.get("labels")
        if checkpoint_labels != config.labels:
            raise ValueError(
                f"checkpoint labels {checkpoint_labels!r} do not match config labels {config.labels!r}"
            )
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.to(self._device).eval()

    def _resolve_device(self, configured: str) -> str:
        if configured != "auto":
            return configured
        return "cuda:0" if self._torch.cuda.is_available() else "cpu"

    @property
    def backend_name(self) -> str:
        return f"skeleton-transformer:{self._device}"

    @property
    def ready(self) -> bool:
        return True

    def predict(self, detection: PoseDetection, sequence: NDArray[np.float32]) -> ActionPrediction:
        assert detection.track_id is not None
        length = self.config.sequence_length
        window = sequence[-length:]
        valid = len(window)
        if valid < length:
            # Right-pad with zeros so real frames keep training-consistent positions;
            # the padding mask removes them from attention and pooling.
            padding = np.zeros((length - valid, *window.shape[1:]), dtype=window.dtype)
            window = np.concatenate((window, padding), axis=0)
        tensor = self._torch.from_numpy(window).reshape(1, length, -1)
        mask = self._torch.zeros(1, length, dtype=self._torch.bool)
        mask[0, valid:] = True
        with self._torch.inference_mode():
            logits = self._model(tensor.to(self._device), mask.to(self._device))
            values = self._torch.softmax(logits, dim=-1)[0].cpu().numpy()
        probabilities = {
            label: float(probability)
            for label, probability in zip(self.config.labels, values, strict=True)
        }
        label = max(probabilities, key=probabilities.__getitem__)
        return ActionPrediction(
            detection.track_id,
            label,
            probabilities[label],
            probabilities,
            True,
        )


def build_temporal_model(
    nn: Any,
    input_dim: int,
    class_count: int,
    max_sequence_length: int,
    model_dim: int = 128,
    heads: int = 4,
    layers: int = 3,
    dropout: float = 0.1,
) -> Any:
    import torch  # type: ignore[import-not-found]

    class SkeletonActionTransformer(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(input_dim, model_dim)
            self.position = nn.Parameter(torch.zeros(max_sequence_length, model_dim))
            nn.init.trunc_normal_(self.position, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=heads,
                dim_feedforward=model_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers,
                enable_nested_tensor=False,
            )
            self.norm = nn.LayerNorm(model_dim)
            self.head = nn.Linear(model_dim, class_count)

        def forward(self, inputs: Any, padding_mask: Any | None = None) -> Any:
            hidden = self.projection(inputs)
            hidden = hidden + self.position[: hidden.shape[1]]
            hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
            if padding_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                # Mean over real frames only; padded positions never contribute.
                valid = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            return self.head(self.norm(pooled))

    return SkeletonActionTransformer()


def create_action_classifier(config: TemporalConfig) -> ActionClassifier:
    if config.backend == "transformer":
        return TransformerActionClassifier(config)
    return HeuristicActionClassifier(config)
