import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from sentinel_vision.models.temporal import build_temporal_model  # noqa: E402


def test_padding_mask_makes_output_independent_of_padded_frames() -> None:
    torch.manual_seed(0)
    model = build_temporal_model(nn, input_dim=51, class_count=5, max_sequence_length=8).eval()

    window = torch.zeros(1, 8, 51)
    window[:, :5] = torch.randn(1, 5, 51)  # five real frames, three padded
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 5:] = True

    noisy = window.clone()
    noisy[:, 5:] = torch.randn(1, 3, 51) * 100.0  # garbage in the padded positions

    with torch.inference_mode():
        clean = model(window, mask)
        garbage = model(noisy, mask)

    # Masked padding must not leak into attention or the pooled representation.
    assert torch.allclose(clean, garbage, atol=1e-5)
