#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

extras=(--extra dev)
if [[ "${1:-}" == "--gpu" ]]; then
  extras+=(--extra ml)
fi

uv sync --python 3.12 "${extras[@]}"
uv run python scripts/download_models.py
uv run sentinel-vision doctor --config config/demo.yaml

echo "Environment ready. Run: make run"

