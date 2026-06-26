.PHONY: install install-ml models test lint typecheck security quality demo demo-transformer dataset-real demo-real benchmark-real run doctor benchmark smoke-models observability

UV_CACHE_DIR ?= .cache/uv
export UV_CACHE_DIR

install:
	uv sync --extra dev

install-ml:
	uv sync --extra dev --extra ml

models:
	uv run python scripts/download_models.py

test:
	uv run pytest --cov=sentinel_vision --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src/sentinel_vision

security:
	mkdir -p .cache
	uv export --quiet --frozen --all-extras --no-emit-project --output-file .cache/security-requirements.txt
	uv run pip-audit --strict --requirement .cache/security-requirements.txt --require-hashes --disable-pip

quality: lint typecheck test

demo:
	uv run sentinel-vision demo --seconds 8

# Generate a synthetic action dataset, train the real temporal transformer on CPU
# (<1 min), then serve it. Requires the ml extra: make install-ml.
demo-transformer:
	uv run python scripts/make_synthetic_windows.py --output artifacts/synthetic-windows.npz --sequence-length 24
	uv run python scripts/train_temporal.py --dataset artifacts/synthetic-windows.npz --output models/temporal-transformer.pt --epochs 40
	uv run sentinel-vision run --config config/demo-transformer.yaml

# Real-data path (requires GPU + ml extra: make install-ml && make models).
# Downloads UR Fall Detection, extracts YOLO26 poses, trains the transformer on a sequence split.
dataset-real:
	uv run python scripts/prepare_urfd_dataset.py
	uv run python scripts/train_temporal.py --dataset artifacts/urfd-train.npz \
		--val-dataset artifacts/urfd-val.npz --labels upright falling fallen --epochs 60

# Serve the full real stack on the real fall video (run dataset-real first).
demo-real:
	uv run sentinel-vision run --config config/demo-real.yaml

benchmark-real:
	uv run python scripts/benchmark_pipeline.py --config config/benchmark-real.yaml --warmup-seconds 3 --seconds 12

run:
	uv run sentinel-vision run --config config/demo.yaml

doctor:
	uv run sentinel-vision doctor --config config/demo.yaml

benchmark:
	uv run python scripts/benchmark_pipeline.py --seconds 15

smoke-models:
	uv run python scripts/smoke_local_models.py

observability:
	docker compose --profile observability up --build
