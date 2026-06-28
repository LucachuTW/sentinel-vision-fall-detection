# Sentinel Vision

Local-first, real-time video analytics for industrial monitoring and behavioural research. The pipeline combines **YOLO26-Pose**, persistent identity tracking, on-demand **SAM 2.1** masks, and a compact skeleton transformer behind an observable FastAPI service.

This repository is an executable engineering system, not a notebook: bounded queues control live-stream latency, models are replaceable adapters, production configuration is fail-closed, secrets are redacted, and the demo works without a camera or GPU.

## Real-data demo: worker fall detection

The full real stack runs end to end on **real video** with **real models** — no synthetic fallbacks. The
temporal head is trained on the [UR Fall Detection Dataset](http://fenix.ur.edu.pl/~mkepski/ds/uf.html)
(real human falls), framed as industrial worker-safety: a fall on the floor is flagged as a **critical
incident**.

![Live fall detection on real video, flagged as a critical incident](docs/demo-real-fall.gif)

_Live pipeline output (real URFD clip). Single frame — a fallen worker flagged `fallen 98%`:_

![Real fall flagged as a critical incident](docs/demo-real-fall.jpg)

- **Perception:** YOLO26n-pose (FP16) → ByteTrack → SAM 2.1 box-prompted masks, all on the RTX 2070.
- **Action model:** skeleton transformer trained on URFD with a **by-sequence split** (held-out falls).
  **Validation macro-F1 = 0.90** on sequences never seen in training — an honest number, not a synthetic 100%.
- **Live result:** a real fall fires `fallen` (critical) / `falling` (warning) at ≥0.96 confidence.
- **Throughput:** 27.98 FPS, p95 49 ms E2E, 1990 MiB VRAM, SLO PASS — see
  [the real-data validation](docs/benchmarks/2026-06-28-rtx2070-real.md).

```bash
make install-ml && make models
make dataset-real     # downloads URFD, extracts YOLO26 poses, trains on a sequence split
make demo-real        # http://localhost:8080 — live fall detection on the real clip
```

The synthetic, no-GPU demo below remains as the CI / no-hardware path; it never pretends to be real inference.

## Validated performance

RTX 2070 8 GiB, driver 595.71.05, PyTorch 2.12.1+cu130, FP16, 640 px model input. Results include a 3-second warmup and report distributions rather than a single best run.

| Path | Throughput | p50 | p95 | Drops | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Full pipeline: YOLO26n-Pose + ByteTrack + conditional SAM 2.1 | 30.00 FPS | 7.66 ms E2E | 47.16 ms E2E | 0% | 2,088 MiB |
| YOLO26n-Pose isolated steady state | 208.42 FPS | 4.80 ms | 5.25 ms | — | shared |
| SAM 2.1 tiny, two box prompts | — | 31.09 ms | 31.76 ms | — | shared |

The full benchmark processed 301/301 frames, tracked four people and passed its throughput, p95 latency and drop-rate SLOs. See [the reviewed benchmark](docs/benchmarks/2026-06-27-rtx2070.md) and [the methodology](docs/performance.md).

```mermaid
flowchart LR
    A[RTSP / file / camera] --> B[GStreamer ingest]
    B -->|bounded latest-frame queue| C[YOLO26-Pose]
    C --> D[ByteTrack]
    D --> E{Segmentation policy}
    E -->|new / stale / uncertain track| F[SAM 2.1]
    E -->|skip expensive call| G[Skeleton window]
    F --> G
    G --> H[Temporal Transformer]
    H --> I[Events + local Ollama narrative]
    I --> J[FastAPI / WebSocket / Prometheus]
```

## What is implemented

- RTSP H.264/H.265, video-file, camera, and deterministic synthetic sources.
- GStreamer NVDEC pipeline construction with low-latency `appsink`, reconnect, and drop-oldest backpressure.
- Native YOLO26 pose adapter for PyTorch, ONNX, or TensorRT artifacts. Configuration rejects non-YOLO26 pose checkpoints.
- ByteTrack via `supervision`, plus a clearly labelled deterministic IoU tracker for CI.
- SAM 2.1 box prompting with dynamic memory and an invocation policy based on new, stale, or uncertain tracks.
- COCO-17 skeleton normalization, per-ID temporal windows, a pre-norm transformer, training script, checkpoint/label validation, and an auditable kinematic fallback.
- Optional Ollama incident summaries using a local `qwen3:1.7b`; it is outside the decision path and cannot create an alert.
- Live dashboard, MJPEG stream, WebSocket telemetry, REST state/events, readiness/liveness, and Prometheus metrics.
- CPU and NVIDIA containers, Compose profiles, test suite, static analysis, model card, benchmark harness, and TensorRT export script.

## Run the self-contained demo

Python 3.12 is required because production ML wheels do not yet consistently target Python 3.14.

```bash
./scripts/bootstrap.sh
UV_CACHE_DIR=.cache/uv uv run sentinel-vision demo --seconds 8
UV_CACHE_DIR=.cache/uv uv run sentinel-vision run --config config/demo.yaml
```

Open [http://localhost:8080](http://localhost:8080). The demo deliberately exposes its fallbacks under `/health/ready`; it does not pretend that synthetic inference is YOLO26/SAM 2.

## Run the real temporal transformer end-to-end

The skeleton transformer is architecture-complete; production refuses to start without a trained
checkpoint. To exercise the genuine transformer path locally (no GPU or labelled corpus needed),
generate a separable synthetic action dataset, train a compact checkpoint on CPU in under a minute,
and serve it:

```bash
make install-ml
make demo-transformer
```

This runs `scripts/make_synthetic_windows.py` → `scripts/train_temporal.py` → `config/demo-transformer.yaml`.
The dataset and `.pt` are regenerated locally and never committed. Swap in a real subject-split dataset
for a meaningful model; the synthetic windows are a runnable smoke demo, not behavioural ground truth.

## Run all local models

```bash
./scripts/bootstrap.sh --gpu
ollama serve
SV_SOURCE_URI='rtsp://user:password@camera/stream' \
  UV_CACHE_DIR=.cache/uv uv run sentinel-vision run --config config/local-gpu.yaml
```

The downloader pins and verifies `yolo26n-pose.pt` and `sam2.1_t.pt` under `models/`. The local profile expects a working CUDA driver; run `sentinel-vision doctor --config config/local-gpu.yaml` first.

Validate the complete local model chain on a known image:

```bash
sha256sum -c models/checksums.sha256
UV_CACHE_DIR=.cache/uv uv run python scripts/smoke_local_models.py
```

Run the reproducible full GPU pipeline benchmark without a camera:

```bash
UV_CACHE_DIR=.cache/uv uv run python scripts/benchmark_pipeline.py \
  --config config/benchmark-gpu.yaml --warmup-seconds 3 --seconds 10
```

## Production TensorRT INT8

INT8 numbers are only meaningful with a representative calibration set and the exact target GPU. The exporter therefore requires a dataset YAML:

```bash
UV_CACHE_DIR=.cache/uv uv run python scripts/export_yolo26.py \
  --model yolo26s-pose.pt \
  --data datasets/site-pose.yaml \
  --batch 8 --device 0
```

Move the resulting engine to `models/yolo26s-pose-int8.engine`, train the temporal head, then start `config/production.yaml`. Production validation refuses demo pose, tracker, segmentation, and temporal backends.

```bash
UV_CACHE_DIR=.cache/uv uv run python scripts/train_temporal.py \
  --dataset datasets/action-windows.npz \
  --output models/temporal-transformer.pt
```

Dataset contract: `x` has shape `[samples, time, 17, 3]` (normalized x/y/confidence) and `y` is the integer class index. Split by subject/session before creating the NPZ; random window leakage produces misleading accuracy.

## GPU and video memory path: exact scope

CUDA inference is verified on the local RTX 2070. The installed PyPI OpenCV wheel has no GStreamer support, so the portable service declares and uses its FFmpeg CPU decode fallback. A GStreamer-capable build can use NVDEC but still downloads at the OpenCV `appsink` boundary. Genuine decode-to-inference zero-copy requires the documented DeepStream boundary (NVMM → `nvinfer`) or a custom GstCudaMemory/DLPack adapter. Readiness reports the active path instead of claiming PCIe-free inference.

See [architecture.md](docs/architecture.md) for the production data plane and [performance.md](docs/performance.md) for the benchmark protocol. No FPS or VRAM figure in this repository is fabricated from a different accelerator.

Current host evidence is recorded in [validation.md](docs/validation.md), including driver, CUDA, model and SLO checks.

## Observability

The service exports Prometheus counters, gauges and histograms for throughput, latency, queue pressure, frame drops, reconnects, detections, events and GPU NVML telemetry. The web console includes a live FPS/E2E graph and GPU utilization/VRAM cards.

The graph is backed by a bounded 600-sample server-side time series (`/v1/telemetry`), so a browser refresh does not reset the visible operating window. It uses separate axes for FPS and latency to avoid the scale collision that previously made the chart unreadable.

```bash
docker compose --profile observability up --build
```

- Application dashboard: `http://localhost:8080`
- Prometheus and alert rules: `http://localhost:9090`
- Provisioned Grafana dashboard: `http://localhost:3000`

Operational procedures are in [runbook.md](docs/runbook.md).

## API

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | process liveness |
| `GET /health/ready` | model/source readiness and active degradations |
| `GET /v1/state` | current tracks, keypoints, actions, and latency |
| `GET /v1/events` | bounded event history |
| `GET /v1/telemetry` | rolling FPS, E2E and stage latency series |
| `GET /v1/system` | local GPU utilization, VRAM, thermals and power |
| `GET /v1/stream.mjpeg` | annotated stream |
| `WS /v1/ws` | per-frame telemetry |
| `GET /metrics` | Prometheus exposition |
| `GET /docs` | OpenAPI UI |

RTSP credentials are accepted through `SV_SOURCE_URI`, never committed, and redacted from the config endpoint and logs.

When `api.require_auth` is true (the default in `config/production.yaml`), every `/v1/*` endpoint,
`/metrics`, the MJPEG stream and the WebSocket require the `$SV_API_TOKEN` token, supplied either as an
`Authorization: Bearer` header or a `?token=` query parameter (browsers cannot set headers on `<img>`
or WebSocket requests). Open the dashboard as `http://host:8080/?token=$SV_API_TOKEN` and it forwards
the token to every call. The service fails to start if the token is unset; health probes stay open for
orchestrators. The default bind address is `127.0.0.1` — set `api.host: 0.0.0.0` explicitly to expose it.

Events are in-memory by default; set `runtime.event_db_path` (on in `production.yaml`) to persist them
to SQLite and reload the recent history on restart.

## Verification

```bash
make quality
make security
make benchmark
make smoke-models
```

The benchmark output records the active backends and degradations with latency percentiles. Compare runs only when input, image size, precision, driver, warmup, and hardware match.

## Design notes and limitations

- Frame dropping is intentional: for live safety/operations views, bounded latency is more useful than processing stale frames.
- SAM 2 is conditionally invoked and low-confidence prompts have a dedicated cooldown; cost still grows with simultaneous objects.
- Pretrained COCO pose detects people. Animal or specialized biological subjects require a domain-labelled YOLO26-Pose fine-tune and a matching skeleton schema.
- The temporal network is architecture-complete but no domain checkpoint is shipped. Production refuses to start without one.
- Alerts are analytical signals, not medical diagnoses or safety interlocks. Validate per site and retain a human escalation path.
- Ultralytics licensing may require an enterprise license for closed commercial deployment; review it before distribution.

## Primary model references

- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics pose documentation](https://docs.ultralytics.com/tasks/pose/)
- [Meta SAM 2 repository](https://github.com/facebookresearch/sam2)
- [ByteTrack paper](https://arxiv.org/abs/2110.06864)

MIT licensed application code. Model weights retain their upstream licenses.
