# Contributing

## Local setup

```bash
./scripts/bootstrap.sh
make quality
```

Use Python 3.12 and `uv`; do not install project packages globally. Model binaries stay under `models/` and are verified by `models/checksums.sha256` but never committed.

## Change contract

1. Add or update tests for behavioural changes.
2. Keep production configuration fail-closed; never silently replace a requested production model with a demo backend.
3. Record benchmark hardware, precision, model digest, warmup and latency distribution. A single FPS number is insufficient.
4. Preserve RTSP credential redaction and avoid raw-frame persistence by default.
5. Run `make quality` before review.

Commit subjects should state the effect of the change. Keep generated benchmark output in `artifacts/`; promote only reviewed summaries to `docs/benchmarks/`.
