# Performance qualification

## Measurement protocol

Use `scripts/benchmark_pipeline.py` after model warmup. Record:

- GPU, driver, CUDA, TensorRT, GStreamer, model digest, input codec/resolution/FPS;
- model image size, batch, precision, calibration dataset digest;
- number of streams and visible objects;
- SAM invocation interval and objects per invocation;
- processed FPS, drop rate, per-stage histograms, and capture-to-output p50/p95/p99;
- steady-state and peak VRAM from `nvidia-smi dmon` or DCGM.

Run for at least five minutes after a 30-second warmup for a portfolio report; run one hour with network impairment and camera reconnects before production acceptance.

## Acceptance targets

Targets must be established on deployment hardware. A reasonable starting service-level objective is:

- p95 capture-to-output latency below 200 ms at nominal stream count;
- no unbounded memory growth over one hour;
- readiness false within five seconds of a failed critical stage;
- reconnect without process restart after a 30-second RTSP outage;
- no event identity cross-over in the site occlusion test set;
- temporal macro-F1 and per-class recall reported by subject-disjoint validation.

Raw throughput above capture FPS is useful for capacity planning, but a live pipeline should also report dropped frames and frame age. Quoting only model kernel FPS hides decode, transfer, tracking, rendering, and conditional segmentation cost.

## INT8 calibration

TensorRT INT8 calibration images must represent actual illumination, camera angle, subject scale, clothing/background, and blur. Report the FP16-to-INT8 delta for pose AP and per-keypoint accuracy. Reject an engine that meets throughput but violates the site accuracy budget.

