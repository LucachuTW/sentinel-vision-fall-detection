# Validation record — 2026-06-27

Evidence from the development host; not a transferable hardware claim.

## Host capability audit

- CPU: AMD Ryzen 7 9800X3D
- GPU: NVIDIA GeForce RTX 2070, 8 GiB, compute capability 7.5
- Driver: `nvidia-open-dkms` 595.71.05, loaded for kernel 7.0.9
- CUDA: system toolkit 13.2; PyTorch 2.12.1+cu130; `torch.cuda.is_available() == True`
- GPU execution: 2048×2048 CUDA matmul passed
- Display: Hyprland 0.55.2 on NVIDIA DRM with explicit sync; 1440p@144 Hz + 1080p@75 Hz
- Hyprland validation: reload successful, `hyprctl configerrors` empty
- GStreamer 1.28.3 NVDEC/NVENC plugin: available and `nvh264dec` loads
- OpenCV wheel: GStreamer disabled; portable ingestion declares the FFmpeg fallback
- FFmpeg: 8.1.1 with NVDEC/NVENC enabled
- Python: 3.12.13
- Local Ollama manifests: `qwen3:1.7b`, `qwen3:8b`, `gemma4:e4b`; server not enabled during benchmark

## Graphics incident resolution

The system driver is healthy. The application environment was the remaining fault: it contained `torch 2.12.1+cpu`, so CUDA was invisible even while `nvidia-smi` worked. Re-syncing the locked `ml` extra installed `torch 2.12.1+cu130` and the matching CUDA runtime libraries. No Hyprland configuration change or reboot was required.

## GPU full-pipeline benchmark

Configuration: `config/benchmark-gpu.yaml`; 3-second warmup; 10.03-second measurement; YOLO26n-Pose FP16 + ByteTrack + conditional SAM 2.1 tiny.

- 301/301 frames, 30.001 FPS, zero drops
- E2E p50/p95/p99: 7.660/47.158/52.304 ms
- Pose p50/p95: 4.751/8.801 ms
- Conditional SAM p95: 33.089 ms
- Peak VRAM: 2,088 MiB
- GPU mean/max: 24.2%/26.0%
- Max temperature/power: 61 °C / 116.5 W
- SLO: PASS

Detailed evidence: [2026-06-27-rtx2070.md](benchmarks/2026-06-27-rtx2070.md).

## Isolated models

- YOLO26n-Pose: 30 samples, p50 4.798 ms, p95 5.250 ms, 208.4 FPS equivalent
- ByteTrack: p50 0.262 ms, IDs stable
- SAM 2.1 tiny, two prompts: p50 31.094 ms, p95 31.757 ms
- Temporal transformer architecture: output `(2, 5)`, 606,469 parameters

## Deterministic CI/demo pipeline

- 240/240 frames at 24.0 FPS
- E2E p50/p95/p99: 2.275/4.456/4.583 ms
- Zero drops; SLO PASS
- Backends are explicitly declared demo fallbacks, not presented as model performance

## Quality gates

- Ruff lint and format: pass
- mypy strict mode: pass
- pytest: 34/34 pass on Python 3.12.13
- Branch-aware coverage: 76.64%, enforced at 75% minimum
- Locked dependency audit: no known vulnerabilities (`pip-audit`); vulnerable pytest 8.4.2 replaced by 9.1.1
- Model SHA-256 verification: pass
- Locked Python environment: `uv.lock`
- Prometheus alert rules and Grafana dashboard: provisioned
- Chromium runtime validation: MJPEG, REST, WebSocket, GPU cards and dual-axis FPS/E2E graph render successfully
