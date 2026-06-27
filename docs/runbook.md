# Operations runbook

## Start and verify

```bash
make doctor
make run
curl -fsS http://127.0.0.1:8080/health/ready
curl -fsS http://127.0.0.1:8080/metrics | head
```

For the complete local stack:

```bash
docker compose --profile observability up --build
```

- Operations UI: `http://localhost:8080`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (anonymous read-only)

## Readiness failure

1. Read `/health/ready`; it names active components, degradations and the latest fatal error.
2. Run `sentinel-vision doctor --config <config>`.
3. Verify model digests: `sha256sum -c models/checksums.sha256`.
4. Verify GPU: `nvidia-smi` and `python -c 'import torch; print(torch.cuda.is_available())'`.
5. Verify decode: `gst-inspect-1.0 nvh264dec`.
6. Verify the camera outside the service with `ffprobe` while keeping credentials out of shell history.

## Latency or frame-drop alert

1. Compare `sentinel_frames_ingested_total` and `sentinel_frames_processed_total` rates.
2. Identify the stage through `sentinel_stage_latency_seconds` p95.
3. Inspect `sentinel_queue_depth` and `sentinel_frames_dropped_total` by boundary.
4. Check VRAM pressure, GPU utilization, temperature and power in Grafana.
5. Reduce SAM objects/frequency before reducing pose input quality.
6. Capture a benchmark artifact with the exact production config.

## Camera disconnect

The source reconnects with bounded exponential backoff. Alert on `sentinel_source_connected == 0` and inspect `sentinel_source_reconnects_total`. Do not restart repeatedly unless the source has recovered and the pipeline remains unready.

## NVIDIA/Wayland verification (Omarchy)

```bash
nvidia-smi
dkms status
hyprctl systeminfo
hyprctl monitors
hyprctl configerrors
gst-inspect-1.0 nvh264dec
```

The validated host uses `nvidia-open-dkms`, DRM modesetting, Hyprland explicit sync and the RTX 2070 as the display GPU. Rebuild initramfs and reboot only after a driver/kernel package change—not for an application-level CPU PyTorch wheel.

## Rollback

- Application: deploy the previous immutable image and matching model volume.
- Configuration: restore the previous reviewed YAML and restart the service.
- Models: restore the previous checksummed artifacts; TensorRT engines are GPU/driver specific.
- Never use an untrained temporal checkpoint as a rollback substitute.

