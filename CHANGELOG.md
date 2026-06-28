# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-06-28

### Added
- Real-time inference pipeline: YOLO26-pose → ByteTrack → conditional SAM 2.1 → temporal skeleton
  transformer, behind an observable FastAPI service with bounded drop-oldest queues.
- Real-data **worker fall detection** trained on the UR Fall Detection Dataset with a by-sequence
  split; validation macro-F1 0.90 on held-out fall sequences.
- Operations dashboard with a live annotated MJPEG stream, WebSocket telemetry, FPS/latency charts,
  GPU cards, and an event timeline; reproducible RTX 2070 benchmarks with SLOs.
- Hot-swappable input sources from the dashboard: fall-demo clip, server webcam, uploaded video,
  RTSP camera, and the synthetic scene.
- Optional bearer-token API auth (header or `?token=`), SQLite event persistence, Prometheus metrics,
  Grafana provisioning, and Docker/Compose deployment.
- Runnable temporal-transformer demo trained on synthetic windows, and a synthetic, no-GPU fallback
  path that is exercised in CI.

### Notes
- Application code is MIT licensed. Model weights and the UR Fall Detection Dataset retain their
  upstream licenses and are downloaded locally, never redistributed by this repository.
