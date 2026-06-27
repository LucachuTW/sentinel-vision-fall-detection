# DeepStream zero-copy deployment contract

Use this boundary for multi-camera 4K deployments where the OpenCV CPU handoff is unacceptable:

`rtspsrc → depay → parse → nvv4l2decoder → nvstreammux → nvinfer (YOLO26 pose TensorRT) → nvtracker → metadata bridge`

Frames remain `NvBufSurface`/NVMM through `nvinfer`; only compact pose/tracking metadata crosses to the Sentinel event/API service. Conditional SAM 2 should run as a separate GPU service receiving selected CUDA IPC frames and box prompts.

This repository does not ship a fake universal output parser. A YOLO26 pose parser must match the exact exported engine bindings and deployed DeepStream/TensorRT ABI. Qualify the parser with golden tensors before enabling `config/production.yaml`. NVIDIA driver and DeepStream are also absent from the portable CI environment.

The portable service explicitly reports `cpu-handoff` when configured with GStreamer NVDEC. This document is the integration contract, not evidence that the current host is zero-copy capable.

