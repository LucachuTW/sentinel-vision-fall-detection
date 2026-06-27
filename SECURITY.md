# Security policy

Report vulnerabilities privately to the repository owner. Do not include camera credentials, private frames, model artifacts, or customer data in an issue.

## Operational baseline

- Supply RTSP credentials through `SV_SOURCE_URI`; the API and logs redact userinfo.
- Mount production models read-only and verify SHA-256 before startup.
- Put authentication and TLS at the reverse proxy; the built-in API is intended for a trusted edge network.
- Restrict `/metrics`, `/docs`, MJPEG and WebSocket endpoints at the network boundary.
- Run containers as the bundled non-root user and grant only the required NVIDIA device access.
- Treat Ollama output as narrative enrichment only. It cannot set event type, severity, confidence or control equipment.
- Do not use analytics events as medical diagnoses or autonomous safety interlocks.

