FROM ghcr.io/astral-sh/uv:0.11.16 AS uv
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg gstreamer1.0-tools gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
      libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY config ./config
COPY models/README.md ./models/README.md
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 sentinel && chown -R sentinel:sentinel /app
USER sentinel
EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=3s --start-period=20s --retries=3 \
  CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=2)"]
CMD [".venv/bin/sentinel-vision", "run", "--config", "config/demo.yaml"]

