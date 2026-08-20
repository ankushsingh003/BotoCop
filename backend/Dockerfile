# BotoCop API service image.
#
# Build context MUST be the repo root (not backend/), since this image
# needs pyproject.toml, uv.lock, and the full backend/ package tree:
#   docker build -f backend/Dockerfile -t botocop-api .
#   docker run -p 8000:8000 --env-file .env botocop-api
#
# This builds the always-on web service (uvicorn). It deliberately does
# NOT install the 'batch' extra (PySpark + a JVM) -- see pyproject.toml's
# comment on [project.optional-dependencies]. To run the Kafka consumer
# worker from this same image instead, override the CMD, e.g.:
#   docker run --env-file .env botocop-api python -m backend.src.ingestion.kafka_consumer

FROM python:3.12-slim

# libgomp1: required by opencv-python-headless / onnxruntime (a chromadb
# dependency) at import time even with the -headless OpenCV build (no
# libGL/X11 needed, but libgomp1 still is). Installing it here avoids the
# classic "ImportError: libgomp.so.1: cannot open shared object file"
# surprise on slim/minimal base images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv gives fast, reproducible installs from uv.lock -- the same tool
# build.sh uses for the Render deploy path, rather than a second, drifting
# pip-based install path just for Docker.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifests first so Docker can cache the dependency layer
# and skip reinstalling on every source-code change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Now copy the actual application code.
COPY . .

ENV PORT=8000
EXPOSE 8000

# init_db() also runs automatically on app startup (see
# backend/src/api/server.py's startup handler) as a safety net, but
# running it explicitly here means the very first request doesn't pay
# that one-time cost.
CMD ["sh", "-c", "uv run python -c 'from backend.src.case.db import init_db; init_db()' && uv run uvicorn backend.src.api.server:app --host 0.0.0.0 --port ${PORT}"]
