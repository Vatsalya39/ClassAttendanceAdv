# ── Stage: base ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Install system deps required by asyncpg / psycopg2 / opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first (layer cache)
COPY pyproject.toml poetry.lock alembic.ini ./

# Install Poetry then all deps (runs from repo root where pyproject.toml now lives)
RUN pip install --no-cache-dir poetry==1.8.3 \
    && poetry config virtualenvs.create false \
    && poetry install --no-root --only main

# Copy application code (+ scripts/ so the migration job and the
# create-superadmin bootstrap work inside containers)
COPY backend/ backend/
COPY ml/ ml/
COPY scripts/ scripts/

# ── Stage: api ────────────────────────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Stage: worker ─────────────────────────────────────────────────────────────
# NOTE: the plain worker stage has NO ML dependencies — it can serve as a
# queue consumer for non-ML tasks, but capture processing requires an
# ml-* image below. Container Apps' worker/ml tiers must use ml-cpu or
# ml-gpu images, otherwise every job detects 0 faces.
FROM base AS worker
CMD ["python", "-m", "backend.worker.main"]

# ── Stage: ml-cpu ─────────────────────────────────────────────────────────────
# Default inference image (works on CPU everywhere: local dev, CI, Azure
# consumption profile). insightface pulls opencv-python; we keep libGL so it
# runs and pin nothing else beyond ml/requirements.txt.
FROM base AS ml-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ml/requirements-cpu.txt ml/requirements-cpu.txt
RUN pip install --no-cache-dir -r ml/requirements-cpu.txt

CMD ["python", "-m", "backend.worker.main"]

# ── Stage: api-ml ─────────────────────────────────────────────────────────────
# The public api image. POST /register runs the face pipeline inline
# (spec decision), so the API tier ships with the same ML stack as the
# inference workers.
FROM ml-cpu AS api-ml
EXPOSE 8000
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Stage: ml-gpu ─────────────────────────────────────────────────────────────
# CUDA-12 runtime base for Azure Container Apps GPU workload profiles
# (NVIDIA drivers are injected by the platform; the container needs the CUDA
# userland that onnxruntime-gpu links against). Larger image by design —
# built/pushed only by the ml-inference pipeline.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04 AS ml-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# Runtime subset of pyproject main deps needed by the worker process,
# installed directly (poetry resolution on this base adds no value).
COPY pyproject.toml poetry.lock alembic.ini ./
RUN python3 -m pip install --upgrade pip \
    && pip install --no-cache-dir \
        "sqlalchemy[asyncio]>=2.0.30,<3" \
        "asyncpg>=0.30,<0.31" \
        "greenlet>=3.1,<4" \
        "alembic>=1.13.1,<2" \
        "pgvector>=0.3.2,<0.4" \
        "redis[hiredis]>=5.0.4,<6" \
        "azure-servicebus>=7.12.1,<8" \
        "azure-storage-blob>=12.20.0,<13" \
        "pydantic>=2.7,<3" \
        "pydantic-settings>=2.3,<3" \
        "passlib[argon2]>=1.7.4,<2" \
        "python-jose[cryptography]>=3.3.0,<4" \
        "structlog>=24.2.0,<25"

COPY backend/ backend/
COPY ml/ ml/
COPY scripts/ scripts/
COPY ml/requirements-gpu.txt ml/requirements-gpu.txt
RUN pip install --no-cache-dir -r ml/requirements-gpu.txt

CMD ["python", "-m", "backend.worker.main"]
