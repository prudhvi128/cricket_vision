# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────────────
# Model weights are NOT baked into the image.
#
# backend/models/ball_detector.pt (6.2 MB) and shot_classifier.ckpt (72 MB) are
# gitignored and are ~78 MB of binary that changes rarely. Mount them at runtime
# and point CRICKET_MODELS_DIR at the mount:
#
#   docker run -v "$PWD/backend/models:/app/backend/models:ro" ...
#
# A missing checkpoint is a loud FileNotFoundError at startup, not a silent
# fall back to random weights — see app/core/config.py:resolve_shot_model_path.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# OpenCV needs libGL and libglib even in a headless container, and ffmpeg is what
# actually writes the delivery clips and overlays.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is only rebuilt when requirements.txt changes,
# not on every source edit.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# `backend/` is copied to /app/backend because `backend/` is the import root:
# main.py does `from app...`, so /app/backend must be on sys.path. Running uvicorn
# with --app-dir keeps that true regardless of WORKDIR.
COPY backend/ ./backend/

# data/ is created empty and mounted over. Nothing generated belongs in a layer.
RUN mkdir -p /app/data/runs /app/data/uploads /app/data/samples

ENV CRICKET_DATA_DIR=/app/data \
    CRICKET_MODELS_DIR=/app/backend/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

# One worker by default. The single-pass design holds the whole analysis in one
# process, and a GPU plus multiple workers competes for the same device. Scale
# with the replica count, not --workers, unless you have measured that the
# bottleneck is CPU rather than GPU.
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]