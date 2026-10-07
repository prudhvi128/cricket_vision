"""
main.py — FastAPI application entry point.

Run from the `backend` directory:

    python -m uvicorn main:app --reload --port 8000

The API serves six routes under /api (docs/API_CONTRACT.md). Nothing else is
mounted: clip bytes go out through `/api/analysis/{id}/clips/{name}`, so the
data directory is never exposed as a static tree.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import DATA_DIR, RUNS_DIR, ensure_base_dirs
from app.core.constants import PIPELINE_VERSION, SCHEMA_VERSION
from app.core.logging import configure_logging

configure_logging()
log = logging.getLogger("cricketunified")

ensure_base_dirs()

app = FastAPI(
    title="Cricket Video Analysis",
    version=PIPELINE_VERSION,
    description=(
        "Single-pass cricket analysis: one decode and one YOLO+Kalman pass per "
        "uploaded video, producing delivery clips, stored trajectories, bowling "
        "analytics, shot labels and overlays."
    ),
)

# The frontend is served from a different origin (vite dev server on :8080), so
# CORS has to be permissive for the API to be reachable at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
def on_startup() -> None:
    log.info("Cricket Video Analysis %s (schema %s) starting", PIPELINE_VERSION, SCHEMA_VERSION)
    log.info("Data directory: %s", DATA_DIR)
    log.info("Runs directory: %s", RUNS_DIR)


@app.on_event("shutdown")
def on_shutdown() -> None:
    log.info("Cricket Video Analysis shutting down")