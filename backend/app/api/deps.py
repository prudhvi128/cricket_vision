"""
deps.py — Shared dependencies and the per-analysis job registry.

WHY A REGISTRY RATHER THAN A GLOBAL
-----------------------------------
CricketTracker's `main.py` keeps `processing_status` as a module-level global,
so two concurrent uploads overwrite each other's progress and one can cancel the
other. State here is keyed by `analysis_id`, so uploads are isolated.

Jobs run on a background thread because a full-match analysis is minutes of
GPU-bound work; doing it inline would block every other request.

MODEL CACHING
-------------
The detector and shot model are process-wide singletons. Loading a 72 MB
checkpoint per request would dominate the runtime of a 40-ball video.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..core.config import Paths, ensure_base_dirs
from ..core.constants import STAGE_PROGRESS_BANDS
from ..pipeline.analysis_pipeline import UnifiedPipeline
from ..tracking.tracking_service import ProgressFn
from ..tracking.detector import BallDetector
from ..storage.persistence import read_json

log = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ProgressEvent:
    at: str
    phase: str
    frames_done: int
    frames_total: int
    detail: str
    percent: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "at": self.at,
            "phase": self.phase,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "percent": self.percent,
            "detail": self.detail,
        }


@dataclass
class Job:
    analysis_id: str
    status: str = "queued"          # queued|running|completed|failed|cancelled
    source_name: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    phase: str = "queued"
    frames_done: int = 0
    frames_total: int = 0
    percent: Optional[float] = None
    detail: str = ""
    events: list[ProgressEvent] = field(default_factory=list)
    error: Optional[str] = None
    cancel_requested: bool = False
    # ── Progress accounting ──────────────────────────────────────────────────
    # `percent` is a MONOTONIC function of the current stage's own measured unit
    # count mapped through STAGE_PROGRESS_BANDS. Two properties matter and both
    # are enforced in `_apply_progress`:
    #
    #   * it never moves backwards, so a client that only draws a bar cannot show
    #     the analysis going "back"; and
    #   * it never claims to be further along than the work actually is, because
    #     every stage's numerator is a real count (bytes written, frames decoded,
    #     deliveries classified) rather than an assumed fraction of time.
    #
    # `_high_water` is the mechanism for the first property.
    _high_water: float = 0.0
    # Which sequential stage the pipeline is in. Distinct from `phase`, which is
    # whatever the stage last reported (see PIPELINE_STAGES).
    stage: str = "queued"
    # Free-form activity label for work interleaved inside the tracking pass
    # (segmentation, per-delivery analytics, clip writing). Never used to compute
    # percent.
    substage: Optional[str] = None
    # Real unit counts per stage, so a client can show "1,204 / 13,565 frames"
    # instead of only a rounded percentage.
    units: dict = field(default_factory=dict)

    def as_dict(self, include_events: bool = False) -> dict:
        d = {
            "analysis_id": self.analysis_id,
            "status": self.status,
            "source_name": self.source_name,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stage": self.stage,
            "substage": self.substage,
            "phase": self.phase,
            "progress": self.percent,
            "units": dict(self.units),
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            # Kept for backwards compatibility with the existing frontend, which
            # reads `percent`. Identical value to `progress`.
            "percent": self.percent,
            "detail": self.detail,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }
        if include_events:
            d["events"] = [e.as_dict() for e in self.events]
        return d


# Which of the pipeline's emitted phases map onto which public stage.
#
# The pipeline emits richer phase names than the API publishes (`opening`,
# `tracking_complete`, `shot`, `models`). The API is the contract, so it collapses
# them here rather than leaking internal vocabulary to clients. Anything unmapped
# keeps the current stage — a client polling during a transition still sees a
# valid, if slightly stale, stage rather than an unknown string.
_PHASE_TO_STAGE = {
    "uploading": "uploading",
    "opening": "tracking",
    "tracking": "tracking",
    "tracking_complete": "validation",
    "models": "tracking",
    "shot": "shot_classification",
    "overlay": "overlay",
    "persist": "persisting",
    "complete": "completed",
}

# Phases that describe interleaved activity rather than a top-level stage. These
# become `substage` so the honest activity is visible without inventing a phase
# boundary that the single-pass architecture does not have.
_PHASE_TO_SUBSTAGE = {
    "delivery_finalise": "analytics",
    "purity_validation": "validation",
    "segmentation": "segmentation",
    "clip_write": "clip_writing",
}


def weighted_progress(stage: str, done: int, total: int) -> Optional[float]:
    """
    Map a stage's real unit count onto 0..100 using its progress band.

    Returns None when `total` is unknown, because a percentage of an unknown
    denominator is a guess. The caller reports `units` instead in that case.
    """
    band = STAGE_PROGRESS_BANDS.get(stage)
    if band is None:
        return None
    low, high = band
    if total <= 0:
        # Stage entered, denominator not yet known. Report the START of the band,
        # which is the honest floor for "this stage has begun".
        return round(low, 1)
    fraction = max(0.0, min(1.0, done / total))
    return round(low + (high - low) * fraction, 1)


class ModelCache:
    """Process-wide, lock-guarded model singletons."""

    def __init__(self) -> None:
        self._detector: Optional[BallDetector] = None
        self._shot_model: Any = None
        # RLock, not Lock: `pipeline()` holds this lock while calling
        # `detector()`, which acquires it again. A plain Lock would self-deadlock
        # on the first request.
        self._lock = threading.RLock()

    def detector(self) -> BallDetector:
        with self._lock:
            if self._detector is None or self._detector.model is None:
                self._detector = BallDetector()
            return self._detector

    def pipeline(self) -> UnifiedPipeline:
        """Build a pipeline bound to the cached models."""
        with self._lock:
            return UnifiedPipeline(
                detector=self.detector(), shot_model=self._shot_model
            )

    def note_shot_model(self, model: Any) -> None:
        """Record the shot model once UnifiedPipeline has loaded it."""
        if model is None:
            return
        with self._lock:
            if self._shot_model is None:
                self._shot_model = model


class JobRegistry:
    """
    In-memory job table keyed by analysis_id.

    Deliberately in-memory: analysis results are persisted to disk as
    `result.json` / `tracking.json`, so the registry only needs to hold
    *progress*, which is transient by nature.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # ── CRUD ─────────────────────────────────────────────────────────────────
    def create(self, source_name: str) -> Job:
        job = Job(
            analysis_id=uuid.uuid4().hex[:16],
            source_name=source_name,
        )
        with self._lock:
            self._jobs[job.analysis_id] = job
        return job

    def get(self, analysis_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(analysis_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    # ── Progress ─────────────────────────────────────────────────────────────
    def progress_cb(self, analysis_id: str) -> ProgressFn:
        """
        The progress callback the pipeline calls on every phase change.

        Translates the pipeline's internal phase names into the public stage
        vocabulary and computes percent from measured units, monotonically. Every
        assignment happens under the registry lock, so a client polling
        `as_dict()` can never observe a half-updated job.
        """
        job = self.get(analysis_id)
        if job is None:
            return lambda *_: None

        def _cb(phase: str, done: int, total: int, detail: str) -> None:
            with self._lock:
                substage = _PHASE_TO_SUBSTAGE.get(phase)
                if substage:
                    # Interleaved activity inside the tracking pass. The top-level
                    # stage does not move, because no phase boundary was crossed.
                    job.substage = substage
                    job.phase = substage
                    job.detail = detail
                    job.events.append(
                        ProgressEvent(
                            at=utc_now_iso(),
                            phase=substage,
                            frames_done=job.frames_done,
                            frames_total=job.frames_total,
                            detail=detail,
                            percent=job.percent,
                        )
                    )
                    return

                stage = _PHASE_TO_STAGE.get(phase)
                if stage is not None and stage != job.stage:
                    job.stage = stage
                    job.substage = None

                job.phase = stage or job.stage
                job.frames_done = done
                job.frames_total = total
                job.detail = detail

                if job.stage in ("uploading", "tracking", "shot_classification",
                                 "overlay", "validation", "persisting"):
                    job.units[job.stage] = {
                        "done": int(done),
                        "total": int(total),
                    }

                candidate = weighted_progress(job.stage, done, total)
                if candidate is not None:
                    # Monotonic clamp. A stage that reports a smaller fraction
                    # than one already reached (which happens at a phase handoff,
                    # because each stage counts its own units) must not rewind the
                    # bar the client is drawing.
                    if candidate < job._high_water:
                        candidate = job._high_water
                    job._high_water = candidate
                    job.percent = candidate

                job.events.append(
                    ProgressEvent(
                        at=utc_now_iso(),
                        phase=job.phase,
                        frames_done=done,
                        frames_total=total,
                        detail=detail,
                        percent=job.percent,
                    )
                )
                # Bound memory on long videos: progress polling does not need
                # every 0.5s tick retained forever.
                if len(job.events) > 500:
                    del job.events[: len(job.events) - 500]

        return _cb

    # ── Execution ────────────────────────────────────────────────────────────
    def start(self, job: Job, source_path: Path, models: ModelCache) -> threading.Thread:
        paths = Paths(analysis_id=job.analysis_id)

        def _work() -> None:
            job.status = "running"
            job.started_at = utc_now_iso()
            job.stage = "tracking"
            job.units["tracking"] = {"done": 0, "total": 0}
            t0 = time.perf_counter()
            try:
                pipeline = models.pipeline()
                result = pipeline.run(
                    source_video=str(source_path),
                    paths=paths,
                    progress=self.progress_cb(job.analysis_id),
                )
                models.note_shot_model(pipeline.shot_model)
                with self._lock:
                    job.frames_total = len(result.deliveries)
                    job.frames_done = len(result.deliveries)
                    job.detail = (
                        f"{len(result.deliveries)} validated deliveries"
                        + (
                            f", {len(result.quarantined)} quarantined"
                            if result.quarantined else ""
                        )
                        + f" in {time.perf_counter() - t0:.1f}s"
                    )
                    job.error = None
                    if job.cancel_requested:
                        job.status = "cancelled"
                        job.stage = "tracking"
                    else:
                        # Only a genuinely finished pipeline claims 100%. A
                        # cancelled run reports how far it actually got.
                        job.status = "completed"
                        job.stage = "completed"
                        job.phase = "completed"
                        job.substage = None
                        job.units["deliveries"] = {
                            "done": len(result.deliveries),
                            "total": len(result.deliveries),
                        }
                        job._high_water = 100.0
                        job.percent = 100.0
            except Exception as exc:
                log.exception("Analysis %s failed", job.analysis_id)
                job.status = "failed"
                job.stage = "failed"
                job.phase = "failed"
                job.error = str(exc)
                job.detail = f"failed: {exc}"
            finally:
                job.finished_at = utc_now_iso()

        thread = threading.Thread(target=_work, name=f"analysis-{job.analysis_id}", daemon=True)
        with self._lock:
            self._threads[job.analysis_id] = thread
        thread.start()
        return thread

    def cancel(self, analysis_id: str) -> bool:
        """
        Request cancellation.

        The tracking pass checks this flag between deliveries. A cancel
        mid-delivery takes effect once that delivery's clip has been written, so
        a clip is never left half-written with its frame map claiming otherwise.
        """
        job = self.get(analysis_id)
        if job is None or job.status in ("completed", "failed", "cancelled"):
            return False
        job.cancel_requested = True
        return True

    def load_result(self, analysis_id: str) -> Optional[dict]:
        """Read a persisted result from disk. Works for jobs not in memory."""
        return read_json(Paths(analysis_id=analysis_id).result_json)


# Module-level singletons, created lazily so importing this module is cheap and
# does not touch the filesystem.
_models: Optional[ModelCache] = None
_jobs: Optional[JobRegistry] = None
_init_lock = threading.Lock()


def get_models() -> ModelCache:
    global _models
    with _init_lock:
        if _models is None:
            ensure_base_dirs()
            _models = ModelCache()
        return _models


def get_jobs() -> JobRegistry:
    global _jobs
    with _init_lock:
        if _jobs is None:
            _jobs = JobRegistry()
        return _jobs