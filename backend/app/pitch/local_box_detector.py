"""
local_box_detector.py — OPTIONAL local pitch-box detector (ultralytics YOLO).

Strictly auxiliary, and OFF unless `CRICKET_PITCH_BOX_MODEL` is set (see
core/config.py -> PitchBoxConfig). It answers one question the Roboflow
keypoint model does not: "is there a pitch-shaped region in this frame, as a
BOX?". The box is then turned into four APPROXIMATE corners by
`pitch/box_corners.py`. Those corners are GUESSES, not measurements: they never
become `pitch.corners`, never feed a homography, and can never make an analysis
`geometry_calibrated`.

THREE RULES THIS MODULE OBEYS
-----------------------------
1. The frame goes to the model at its ORIGINAL resolution. Ultralytics
   letterboxes it to `imgsz` itself. Pre-resizing to a fixed 640x360 first is
   what squashed frames and broke the ball detector's aspect ratio; the same
   mistake is not repeated here.
2. `detect()` never raises and never blocks the caller for longer than one
   inference. Every failure is logged, counted, and answered with None — the
   analysis carries on exactly as it would have.
3. Nothing is dropped silently. Every call increments exactly one of
   frames_seen → {boxes, no_box, errors}, and `stats()` reports the tally.

WHERE THE SLOW PART RUNS
------------------------
One inference costs ~15-250 ms on this machine (measured on real_cricket.mp4,
CUDA). That is too much to pay inline in the tracking loop at any useful rate,
so the loop only OFFERS frames: `PitchBoxSlot` holds a one-deep, latest-wins
queue drained by a single daemon worker thread. An offered frame the worker has
not reached yet is counted as `evicted`, never hidden.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .box_corners import box_to_corners, corners_as_dict, corners_valid

log = logging.getLogger(__name__)

# Ultralytics' own model cache, keyed by the RESOLVED path. There is no
# pre-existing YOLO cache in this project to reuse — `tracking/detector.py`
# builds a fresh `YOLO(...)` per BallDetector — so this one is introduced here
# and scoped to this module: two analyses in one server process pay the load
# cost once, and the weights file is never copied or renamed.
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PitchBox:
    """One detected pitch box, in ORIGINAL frame pixels."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    def as_list(self) -> list[float]:
        return [round(self.x1, 1), round(self.y1, 1),
                round(self.x2, 1), round(self.y2, 1)]


def load_pitch_box_model(model_path: str, imgsz: int = 640):
    """
    Load (or reuse) the YOLO weights at `model_path`, warmed up once.

    The warm-up costs a few seconds the first time a process sees the model
    (CUDA context + graph setup, measured 3.0-4.4 s on this machine) and would
    otherwise land on the first frame the loop offers. It is done here, under
    the cache lock, so it happens exactly once per process — and any warm-up
    failure is logged and swallowed: a detector that cannot warm up can still
    be asked to detect, and the per-call error path will report it then.
    """
    import numpy as np

    key = str(Path(model_path).resolve())
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            return model
        from ultralytics import YOLO

        log.info("Loading local pitch-box weights: %s", key)
        model = YOLO(key)
        try:
            model.predict(
                source=np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                conf=0.0, imgsz=imgsz, verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 — warm-up is best effort
            log.warning("Pitch-box model warm-up failed (continuing): %s", exc)
        _MODEL_CACHE[key] = model
        return model


class LocalPitchBoxDetector:
    """
    One frame in, one `PitchBox` (or None) out.

    `model_path` must already be resolved — a bare filename is resolved against
    `backend/models/` by `PitchBoxConfig.model_path`, and this class never
    copies, renames or downloads weights.
    """

    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 640) -> None:
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self._model = None
        self._load_error: Optional[str] = None

        self.frames_seen = 0
        self.boxes = 0
        self.no_box = 0
        self.errors = 0
        self.last_error: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, frame_bgr) -> Optional[PitchBox]:
        """
        Most confident pitch box in `frame_bgr`, or None. Never raises.

        The frame is handed to the model untouched (letterboxed by ultralytics
        to `imgsz`), and only the single highest-confidence box survives.
        """
        self.frames_seen += 1
        try:
            model = self._model_or_error()
            results = model.predict(
                source=frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False
            )[0]
            box = self._best_box(results)
        except Exception as exc:  # noqa: BLE001 — a soft failure, always
            self.errors += 1
            self.last_error = str(exc)[:300]
            log.warning("Local pitch-box detection failed: %s", self.last_error)
            return None

        if box is None:
            self.no_box += 1
        else:
            self.boxes += 1
        return box

    def stats(self) -> dict:
        """The tally. `frames_seen == boxes + no_box + errors`, always."""
        return {
            "model_path": self.model_path,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "frames_seen": self.frames_seen,
            "boxes": self.boxes,
            "no_box": self.no_box,
            "errors": self.errors,
            "last_error": self.last_error,
        }

    # ── Internals ─────────────────────────────────────────────────────────────
    def _model_or_error(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        try:
            self._model = load_pitch_box_model(self.model_path, self.imgsz)
        except Exception as exc:  # noqa: BLE001
            # Cached as an error so every later call fails the same, cheap way
            # instead of retrying a load that is not going to succeed.
            self._load_error = f"pitch-box weights unavailable: {exc}"
            raise RuntimeError(self._load_error) from exc
        return self._model

    @staticmethod
    def _best_box(results) -> Optional[PitchBox]:
        """The single most confident box, in original frame pixels."""
        boxes = getattr(results, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        best = int(boxes.conf.argmax())
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[best].tolist()]
        return PitchBox(x1, y1, x2, y2, float(boxes.conf[best]))


class PitchBoxSlot:
    """
    The loop's one call site: offer a frame, never wait, never raise.

    Every `every_n`-th frame is offered; the worker thread runs the detector on
    whichever offered frame is current, keeping only the LATEST finished
    estimate. `latest_before(frame)` then answers, for a delivery being
    finalised, "what did the box detector most recently say about a frame at or
    before this one?" — never about a frame in the future, because a delivery
    emitted late (merge deferral) must not be handed an estimate measured after
    its own window ended.
    """

    def __init__(self, detector: LocalPitchBoxDetector, *, every_n: int = 15,
                 inset: float = 0.08) -> None:
        self.detector = detector
        self.every_n = max(1, int(every_n))
        self.inset = float(inset)
        self.offered = 0
        self.evicted = 0

        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._stopped = False
        self._thread = threading.Thread(
            target=self._worker, name="pitch-box-detector", daemon=True
        )
        self._thread.start()

    # ── The loop's call ───────────────────────────────────────────────────────
    def offer(self, frame_number: int, frame) -> None:
        """Offer every `every_n`-th frame. Non-blocking; a stale queued frame
        is evicted (and counted) rather than ever making the loop wait."""
        if self._stopped or frame_number % self.every_n != 0:
            return
        self.offered += 1
        item = (frame_number, frame)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self.evicted += 1
                self._queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                self.evicted += 1

    def latest_before(self, max_frame: Optional[int]) -> Optional[dict]:
        """A copy of the newest estimate measured at or before `max_frame`."""
        with self._lock:
            est = self._latest
        if est is None:
            return None
        if max_frame is not None and est["frame_index"] > int(max_frame):
            return None
        return {
            **est,
            "corners": dict(est["corners"]),
            "box": list(est["box"]),
        }

    def summary(self) -> dict:
        """Detector tally plus the slot's own offered/evicted counts."""
        summary = dict(self.detector.stats())
        summary.update({
            "enabled": True,
            "every_n": self.every_n,
            "inset": self.inset,
            "offered": self.offered,
            "evicted": self.evicted,
        })
        return summary

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the worker to finish what it holds and join it. Bounded: a hung
        inference is abandoned (the thread is a daemon), never waited on."""
        self._stopped = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    # ── Worker ────────────────────────────────────────────────────────────────
    def _worker(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stopped:
                    return
                continue
            if item is None:
                return
            frame_number, frame = item
            try:
                box = self.detector.detect(frame)
            except Exception as exc:  # noqa: BLE001 — detect() swallows already
                log.warning("Pitch-box worker error: %s", exc)
                box = None
            if box is not None:
                self._record(frame_number, box)
            if self._stopped and self._queue.empty():
                return

    def _record(self, frame_number: int, box: PitchBox) -> None:
        """Box -> four approximate corners -> the stored estimate."""
        corners = box_to_corners(
            box.x1, box.y1, box.x2, box.y2, top_inset=self.inset
        )
        valid, reason = corners_valid(corners)
        estimate = {
            "corners": corners_as_dict(corners),
            "corners_source": "box_estimate",
            "corners_valid": bool(valid),
            "corners_reason": reason,
            "box": box.as_list(),
            "box_conf": round(float(box.conf), 4),
            "inset": self.inset,
            "frame_index": int(frame_number),
        }
        with self._lock:
            self._latest = estimate


def pitch_box_enabled(cfg) -> bool:
    """Is the feature configured ON (a model was named)?"""
    return bool(getattr(cfg, "model", ""))


def describe_pitch_box(cfg) -> str:
    """The startup line: ON/OFF, and when ON the resolved path and its size."""
    if not pitch_box_enabled(cfg):
        return "pitch box detector: OFF (CRICKET_PITCH_BOX_MODEL not set)"
    path = cfg.model_path
    if path is None:
        return f"pitch box detector: ON but unresolved (CRICKET_PITCH_BOX_MODEL={cfg.model!r})"
    if not Path(path).is_file():
        return (
            f"pitch box detector: ON but weights NOT FOUND at {path} "
            f"(CRICKET_PITCH_BOX_MODEL={cfg.model!r})"
        )
    size = Path(path).stat().st_size
    return (
        f"pitch box detector: ON — {path} "
        f"({size / 1048576:.1f} MiB, conf={cfg.conf}, inset={cfg.inset}, "
        f"every_n={cfg.every_n})"
    )


__all__ = [
    "LocalPitchBoxDetector",
    "PitchBox",
    "PitchBoxSlot",
    "describe_pitch_box",
    "load_pitch_box_model",
    "pitch_box_enabled",
]
