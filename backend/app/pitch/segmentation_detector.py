"""
segmentation_detector.py — the local pitch-SEGMENTATION detector (YOLO11s-seg).

A local ultralytics model that answers with a MASK over the complete
wicket-to-wicket pitch. The mask boundary is simplified to a four-corner quad
by `pitch/mask_geometry.py`, and the quad becomes the four named keypoints
`PitchService` consumes. Nothing here calls a remote API or reads a
credential; the weights are the shipped backend/models/pitch_seg_yolo11s.pt.

WHAT THIS OUTPUT IS, AND WHAT IT IS NOT
---------------------------------------
It is a detection of where the pitch is in this frame, in source pixels,
with the mask's own confidence. It is NOT a measurement of the ground: the
source stamped on every detection is `pitch_segmentation`, and
`analytics/pitch_geometry.py` refuses any source other than
`pitch_keypoints`, so a segmentation quad can never produce a speed, a
length, a line or a swing, and can never make an analysis
`geometry_calibrated`. It feeds the pitch block and the frontend's drawn
boundary only.

THREE RULES THIS MODULE OBEYS
-----------------------------
1. The frame goes to the model at its ORIGINAL resolution. Ultralytics
   letterboxes it to `imgsz` itself (same rule as local_box_detector.py).
2. Every instance the model returns is either ACCEPTED or REJECTED WITH A
   REASON — low confidence, implausible area, absurd aspect, unshapable
   quad, wrong class — and `stats()` reports the tally. Nothing is dropped
   silently.
3. `detect()` raises only the errors `PitchService` knows how to file:
   `PitchDetectorUnavailable` when the weights cannot load (submissions are
   futile; the service disables itself), `PitchDetectError` for everything
   else (a content answer the scene could not support, or a soft failure).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from ..core.config import PitchConfig, PitchSegConfig
from ..core.constants import (
    PITCH_DETECTION_SOURCE_SEGMENTATION,
    PITCH_MAX_LOST_FRAMES,
    PITCH_SEG_CONF_TIE_BAND,
    PITCH_SEG_CONSISTENCY_WINDOW,
    PITCH_SMOOTHING_ALPHA,
)
from .detector import Keypoint, PitchDetectError, PitchDetection, PitchDetectorUnavailable
from .mask_geometry import (
    corners_as_named,
    mask_centroid,
    mask_polygon_area_fraction,
    quad_aspect_ratio,
    quad_from_mask,
    validate_mask,
)

log = logging.getLogger(__name__)

Size = tuple[int, int]

# How many accepted predictions the detector remembers for diagnostics and
# for the consistency tie-break. A LOOKBACK, not a log: at every_n=5 this
# covers 1,280 source frames (~51 s at 25 fps).
PITCH_SEG_HISTORY = 256
# Latency samples kept for the percentile report. Bounded so a nine-minute
# video costs one float per sample, never an unbounded list.
PITCH_SEG_LATENCY_CAP = 4096

# Ultralytics model cache, keyed by the RESOLVED weights path. Same idea as
# local_box_detector's cache: two analyses in one server process pay the load
# cost once, and the weights file is never copied or renamed.
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


# ── One accepted (or rejected) prediction ───────────────────────────────────
@dataclass(frozen=True)
class PitchSegPrediction:
    """One mask quad, in ORIGINAL frame pixels, with its provenance."""

    frame_index: int
    conf: float
    polygon: tuple[tuple[float, float], ...]   # raw mask boundary
    corners: dict[str, tuple[float, float]]    # top_left … bottom_left
    quad_method: str                           # approx_polydp | hull_reduce | direct_hull
    area_fraction: float
    aspect_ratio: float
    center: tuple[float, float]
    instance_count: int

    def as_dict(self, *, include_polygon: bool = False) -> dict:
        out = {
            "frame_index": self.frame_index,
            "conf": round(self.conf, 4),
            "corners": {k: [round(v[0], 2), round(v[1], 2)] for k, v in self.corners.items()},
            "quad_method": self.quad_method,
            "area_fraction": round(self.area_fraction, 4),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "instance_count": self.instance_count,
        }
        if include_polygon:
            out["polygon"] = [[round(x, 1), round(y, 1)] for x, y in self.polygon]
        return out


def resolve_seg_device(prefer: str = "") -> str:
    """
    The device inference actually runs on: CUDA when available (or asked for),
    else CPU — never a crash on a machine without a GPU.

    An explicit `cpu` is always honoured. An explicit `cuda*` that cannot be
    honoured falls back to CPU with a warning, because a segmentation
    detector that refuses to run is worse than a slow one.
    """
    want = (prefer or "").strip().lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — torch missing or broken: CPU it is
        cuda = False
    if want.startswith("cuda"):
        if cuda:
            return want
        log.warning(
            "pitch segmentation: CUDA requested (%s) but unavailable; using CPU", want
        )
        return "cpu"
    return "cuda" if cuda else "cpu"


def load_pitch_seg_model(model_path: str, imgsz: int = 640, device: str = "cpu"):
    """
    Load (or reuse) the YOLO weights at `model_path`, warmed up once.

    The warm-up costs a few seconds the first time a process sees the model
    (CUDA context + graph setup) and would otherwise land on the first frame
    the service submits. It is done here, under the cache lock, so it happens
    exactly once per process — and any warm-up failure is logged and
    swallowed: a model that cannot warm up can still be asked to detect, and
    the per-call error path will report it then.
    """
    import numpy as np

    key = str(Path(model_path).resolve())
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            return model
        from ultralytics import YOLO

        log.info("Loading local pitch-segmentation weights: %s", key)
        model = YOLO(key)
        try:
            model.predict(
                source=np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                conf=0.0, imgsz=imgsz, device=device, verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 — warm-up is best effort
            log.warning("Pitch-segmentation model warm-up failed (continuing): %s", exc)
        _MODEL_CACHE[key] = model
        return model


class LocalPitchSegDetector:
    """
    One frame in, one `PitchDetection` (four mask-derived keypoints) out.

    Implements the `PitchService` detector interface: `detect(frame,
    frame_number, image_size)`. `model_path` must already be resolved — a bare
    filename is resolved against `backend/models/` by
    `PitchSegConfig.model_path`, and this class never copies, renames or
    downloads weights.
    """

    # PitchService reads this to notice a timeout-driven disable. A local
    # model has no such path, so it is permanently None.
    disabled_reason: Optional[str] = None

    def __init__(
        self,
        model_path: str,
        *,
        conf: float = 0.5,
        imgsz: int = 640,
        min_area_fraction: float = 0.01,
        max_area_fraction: float = 0.25,
        min_aspect_ratio: float = 1.0,
        max_aspect_ratio: float = 6.0,
        device: str = "",
    ) -> None:
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.min_area_fraction = float(min_area_fraction)
        self.max_area_fraction = float(max_area_fraction)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.device = resolve_seg_device(device)
        self._model = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

        self.samples = 0
        self.accepted = 0
        self.no_mask = 0
        self.errors = 0
        self.rejected: dict[str, int] = {}
        self.last_error: Optional[str] = None
        self.last_prediction: Optional[PitchSegPrediction] = None
        self._history: deque[PitchSegPrediction] = deque(maxlen=PITCH_SEG_HISTORY)
        self._latency_ms: deque[float] = deque(maxlen=PITCH_SEG_LATENCY_CAP)

    @classmethod
    def from_config(cls, cfg: PitchSegConfig) -> "LocalPitchSegDetector":
        """Build from the environment's configuration. Weights must exist."""
        path = cfg.model_path
        if path is None:
            raise PitchDetectorUnavailable("pitch segmentation is not configured")
        return cls(
            str(path),
            conf=cfg.conf,
            imgsz=cfg.imgsz,
            min_area_fraction=cfg.min_area_fraction,
            max_area_fraction=cfg.max_area_fraction,
            min_aspect_ratio=cfg.min_aspect_ratio,
            max_aspect_ratio=cfg.max_aspect_ratio,
            device=cfg.device,
        )

    # ── The PitchService interface ───────────────────────────────────────────
    def detect(self, frame, frame_number: int, image_size: Size) -> PitchDetection:
        """
        One inference; the best VALID mask quad in `frame`, as four keypoints.

        Raises `PitchDetectorUnavailable` when the weights cannot load (the
        service then stops submitting), and `PitchDetectError` for a soft
        failure or a frame with no usable pitch mask. Every outcome is
        counted before it leaves this method.
        """
        with self._lock:
            self.samples += 1
        started = time.perf_counter()
        try:
            model = self._model_or_error()
            results_list = model.predict(
                source=frame, conf=self.conf, imgsz=self.imgsz,
                device=self.device, verbose=False,
            )
        except PitchDetectorUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 — a soft failure, always
            with self._lock:
                self.errors += 1
                self.last_error = str(exc)[:300]
            raise PitchDetectError(f"pitch segmentation inference failed: {exc}") from exc
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._latency_ms.append(elapsed)

        results = results_list[0] if results_list else None
        candidates, rejections = self._candidates(results, image_size, frame_number)
        if not candidates:
            with self._lock:
                self.no_mask += 1
                for reason, count in rejections.items():
                    self.rejected[reason] = self.rejected.get(reason, 0) + count
            reason_text = _reason_text(rejections)
            raise PitchDetectError(f"no keypoints: {reason_text}")

        best = self._select(candidates, frame_number)
        keypoints = [
            Keypoint(name, xy[0], xy[1], best.conf)
            for name, xy in best.corners.items()
        ]
        prediction = PitchSegPrediction(
            frame_index=frame_number,
            conf=best.conf,
            polygon=tuple(best.polygon),
            corners=best.corners,
            quad_method=best.quad_method,
            area_fraction=best.area_fraction,
            aspect_ratio=best.aspect_ratio,
            center=best.center,
            instance_count=best.instance_count,
        )
        with self._lock:
            self.accepted += 1
            for reason, count in rejections.items():
                self.rejected[reason] = self.rejected.get(reason, 0) + count
            self._history.append(prediction)
            self.last_prediction = prediction
        return PitchDetection(
            keypoints=keypoints,
            frame_number=frame_number,
            image_size=image_size,
            confidence=round(best.conf, 4),
            source=PITCH_DETECTION_SOURCE_SEGMENTATION,
        )

    # ── Diagnostics ──────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """
        The tally. `samples == accepted + no_mask + errors`, always: every
        frame handed to the detector is accounted for exactly once.
        """
        with self._lock:
            latency = _latency_stats(list(self._latency_ms))
            return {
                "enabled": True,
                "model_path": self.model_path,
                "device": self.device,
                "conf": self.conf,
                "imgsz": self.imgsz,
                "samples": self.samples,
                "accepted": self.accepted,
                "no_mask": self.no_mask,
                "errors": self.errors,
                "rejected": dict(sorted(
                    self.rejected.items(), key=lambda kv: (-kv[1], kv[0])
                )),
                "last_error": self.last_error,
                "latency_ms": latency,
                "last_prediction": (
                    self.last_prediction.as_dict() if self.last_prediction else None
                ),
            }

    def history(self) -> list[dict]:
        """Accepted predictions, newest last, with their mask polygons."""
        with self._lock:
            return [p.as_dict(include_polygon=True) for p in self._history]

    # ── Internals ────────────────────────────────────────────────────────────
    def _model_or_error(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise PitchDetectorUnavailable(self._load_error)
        try:
            self._model = load_pitch_seg_model(self.model_path, self.imgsz, self.device)
        except Exception as exc:  # noqa: BLE001
            # Cached as an error so every later call fails the same, cheap way
            # instead of retrying a load that is not going to succeed.
            self._load_error = f"pitch-segmentation weights unavailable: {exc}"
            raise PitchDetectorUnavailable(self._load_error) from exc
        return self._model

    def _candidates(self, results, image_size: Size, frame_number: int):
        """
        Every VALID instance quad, plus the per-instance rejection tally.

        A frame with several pitch-shaped regions keeps them all until
        `_select` chooses; a frame whose only region is a crowd is refused
        with that region's reason rather than silently answering "nothing".
        """
        candidates: list[_Candidate] = []
        rejections: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejections[reason] = rejections.get(reason, 0) + 1

        if results is None:
            return candidates, rejections
        masks = getattr(results, "masks", None)
        polygons = getattr(masks, "xy", None) if masks is not None else None
        if not polygons:
            return candidates, rejections

        boxes = getattr(results, "boxes", None)
        confs = getattr(boxes, "conf", None) if boxes is not None else None
        clss = getattr(boxes, "cls", None) if boxes is not None else None

        instance_count = len(polygons)
        for i, poly in enumerate(polygons):
            conf = _float_at(confs, i)
            if conf is None or conf < self.conf:
                reject("below_confidence")
                continue
            cls_id = _int_at(clss, i)
            if cls_id is not None and cls_id != 0:
                reject("wrong_class")
                continue
            points = _as_points(poly)
            ok, reason = validate_mask(
                points, image_size,
                min_area_fraction=self.min_area_fraction,
                max_area_fraction=self.max_area_fraction,
                min_aspect_ratio=self.min_aspect_ratio,
                max_aspect_ratio=self.max_aspect_ratio,
            )
            if not ok:
                reject(reason)
                continue
            quad, method = quad_from_mask(points)
            corners = corners_as_named(quad) if quad is not None else None
            if corners is None:
                reject("quad_unavailable")
                continue
            center = mask_centroid(points)
            if center is None:
                reject("quad_unavailable")
                continue
            candidates.append(_Candidate(
                frame_number=frame_number,
                conf=float(conf),
                polygon=points,
                corners=corners,
                quad_method=method,
                area_fraction=_area_fraction(points, image_size),
                aspect_ratio=quad_aspect_ratio(points),
                center=center,
                instance_count=instance_count,
            ))
        return candidates, rejections

    def _select(self, candidates: Sequence["_Candidate"], frame_number: int) -> "_Candidate":
        """
        The best VALID candidate: highest confidence, ties broken by
        consistency with the previous accepted estimate.

        Consistency only decides a TIE (confidences within
        PITCH_SEG_CONF_TIE_BAND): a 0.9 mask somewhere new beats a 0.88 mask
        next door, because confidence is the model's own evidence and
        position is only ours. But when two regions are equally believable —
        two pitch-shaped patches under a split screen — the one where the
        pitch has been is the honest choice.
        """
        best_conf = max(c.conf for c in candidates)
        contenders = [
            c for c in candidates if c.conf >= best_conf - PITCH_SEG_CONF_TIE_BAND
        ]
        if len(contenders) == 1:
            return contenders[0]
        with self._lock:
            previous = self.last_prediction
        if (
            previous is not None
            and frame_number - previous.frame_index <= PITCH_SEG_CONSISTENCY_WINDOW
        ):
            px, py = previous.center
            return min(
                contenders,
                key=lambda c: (c.center[0] - px) ** 2 + (c.center[1] - py) ** 2,
            )
        return max(contenders, key=lambda c: c.conf)


@dataclass(frozen=True)
class _Candidate:
    """One validated mask quad, awaiting selection."""

    frame_number: int
    conf: float
    polygon: list[tuple[float, float]]
    corners: dict[str, tuple[float, float]]
    quad_method: str
    area_fraction: float
    aspect_ratio: float
    center: tuple[float, float]
    instance_count: int


# ── Service wiring ──────────────────────────────────────────────────────────
def seg_pitch_config(cfg: PitchSegConfig) -> PitchConfig:
    """
    The `PitchConfig` the state machine runs under when the mask detector is
    the pitch-detection source.

    Remote everything is off (this path never touches a network), and the
    validation floors are the mask's own: a quad needs all four corners, and
    the hull-area floor is the segmentation minimum rather than the keypoint
    one. `model_id` names the actual weights so the pitch block says which
    model produced its corners.
    """
    return PitchConfig(
        api_key="",
        remote_enabled=False,
        api_url="",
        model_id=f"pitch_seg:{Path(cfg.model).name}",
        interval_frames=max(1, cfg.every_n),
        sampling_mode=cfg.sampling_mode,
        min_confidence=cfg.conf,
        min_keypoints=4,
        homography_min_keypoints=4,
        min_area_fraction=cfg.min_area_fraction,
        max_lost_frames=PITCH_MAX_LOST_FRAMES,
        smoothing_alpha=PITCH_SMOOTHING_ALPHA,
    )


def build_pitch_seg_summary(detector: LocalPitchSegDetector, service_summary: dict) -> dict:
    """
    The `pitch_seg_detection` record: detector tally, service filing, and the
    explicit statement that this source may not become metric geometry.

    Configuration and counts only — never a credential, never a raw mask
    polygon beyond the single last accepted prediction.
    """
    diagnostics = service_summary.get("diagnostics") or {}
    summary = detector.stats()
    summary["service"] = {
        "submitted": service_summary.get("submitted"),
        "detections": service_summary.get("detections"),
        "failures": service_summary.get("failures"),
        "refusals": service_summary.get("refusals"),
        "queue_drops": service_summary.get("queue_drops"),
        "categories": diagnostics.get("categories"),
        "refusals_by_reason": diagnostics.get("refusals"),
        "latency_ms": diagnostics.get("latency_ms"),
        "scene_changes": diagnostics.get("scene_changes"),
        "last_scene_cut_frame": diagnostics.get("last_scene_cut_frame"),
        "detector_error": service_summary.get("detector_error"),
    }
    summary["calibration"] = {
        # A segmentation quad MAY become metric geometry when its geometry
        # passes the checks in analytics/pitch_geometry.py (aspect agrees with
        # the ball's travel, homography finite, corners map inside the pitch).
        # The per-delivery verdict and its reason live in the run's
        # `pitch_geometry_stats`, not here: this block states the source's
        # standing, and that standing is now "eligible, checked per delivery".
        "geometry_calibrated": "per_delivery",
        "reason": None,
        "metric_analytics": "subject_to_geometric_validation",
    }
    return summary


def pitch_seg_enabled(cfg) -> bool:
    """Is the feature configured ON (weights named, and not one of `off`)?"""
    enabled = getattr(cfg, "enabled", None)
    if enabled is not None:
        return bool(enabled)
    token = str(getattr(cfg, "model", "")).strip().lower()
    return bool(token) and token not in {"off", "none", "disabled", "0", "false"}


def describe_pitch_seg(cfg) -> str:
    """The startup line: ON/OFF, and when ON the resolved path, size, device."""
    if not pitch_seg_enabled(cfg):
        return (
            "pitch segmentation detector: OFF "
            f"(CRICKET_PITCH_SEG_MODEL={getattr(cfg, 'model', '')!r})"
        )
    path = cfg.model_path
    if path is None:
        return (
            f"pitch segmentation detector: ON but unresolved "
            f"(CRICKET_PITCH_SEG_MODEL={cfg.model!r})"
        )
    if not Path(path).is_file():
        return (
            f"pitch segmentation detector: ON but weights NOT FOUND at {path} "
            f"(CRICKET_PITCH_SEG_MODEL={cfg.model!r})"
        )
    size = Path(path).stat().st_size
    return (
        f"pitch segmentation detector: ON — {path} "
        f"({size / 1048576:.1f} MiB, conf={cfg.conf}, every_n={cfg.every_n}, "
        f"imgsz={cfg.imgsz})"
    )


# ── helpers ─────────────────────────────────────────────────────────────────
def _as_points(poly: Any) -> list[tuple[float, float]]:
    """An (N, 2) array/list of mask vertices, or a best-effort partial read."""
    try:
        import numpy as np
        arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in arr.tolist()]
    except Exception:  # noqa: BLE001 — malformed polygon is a refusal, not a crash
        try:
            return [(float(p[0]), float(p[1])) for p in poly]
        except Exception:  # noqa: BLE001
            return []


def _float_at(values: Any, index: int) -> Optional[float]:
    if values is None:
        return None
    try:
        return float(values[index])
    except Exception:  # noqa: BLE001
        return None


def _int_at(values: Any, index: int) -> Optional[int]:
    if values is None:
        return None
    try:
        return int(values[index])
    except Exception:  # noqa: BLE001
        return None


def _area_fraction(points: Sequence[tuple[float, float]], image_size: Size) -> float:
    return mask_polygon_area_fraction(points, image_size)


def _reason_text(rejections: dict[str, int]) -> str:
    if not rejections:
        return "model returned no masks"
    parts = [f"{count}x {reason}" for reason, count in sorted(rejections.items())]
    return "no usable pitch mask (" + ", ".join(parts) + ")"


def _latency_stats(values: list[float]) -> dict:
    """Count, mean and percentiles of the detector's own inference latency."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def quantile(p: float) -> float:
        return round(ordered[int(round(p * (len(ordered) - 1)))], 2)

    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 2),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "max": round(ordered[-1], 2),
    }


__all__ = [
    "LocalPitchSegDetector",
    "PitchSegPrediction",
    "build_pitch_seg_summary",
    "describe_pitch_seg",
    "load_pitch_seg_model",
    "pitch_seg_enabled",
    "resolve_seg_device",
    "seg_pitch_config",
]
