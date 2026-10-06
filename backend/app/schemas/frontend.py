"""
frontend.py — The clean, frontend-facing view of one analysis result.

ARCHITECTURE (why this file exists)
-----------------------------------
The internal result document (`result.json`) is a DIAGNOSTIC artefact. It keeps
everything the pipeline produced — segmenter evidence and boundaries, purity
validation verdicts, provenance, speed samples, clip frame maps, shot sampling
audit — because a wrong boundary or a suspicious label has to be explainable
without re-running the pass.

The frontend needs none of that. It needs, per delivery: a video to play, a
trajectory to draw, a bounce to mark, the bowling numbers, a shot label, and a
small amount of trust information. Serving the internal document to a browser
leaks filesystem paths, model internals and hundreds of kilobytes of debug data
per analysis.

So the pipeline keeps writing the internal document unchanged, and this module
is the serializer between it and the API:

    INTERNAL PIPELINE DATA
            ↓
    internal analysis object  →  result.json (unchanged, still complete)
            ↓
    serialize_result_document()  ← this module
            ↓
    CLEAN FRONTEND JSON  (a flat ARRAY of deliveries)

Nothing here computes, repairs or invents a measurement. It renames, selects and
drops. A value that does not exist internally comes out as `null`, never as an
estimate: `speed_kmh: null` is a fact about the run, a filled-in number would be
a lie.

THE RESPONSE IS AN ARRAY
------------------------
    [ delivery_1, delivery_2, delivery_3, ... ]

so a client does `deliveries[currentIndex]` for Previous/Next navigation and
gets the video, trajectory, bounce, bowling analytics, shot and quality state of
THAT delivery in one object. No client ever has to reassemble a delivery from
two internal documents.

KEY MAPPING (internal → frontend)
---------------------------------
    clip_path, clip.file_name      → dropped   (filesystem)
    clip.url / clip_url            → video.clip_url   (API/static URL only)
    clip.start_frame/end_frame/…   → video.start_frame/end_frame/fps/…
    trajectory.points[].original_frame → trajectory.points[].frame  (source timebase)
    trajectory.points[].clip_frame     → dropped   (derivable: frame - start + 1)
    trajectory.detected_count/predicted_count/continuity
                                   → trajectory.detected_points/predicted_points/continuity
    bounce.original_frame/x_px/y_px → bounce.frame/x/y
    bowling.*                      → bowling.*  (plus `calibrated`, the pitch
                                    geometry calibration flag)
    shot.type                      → shot.classification
    quality.* + validation.*       → quality.{trajectory_valid, tracking_confidence,
                                    requires_human_review, flags}   (verbose
                                    validation collapsed into concise flags)

Model-independent: everything is driven by `dict.get`, so it works on the
persisted document of any pipeline version, on disk or in memory.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

# Validation verdicts that are worth a flag on the frontend record. SINGLE_EVENT
# is the good case and gets no flag — a flags list that always contains an entry
# cannot be used to spot the deliveries that actually need attention.
_VALIDATION_FLAG = {
    "MULTIPLE_EVENTS": "multiple_events_in_clip",
    "NO_EVENT": "no_event_in_clip",
    "AMBIGUOUS": "ambiguous_event",
}


class _FrontendModel(BaseModel):
    # `forbid` is the whole point of this module: a field that is not declared
    # here cannot reach the browser, however the internal document grows.
    model_config = ConfigDict(extra="forbid")


# ── Response schema ───────────────────────────────────────────────────────────
class VideoOut(_FrontendModel):
    """Everything needed to load and seek the delivery clip. No paths."""

    clip_url: Optional[str] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None


class TrajectoryPointOut(_FrontendModel):
    """
    One ball position.

    `frame` is the SOURCE frame number — the same timebase as
    `video.start_frame`, so a client overlays a point at
    `frame - video.start_frame + 1` inside the clip.

    `source` is never dropped: `detected` is a real detector measurement,
    `predicted` is a Kalman prediction. The distinction is the honesty of the
    whole trajectory, so it travels with every point.
    """

    frame: int
    x: int
    y: int
    source: str
    confidence: Optional[float] = None


class TrajectoryOut(_FrontendModel):
    points: list[TrajectoryPointOut]
    detected_points: int
    predicted_points: int
    continuity: str


class BounceOut(_FrontendModel):
    """
    Bounce location in source pixels.

    `detected: false` means the score-based bounce detector did not fire and the
    coordinates are the real lowest tracked point used as a fallback — they are
    measurements, never invented, and the flag says which they are.

    Ground-plane metres are null unless the pitch corners were measured on this
    video.
    """

    detected: bool
    frame: Optional[int] = None
    x: Optional[int] = None
    y: Optional[int] = None
    ground_x_m: Optional[float] = None
    ground_y_m: Optional[float] = None


class BowlingOut(_FrontendModel):
    """
    Bowling analytics. Null means "could not be computed", not "compute it".

    `calibrated` is the per-video pitch-geometry calibration: when false, line,
    length, swing, bounce angle and speed are unavailable on the ground plane
    and are returned null. Speed is an estimate even when `calibrated` is true
    (Phase 3 decision D2) — it is never presented as a measurement.
    """

    speed_kmh: Optional[float] = None
    line: Optional[str] = None
    length: Optional[str] = None
    swing: Optional[str] = None
    release_angle: Optional[float] = None
    bounce_angle: Optional[float] = None
    calibrated: bool = False


class ShotOut(_FrontendModel):
    """Null/null when classification was not performed or failed."""

    classification: Optional[str] = None
    confidence: Optional[float] = None


class QualityOut(_FrontendModel):
    """
    How much to trust this delivery, in four fields.

    The internal validation verdict (classification, event ranges, margins,
    reason) is collapsed into `flags` — a client shows a badge, it does not need
    the segmenter's working.
    """

    trajectory_valid: bool
    tracking_confidence: Optional[float] = None
    requires_human_review: bool
    flags: list[str]


class DeliveryOut(_FrontendModel):
    """One delivery, complete and self-contained. Previous/Next swaps this."""

    delivery_id: int
    delivery_ref: str
    video: VideoOut
    trajectory: TrajectoryOut
    bounce: BounceOut
    bowling: BowlingOut
    shot: ShotOut
    quality: QualityOut


# ── Serialization ─────────────────────────────────────────────────────────────
def serialize_delivery(delivery: dict, *, analysis_id: str) -> DeliveryOut:
    """
    Project one internal delivery record onto the frontend schema.

    `delivery` is the persisted form (a `result.json` entry) or any dict with
    the same keys. Internal fields are ignored by construction rather than
    filtered: only the fields declared on the models above are read.
    """
    delivery_id = int(delivery["delivery_id"])
    clip = delivery.get("clip") or {}
    quality = delivery.get("quality") or {}
    validation = delivery.get("validation")
    points = (delivery.get("trajectory") or {}).get("points") or []

    return DeliveryOut(
        delivery_id=delivery_id,
        delivery_ref=delivery.get("delivery_ref") or f"delivery_{delivery_id:03d}",
        video=_video(delivery, clip, analysis_id),
        trajectory=_trajectory(delivery, points),
        bounce=_bounce(delivery),
        bowling=_bowling(delivery),
        shot=_shot(delivery),
        quality=_quality(quality, validation, delivery, points),
    )


def serialize_result_document(doc: dict) -> list[DeliveryOut]:
    """
    The whole result document as the array the frontend iterates.

    Order is the document's order (delivery id order), so
    `deliveries[currentIndex ± 1]` is Previous/Next.
    """
    return [
        serialize_delivery(d, analysis_id=str(doc.get("analysis_id") or ""))
        for d in doc.get("deliveries") or []
    ]


# ── Field groups ──────────────────────────────────────────────────────────────
def _video(delivery: dict, clip: dict, analysis_id: str) -> VideoOut:
    start = _int(clip.get("start_frame"))
    end = _int(clip.get("end_frame"))
    fps = _float(clip.get("fps"))
    duration = None
    if start is not None and end is not None and end >= start and fps:
        duration = round((end - start + 1) / fps, 3)
    return VideoOut(
        clip_url=_clip_url(delivery, clip, analysis_id),
        start_frame=start,
        end_frame=end,
        fps=fps,
        width=_int(clip.get("width")),
        height=_int(clip.get("height")),
        duration_seconds=duration,
    )


def _clip_url(delivery: dict, clip: dict, analysis_id: str) -> Optional[str]:
    """
    A URL a browser can fetch, or null. Never a path.

    The stored `clip_url` is normally `/data/runs/<id>/clips/delivery_001.mp4`
    (the static mount). Anything that looks like a filesystem path — which is
    what `clip_path` holds, and what a stale record might hold in `clip_url` —
    is replaced by the stable API route for the same bytes, which needs no
    static mount and cannot leak where the file lives.
    """
    stored = delivery.get("clip_url") or clip.get("url")
    if isinstance(stored, str) and stored and _is_url(stored):
        return stored
    if delivery.get("clip") is not None or delivery.get("clip_path"):
        did = delivery.get("delivery_id")
        if analysis_id and did is not None:
            return f"/api/analysis/{analysis_id}/deliveries/{did}/clip"
    return None


def _is_url(value: str) -> bool:
    return value.startswith(("/", "http://", "https://")) and "\\" not in value


def _trajectory(delivery: dict, points: list) -> TrajectoryOut:
    traj = delivery.get("trajectory") or {}
    clean: list[TrajectoryPointOut] = []
    for p in points:
        if not isinstance(p, dict):
            continue
        # `original_frame` is the source timebase; `frame` is accepted as the
        # spelling a future schema might use. A point with no usable frame or
        # coordinate cannot be drawn, so it is not handed to a client that
        # would draw it in the wrong place.
        frame = _int(p.get("original_frame"))
        if frame is None:
            frame = _int(p.get("frame"))
        x, y = _int(p.get("x")), _int(p.get("y"))
        if frame is None or x is None or y is None:
            continue
        clean.append(
            TrajectoryPointOut(
                frame=frame,
                x=x,
                y=y,
                source=str(p.get("source") or ""),
                confidence=_float(p.get("confidence")),
            )
        )
    detected = sum(1 for p in clean if p.source == "detected")
    predicted = sum(1 for p in clean if p.source == "predicted")
    return TrajectoryOut(
        points=clean,
        # Counted from the points actually emitted, so a client's
        # detected + predicted == points.length always holds.
        detected_points=detected,
        predicted_points=predicted,
        continuity=str(traj.get("continuity") or "continuous"),
    )


def _bounce(delivery: dict) -> BounceOut:
    b = delivery.get("bounce") or {}
    return BounceOut(
        detected=bool(b.get("detected")),
        frame=_int(b.get("original_frame")),
        x=_int(b.get("x_px")),
        y=_int(b.get("y_px")),
        ground_x_m=_float(b.get("ground_x_m")),
        ground_y_m=_float(b.get("ground_y_m")),
    )


def _bowling(delivery: dict) -> BowlingOut:
    b = delivery.get("bowling") or {}
    return BowlingOut(
        speed_kmh=_float(b.get("speed_kmh")),
        line=b.get("line"),
        length=b.get("length"),
        swing=b.get("swing"),
        release_angle=_float(b.get("release_angle")),
        bounce_angle=_float(b.get("bounce_angle")),
        calibrated=bool(b.get("geometry_calibrated")),
    )


def _shot(delivery: dict) -> ShotOut:
    shot = delivery.get("shot")
    if not isinstance(shot, dict) or shot.get("error"):
        # Not classified, or the classifier failed: null/null, never a guess.
        return ShotOut()
    return ShotOut(
        classification=shot.get("type"),
        confidence=_float(shot.get("confidence")),
    )


def _quality(
    quality: dict, validation: Optional[dict], delivery: dict, points: list
) -> QualityOut:
    trajectory_valid = bool(quality.get("trajectory_valid", True))
    flags = _flags(quality, validation, delivery)
    return QualityOut(
        trajectory_valid=trajectory_valid,
        tracking_confidence=_tracking_confidence(quality, points),
        requires_human_review=_requires_human_review(
            validation, trajectory_valid, delivery
        ),
        flags=flags,
    )


def _flags(quality: dict, validation: Optional[dict], delivery: dict) -> list[str]:
    """
    Concise flags, in place of the whole validation object.

    Deliberately NOT derived from `validation.classification == SINGLE_EVENT`:
    an unambiguous delivery earns no flag, so a non-empty list always means
    something a client should surface.
    """
    flags: list[str] = list(quality.get("flags") or [])

    if validation is None:
        # The pipeline quarantines unvalidated clips, so this should be
        # unreachable — but an unvalidated record must not read as validated.
        flags.append("unvalidated_clip")
    else:
        mapped = _VALIDATION_FLAG.get(validation.get("classification"))
        if mapped:
            flags.append(mapped)
        flags.extend(validation.get("flags") or [])

    if not delivery.get("clip"):
        flags.append("clip_unavailable")
    if quality.get("frame_map_valid") is False:
        # The clip's frame numbering could not be verified, so a trajectory
        # drawn over it may be offset. A client must be told.
        flags.append("frame_map_invalid")

    seen: set[str] = set()
    ordered: list[str] = []
    for flag in flags:
        if isinstance(flag, str) and flag and flag not in seen:
            seen.add(flag)
            ordered.append(flag)
    return ordered


def _requires_human_review(
    validation: Optional[dict], trajectory_valid: bool, delivery: dict
) -> bool:
    """
    The validation verdict's own answer, with two overrides that the verdict
    structurally cannot know: an unvalidated clip and a trajectory that did not
    survive validation both need eyes on them.
    """
    if validation is None or not trajectory_valid:
        return True
    if (delivery.get("quality") or {}).get("frame_map_valid") is False:
        return True
    return bool(validation.get("requires_human_review"))


def _tracking_confidence(quality: dict, points: list) -> Optional[float]:
    """Mean confidence over DETECTED points, from the stored record or recomputed."""
    stored = _float(quality.get("tracking_confidence"))
    if stored is not None:
        return stored
    values = [
        _float(p.get("confidence"))
        for p in points
        if p.get("source") == "detected" and p.get("confidence") is not None
    ]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


# ── Coercion helpers ──────────────────────────────────────────────────────────
# `None` stays `None`; a present value is converted. Nothing is defaulted into
# existence, because a default is a measurement this run never made.
def _int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BounceOut",
    "BowlingOut",
    "DeliveryOut",
    "QualityOut",
    "ShotOut",
    "TrajectoryOut",
    "TrajectoryPointOut",
    "VideoOut",
    "serialize_delivery",
    "serialize_result_document",
]
