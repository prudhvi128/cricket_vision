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
    CLEAN FRONTEND JSON  (one envelope, not the diagnostic document)

Nothing here computes, repairs or invents a measurement. It renames, selects and
drops. A value that does not exist internally comes out as `null`, never as an
estimate: `speed_kmh: null` is a fact about the run, a filled-in number would be
a lie.

THE RESPONSE IS ONE ENVELOPE
----------------------------
    {
      "analysis_id": "f24b6eb0f6184ac2",
      "status": "completed",
      "pitch":     { ... the calibration in effect for this analysis ... },
      "deliveries": [ delivery_1, delivery_2, ... ]
    }

so a client does `deliveries[currentIndex]` for Previous/Next navigation and
gets the video, trajectory, bounce, bowling analytics, shot and quality state of
THAT delivery in one object. No client ever has to reassemble a delivery from
two internal documents. `analysis_id` and `status` identify what is being shown;
`pitch` repeats the calibration of the LAST delivery that had one, so a header or
overlay can show the state of the analysis without walking the list.

KEY MAPPING (internal → frontend)
---------------------------------
    clip_path, clip.file_name      → dropped   (filesystem)
    clip.url / clip_url            → video.clip_url   (/api/analysis/{id}/clips/
                                      {name}, rebuilt here; the static mount is
                                      gone, so a stored /data/... url is a
                                      filename hint, never served as-is)
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
    delivery.pitch.*               → pitch.{state, detected, calibrated, confidence,
                                     using_previous_calibration, keypoints, corners,
                                     center, homography_available} — the optional
                                     Roboflow pitch calibration in source pixels.
                                     The stored homography is applied ONCE, into
                                     trajectory.points[].pitch_position (0..1 on
                                     the pitch), and never sent to the client.

Model-independent: everything is driven by `dict.get`, so it works on the
persisted document of any pipeline version, on disk or in memory.
"""

from __future__ import annotations

import math
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

    `pitch_position` places the same point on the PITCH (0..1 across the quad
    the pitch model found), for a client drawing a top-down pitch map. It is
    null whenever no calibration with a homography was in effect — a missing
    position is a fact, not a zero.
    """

    frame: int
    x: int
    y: int
    source: str
    confidence: Optional[float] = None
    pitch_position: Optional["PitchPositionOut"] = None


class PitchPositionOut(_FrontendModel):
    """Normalised ball position on the detected pitch quad."""

    x: float
    y: float
    inside_pitch: bool


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


class PitchOrientationOut(_FrontendModel):
    """
    Which end of the detected quad the batter stands at.

    The raw homography says "this is where the ball was as a fraction of the
    quad"; it says nothing about which end of that quad is the batting end, and
    a client that draws a top-down pitch without it can put a ball at the wrong
    end of the pitch — contradicting the very length label sitting next to it.
    So the orientation the analytics used is served alongside the points.

    `axis` is which unit-square axis runs along the pitch (`"u"` across the
    frame's width, `"v"` down it), `batting_end` is where the batter is along
    that axis (0.0 or 1.0), and `travel_delta` is how far the ball actually
    travelled along it — the evidence behind the choice. Null when no quad
    could be oriented, which is why a pitch map may legitimately show nothing.
    """

    axis: str
    batting_end: float
    travel_delta: Optional[float] = None
    calibration_frame: Optional[int] = None


class PitchOut(_FrontendModel):
    """
    Pitch keypoint calibration for this delivery, in SOURCE pixels.

    Every key is present on every delivery, so a client reads `pitch.detected`
    without having to test for the block first.

    The four states a client should switch on:

        state == "calibrated"          fresh detection → draw the pitch
        state == "temporary_loss"      detections failed briefly → draw the LAST
                                       known pitch, labelled as previous
        state == "no_calibration"      nothing found yet → draw nothing
        state == "calibration_expired" too old to trust → draw nothing

    `keypoints` and `corners` are `{"name": [x, y]}` in source pixels; `corners`
    is empty when fewer than four keypoints survived validation, because three
    keypoints cannot describe a rectangle and the fourth corner must never be
    invented. `confidence` is the model's own, not a derived number.
    """

    state: str = "no_calibration"
    detected: bool = False
    calibrated: bool = False
    confidence: Optional[float] = None
    using_previous_calibration: bool = False
    keypoints: dict[str, list[float]] = {}
    corners: dict[str, list[float]] = {}
    center: Optional[list[float]] = None
    homography_available: bool = False
    orientation: Optional[PitchOrientationOut] = None


class DeliveryOut(_FrontendModel):
    """One delivery, complete and self-contained. Previous/Next swaps this."""

    id: int
    video: VideoOut
    trajectory: TrajectoryOut
    bounce: BounceOut
    bowling: BowlingOut
    shot: ShotOut
    quality: QualityOut
    pitch: PitchOut = PitchOut()


class ResultDocument(_FrontendModel):
    """
    `GET /api/analysis/{analysis_id}/result` — the whole contract in one object.

    `pitch` is the calibration of the last delivery that had a real pitch block
    (or the empty default when none did), so a client shows the analysis-level
    state without scanning every delivery.
    """

    analysis_id: str
    status: str = "completed"
    pitch: PitchOut = PitchOut()
    deliveries: list[DeliveryOut]


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
    pitch = _pitch(delivery)

    return DeliveryOut(
        id=delivery_id,
        video=_video(delivery, clip, analysis_id),
        trajectory=_trajectory(delivery, points),
        bounce=_bounce(delivery),
        bowling=_bowling(delivery),
        shot=_shot(delivery),
        quality=_quality(quality, validation, delivery, points),
        pitch=pitch,
    )


def serialize_result_document(doc: dict) -> ResultDocument:
    """
    The whole result document as the envelope the frontend renders.

    Delivery order is the document's order (delivery id order), so
    `deliveries[currentIndex ± 1]` is Previous/Next. The analysis-level `pitch`
    is the last delivery's block, because calibration state only moves forward.
    """
    analysis_id = str(doc.get("analysis_id") or "")
    deliveries = [
        serialize_delivery(d, analysis_id=analysis_id)
        for d in doc.get("deliveries") or []
    ]
    pitch = PitchOut()
    for d in reversed(deliveries):
        if d.pitch.detected or d.pitch.state != "no_calibration":
            pitch = d.pitch
            break
    return ResultDocument(
        analysis_id=analysis_id,
        pitch=pitch,
        deliveries=deliveries,
    )


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
    The clips route for this delivery, or null. Never a path.

    The stored value (`/data/runs/<id>/clips/delivery_001.mp4`, or a bare
    filename on an older record) says WHERE the bytes are, which is internal —
    so only its basename is kept and it is reissued as the one public route that
    serves those bytes. A record with no clip at all gets null rather than a URL
    that would 404.
    """
    if delivery.get("clip") is None and not delivery.get("clip_path") and not delivery.get("clip_url"):
        return None
    if not analysis_id:
        return None
    return f"/api/analysis/{analysis_id}/clips/{_clip_name(delivery, clip)}"


def _clip_name(delivery: dict, clip: dict) -> str:
    """The clip's filename, or the id-derived name when a record omits it."""
    for candidate in (delivery.get("clip_url"), clip.get("url"), clip.get("file_name")):
        if not isinstance(candidate, str) or not candidate:
            continue
        name = candidate.replace("\\", "/").rsplit("/", 1)[-1]
        if name.startswith("delivery_") and name.endswith(".mp4"):
            return name
    return f"delivery_{int(delivery.get('delivery_id') or 0):03d}.mp4"


def _trajectory(delivery: dict, points: list) -> TrajectoryOut:
    traj = delivery.get("trajectory") or {}
    homography = (delivery.get("pitch") or {}).get("homography")
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
                pitch_position=_pitch_position(x, y, homography),
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


def _pitch(delivery: dict) -> PitchOut:
    """
    The pitch block, projected onto the frontend schema.

    Only the fields a client draws are kept: `model_id`, `image_size`, the
    calibration frame and the raw homography stay internal — the homography is
    applied once, here, into `pitch_position` on every trajectory point.
    """
    p = delivery.get("pitch")
    if not isinstance(p, dict):
        # A record written before this feature existed, or a run with no API
        # key: the honest empty state, with every key still present.
        return PitchOut()
    orientation = p.get("orientation") if isinstance(p.get("orientation"), dict) else None
    return PitchOut(
        state=str(p.get("state") or "no_calibration"),
        detected=bool(p.get("detected")),
        calibrated=bool(p.get("calibrated")),
        confidence=_float(p.get("confidence")),
        using_previous_calibration=bool(p.get("using_previous_calibration")),
        keypoints=_pairs(p.get("keypoints")),
        corners=_pairs(p.get("corners")),
        center=_pair(p.get("center")),
        # Derived from the matrix itself, not from a flag: a client must never
        # be told a transform exists while `pitch_position` is null on every
        # point because the stored one was unusable.
        homography_available=_matrix3(p.get("homography")) is not None,
        orientation=(
            PitchOrientationOut(
                axis=str(orientation.get("axis")),
                batting_end=float(orientation.get("batting_end")),
                travel_delta=_float(orientation.get("travel_delta")),
                calibration_frame=_int(orientation.get("calibration_frame")),
            )
            if orientation is not None and orientation.get("axis") is not None
            else None
        ),
    )


def _pitch_position(
    x: Any, y: Any, homography: Any
) -> Optional[PitchPositionOut]:
    """
    Source pixel → normalised pitch coordinate through the stored homography.

    Not a new measurement: the matrix comes from the record, so the position a
    client draws can never disagree with the calibration that produced it. Null
    when there is no matrix — a three-keypoint detection defines no transform,
    and a position computed from one that does not exist would be a fabrication.
    """
    px, py = _float(x), _float(y)
    if px is None or py is None:
        return None
    h = _matrix3(homography)
    if h is None:
        return None
    denom = h[2][0] * px + h[2][1] * py + h[2][2]
    if not math.isfinite(denom) or abs(denom) < 1e-9:
        return None
    u = (h[0][0] * px + h[0][1] * py + h[0][2]) / denom
    v = (h[1][0] * px + h[1][1] * py + h[1][2]) / denom
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return PitchPositionOut(
        x=round(u, 4),
        y=round(v, 4),
        inside_pitch=(-1e-6 <= u <= 1 + 1e-6 and -1e-6 <= v <= 1 + 1e-6),
    )


def _matrix3(value: Any) -> Optional[list[list[float]]]:
    """A 3x3 matrix of finite numbers, or None."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return None
        numbers = [_float(v) for v in row]
        if any(v is None or not math.isfinite(v) for v in numbers):
            return None
        rows.append(numbers)  # type: ignore[arg-type]
    return rows


def _pairs(value: Any) -> dict[str, list[float]]:
    """`{"name": [x, y]}`, dropping anything unreadable."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[float]] = {}
    for key, raw in value.items():
        pair = _pair(raw)
        if pair is not None:
            out[str(key)] = pair
    return out


def _pair(value: Any) -> Optional[list[float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = _float(value[0]), _float(value[1])
        if x is None or y is None:
            return None
        return [round(x, 2), round(y, 2)]
    return None


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
    "PitchOut",
    "PitchPositionOut",
    "QualityOut",
    "ResultDocument",
    "ShotOut",
    "TrajectoryOut",
    "TrajectoryPointOut",
    "VideoOut",
    "serialize_delivery",
    "serialize_result_document",
]
