"""
schemas.py — Typed records for the persisted tracking result.

These dataclasses define `tracking.json` (schema_version 2.0). The schema is the
contract between the single tracking pass and every downstream consumer, so it
is expressed in code rather than assembled ad hoc inside the pipeline.

Invariants enforced here rather than trusted (UNIFIED_ARCHITECTURE.md §4.6):
  * ClipFrameMap guarantees original_frame <-> clip_frame round-trips.
  * Trajectory points are never synthesised. A point exists only if the tracker
    produced a position for that frame, and it always records which it was.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

from ..core.constants import SHOT_MODEL_PROVENANCE_NAME, SHOT_N_FRAMES

TrajectorySource = Literal["detected", "predicted"]
TrackingSource = Literal["single_pass", "clip_fallback"]
BounceMethod = Literal["score", "fallback_max_y"]


class ClipMapError(Exception):
    """Raised when a clip's frame numbering cannot be trusted for overlay rendering."""


# ── Trajectory ────────────────────────────────────────────────────────────────
@dataclass
class TrajectoryPoint:
    """
    One tracked ball position.

    original_frame : 1-based frame number in the SOURCE video. This is the
                     authoritative timebase for the whole system.
    clip_frame     : 1-based frame number within the delivery clip. Only set
                     once the clip window is known.
    x, y           : full-resolution SOURCE pixel coordinates. Never rescaled —
                     the clip is written at source resolution so this mapping
                     to the clip is the identity (§6.3).
    source         : "detected" (a real YOLO measurement on this frame) or
                     "predicted" (a Kalman extrapolation). Never fabricated.
    confidence     : YOLO confidence for detected points, null for predicted.
    """
    original_frame: int
    x: int
    y: int
    source: TrajectorySource
    confidence: Optional[float] = None
    clip_frame: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trajectory:
    """
    THE single source of truth for a delivery's ball path.

    Consumed by, unmodified: the overlay renderer, the bounce marker, the pitch
    map, the bowling analytics and the API response (§1.2 C3).
    """
    points: list[TrajectoryPoint] = field(default_factory=list)

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def detected_count(self) -> int:
        return sum(1 for p in self.points if p.source == "detected")

    @property
    def predicted_count(self) -> int:
        return sum(1 for p in self.points if p.source == "predicted")

    @property
    def continuity(self) -> str:
        return "continuous" if not self.gaps else "gapped"

    @property
    def gaps(self) -> list[list[int]]:
        """
        Contiguous runs of original frames with no tracked point at all.

        Computed from the points that exist; frames without a point are simply
        not represented. No point is ever interpolated to fill a gap.
        """
        if len(self.points) < 2:
            return []
        frames = sorted(p.original_frame for p in self.points)
        gaps: list[list[int]] = []
        run_start = None
        for prev, cur in zip(frames, frames[1:]):
            if cur - prev > 1:
                if run_start is None:
                    run_start = prev + 1
            elif run_start is not None:
                # `prev` itself HAS a point, so the gap ends at prev - 1.
                gaps.append([run_start, prev - 1])
                run_start = None
        if run_start is not None:
            gaps.append([run_start, frames[-1] - 1])
        return gaps

    def detected_points(self) -> list[TrajectoryPoint]:
        """Detection-confirmed points only — the valid input for physics fits."""
        return [p for p in self.points if p.source == "detected"]

    def as_dict(self) -> dict:
        return {
            "points": [p.as_dict() for p in self.points],
            "point_count": self.point_count,
            "detected_count": self.detected_count,
            "predicted_count": self.predicted_count,
            "continuity": self.continuity,
            "gaps": self.gaps,
        }


# ── Clip + deterministic frame mapping ───────────────────────────────────────
@dataclass
class ClipFrameMap:
    """
    Deterministic original_frame <-> clip_frame mapping.

    The clip is written by streaming frames sequentially out of the tracking
    loop — never by seeking — so the mapping is exactly:
        clip_frame (1-based) = original_frame - frame_offset
        frame_offset         = clip_start_frame - 1

    ONE CLIP = ONE DELIVERY = ONE BATTING EVENT
    ------------------------------------------------
    A delivery clip must never contain a second delivery. The window is
    therefore bounded on BOTH sides by the neighbouring deliveries:

        start = max(detection_start - PRE_ROLL, previous_clip_end + 1)
        end   = min(ball_lost + POST_ROLL, next_delivery_start - 1)

    Both bounds can bind. When deliveries are closer together than
    PRE_ROLL + POST_ROLL the windows would overlap, and an overlapping clip
    contains two batting events — which is exactly the defect this prevents.
    Contiguity wins over roll length: when the two conflict, the later
    delivery loses pre-roll rather than the earlier one losing post-roll,
    because post-roll is what contains the stroke.

    `frame_offset` is derived from clip_start_frame and never changes when the
    window is truncated, so clip_frame numbering stays exact for every frame
    actually written.
    """

    clip_start_frame: int
    clip_end_frame: int
    frame_count: int
    width: int
    height: int
    fps: float
    codec: str
    frame_offset: int = field(init=False)
    frames_written: int = 0
    frame_map_valid: bool = True
    frame_map_error: Optional[str] = None
    pre_roll_actual_sec: Optional[float] = None
    post_roll_frames: int = 0
    # ── Window bounding provenance ──────────────────────────────────────────
    # The delivery's ball-loss frame, so the post-roll actually granted can be
    # reported separately from the post-roll requested.
    ball_lost_frame: Optional[int] = None
    # Set when the window was cut short because the NEXT delivery began. This is
    # the normal, correct outcome for closely spaced deliveries — it is recorded
    # rather than hidden so a consumer can see the post-roll was shortened.
    truncated_before_frame: Optional[int] = None
    # WHY the window was cut: "next_event" | "black_gap" | "scene_cut" |
    # "untracked_activity" | "inactivity_valley". A truncation caused by a black
    # gap or a camera cut is not evidence that the stroke was cut short, so the
    # validator must not treat every truncation as ambiguous.
    truncated_reason: Optional[str] = None
    # Set when the start was pushed forward because the PREVIOUS delivery's clip
    # already occupied those frames.
    start_clamped_by_previous: bool = False
    post_roll_actual_frames: Optional[int] = None

    def __post_init__(self) -> None:
        self.frame_offset = self.clip_start_frame - 1
        self.frame_count = self.clip_end_frame - self.clip_start_frame + 1

    def truncate_end(self, new_end: int, reason: Optional[str] = None) -> None:
        """
        Shorten the window to end at *new_end* (inclusive).

        Used when the next delivery starts earlier than the requested post-roll
        reaches. frame_offset is untouched, so every frame already written keeps
        its clip_frame number and the mapping stays exact and continuous.

        Trajectory points that fall outside the shortened window have their
        clip_frame cleared. They are real tracking data that simply has no
        position in this clip, and leaving a stale number on them would put the
        overlay in the wrong place.
        """
        new_end = int(new_end)
        if new_end < self.clip_end_frame:
            self.truncated_before_frame = new_end + 1
            self.truncated_reason = reason or self.truncated_reason
        self.clip_end_frame = max(self.clip_start_frame - 1, new_end)
        self.frame_count = self.clip_end_frame - self.clip_start_frame + 1
        if self._trajectory is not None:
            self.assign_clip_frames(self._trajectory)

    def recompute_post_roll(self) -> None:
        """Post-roll actually granted, which may be less than requested."""
        if self.ball_lost_frame is None:
            self.post_roll_actual_frames = None
        else:
            self.post_roll_actual_frames = max(
                0, self.clip_end_frame - self.ball_lost_frame
            )

    def to_clip_frame(self, original_frame: int) -> int:
        """1-based clip frame for a source frame. Round-trips exactly."""
        return original_frame - self.frame_offset

    def to_original_frame(self, clip_frame: int) -> int:
        """1-based source frame for a clip frame. Round-trips exactly."""
        return clip_frame + self.frame_offset

    def contains_original_frame(self, original_frame: int) -> bool:
        return self.clip_start_frame <= original_frame <= self.clip_end_frame

    def assign_clip_frames(self, trajectory: Trajectory) -> None:
        """
        Stamp clip_frame onto every trajectory point inside the clip window.

        Points outside the window keep clip_frame=None — they are still real
        tracking data, they just have no position in this clip. The trajectory is
        retained so a later `truncate_end` can re-stamp it consistently.
        """
        self._trajectory = trajectory
        for p in trajectory.points:
            p.clip_frame = (
                self.to_clip_frame(p.original_frame)
                if self.contains_original_frame(p.original_frame)
                else None
            )

    # Retained so truncate_end can re-stamp after the window shrinks.
    _trajectory: Optional["Trajectory"] = None

    def finalize(self) -> None:
        """
        Assert INVARIANT B (frames_written == frame_count) after writing.

        A writer that silently drops frames destroys the mapping, and a
        trajectory drawn onto a mis-numbered clip is worse than no trajectory
        at all. On failure we mark the map invalid rather than render anyway.
        """
        self.frames_written = self._frames_written
        self.recompute_post_roll()
        if self.frames_written != self.frame_count:
            self.frame_map_valid = False
            self.frame_map_error = (
                f"wrote {self.frames_written} of {self.frame_count} frames"
            )

    # set by the writer before finalize()
    _frames_written: int = 0

    def as_dict(self) -> dict:
        return {
            "start_frame": self.clip_start_frame,
            "end_frame": self.clip_end_frame,
            "frame_count": self.frame_count,
            "frame_offset": self.frame_offset,
            "fps": round(self.fps, 6),
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "frames_written": self.frames_written,
            "frame_map_valid": self.frame_map_valid,
            "frame_map_error": self.frame_map_error,
            "pre_roll_actual_sec": (
                round(self.pre_roll_actual_sec, 3)
                if self.pre_roll_actual_sec is not None
                else None
            ),
            "post_roll_frames": self.post_roll_frames,
            "post_roll_actual_frames": self.post_roll_actual_frames,
            "ball_lost_frame": self.ball_lost_frame,
            "truncated_before_frame": self.truncated_before_frame,
            "truncated_reason": self.truncated_reason,
            "start_clamped_by_previous": self.start_clamped_by_previous,
        }


# ── Physics ───────────────────────────────────────────────────────────────────
@dataclass
class SpeedSample:
    """A detection actually used to fit speed, with its ground-plane projection."""
    original_frame: int
    x: int
    y: int
    ground_x_m: float
    ground_y_m: float
    t_sec: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Bounce:
    detected: bool
    original_frame: Optional[int] = None
    x_px: Optional[int] = None
    y_px: Optional[int] = None
    x_norm: Optional[float] = None
    y_norm: Optional[float] = None
    ground_x_m: Optional[float] = None
    ground_y_m: Optional[float] = None
    method: BounceMethod = "score"

    def as_dict(self, include_none: bool = True) -> dict:
        d = asdict(self)
        if not include_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


@dataclass
class Bowling:
    """
    Bowling analytics for one delivery.

    speed_kmh is an ESTIMATE produced by a hand-tuned homography calibration
    (scale 2.0, offset -20 km/h), not a physical measurement. speed_calibrated
    is always true when a speed exists, and the calibration constants are also
    persisted at analysis level. Never display this as a calibrated reading.

    speed_kmh is null when speed could not be computed reliably. Per Phase 3
    decision D1 it is NEVER filled from a median or from another delivery.

    When `geometry_calibrated` is False the pitch corners were never measured on
    this video, so the homography describes a different camera angle. speed_kmh,
    length, line, swing and bounce_angle are then ALL null — not estimated, and
    not cosmetically relabelled. A number derived from the wrong corners is not
    a rough measurement but a confident falsehood, and Phase 3 measured exactly
    that: 42-199 km/h, with bounces landing off the end of a 20.12 m pitch.
    """
    speed_kmh: Optional[float] = None
    speed_method: Optional[str] = None
    # Defaults are False, not True. A default-constructed Bowling has established
    # no calibration and computed no speed, so claiming otherwise on the default
    # path is a statement the object cannot support.
    speed_calibrated: bool = False
    speed_imputed: bool = False
    length: Optional[str] = None
    line: Optional[str] = None
    swing: Optional[str] = None
    release_angle: Optional[float] = None
    bounce_angle: Optional[float] = None
    geometry_calibrated: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# ── Delivery record ───────────────────────────────────────────────────────────
@dataclass
class Quality:
    trajectory_valid: bool = True
    analytics_complete: bool = True
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Shot:
    type: Optional[str] = None
    confidence: Optional[float] = None
    class_probabilities: dict[str, float] = field(default_factory=dict)
    input_source: Optional[str] = None
    frames_sampled: Optional[int] = None
    # ── Frame-mapping audit ────────────────────────────────────────────────
    # The model has a fixed 30-frame temporal input, so a clip shorter than that
    # necessarily repeats positions. That is required by the architecture rather
    # than chosen here, but it must never be invisible: a label produced from 19
    # distinct frames is weaker evidence than one from 30, and a consumer cannot
    # tell the difference without these fields.
    unique_frames_sampled: Optional[int] = None
    frames_repeated: Optional[bool] = None
    # Which clip positions (0-based) were fed, and the same positions expressed as
    # ORIGINAL source frame numbers. Populated when the clip's frame offset is
    # known; null when the mapping could not be established.
    sampled_clip_frames: Optional[list[int]] = None
    sampled_original_frames: Optional[list[int]] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class OverlayInfo:
    path: Optional[str] = None
    url: Optional[str] = None
    rendered: bool = False
    trajectory_source: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Delivery:
    delivery_id: int

    # Boundaries, in SOURCE frame numbers (1-based).
    detection_start_frame: int
    detection_end_frame: int
    ball_lost_frame: int
    release_frame: Optional[int] = None

    clip: Optional[ClipFrameMap] = None
    clip_path: Optional[str] = None
    clip_url: Optional[str] = None

    trajectory: Trajectory = field(default_factory=Trajectory)

    speed_samples: list[SpeedSample] = field(default_factory=list)
    bounce: Bounce = field(default_factory=lambda: Bounce(detected=False))
    bowling: Bowling = field(default_factory=Bowling)

    shot: Optional[Shot] = None
    overlay: OverlayInfo = field(default_factory=OverlayInfo)

    tracking_source: TrackingSource = "single_pass"
    fallback_reason: Optional[str] = None
    quality: Quality = field(default_factory=Quality)

    # ── Event segmentation provenance ────────────────────────────────────────
    # `event` is the full evidence the segmenter used to draw this delivery's
    # boundaries: first/last confirmed detection, confirmation runs, coast
    # duration, reacquisitions, displacement, and WHY the boundary sits where it
    # does. `validation` is the independent single-event verdict for the clip.
    #
    # Both are stored so a wrong boundary can be diagnosed without re-running
    # the pass. Neither is required to make use of a delivery.
    event: Optional[dict] = None
    validation: Optional[dict] = None

    @property
    def delivery_ref(self) -> str:
        """Stable string identifier (`delivery_001`) used in media file names."""
        return f"delivery_{self.delivery_id:03d}"

    def provenance(self) -> dict:
        """
        Where each part of this record came from.

        Published per delivery rather than once per analysis because it is the only
        place a consumer can check the two architectural claims that matter: the
        ball data is the single tracking pass, and the shot label came from a model
        run on this delivery's own validated frames.
        """
        return {
            "tracking_source": "crickettracker",
            "tracking_pass": self.tracking_source,
            "trajectory_source": "stored_tracking",
            "shot_model": SHOT_MODEL_PROVENANCE_NAME,
            "shot_model_input_frames": SHOT_N_FRAMES,
            "source_video_decodes": 1,
            "yolo_passes": 1,
            "kalman_passes": 1,
        }

    def quality_report(self) -> dict:
        """
        `quality` plus the two numbers a client needs to decide how much to trust
        the record, both computed from the stored trajectory rather than asserted.
        """
        base = self.quality.as_dict()
        points = self.trajectory.points
        detected = self.trajectory.detected_count
        base["trajectory_points"] = len(points)
        base["trajectory_detected_points"] = detected
        base["trajectory_predicted_points"] = self.trajectory.predicted_count
        base["tracking_confidence"] = self._mean_detection_confidence()
        base["frame_map_valid"] = bool(
            self.clip.frame_map_valid if self.clip else False
        )
        return base

    def _mean_detection_confidence(self) -> Optional[float]:
        """Mean YOLO confidence over detected points. Null when there are none.

        Deliberately over DETECTED points only. Averaging in the null confidences
        of predicted points would drag the number toward zero for an event that
        was in fact tracked cleanly.
        """
        values = [
            p.confidence for p in self.trajectory.points
            if p.source == "detected" and p.confidence is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    def as_dict(self) -> dict:
        clip = self.clip.as_dict() if self.clip else None
        if clip is not None:
            clip["url"] = self.clip_url
            clip["file_name"] = f"{self.delivery_ref}.mp4"
        return {
            "delivery_id": self.delivery_id,
            "delivery_ref": self.delivery_ref,
            "index": self.delivery_id,
            "detection_start_frame": self.detection_start_frame,
            "detection_end_frame": self.detection_end_frame,
            "ball_lost_frame": self.ball_lost_frame,
            "release_frame": self.release_frame,
            "clip_path": self.clip_path,
            "clip_url": self.clip_url,
            "clip": clip,
            "trajectory": self.trajectory.as_dict(),
            "speed_samples": [s.as_dict() for s in self.speed_samples],
            "bounce": self.bounce.as_dict(),
            "bowling": self.bowling.as_dict(),
            "shot": self.shot.as_dict() if self.shot else None,
            "overlay": self.overlay.as_dict(),
            "overlay_url": self.overlay.url,
            "tracking_source": self.tracking_source,
            "fallback_reason": self.fallback_reason,
            "provenance": self.provenance(),
            "quality": self.quality_report(),
            "event": self.event,
            "validation": self.validation,
        }