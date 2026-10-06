"""
event_segmenter.py — ONE BATTING EVENT = ONE DELIVERY.

THE PROBLEM THIS SOLVES
-----------------------
A delivery used to be whatever fell out of the Kalman filter's `in_delivery`
flag: the flag turned on at the first accepted detection and off after
`MAX_MISSED` consecutive frames with no detection at all. Two consequences,
both observed on real footage:

  1. TWO EVENTS, ONE DELIVERY. When the next ball is detected while the filter
     is still coasting — or is accepted by the gate because it happens to fall
     within `GATE_RADIUS` of the stale prediction — `missed_frames` resets and
     the flag never clears. Two batting events become one delivery, and the
     clip contains both strokes.

  2. THE BLACK-GAP TRAP. The obvious fix is to treat a visual gap as a
     boundary. That is worse: footage where two events abut with no black frame
     between them has no gap to find, so those events stay merged. Meanwhile the
     41 contiguous footage regions a pixel scan found are just 41 camera
     segments — each may hold one event or several, and the scan cannot tell.

So neither "delivery = black-gap region" nor "delivery = detection cluster" is
a definition of a delivery. This module defines it properly:

    A DELIVERY IS ONE CONTINUOUS, TRACKED BALL.

THE THREE THINGS THIS IS NOT
----------------------------
  * Not a pixel-region counter. The number of contiguous non-black regions in a
    video is not recorded, configured, or assumed anywhere here.
  * Not a delivery-count target. There is no expected number anywhere in this
    module, and none may be added.
  * Not derived from the shot classifier. Shot classification needs a finished
    clip, so it cannot be what decides where the clip begins and ends. The
    dependency runs tracking -> boundaries -> clip -> shot, never the reverse.

THE EVIDENCE
------------
Every decision uses only what the single pass already produced, per frame:

  * was the ball CONFIRMED by YOLO, or only PREDICTED by the filter;
  * how many frames since the last confirmation (the COAST);
  * the tracker's own state;
  * whether the ball could physically have reached the new position.

No second decode, no second detector pass, no re-run of the filter. This module
is fed one evidence record per decoded frame inside the existing loop.

DETECTION GAP vs NEW DELIVERY
-----------------------------
A missed detection during a delivery is ordinary — the ball goes behind the
bat, behind a fielder's arm, briefly out of focus. That is a SHORT TRACKING
INTERRUPTION and must not split the delivery. But a Kalman filter extrapolates
forever, so prediction gets a hard ceiling (`max_coast_frames`): past it, the
event is over no matter what the filter believes. That ceiling is what stops an
old delivery from absorbing a genuinely new one.

Gaps are banded, and each band behaves differently:

    coast <= max_coast_frames      SHORT     same ball unless kinematics say
                                                  otherwise
    max_coast < coast <= hard      AMBIGUOUS kinematics decide; flagged either
    coast > hard_split_frames      LONG      new event, unconditionally

MULTIPLE EVENTS INSIDE ONE CONTIGUOUS REGION
--------------------------------------------
This is the case a pixel scan cannot solve, and the reason the SHORT and
AMBIGUOUS bands consult kinematics. When a confirmed detection arrives after a
gap, it is tested for REACHABILITY: could the ball have travelled from its last
confirmed position to this one, at its last measured speed, during the gap?

  * inside the reachable radius, moving in roughly the same direction  -> same ball
  * outside it, or reversed in direction                                -> new ball

Both tests are scale-free. The reachable radius is the measured speed times the
gap, plus slack expressed as a fraction of the frame DIAGONAL, so the same code
behaves identically on 320x180 test footage and 1080p broadcast footage. No
pixel threshold anywhere here was fitted to a particular video.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from ..core.config import ResolvedSegmentation, SegmentationConfig

__all__ = [
    "BoundaryDecision",
    "ConfirmationMethod",
    "ContinuationKind",
    "EventSegment",
    "EventSegmenter",
    "FrameEvidence",
    "SegmentDecision",
    "WeakSignal",
]


# ── Why a frame did or did not start a new event ──────────────────────────────
class ContinuationKind:
    """Machine-readable outcome of one `observe()` call."""

    IDLE = "idle"                        # nothing open, nothing confirmed
    COASTING = "coasting"                # no evidence; an open event is coasting
    SAME_EVENT = "same_event"            # confirmed, continues the open event
    REACQUIRED = "reacquired"            # confirmed after a coast: same event
    GATED_CONTINUATION = "gated_continuation"  # gate-rejected, kept in-event
    NEW_EVENT = "new_event"              # confirmed, starts a new event
    EVENT_EXPIRED = "event_expired"      # coast ceiling reached, event closed


# Why a boundary was drawn. Persisted per event so a human can audit any split.
class BoundaryReason:
    FIRST_DETECTION = "first_detection"
    LONG_SILENCE = "long_silence"                    # coast > hard split
    COAST_CEILING = "coast_ceiling"                  # coast > max coast
    UNREACHABLE = "unreachable_displacement"         # teleport out of the envelope
    DIRECTION_REVERSAL = "direction_reversal"
    SPEED_JUMP = "speed_discontinuity"
    END_OF_VIDEO = "end_of_video"
    NO_CONFIRMED_DETECTIONS = "no_confirmed_detections"


# How an event earned the right to be called a delivery. Recorded per event and
# persisted, so a consumer can tell a straightforward detection from a rescued one
# without re-deriving anything.
class ConfirmationMethod:
    STANDARD = "standard"
    # Below the confirmed-detection bar, promoted because several INDEPENDENT
    # signals agreed that gate-refused real measurements belonged to this ball.
    # The measurements stay evidence: they are not detections, they are not
    # trajectory points, and they never entered the Kalman filter.
    COHERENT_WEAK = "coherent_weak_detection"


class WeakSignal:
    """
    The six independent coherence signals a weak promotion must satisfy.

    Each is a separate claim about the SAME evidence, so they fail independently:
    noise scattered along the pitch can look continuous (1) while being
    physically impossible (2); a second ball can be perfectly continuous (1, 2)
    while sitting outside the pitch (3) or in the wrong place entirely (5).

    `MANDATORY` are the two that define "coherent" at all. Without continuity or
    without a physically possible path, the remaining signals prove nothing.
    """

    TEMPORAL_CONTINUITY = "temporal_continuity"
    PATH_COHERENCE = "path_coherence"
    CORRIDOR_RESIDENCY = "pitch_corridor_residency"
    PHYSICAL_MOVEMENT = "physically_possible_speed"
    RIVAL_ABSENCE = "single_object_not_two"
    PITCH_CONTEXT = "cricket_context_present"

    ALL = (
        TEMPORAL_CONTINUITY,
        PATH_COHERENCE,
        CORRIDOR_RESIDENCY,
        PHYSICAL_MOVEMENT,
        RIVAL_ABSENCE,
        PITCH_CONTEXT,
    )
    MANDATORY = (TEMPORAL_CONTINUITY, PATH_COHERENCE)


@dataclass(frozen=True)
class FrameEvidence:
    """
    One decoded frame's contribution to segmentation.

    Produced inside the tracking loop from values it already has. It carries
    provenance, not conclusions: the segmenter decides what this means.

    confirmed   : a real YOLO detection was ACCEPTED this frame.
    candidate   : a real detection existed but the tracker's gate rejected it.
                  It is still evidence that a ball is on screen — sometimes the
                  same ball, briefly out of the gate after an occlusion.
    position    : the detector's position for confirmed/candidate frames.
    tracker_position : whatever the filter reported (predicted while coasting).
    tracker_state : "untracked" | "coasting" | "tracking" | "lost"
    is_black    : frame luminance is below threshold (black/gap frame).
    is_scene_cut: frame difference indicates a camera switch/cut.
    motion_energy: full-frame inter-frame difference energy.
    pitch_motion : inter-frame difference energy focused on pitch corridor.
    """

    frame: int
    confirmed: bool = False
    candidate: bool = False
    position: Optional[tuple] = None
    confidence: Optional[float] = None
    tracker_position: Optional[tuple] = None
    tracker_state: str = "untracked"
    is_black: bool = False
    is_scene_cut: bool = False
    motion_energy: float = 0.0
    pitch_motion: float = 0.0
    # Whether THIS frame's ball position lies inside the pitch corridor. Carried
    # per measurement rather than derived later, because a fragment can only be
    # judged on the corridor context of its own confirmed points, and the
    # corridor is defined in frame fractions so it means the same thing at any
    # resolution. See `PITCH_CORRIDOR` in core/constants.py.
    in_pitch_corridor: bool = False

    @property
    def has_evidence(self) -> bool:
        return self.confirmed or self.candidate


@dataclass
class BoundaryDecision:
    """
    The full reasoning behind one boundary, kept for the audit trail.

    Every field is measured, not asserted. `reason` says which test fired; the
    numbers say what it saw.
    """

    at_frame: int
    kind: str
    reason: str
    gap_frames: int
    gap_seconds: float
    displacement_px: Optional[float] = None
    implied_speed_px_per_frame: Optional[float] = None
    reference_speed_px_per_frame: Optional[float] = None
    reachable_radius_px: Optional[float] = None
    direction_change_deg: Optional[float] = None
    speed_ratio: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "at_frame": self.at_frame,
            "kind": self.kind,
            "reason": self.reason,
            "gap_frames": self.gap_frames,
            "gap_seconds": round(self.gap_seconds, 4),
            "displacement_px": (
                round(self.displacement_px, 2)
                if self.displacement_px is not None else None
            ),
            "implied_speed_px_per_frame": (
                round(self.implied_speed_px_per_frame, 3)
                if self.implied_speed_px_per_frame is not None else None
            ),
            "reference_speed_px_per_frame": (
                round(self.reference_speed_px_per_frame, 3)
                if self.reference_speed_px_per_frame is not None else None
            ),
            "reachable_radius_px": (
                round(self.reachable_radius_px, 2)
                if self.reachable_radius_px is not None else None
            ),
            "direction_change_deg": (
                round(self.direction_change_deg, 1)
                if self.direction_change_deg is not None else None
            ),
            "speed_ratio": (
                round(self.speed_ratio, 3)
                if self.speed_ratio is not None else None
            ),
            "notes": list(self.notes),
        }


@dataclass
class SegmentDecision:
    """What `observe()` concluded, for one frame."""

    frame: int
    kind: str
    # An event opened on THIS frame.
    opened: Optional["EventSegment"] = None
    # An event closed on THIS frame. Always set together with `boundary`.
    closed: Optional["EventSegment"] = None
    boundary: Optional[BoundaryDecision] = None
    # The caller should re-seed the Kalman filter at `reseed_position`, keeping
    # the SAME event open. This is what makes a short detector dropout a
    # continued delivery instead of the end of one.
    reseed_position: Optional[tuple] = None

    @property
    def opened_new_event(self) -> bool:
        return self.kind == ContinuationKind.NEW_EVENT

    @property
    def continued_same_event(self) -> bool:
        return self.kind in (
            ContinuationKind.SAME_EVENT,
            ContinuationKind.REACQUIRED,
            ContinuationKind.GATED_CONTINUATION,
        )


@dataclass
class EventSegment:
    """
    One batting event, as far as the tracking evidence supports one.

    This is the unit a delivery is built from. Its `start_frame`/`end_frame`
    span the CONFIRMED detections of one continuous ball; everything outside
    that span is pre-roll or post-roll, and the clip window is bounded
    separately (clip_writer).
    """

    index: int
    start_frame: int
    end_frame: int
    closed_at_frame: int
    confirmed_frames: list[int] = field(default_factory=list)
    predicted_frames: list[int] = field(default_factory=list)
    reacquired_frames: list[int] = field(default_factory=list)
    gated_continuation_frames: list[int] = field(default_factory=list)
    close_reason: str = ""
    # How this event's detections earned the right to be called a delivery.
    #   "standard"  — confirmed by the 8-frame threshold on its own.
    #   "coherent_weak_detection" — below that threshold, promoted by the
    #     secondary multi-signal path (see `_promote_weak_evidence`), which
    #     requires independent agreement from several other signals.
    # Recorded per event so a consumer can tell a straightforward detection from
    # a rescued one without re-deriving anything.
    confirmation_method: str = ConfirmationMethod.STANDARD
    # Gate-refused real measurements attributed to this event. EVIDENCE ONLY:
    # they are never appended to `confirmed_frames`, never stored as trajectory
    # points, and never fed back to the Kalman filter. They are kept here, with
    # their frames, so the coherence decision that used them can be audited.
    weak_evidence_frames: list[int] = field(default_factory=list)
    # How many of those refused measurements lay inside the pitch corridor.
    weak_in_corridor_frames: int = 0
    # The full coherence report, present only for a promoted event. Every signal,
    # its measured value, and the reason it passed or failed.
    weak_confirmation: Optional[dict] = None
    boundary: Optional[BoundaryDecision] = None
    confidence: float = 1.0
    ambiguous: bool = False
    ambiguous_reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    # The frame the ball was last accounted for, i.e. how far past its last
    # confirmed detection this event is entitled to keep its own footage. Set at
    # close time, because only then is it known.
    ball_lost_frame: Optional[int] = None

    # ── Physical context, needed to decide whether two fragments are one ball ──
    # Endpoints are the CONFIRMED detector positions at the fragment's first and
    # last confirmations. Fragment merging re-tests continuity from BOTH sides
    # using these, because the outgoing-velocity test that caused the split in
    # the first place is fitted over a window that can straddle a bounce.
    first_position: Optional[tuple] = None
    last_position: Optional[tuple] = None
    # Exit and entry velocities in px/frame, least-squares fitted within the
    # fragment. `None` when the fragment is too short to fit one.
    exit_velocity: Optional[tuple] = None
    entry_velocity: Optional[tuple] = None
    in_corridor_frames: int = 0

    # ── Merge provenance, populated only when this event absorbs another ──────
    # `merged_from` lists the candidate indices that were folded in, so a
    # delivered clip can always be decomposed back into what it was built from.
    merged_from: list[int] = field(default_factory=list)
    merged_gap_frames: int = 0
    merge_reason: str = ""
    merge_checks: dict = field(default_factory=dict)
    duplicate_of: Optional[int] = None

    @property
    def corridor_ratio(self) -> float:
        """Share of this event's confirmed detections inside the pitch corridor."""
        return self.in_corridor_frames / max(1, self.confirmed_count)

    # ── Derived facts the report and validator need ──────────────────────────

    @property
    def confirmed_count(self) -> int:
        return len(self.confirmed_frames)

    @property
    def weak_evidence_count(self) -> int:
        return len(self.weak_evidence_frames)

    @property
    def total_measurement_count(self) -> int:
        """Real YOLO measurements backing this event: confirmed plus refused."""
        return len(self.confirmed_frames) + len(self.weak_evidence_frames)

    @property
    def is_weak_confirmation(self) -> bool:
        return self.confirmation_method == ConfirmationMethod.COHERENT_WEAK

    @property
    def predicted_count(self) -> int:
        return len(self.predicted_frames)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def ball_lost_duration(self) -> int:
        """Frames between the last confirmation and the moment it was closed."""
        return max(0, self.closed_at_frame - self.end_frame)

    def contains_frame(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame

    def as_dict(self) -> dict:
        d = {
            "event_index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "closed_at_frame": self.closed_at_frame,
            "duration_frames": self.duration_frames,
            "confirmed_count": self.confirmed_count,
            "weak_evidence_count": self.weak_evidence_count,
            "total_measurement_count": self.total_measurement_count,
            "predicted_count": self.predicted_count,
            "reacquired_count": len(self.reacquired_frames),
            "gated_continuation_count": len(self.gated_continuation_frames),
            "ball_lost_duration_frames": self.ball_lost_duration,
            "close_reason": self.close_reason,
            "confirmation_method": self.confirmation_method,
            "ball_lost_frame": self.ball_lost_frame,
            "confidence": round(self.confidence, 4),
            "ambiguous": self.ambiguous,
            "ambiguous_reasons": list(self.ambiguous_reasons),
            "boundary": self.boundary.as_dict() if self.boundary else None,
            "evidence": dict(self.evidence),
        }
        # Frame lists are large and already reproducible from the trajectory;
        # only the ranges are persisted.
        d["confirmed_frame_ranges"] = _ranges(self.confirmed_frames)
        d["predicted_frame_ranges"] = _ranges(self.predicted_frames)
        d["reacquired_frames"] = list(self.reacquired_frames)
        # Merge provenance travels with the delivery, so a merged clip can always
        # be decomposed back into the fragments it was built from. Recorded even
        # when empty, because its ABSENCE is meaningful: it distinguishes a clip
        # that was assembled from fragments from one that never needed to be.
        d["merged_from"] = list(self.merged_from)
        d["merged_gap_frames"] = self.merged_gap_frames
        d["merge_reason"] = self.merge_reason
        d["merge_checks"] = dict(self.merge_checks)
        d["duplicate_of"] = self.duplicate_of
        d["corridor_ratio"] = round(self.corridor_ratio, 4)
        # The coherence decision travels with the event, including when it failed:
        # an event that was NOT rescued should be auditable too, and that is what
        # distinguishes "the evidence was thin" from "the evidence was examined
        # and did not agree".
        d["weak_confirmation"] = dict(self.weak_confirmation) if self.weak_confirmation else None
        return d


def _ranges(frames: list[int]) -> list[list[int]]:
    """Collapse a sorted frame list into inclusive [start, end] runs."""
    if not frames:
        return []
    out: list[list[int]] = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        out.append([start, prev])
        start = prev = f
    out.append([start, prev])
    return out


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_between_deg(v1: tuple, v2: tuple) -> float:
    """Angle between two vectors, in degrees, 0..180. Undefined if either is zero."""
    m1 = math.hypot(v1[0], v1[1])
    m2 = math.hypot(v2[0], v2[1])
    if m1 < 1e-6 or m2 < 1e-6:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


class EventSegmenter:
    """
    Streaming segmenter: one evidence record in, one boundary decision out.

    It is fed every decoded frame in order, so it sees exactly the evidence the
    single pass produced and needs no second pass of its own. Events it closes
    are appended to `events`, newest last.

    Thread-safety is deliberately not claimed: one instance belongs to one
    tracking run and is driven from that run's single loop.
    """

    def __init__(
        self,
        fps: float,
        width: int,
        height: int,
        config: Optional[SegmentationConfig] = None,
        resolved: Optional[ResolvedSegmentation] = None,
    ) -> None:
        self.resolved = resolved or (config or SegmentationConfig()).resolve(
            fps, width, height
        )
        self.events: list[EventSegment] = []
        self.open_event: Optional[EventSegment] = None
        self._next_index = 1

        # Reference-velocity window for the open event: (frame, x, y).
        self._window: Deque[tuple[int, float, float]] = deque(
            maxlen=self.resolved.velocity_window
        )
        self._last_confirmed: Optional[tuple[int, float, float]] = None
        # Every confirmed position of the open event, for the displacement
        # figure reported in the evidence block.
        self._positions: list[tuple[float, float]] = []
        # Every confirmed (frame, x, y) of the open event, unbounded. Needed at
        # CLOSE time to fit how the event entered and left, which the fragment
        # merger uses to re-test continuity from the opposite side.
        self._confirmations: list[tuple[int, float, float]] = []
        # Every REAL measurement of the open event, in arrival order, whether the
        # tracker accepted it or refused it at the gate:
        #   (frame, x, y, in_pitch_corridor, motion_energy, pitch_motion)
        # Held only while one event is open and cleared with it. This is the
        # evidence pool `WeakSignal` is evaluated against; nothing here is ever
        # promoted into `confirmed_frames` or into a trajectory.
        self._measurements: list[tuple[int, float, float, bool, float, float]] = []
        # The gate-refused subset of the above — the weak-evidence channel.
        self._refused: list[tuple[int, float, float, bool, float, float]] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def observe(self, ev: FrameEvidence) -> SegmentDecision:
        """
        Feed one frame's evidence. Never raises on odd input; an unusable frame
        simply produces no boundary, because losing a frame must not invent one.
        """
        r = self.resolved
        open_event = self.open_event

        if not ev.has_evidence:
            # ── No measurement this frame ──────────────────────────────────
            if open_event is None:
                return SegmentDecision(frame=ev.frame, kind=ContinuationKind.IDLE)
            self._absorb_prediction(ev)
            gap = ev.frame - open_event.end_frame
            if gap > r.max_coast_frames:
                # CASE E. The Kalman filter would happily keep extrapolating;
                # the event does not get to live that long. Closing here is
                # what stops an old delivery absorbing a new one.
                # A coast ceiling is the NORMAL, confident end of a delivery:
                # the ball was not seen for longer than the prediction ceiling,
                # so the event is over. It is deliberately NOT flagged
                # ambiguous — treating every ordinary ending as uncertain would
                # make AMBIGUOUS meaningless. Only an uncorroborated ending
                # (video ended) or too little measurement is ambiguous.
                closed = self._close(
                    at_frame=ev.frame,
                    reason=BoundaryReason.COAST_CEILING,
                    kind=ContinuationKind.EVENT_EXPIRED,
                    ambiguous=False,
                    note=(
                        f"{gap} frames ({gap / r.fps:.2f}s) since the last "
                        f"confirmed detection exceeds the prediction ceiling "
                        f"of {r.max_coast_frames} frames ({r.max_coast_frames / r.fps:.2f}s)"
                    ),
                )
                return SegmentDecision(
                    frame=ev.frame,
                    kind=ContinuationKind.EVENT_EXPIRED,
                    closed=closed,
                    boundary=closed.boundary,
                )
            return SegmentDecision(frame=ev.frame, kind=ContinuationKind.COASTING)

        # ── There IS a measurement this frame ──────────────────────────────
        position = ev.position
        if position is None:
            # `has_evidence` without a position is malformed input. Treat it as
            # no evidence rather than crashing the pass over it.
            return SegmentDecision(frame=ev.frame, kind=ContinuationKind.IDLE)
        px, py = float(position[0]), float(position[1])

        if open_event is None:
            return self._open(ev, px, py)

        gap = ev.frame - open_event.end_frame
        if ev.candidate and not ev.confirmed:
            return self._consider_gated(ev, open_event, px, py, gap)

        return self._consider_confirmed(ev, open_event, px, py, gap)

    def close_open_event(
        self, at_frame: int, reason: str = BoundaryReason.END_OF_VIDEO
    ) -> Optional[EventSegment]:
        """
        Close whatever is still open at end of video.

        The event is CLOSED, not deleted: if it holds too few confirmed
        detections to be a delivery it is marked low-confidence and the caller
        discards it. Silence must never fabricate a delivery (case F) and must
        never delete a real one.
        """
        if self.open_event is None:
            return None
        return self._close(
            at_frame=max(at_frame, self.open_event.end_frame),
            reason=reason,
            kind=ContinuationKind.EVENT_EXPIRED,
            ambiguous=False,
        )

    def primary_events(self) -> list[EventSegment]:
        """
        Events with enough evidence to be a delivery.

        `events` keeps everything the segmenter found, including ones too thin
        to be a delivery, so a diagnostic can report them. This is the subset
        the pipeline turns into deliveries: events that cleared the confirmed
        detection bar on the standard path, plus the ones explicitly promoted on
        coherent weak evidence.
        """
        floor = self.resolved.min_event_confirmed_frames
        return [
            e for e in self.events
            if e.confirmed_count >= floor
            or e.confirmation_method == ConfirmationMethod.COHERENT_WEAK
        ]

    def summary(self) -> dict:
        return {
            "events_total": len(self.events),
            "events_primary": len(self.primary_events()),
            "events_ambiguous": sum(1 for e in self.events if e.ambiguous),
            "events_rejected": len(self.events) - len(self.primary_events()),
            "confirmation_methods": _tally(e.confirmation_method for e in self.events),
            "weak_evidence_total": sum(
                e.weak_evidence_count for e in self.events
            ),
            "resolved_thresholds": self.resolved.as_metadata(),
            "boundary_reasons": _tally(e.close_reason for e in self.events),
        }

    # ── Decision internals ───────────────────────────────────────────────────

    def _open(
        self, ev: FrameEvidence, px: float, py: float
    ) -> SegmentDecision:
        """Start an event on this confirmed detection."""
        r = self.resolved
        seg = EventSegment(
            index=self._next_index,
            start_frame=ev.frame,
            end_frame=ev.frame,
            closed_at_frame=ev.frame,
            close_reason="",
            boundary=BoundaryDecision(
                at_frame=ev.frame,
                kind=ContinuationKind.NEW_EVENT,
                reason=BoundaryReason.FIRST_DETECTION,
                gap_frames=0,
                gap_seconds=0.0,
                displacement_px=0.0,
                reference_speed_px_per_frame=0.0,
            ),
        )
        self._next_index += 1
        self.open_event = seg
        self._reset_window()
        self._absorb_confirmation(ev, px, py)
        return SegmentDecision(
            frame=ev.frame,
            kind=ContinuationKind.NEW_EVENT,
            opened=seg,
            boundary=seg.boundary,
        )

    def _absorb_confirmation(self, ev: FrameEvidence, px: float, py: float) -> None:
        seg = self.open_event
        assert seg is not None
        self._record_measurement(ev, px, py)
        if ev.confirmed:
            seg.confirmed_frames.append(ev.frame)
            self._last_confirmed = (ev.frame, px, py)
            self._window.append((ev.frame, px, py))
            self._positions.append((px, py))
            self._confirmations.append((ev.frame, px, py))
            # Endpoints and corridor residency are captured per CONFIRMED
            # measurement. They are what a later merge test re-checks from the
            # opposite side, so they must be recorded while the frame is in hand
            # rather than reconstructed afterwards from a single velocity fit.
            if seg.first_position is None:
                seg.first_position = (px, py)
            seg.last_position = (px, py)
            if ev.in_pitch_corridor:
                seg.in_corridor_frames += 1
        elif ev.candidate:
            # A gate-rejected detection kept inside the event. It is a real
            # measurement, just not one the filter would accept, so it counts
            # as evidence of the ball being present without being trusted as a
            # position.
            seg.gated_continuation_frames.append(ev.frame)

    def _absorb_prediction(self, ev: FrameEvidence) -> None:
        """A frame with no measurement, but the filter still had an opinion."""
        seg = self.open_event
        if seg is not None and ev.tracker_position is not None:
            seg.predicted_frames.append(ev.frame)

    def _record_measurement(
        self, ev: FrameEvidence, px: float, py: float
    ) -> None:
        """
        File one REAL detection against the open event, with its context.

        Called for every frame in which the detector returned a box, whether the
        tracker accepted it (`confirmed`) or refused it at the gate (`candidate`).
        The refused measurements are what `coherent_weak_detection` is later judged
        on, so they are kept with their positions — but only as evidence: nothing
        here writes to `confirmed_frames`, to a trajectory point, or back to the
        filter. `_consider_gated` records unconditionally, including on the frames
        it then declines to act on, so a run of refusals the gate rejected as
        unreachable is still available to the coherence test rather than being
        silently discarded.
        """
        seg = self.open_event
        if seg is None:
            return
        row = (
            int(ev.frame),
            float(px),
            float(py),
            bool(ev.in_pitch_corridor),
            float(ev.motion_energy),
            float(ev.pitch_motion),
        )
        self._measurements.append(row)
        if ev.candidate and not ev.confirmed:
            self._refused.append(row)
            seg.weak_evidence_frames.append(int(ev.frame))
            if ev.in_pitch_corridor:
                seg.weak_in_corridor_frames += 1

    def _reference_velocity(self) -> tuple[float, float, float]:
        """
        Event's recent velocity in px/frame, from a least-squares fit.

        Returns (vx, vy, speed). A fit over several confirmed points rather
        than the last pair, because the ball bounces: any two-point estimate
        taken across a bounce is noise, and a noisy reference makes the
        reachability test meaningless in both directions.
        """
        n = len(self._window)
        if n == 0:
            return 0.0, 0.0, 0.0
        if n == 1:
            return 0.0, 0.0, 0.0
        tf = sum(p[0] for p in self._window)
        tx = sum(p[1] for p in self._window)
        ty = sum(p[2] for p in self._window)
        tff = sum(p[0] * p[0] for p in self._window)
        tfx = sum(p[0] * p[1] for p in self._window)
        tfy = sum(p[0] * p[2] for p in self._window)
        denom = n * tff - tf * tf
        if abs(denom) < 1e-9:
            return 0.0, 0.0, 0.0
        vx = (n * tfx - tf * tx) / denom
        vy = (n * tfy - tf * ty) / denom
        return vx, vy, math.hypot(vx, vy)

    def _kinematics(self, px: float, py: float, gap: int) -> dict:
        """
        Everything measurable about a returning ball, for one gap.

        reach_radius : how far the ball could plausibly have travelled given
                       its last measured speed and the time available, plus
                       slack. Expressed as a fraction of the frame diagonal so
                       the test is resolution-independent.
        reachable    : the observed displacement is inside that radius.
        direction_change : angle between the reference velocity and the
                       displacement from the last confirmed point.
        speed_ratio  : observed speed / reference speed.
        """
        r = self.resolved
        last = self._last_confirmed
        if last is None:
            return {
                "reachable": True, "reach_radius": None, "displacement": None,
                "direction_change": None, "speed_ratio": None,
                "implied_speed": None, "reference_speed": None,
                "unknown": True,
            }

        _, lx, ly = last
        dx, dy = px - lx, py - ly
        displacement = math.hypot(dx, dy)
        span = max(1, gap)
        implied = displacement / span
        _, _, ref_speed = self._reference_velocity()

        # Reach uses the last CONFIRMED displacement when the fit is
        # unavailable (the first two points of an event), so a brand-new track
        # is not judged against a zero reference velocity.
        reach_speed = ref_speed if ref_speed > 1e-6 else displacement / span
        reach_radius = reach_speed * span + r.reach_base_px + r.reach_rate_px * span
        reachable = displacement <= reach_radius

        direction_change = None
        vx, vy, _ = self._reference_velocity()
        if math.hypot(vx, vy) > 1e-6 and displacement > 1e-6:
            direction_change = _angle_between_deg((vx, vy), (dx, dy))

        speed_ratio = (implied / ref_speed) if ref_speed > 1e-6 else None

        return {
            "reachable": reachable,
            "reach_radius": reach_radius,
            "displacement": displacement,
            "direction_change": direction_change,
            "speed_ratio": speed_ratio,
            "implied_speed": implied,
            "reference_speed": ref_speed,
            "unknown": False,
        }

    def _judge(self, kin: dict, gap: int) -> tuple[bool, str, list[str]]:
        """
        Decide: is this returning ball the SAME ball, or a NEW one?

        Returns (is_new_event, reason, notes). Split into bands by gap length,
        because silence and kinematics carry different weight depending on how
        much silence there is.
        """
        r = self.resolved
        if kin.get("unknown"):
            return False, "insufficient_motion_history", []

        reversal = kin["direction_change"]
        reversed_hard = reversal is not None and reversal >= r.reversal_degrees
        ratio = kin["speed_ratio"]
        speed_jumped = ratio is not None and (
            ratio > r.speed_jump_up or ratio < r.speed_jump_down
        )
        unreachable = not kin["reachable"]

        notes: list[str] = []

        # ── LONG band: silence is conclusive on its own ────────────────────
        if gap > r.hard_split_frames:
            notes.append(
                f"gap {gap}f ({gap / r.fps:.2f}s) exceeds the hard split "
                f"threshold {r.hard_split_frames}f "
                f"({r.hard_split_frames / r.fps:.2f}s)"
            )
            return True, BoundaryReason.LONG_SILENCE, notes

        # ── AMBIGUOUS band ─────────────────────────────────────────────────
        if gap > r.max_coast_frames:
            if unreachable:
                notes.append(
                    f"ball moved {kin['displacement']:.0f}px in {gap}f, outside "
                    f"the {kin['reach_radius']:.0f}px reachable radius"
                )
                return True, BoundaryReason.UNREACHABLE, notes
            if reversed_hard:
                notes.append(f"direction reversed by {reversal:.0f}deg")
                return True, BoundaryReason.DIRECTION_REVERSAL, notes
            if speed_jumped:
                notes.append(f"implied speed is {ratio:.2f}x the event's own")
                return True, BoundaryReason.SPEED_JUMP, notes
            notes.append(
                "kinematics stayed continuous across a gap longer than the "
                "prediction ceiling; treated as one event and flagged"
            )
            return False, "ambiguous_band_continuous", notes

        # ── SHORT band: an occlusion, unless kinematics say otherwise ────────
        # Within the prediction ceiling a gap is ordinary, so silence ALONE
        # never splits. Two independent kinematic failures together do.
        if unreachable and reversed_hard:
            notes.append(
                f"unreachable ({kin['displacement']:.0f}px vs "
                f"{kin['reach_radius']:.0f}px) and reversed by {reversal:.0f}deg"
            )
            return True, BoundaryReason.UNREACHABLE, notes
        if unreachable and speed_jumped:
            notes.append(
                f"unreachable ({kin['displacement']:.0f}px vs "
                f"{kin['reach_radius']:.0f}px) and speed ratio {ratio:.2f}"
            )
            return True, BoundaryReason.UNREACHABLE, notes

        notes.append(
            f"gap {gap}f within the prediction ceiling "
            f"({r.max_coast_frames}f); treated as a tracking interruption"
        )
        return False, "short_interruption", notes

    def _consider_confirmed(
        self, ev: FrameEvidence, seg: EventSegment, px: float, py: float, gap: int
    ) -> SegmentDecision:
        """A confirmed detection: continuation of this event, or a new one."""
        r = self.resolved
        kin = self._kinematics(px, py, gap)
        is_new, reason, notes = self._judge(kin, gap)

        boundary = BoundaryDecision(
            at_frame=ev.frame,
            kind=ContinuationKind.NEW_EVENT if is_new else ContinuationKind.SAME_EVENT,
            reason=reason,
            gap_frames=gap,
            gap_seconds=gap / r.fps if r.fps else 0.0,
            displacement_px=kin["displacement"],
            implied_speed_px_per_frame=kin["implied_speed"],
            reference_speed_px_per_frame=kin["reference_speed"],
            reachable_radius_px=kin["reach_radius"],
            direction_change_deg=kin["direction_change"],
            speed_ratio=kin["speed_ratio"],
            notes=notes,
        )

        if not is_new:
            kind = (
                ContinuationKind.REACQUIRED if gap > 1
                else ContinuationKind.SAME_EVENT
            )
            self._absorb_confirmation(ev, px, py)
            seg.end_frame = ev.frame
            seg.boundary = boundary
            self._record_evidence(seg, boundary, reacquired=(gap > 1))
            return SegmentDecision(
                frame=ev.frame,
                kind=kind,
                boundary=boundary,
                # The tracker may already have declared the ball lost. Hand the
                # caller the position to restart the filter on, WITHOUT ending
                # the event. This is what turns a short detector dropout into a
                # continued delivery rather than two deliveries (case B).
                reseed_position=(int(px), int(py)),
            )

        # ── New event. Close this one, capped so its clip cannot reach here. ──
        seg.end_frame = max(seg.end_frame, ev.frame - 1)
        closed = self._close(
            at_frame=ev.frame,
            reason=reason,
            kind=ContinuationKind.NEW_EVENT,
            boundary=boundary,
            notes=notes,
        )
        opened = self._open(ev, px, py)
        return SegmentDecision(
            frame=ev.frame,
            kind=ContinuationKind.NEW_EVENT,
            closed=closed,
            opened=opened.opened,
            boundary=boundary,
        )

    def _consider_gated(
        self, ev: FrameEvidence, seg: EventSegment, px: float, py: float, gap: int
    ) -> SegmentDecision:
        """
        A detection the tracker's GATE rejected.

        The gate is a 120 px radius around the filter's prediction. After an
        occlusion the true ball often lands just outside it, so throwing the
        measurement away can end a delivery that is still happening.

        It is only ever allowed to CONTINUE an event, and only inside the
        prediction ceiling. Beyond the ceiling this handler is never reached:
        `observe` closes the event first. A gated detection can therefore never
        merge two deliveries — the failure mode being fixed here is merging, and
        this path cannot cause it.

        Whatever this handler decides, the measurement is recorded as evidence
        for the open event first. A refusal the gate judges unreachable may still
        be part of one continuous ball that the filter lost track of, and that is
        precisely the case `coherent_weak_detection` exists to rescue; dropping
        the frame here would make the evidence channel blind exactly where it is
        needed. Recording is not trusting: the segment boundary decision below is
        unchanged.
        """
        r = self.resolved
        self._record_measurement(ev, px, py)
        kin = self._kinematics(px, py, gap)
        reachable = kin["reachable"] and kin.get("unknown") is not True

        if gap <= r.max_coast_frames and reachable:
            seg.gated_continuation_frames.append(ev.frame)
            seg.end_frame = ev.frame
            boundary = BoundaryDecision(
                at_frame=ev.frame,
                kind=ContinuationKind.GATED_CONTINUATION,
                reason="gate_rejected_but_reachable",
                gap_frames=gap,
                gap_seconds=gap / r.fps if r.fps else 0.0,
                displacement_px=kin["displacement"],
                reachable_radius_px=kin["reach_radius"],
                direction_change_deg=kin["direction_change"],
                notes=[
                    "detection fell outside the tracker gate but within the "
                    "reachable radius, so it is kept as evidence that this "
                    "event is still running"
                ],
            )
            seg.boundary = boundary
            self._record_evidence(seg, boundary)
            return SegmentDecision(
                frame=ev.frame,
                kind=ContinuationKind.GATED_CONTINUATION,
                boundary=boundary,
                reseed_position=(int(px), int(py)),
            )

        # Not reachable, or too late: treat it as neither. The event keeps
        # coasting and the next clean confirmation re-decides the boundary.
        return SegmentDecision(frame=ev.frame, kind=ContinuationKind.COASTING)

    def _fit_velocity(self, points) -> tuple[float, float]:
        """
        Least-squares velocity in px/frame over a fragment's own confirmed points.

        Kept separate from `_reference_velocity`, which fits only the open
        event's sliding window. A merge needs the velocity each CLOSED fragment
        arrived and departed with, over its whole span.
        """
        n = len(points)
        if n < 2:
            return 0.0, 0.0
        tf = sum(p[0] for p in points)
        tx = sum(p[1] for p in points)
        ty = sum(p[2] for p in points)
        tff = sum(p[0] * p[0] for p in points)
        tfx = sum(p[0] * p[1] for p in points)
        tfy = sum(p[0] * p[2] for p in points)
        denom = n * tff - tf * tf
        if abs(denom) < 1e-9:
            return 0.0, 0.0
        return (n * tfx - tf * tx) / denom, (n * tfy - tf * ty) / denom

    # ── Coherent weak confirmation ───────────────────────────────────────────
    #
    # A batting event is sometimes shredded by the ASSOCIATION GATE rather than
    # by segmentation. The filter's prediction drifts after an occlusion, and then
    # a run of perfectly real YOLO detections of the same ball lands beyond
    # GATE_RADIUS and is refused, one frame after another. The filter will not
    # recover on its own — a refusal advances the same counter that would have let
    # it recover — and the event ends up with fewer confirmed detections than the
    # delivery bar requires.
    #
    # This path is the ONLY way such an event becomes a delivery, and it is
    # deliberately expensive in evidence:
    #
    #   * the event must already fail the confirmed-detection bar, so a healthy
    #     event is never routed through here and the ordinary path is untouched;
    #   * the TOTAL number of real measurements must still reach MIN_DELIVERY_FRAMES
    #     (asserted at import in core/constants.py), so nothing about how much
    #     evidence is required changes — only where part of it came from;
    #   * six INDEPENDENT signals must agree, two of them mandatory.
    #
    # NOTHING IS FABRICATED. The refused measurements were produced by the same
    # single YOLO pass; they are not re-run, not smoothed, not invented. They
    # never become confirmed detections, never become trajectory points, and never
    # re-enter the Kalman filter. The only thing that changes is that the event
    # carrying them is allowed to be called a delivery, and that it is permanently
    # labelled `coherent_weak_detection` so nothing downstream mistakes it for a
    # straightforward detection.

    def _coherence_signals(self, seg: EventSegment) -> dict:
        """
        Measure all six coherence signals for the open event's evidence.

        Returns a report containing, for every signal, the boolean verdict and the
        numbers behind it. The numbers are kept whether the signal passed or
        failed: "the evidence was thin" and "the evidence was examined and did not
        agree" are different diagnoses and a diagnostic has to be able to tell them
        apart.
        """
        r = self.resolved
        meas = self._measurements
        refused = self._refused
        report: dict = {}

        frames = sorted({m[0] for m in meas})
        span = (frames[-1] - frames[0] + 1) if frames else 0
        n_meas = len(frames)

        # ── 1. TEMPORAL_CONTINUITY ──────────────────────────────────────────
        # A ball being followed produces measurements in an unbroken run. Silence
        # inside the span, and a span broken into several islands, both mean the
        # measurements are not describing one continuous sighting.
        longest_run = 0
        run = 0
        prev: Optional[int] = None
        for f in frames:
            run = run + 1 if (prev is not None and f == prev + 1) else 1
            longest_run = max(longest_run, run)
            prev = f
        silence = max(0, span - n_meas)
        silence_fraction = (silence / span) if span > 0 else 1.0
        contiguous_fraction = (longest_run / n_meas) if n_meas else 0.0
        report[WeakSignal.TEMPORAL_CONTINUITY] = {
            "pass": bool(
                span > 0
                and contiguous_fraction >= r.weak_min_contiguous_fraction
                and silence_fraction <= r.weak_max_silence_fraction
            ),
            "measurements": n_meas,
            "span_frames": span,
            "longest_run_frames": longest_run,
            "contiguous_fraction": round(contiguous_fraction, 4),
            "silence_fraction": round(silence_fraction, 4),
            "contiguous_fraction_min": r.weak_min_contiguous_fraction,
            "silence_fraction_max": r.weak_max_silence_fraction,
            "statement": (
                "measurements must form one unbroken run with little silence "
                "between them, or they are not one continuous sighting"
            ),
        }

        # ── 2. PATH_COHERENCE (mandatory) ───────────────────────────────────
        # Every consecutive pair of real measurements must be connected by a
        # physically possible travel. The radius is built from the event's OWN
        # measured speed, so this asks "could this ball have gone from where it was
        # to where it now is", not "does the path look smooth". Two independent
        # ball sightings at the same moment of play fail this even when both are
        # continuous on their own.
        ref_speed = self._reference_velocity()[2]
        if ref_speed < 1e-6 and span > 0:
            # A fit over the sliding window can be degenerate at the start of an
            # event. Fall back to the event's average travel rate, which is still
            # this ball's own speed and still scale-free.
            path_len = sum(
                _dist((meas[i - 1][1], meas[i - 1][2]), (meas[i][1], meas[i][2]))
                for i in range(1, len(meas))
            )
            ref_speed = path_len / span
        worst_ratio = 0.0
        unreachable_pairs = 0
        if len(meas) >= 2:
            for i in range(1, len(meas)):
                gap = max(1, meas[i][0] - meas[i - 1][0])
                displacement = _dist(
                    (meas[i - 1][1], meas[i - 1][2]), (meas[i][1], meas[i][2])
                )
                radius = (
                    ref_speed * gap * r.weak_reach_tolerance
                    + r.reach_base_px
                    + r.reach_rate_px * gap
                )
                ratio = (displacement / radius) if radius > 1e-9 else float("inf")
                worst_ratio = max(worst_ratio, ratio)
                if ratio > 1.0:
                    unreachable_pairs += 1
        report[WeakSignal.PATH_COHERENCE] = {
            "pass": bool(len(meas) >= 2 and unreachable_pairs == 0),
            "reference_speed_px_per_frame": round(ref_speed, 4),
            "worst_reach_ratio": (
                round(worst_ratio, 3) if worst_ratio != float("inf") else None
            ),
            "unreachable_pairs": unreachable_pairs,
            "reach_tolerance": r.weak_reach_tolerance,
            "statement": (
                "every consecutive measurement must lie inside a reachable "
                "radius built from this event's own measured speed"
            ),
        }

        # ── 3. CORRIDOR_RESIDENCY ───────────────────────────────────────────
        # A ball bowled or struck stays on the pitch; a fielder's boot and a
        # spectator's hat do not. Recorded per measurement rather than derived
        # later, because the corridor is stated in frame fractions and therefore
        # means the same thing at any resolution.
        corridor_ratio = (
            seg.weak_in_corridor_frames / len(refused) if refused else 0.0
        )
        report[WeakSignal.CORRIDOR_RESIDENCY] = {
            "pass": bool(refused and corridor_ratio >= r.weak_corridor_ratio),
            "in_corridor": seg.weak_in_corridor_frames,
            "of": len(refused),
            "ratio": round(corridor_ratio, 4),
            "ratio_min": r.weak_corridor_ratio,
            "statement": (
                "most refused measurements must lie inside the pitch corridor, "
                "which is where a played ball is"
            ),
        }

        # ── 4. PHYSICAL_MOVEMENT ─────────────────────────────────────────────
        # How fast the refused measurements claim the ball travelled, against how
        # fast this event's confirmed detections say it travels. A bat's edge can
        # multiply a ball's speed; it cannot multiply it twentyfold between two
        # adjacent frames, and it cannot slow it to a fifth.
        implied = None
        ratio = None
        if len(refused) >= 2:
            w_span = max(1, refused[-1][0] - refused[0][0])
            implied = _dist(
                (refused[0][1], refused[0][2]), (refused[-1][1], refused[-1][2])
            ) / w_span
            if ref_speed > 1e-6:
                ratio = implied / ref_speed
        report[WeakSignal.PHYSICAL_MOVEMENT] = {
            "pass": bool(
                ratio is not None
                and r.weak_speed_ratio_down <= ratio <= r.weak_speed_ratio_up
            ),
            "implied_speed_px_per_frame": (
                round(implied, 4) if implied is not None else None
            ),
            "reference_speed_px_per_frame": round(ref_speed, 4),
            "speed_ratio": round(ratio, 4) if ratio is not None else None,
            "speed_ratio_band": [r.weak_speed_ratio_down, r.weak_speed_ratio_up],
            "statement": (
                "the refused measurements must imply a speed the event's own "
                "confirmed detections also support"
            ),
        }

        # ── 5. RIVAL_ABSENCE ────────────────────────────────────────────────
        # One ball, or two? Two balls detected during the same passage of play sit
        # in different parts of the frame; one ball's flight traces a bounded
        # region. Measured as the diagonal of the bounding box of every refused
        # position, as a fraction of the frame diagonal so it is resolution
        # independent.
        spread_px = None
        if refused:
            xs = [m[1] for m in refused]
            ys = [m[2] for m in refused]
            spread_px = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        spread_limit = r.weak_spread_fraction * r.frame_diagonal
        report[WeakSignal.RIVAL_ABSENCE] = {
            "pass": bool(
                spread_px is not None and spread_px <= spread_limit
            ),
            "spread_px": round(spread_px, 2) if spread_px is not None else None,
            "spread_limit_px": round(spread_limit, 2),
            "spread_fraction_of_diagonal": r.weak_spread_fraction,
            "statement": (
                "the refused positions must fit in one bounded region; two "
                "balls in the same passage of play do not"
            ),
        }

        # ── 6. PITCH_CONTEXT ────────────────────────────────────────────────
        # Something was happening in the cricket area while these measurements
        # arrived. Measured two ways, because either is sufficient on its own: the
        # pitch corridor was not dead quiet, or the event's own accepted
        # detections were already inside that corridor. A static artefact cannot
        # satisfy either.
        pitch_energy = max(
            (max(m[4], m[5]) for m in refused), default=0.0
        )
        corridor_seen = seg.in_corridor_frames > 0
        report[WeakSignal.PITCH_CONTEXT] = {
            "pass": bool(
                pitch_energy >= r.pitch_activity_quiet_threshold or corridor_seen
            ),
            "peak_pitch_energy": round(pitch_energy, 3),
            "quiet_threshold": r.pitch_activity_quiet_threshold,
            "event_confirmed_in_corridor": corridor_seen,
            "statement": (
                "the pitch area must have been in motion while the refused "
                "measurements arrived, or the event's own detections were in it"
            ),
        }

        return report

    def _promote_weak_evidence(self, seg: EventSegment) -> Optional[dict]:
        """
        Can this thin event be confirmed on coherent weak evidence?

        Returns the full coherence report when the event is PROMOTED, and the same
        report (with `promoted: false` and the reasons) when it is not. Never
        returns None, and never changes a threshold: the caller only decides
        whether to accept the promotion.
        """
        r = self.resolved
        refused = self._refused
        total = len(self._measurements)

        signals = self._coherence_signals(seg)
        agreed = sum(1 for s in WeakSignal.ALL if signals[s]["pass"])
        mandatory_ok = all(signals[s]["pass"] for s in WeakSignal.MANDATORY)

        reasons: list[str] = []
        if seg.confirmed_count < r.weak_min_confirmed_frames:
            reasons.append(
                f"only {seg.confirmed_count} confirmed detections; at least "
                f"{r.weak_min_confirmed_frames} are needed to anchor a refused "
                f"measurement against anything"
            )
        if total < r.weak_min_total_measurements:
            reasons.append(
                f"{total} real measurements in total, below the delivery floor of "
                f"{r.weak_min_total_measurements} (this floor counts confirmed "
                f"AND refused detections and is never lowered)"
            )
        if len(refused) < r.weak_min_candidate_frames:
            reasons.append(
                f"only {len(refused)} gate-refused measurements; at least "
                f"{r.weak_min_candidate_frames} are needed before this is more "
                f"than a stray box"
            )
        if not mandatory_ok:
            missing = [s for s in WeakSignal.MANDATORY if not signals[s]["pass"]]
            reasons.append(
                "the mandatory coherence signals did not agree: " + ", ".join(missing)
            )
        if agreed < r.weak_min_signal_agreement:
            reasons.append(
                f"only {agreed} of {len(WeakSignal.ALL)} coherence signals agreed; "
                f"{r.weak_min_signal_agreement} are required"
            )

        promoted = not reasons
        return {
            "promoted": promoted,
            "signals": signals,
            "signals_agreed": agreed,
            "signals_total": len(WeakSignal.ALL),
            "signals_required": r.weak_min_signal_agreement,
            "confirmed_count": seg.confirmed_count,
            "weak_evidence_count": len(refused),
            "total_measurements": total,
            "total_measurements_required": r.weak_min_total_measurements,
            "confidence": r.weak_confirmation_confidence if promoted else 0.0,
            "reasons": reasons,
            "evidence_only_note": (
                "these measurements were refused by the association gate. They are "
                "recorded as evidence and were not used as detections, as "
                "trajectory points, or as Kalman input."
            ),
        }

    def _close(
        self,
        at_frame: int,
        reason: str,
        kind: str,
        boundary: Optional[BoundaryDecision] = None,
        ambiguous: bool = False,
        note: str | None = None,
        notes: list[str] | None = None,
    ) -> Optional[EventSegment]:
        """Close the open event, record why, and file it."""
        seg = self.open_event
        if seg is None:
            return None

        r = self.resolved
        seg.closed_at_frame = max(int(at_frame), seg.end_frame)
        seg.close_reason = reason

        # HOW FAR PAST ITS LAST CONFIRMED DETECTION AN EVENT MAY REACH
        # -----------------------------------------------------------
        # It may coast for at most the prediction ceiling, and it may never
        # reach the frame on which a new event was found. Both bounds matter:
        # the first stops extrapolation from inventing footage, the second
        # stops this event's window from containing the next one's ball.
        seg.ball_lost_frame = max(
            seg.end_frame,
            min(seg.closed_at_frame - 1, seg.end_frame + r.max_coast_frames),
        )

        if boundary is None:
            boundary = BoundaryDecision(
                at_frame=seg.closed_at_frame,
                kind=kind,
                reason=reason,
                gap_frames=seg.ball_lost_duration,
                gap_seconds=seg.ball_lost_duration / r.fps if r.fps else 0.0,
                notes=list(notes or ([note] if note else [])),
            )
        elif notes:
            boundary.notes.extend(n for n in notes if n not in boundary.notes)
        seg.boundary = boundary
        seg.boundary.reason = reason

        if seg.confirmed_count < r.min_event_confirmed_frames:
            # CASE F. Not enough ACCEPTED measurement to call this a delivery on
            # the standard path. There is one way out, and only one: the
            # gate-refused measurements this event also collected may be coherent
            # enough to confirm it. That decision is measured and recorded either
            # way, and it never invents a detection.
            report = self._promote_weak_evidence(seg)
            seg.weak_confirmation = report
            if report["promoted"]:
                seg.confirmation_method = ConfirmationMethod.COHERENT_WEAK
                seg.confidence = r.weak_confirmation_confidence
                if reason == BoundaryReason.END_OF_VIDEO:
                    # The end of the clip is uncorroborated regardless of how the
                    # event was confirmed, so the video-ended caveat still applies.
                    seg.confidence = min(seg.confidence, 0.9)
                    seg.ambiguous = True
                    seg.ambiguous_reasons.append(
                        "video ended while this event was still open, so its end "
                        "is not corroborated by a subsequent gap"
                    )
            else:
                seg.ambiguous = True
                seg.ambiguous_reasons.append(
                    f"only {seg.confirmed_count} confirmed detections, below the "
                    f"minimum of {r.min_event_confirmed_frames}"
                    + (
                        f"; coherent weak confirmation was attempted and refused "
                        f"({report['signals_agreed']}/{report['signals_total']} "
                        f"signals agreed)"
                        if report["weak_evidence_count"]
                        else "; no gate-refused evidence was available to rescue it"
                    )
                )
                seg.confidence = 0.0
        elif ambiguous:
            seg.ambiguous = True
            seg.confidence = min(seg.confidence, r.ambiguous_confidence)
            seg.ambiguous_reasons.append(
                f"closed as {reason} after {seg.ball_lost_duration} frames "
                f"without a confirmed detection"
            )
        elif reason == BoundaryReason.END_OF_VIDEO:
            seg.confidence = min(seg.confidence, 0.9)
            seg.ambiguous = True
            seg.ambiguous_reasons.append(
                "video ended while this event was still open, so its end is "
                "not corroborated by a subsequent gap"
            )

        self._finalise_evidence(seg)
        self.events.append(seg)
        self.open_event = None
        self._reset_window()
        return seg

    def _reset_window(self) -> None:
        self._window.clear()
        self._positions.clear()
        self._confirmations.clear()
        self._last_confirmed = None
        self._measurements.clear()
        self._refused.clear()

    # ── The audit trail ──────────────────────────────────────────────────────

    def _record_evidence(
        self,
        seg: EventSegment,
        boundary: BoundaryDecision,
        reacquired: bool = False,
    ) -> None:
        if reacquired and seg.confirmed_frames:
            if seg.confirmed_frames[-1] not in seg.reacquired_frames:
                seg.reacquired_frames.append(seg.confirmed_frames[-1])
        if boundary.notes:
            for n in boundary.notes:
                if n not in seg.evidence.setdefault("notes", []):
                    seg.evidence["notes"].append(n)

    def _finalise_evidence(self, seg: EventSegment) -> None:
        """
        Assemble the per-event evidence block the report and validator read.

        Everything here is measured from the single pass. `batting_event_*` is
        recorded as null deliberately: this pipeline has no independent batting
        event detector, so the field exists and is honestly empty rather than
        being back-filled from the shot classifier (which runs later, and must
        not be the segmenter).
        """
        r = self.resolved
        frames = seg.confirmed_frames
        runs = _ranges(frames)
        longest_run = max((b - a + 1 for a, b in runs), default=0)
        internal_gaps = [
            [runs[i][1] + 1, runs[i + 1][0] - 1]
            for i in range(len(runs) - 1)
            if runs[i + 1][0] - runs[i][1] > 1
        ]
        vx, vy, speed = self._reference_velocity()
        positions = list(self._positions)

        # How this event ENTERED and how it LEFT, each fitted over its own end of
        # the confirmed sequence. A fragment that ends mid-flight and one that
        # begins after a split are only comparable when both velocities are known
        # from both sides, which is what `FragmentMerger` uses to decide whether
        # the split was physical or merely a bad local fit.
        window = max(2, r.velocity_window)
        conf = self._confirmations
        if len(conf) >= 2:
            seg.entry_velocity = self._fit_velocity(conf[:window])
            seg.exit_velocity = self._fit_velocity(conf[-window:])

        seg.evidence.update({
            "first_confirmed_detection": frames[0] if frames else None,
            "last_confirmed_detection": frames[-1] if frames else None,
            "confirmed_runs": runs,
            "longest_confirmed_run": longest_run,
            "predicted_frames": seg.predicted_count,
            "confirmed_ratio": (
                seg.confirmed_count / max(1, seg.confirmed_count + seg.predicted_count)
            ),
            "ball_loss_duration_frames": seg.ball_lost_duration,
            "ball_loss_duration_seconds": (
                round(seg.ball_lost_duration / r.fps, 4) if r.fps else None
            ),
            "tracker_state_at_close": seg.evidence.get("tracker_state_at_close"),
            "ball_reacquisitions": list(seg.reacquired_frames),
            "gated_continuations": list(seg.gated_continuation_frames),
            "trajectory_displacement_px": (
                round(_dist(positions[0], positions[-1]), 2)
                if len(positions) >= 2 else None
            ),
            "path_length_px": (
                round(
                    sum(
                        _dist(positions[i - 1], positions[i])
                        for i in range(1, len(positions))
                    ),
                    2,
                )
                if len(positions) >= 2 else None
            ),
            "reference_velocity_px_per_frame": [round(vx, 4), round(vy, 4)],
            "reference_speed_px_per_frame": round(speed, 4),
            "internal_gaps": internal_gaps,
            "max_internal_gap": max(
                (b - a + 1 for a, b in internal_gaps), default=0
            ),
            "batting_event_timing": None,
            "batting_event_timing_note": (
                "no independent batting-event detector exists in this pipeline; "
                "recorded as null rather than inferred from the shot classifier, "
                "which runs after segmentation and must not drive it"
            ),
            "in_pitch_corridor_frames": seg.in_corridor_frames,
            "corridor_ratio": round(seg.corridor_ratio, 4),
            # The evidence-only channel, summarised alongside the confirmed one so
            # a consumer can see at a glance how much of this event the filter
            # accepted and how much it merely refused.
            "weak_evidence_frames": seg.weak_evidence_count,
            "weak_evidence_frame_ranges": _ranges(seg.weak_evidence_frames),
            "weak_in_pitch_corridor_frames": seg.weak_in_corridor_frames,
            "weak_corridor_ratio": (
                round(seg.weak_in_corridor_frames / max(1, seg.weak_evidence_count), 4)
            ),
            "total_measurement_count": seg.total_measurement_count,
            "confirmation_method": seg.confirmation_method,
            "first_position": list(seg.first_position) if seg.first_position else None,
            "last_position": list(seg.last_position) if seg.last_position else None,
            "entry_velocity": (
                [round(v, 4) for v in seg.entry_velocity] if seg.entry_velocity else None
            ),
            "exit_velocity": (
                [round(v, 4) for v in seg.exit_velocity] if seg.exit_velocity else None
            ),
        })


def _tally(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        if v:
            out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))