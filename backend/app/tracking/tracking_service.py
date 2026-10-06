"""
tracking_service.py — THE single pass over the uploaded video.

This module is the architectural centre of CricketUnified. Everything below
exists to keep one promise (UNIFIED_ARCHITECTURE.md §1, C1–C5):

    The source video is opened once, decoded once, and passed through YOLO +
    Kalman exactly once. Everything else — delivery boundaries, clips, stored
    trajectories, physics — is a by-product of that one pass.

WHAT HAPPENS PER FRAME
----------------------
    1. Decode a frame (the only time the source is ever read).
    2. Push it into the ring buffer so a clip can later start BEFORE this
       delivery was detected.
    3. Hand it to the open clip writer, if a delivery clip is currently open.
       One delivery can be open at a time thanks to the delivery debounce.
    4. Run YOLO detection.
    5. Run the Kalman update, recording the resulting position WITH PROVENANCE
       (detected vs extrapolated).
    6. If the ball has been lost long enough, finalise the delivery: derive
       physics, open its clip, and persist the record.

WHY CLIP EXTRACTION LIVES INSIDE THIS LOOP
-------------------------------------------
Upstream ran the tracker over the whole video and then re-opened the file to cut
clips. That second decode is the cost this project exists to remove. Clips can
only be cut here, while the frames are still in hand.

PROVENANCE IS RECORDED, NEVER INFERRED
--------------------------------------
Every trajectory point records whether it came from a YOLO detection on that
frame or from a Kalman extrapolation. Nothing is interpolated, smoothed into
existence, or back-filled. A frame with no tracked point simply has no point,
and `Trajectory.gaps` reports the hole rather than hiding it.

NO SHOT INFERENCE HERE
----------------------
Shot classification needs the finished clip. It is run by `shot_classifier`
against the frames this pass already retained, or against the written clip
file. It never re-decodes the source video.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from ..core.config import PipelineConfig, Paths
from ..core.constants import (
    PITCH_CORRIDOR,
    MAX_MISSED,
    MIN_DELIVERY_FRAMES,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
)
from ..schemas.tracking import (
    Delivery,
    OverlayInfo,
    Quality,
    Trajectory,
    TrajectoryPoint,
)
from ..storage.persistence import atomic_write_json
from ..analytics.bowling import analyse_delivery
from ..analytics.calibration import calibration_metadata, make_homography
from ..segmentation.purity_validator import (
    ClipValidator,
    SINGLE_EVENT,
    ValidationReport,
)
from ..video.clips import ClipWriteResult, ClipWriterRegistry
from ..segmentation.barriers import (
    ActivityWindowTracker,
    FrameSignal,
    SignalTimeline,
)
from ..segmentation.candidate_generation import (
    BoundaryReason,
    ContinuationKind,
    EventSegment,
    EventSegmenter,
    FrameEvidence,
)
from ..segmentation.fragment_merger import BarrierSet, FragmentMerger
from ..video.frame_buffer import FrameRingBuffer
from ..video.video_io import get_fps
from .detector import BallDetector
from .kalman_tracker import BallTracker

log = logging.getLogger(__name__)

# Progress callback: (phase, frames_done, frames_total, detail)
ProgressFn = Callable[[str, int, int, str], None]


def _noop_progress(phase: str, done: int, total: int, detail: str) -> None:
    pass


class VideoOpenError(Exception):
    """Raised when the source video cannot be opened at all."""


def _tracker_state(track, tracker) -> str:
    """Helper to derive readable tracker state for evidence logging."""
    if track.source == "detected":
        return "tracking"
    elif track.source == "predicted":
        return "coasting"
    elif getattr(tracker, "in_delivery", False):
        return "coasting"
    return "untracked"


def _needs_reseed(decision, track, tracker, detection) -> bool:
    """
    Should the filter be restarted on this frame's real detection?

    Event identity belongs to the segmenter; this filter only smooths between
    frames. When the segmenter rules "same ball, keep the event open" it does so
    on the strength of a REAL measurement — `reseed_position` is only ever set
    from a confirmed or a gate-rejected detection (event_segmenter.py, the two
    `reseed_position=(int(px), int(py))` returns), and never from a prediction.

    WHY THE OLD GUARD COULD NOT FIRE
    ---------------------------------
    The call site additionally required `track.position is None`. That is
    unreachable in combination. `BallTracker.update` returns a null position only
    on the `detection is None` branch, once `missed_frames > MAX_MISSED_FRAMES`;
    but with no detection there is no measurement for the segmenter to rule on,
    so `reseed_position` is never set. The two conditions were mutually
    exclusive, so `BallTracker.reseed()` was dead code on every run. Measured on
    real_cricket.mp4: all 120 reacquisitions carried `kalman_missed=0`, i.e. the
    filter happened to still be healthy and the reseed was redundant even if it
    had been reachable.

    The case reseed exists for therefore never ran: the ball is plainly on screen
    and the segmenter has confirmed it belongs to the open event, yet the filter's
    own prediction has drifted so far that its 120 px gate refuses the box. The
    filter is then disowned AND the refusal advances `missed_frames`, so it cannot
    recover on its own — the real trajectory gets shredded into successive short
    fragments (the #34+#35+#36 and #77+#78+#79 chains in the Stage 4 diagnostic).

    Reseed exactly when the filter did not accept a measurement the segmenter
    trusts. A detection the filter absorbed cleanly leaves it healthy and is left
    alone, so the ordinary path — including every Stage 3 recovery — is unchanged.
    """
    if decision is None or decision.reseed_position is None:
        return False
    # Never fabricate: re-seeding needs a real box, not a prediction.
    if detection is None:
        return False
    if getattr(track, "gated_out", False):
        return True          # gate refused a ball we know is on screen
    if tracker is not None and bool(getattr(tracker, "is_lost", False)):
        return True          # filter disowned itself; re-acquire in place
    return False


@dataclass
class _PendingEvent:
    """
    A closed event awaiting its merge decision, with the trajectory points it
    collected.

    The points travel with the segment because emission is deferred: a candidate
    that later absorbs a follower has to hand the follower its points too, and
    the single `cur_points` accumulator has already been reset for the new event
    by then.
    """

    segment: EventSegment
    points: list = field(default_factory=list)
    # Frame cap from the split that closed this candidate: the frame before the
    # next event was first seen. None when it closed at end of video.
    hard_end_frame: Optional[int] = None
    # Every merge attempt made about this candidate, passed or failed, so the
    # audit trail records why a candidate was NOT merged too.
    merges: list = field(default_factory=list)


@dataclass
class TrackingResult:
    """Everything the single pass produced, ready for downstream stages."""

    analysis_id: str
    deliveries: list[Delivery] = field(default_factory=list)
    fps: float = 25.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    frames_decoded: int = 0
    wall_seconds: float = 0.0
    homography: Optional[np.ndarray] = None
    source_caps: dict = field(default_factory=dict)
    # Whether real per-video pitch corners were supplied. False means every
    # ground-plane quantity (speed, length, line, swing, bounce angle) is null by
    # policy, not by accident.
    geometry_calibrated: bool = False
    ring_stats: dict = field(default_factory=dict)
    clip_stats: dict = field(default_factory=dict)
    # Per-delivery clip results, including the frames retained in memory for
    # shot classification. Populated only after the pass completes, because a
    # clip is not finished being written until its post-roll has streamed past.
    clip_results: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    trajectory_global: list[dict] = field(default_factory=list)
    # Every batting event the segmenter found, including events too thin to
    # become a delivery. Kept so a diagnostic can report what was rejected and
    # why, which is impossible once only the accepted deliveries survive.
    segments: list = field(default_factory=list)
    # The independent single-event verdict for each written clip.
    validation: Optional[ValidationReport] = None
    # Event horizons (exclusion zones) the pass discovered: rejected candidates
    # and untracked activity windows. A clip may never cross one.
    event_horizons: list = field(default_factory=list)
    segmentation_summary: dict = field(default_factory=dict)
    # Frames on which the filter was re-acquired in place on a real detection
    # after the segmenter had ruled the ball was never lost. Previously always 0:
    # `BallTracker.reseed()` sat behind an unreachable guard, so this counter is
    # also the proof that the fix is live.
    reseeds_applied: int = 0
    # Every fragment-merge decision the pass made, merged or not, for audit.
    fragment_merges: list = field(default_factory=list)
    # Every window the pre-persistence purity gate refused, with the verdict and,
    # where one was possible, the narrower window it could not be repaired to. Kept
    # so a refusal is as auditable as an acceptance: "we dropped a delivery and
    # here is why" is only useful if the reason survives the run.
    purity_rejections: list = field(default_factory=list)
    # Cumulative seconds spent in each part of the loop. The pass is
    # detector-bound, but "detector-bound" is a claim, and this turns it into a
    # measurement the report can cite.
    stage_seconds: dict = field(default_factory=dict)

    def time_stage(self, name: str, seconds: float) -> None:
        self.stage_seconds[name] = (
            self.stage_seconds.get(name, 0.0) + seconds
        )

    def tracking_document(self, config: PipelineConfig) -> dict:
        """
        The persisted `tracking.json` (schema 2.0).

        This is the artefact downstream stages consume. It deliberately contains
        no shot predictions and no overlay paths — those are attached later to
        `result.json`, so a consumer can tell what the tracker actually measured
        from what was inferred later.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "analysis_id": self.analysis_id,
            "video": {
                "fps": round(self.fps, 6),
                "width": self.width,
                "height": self.height,
                "total_frames": self.total_frames,
                "frames_decoded": self.frames_decoded,
                "duration_sec": (
                    round(self.frames_decoded / self.fps, 3) if self.fps > 0 else None
                ),
                # Read once, at open time, from the container. Not trusted for
                # frame numbering — see frame_numbering below.
                "reported_frame_count": self.source_caps.get("reported_frame_count"),
            },
            "calibration": calibration_metadata(
                self.pitch_corners_px if self.geometry_calibrated else None
            ),
            "config": config.as_metadata(),
            "performance": {
                "wall_seconds": round(self.wall_seconds, 3),
                "source_decodes": 1,
                "fps_processed": (
                    round(self.frames_decoded / self.wall_seconds, 2)
                    if self.wall_seconds > 0
                    else None
                ),
                "stage_seconds": {
                    k: round(v, 3) for k, v in sorted(
                        self.stage_seconds.items(), key=lambda kv: -kv[1]
                    )
                },
            },
            "ring_buffer": self.ring_stats,
            "clip_writer": self.clip_stats,
            "warnings": self.warnings,
            "segmentation": {
                **self.segmentation_summary,
                "reseeds_applied": self.reseeds_applied,
                "fragment_merges_attempted": len(self.fragment_merges),
                "fragment_merges_accepted": sum(
                    1 for m in self.fragment_merges if getattr(m, "merged", False)
                ),
                "fragment_merges": [
                    {
                        "merged": getattr(m, "merged", False),
                        "reason": getattr(m, "reason", ""),
                        "gap_frames": getattr(m, "gap_frames", 0),
                        "gap_seconds": round(getattr(m, "gap_seconds", 0.0), 4),
                        "displacement_px": (
                            round(m.displacement_px, 2)
                            if getattr(m, "displacement_px", None) is not None
                            else None
                        ),
                        "reachable_radius_px": (
                            round(m.reachable_radius_px, 2)
                            if getattr(m, "reachable_radius_px", None) is not None
                            else None
                        ),
                        "direction_change_deg": (
                            round(m.direction_change_deg, 1)
                            if getattr(m, "direction_change_deg", None) is not None
                            else None
                        ),
                        "combined_confirmed": getattr(m, "combined_confirmed", 0),
                        "absorbed": list(getattr(m, "absorbed", [])),
                        "checks": dict(getattr(m, "checks", {})),
                    }
                    for m in self.fragment_merges
                ],
                "event_horizons": list(self.event_horizons),
                "purity_rejections": list(self.purity_rejections),
                "purity_rejected_count": len(self.purity_rejections),
                "events": [
                    s.as_dict() if hasattr(s, "as_dict") else s
                    for s in self.segments
                ],
            },
            "validation": self.validation.as_dict() if self.validation else None,
            "deliveries": [d.as_dict() for d in self.deliveries],
        }


class TrackingService:
    """
    Runs the one and only ball-detection pass over a video.

    One instance handles exactly one video. `analyse()` opens the source, streams
    it once, and returns a TrackingResult.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    # ── Public API ───────────────────────────────────────────────────────────
    def analyse(
        self,
        video_path: str,
        paths: Paths,
        progress: Optional[ProgressFn] = None,
        detector: Optional[BallDetector] = None,
    ) -> TrackingResult:
        """
        Decode and track *video_path* in a single pass.

        *detector* may be supplied so several analyses share one loaded YOLO
        model instead of paying the load cost each time.
        """
        emit = progress or _noop_progress
        t_start = time.perf_counter()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoOpenError(f"Could not open video: {video_path}")
        try:
            return self._run(cap, video_path, paths, emit, t_start, detector)
        finally:
            cap.release()

    # ── The pass ─────────────────────────────────────────────────────────────
    def _run(
        self,
        cap: cv2.VideoCapture,
        video_path: str,
        paths: Paths,
        emit: ProgressFn,
        t_start: float,
        detector: Optional[BallDetector],
    ) -> TrackingResult:
        cfg = self.config

        # ── Source properties, read ONCE ─────────────────────────────────────
        fps = get_fps(cap, default=25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Recorded for diagnostics only. Frame numbering below comes from the
        # decode counter, NOT from this value: CAP_PROP_FRAME_COUNT is derived
        # from container metadata and is routinely wrong for variable frame rate
        # and B-frame video, which would break every clip mapping.
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if width <= 0 or height <= 0:
            raise VideoOpenError(
                f"Video reports {width}x{height}; cannot track without real dimensions"
            )

        emit("opening", 0, reported_frames or 0, "loading detector")

        if detector is None:
            detector = BallDetector()
        if detector.model is None:
            raise VideoOpenError(
                "Ball detector failed to load; no analysis possible. "
                f"Model: {detector.model_path}"
            )

        # Ground-plane geometry comes from THIS video's pitch corners when the
        # operator measured them, and from the fixed template fractions
        # otherwise. `cfg.geometry_calibrated` is the single gate, and it is
        # False unless real corners were supplied — the template corners are not a
        # calibration and every ground-plane number derived from them describes a
        # different camera angle.
        geometry_calibrated = cfg.geometry_calibrated
        homography = make_homography(
            width, height, corners_px=cfg.pitch_corners_px
        )
        tracker = BallTracker()

        # The segmenter owns delivery identity. The Kalman filter smooths the ball
        # between frames; it does not decide where one batting event ends.
        # Boundaries come from here, from the provenance this same pass recorded,
        # so no second decode, no second detector run and no second filter run is
        # involved. Built before the ring buffer because the buffer has to be
        # large enough to still hold a clip's pre-roll after the merge deferral
        # below — see `ring_capacity`.
        segmenter = EventSegmenter(
            fps=fps, width=width, height=height, config=cfg.segmentation
        )
        resolved = segmenter.resolved

        # A candidate is held for at most the merge horizon before it is emitted,
        # so by the time its clip window is computed up to
        # `max_pre_roll_frames` more frames have streamed past. The buffer must
        # cover the pre-roll, the deferral, the post-roll and a little slack, or
        # the pre-roll has already been evicted and the clip silently starts late.
        # RING_BUFFER_MAX_BYTES still governs memory; this only raises the frame
        # ceiling when the timing genuinely requires it.
        ring_capacity = max(
            cfg.ring_capacity_frames,
            resolved.max_pre_roll_frames
            + resolved.merge_horizon_frames
            + resolved.max_post_roll_frames
            + 10,
        )
        ring = FrameRingBuffer(
            capacity_frames=ring_capacity,
            max_bytes=cfg.ring_max_bytes,
            jpeg_quality=cfg.ring_jpeg_quality,
        )
        clips = ClipWriterRegistry(ring, codec_preference=cfg.codec_preference)

        result = TrackingResult(
            analysis_id=paths.analysis_id,
            fps=fps,
            width=width,
            height=height,
            total_frames=reported_frames,
            homography=homography,
            source_caps={
                "reported_frame_count": reported_frames,
                "path": str(video_path),
            },
            geometry_calibrated=geometry_calibrated,
        )

        pre_roll = cfg.pre_roll_frames(fps)
        post_roll = cfg.post_roll_frames(fps)

        # ── Event horizons: the per-frame barrier evidence ─────────────────
        # Recorded from the SAME single pass. The clip-window search reads it to
        # bound pre-roll/post-roll, so a window stops at a black gap, a camera
        # cut, a reset valley or a neighbouring event instead of blindly
        # extending. Costs a few scalars per frame.
        signals = SignalTimeline(
            quiet_threshold=resolved.pitch_activity_quiet_threshold,
            active_threshold=resolved.pitch_activity_active_threshold,
            black_run_barrier=resolved.black_frame_run_barrier,
            fps=fps,
        )
        activity = ActivityWindowTracker(
            quiet_threshold=resolved.pitch_activity_quiet_threshold,
            active_threshold=resolved.pitch_activity_active_threshold,
            quiet_hold_frames=resolved.activity_quiet_hold_frames,
            min_window_frames=resolved.activity_min_window_frames,
        )

        delivery_id = 1
        frame_number = 0

        # Per-delivery accumulators. Reset at every event boundary the segmenter
        # announces — which is the only place a boundary is ever created.
        cur_points: list[TrajectoryPoint] = []

        # Speeds computed so far in this analysis, used ONLY if the operator
        # explicitly enables imputation (decision D1 — off by default).
        speeds_seen: list[float] = []

        # ── Fragment merging ─────────────────────────────────────────────────
        # A candidate cannot be turned into a delivery until we know whether the
        # next candidate is the rest of it, so closed candidates are HELD for at
        # most the merge horizon before being emitted. The queue is bounded by
        # that horizon, and it holds no frames — only segments and their already
        # recorded points — so nothing here costs a second pass.
        merger = FragmentMerger(resolved=resolved)
        pending: list[_PendingEvent] = []

        def commit(entry: _PendingEvent) -> None:
            """Emit one finished (and possibly merged) candidate."""
            emit_event(
                entry.segment,
                points=entry.points,
                hard_end_frame=entry.hard_end_frame,
            )
            result.fragment_merges.extend(entry.merges)
            # Whatever could be merged with this has now been decided, so its own
            # span becomes a barrier for anything that follows.
            clips.register_exclusion_zone(
                entry.segment.start_frame,
                entry.segment.ball_lost_frame or entry.segment.end_frame,
                "COMMITTED_EVENT",
            )

        def flush_pending(now_frame: int, force: bool = False) -> None:
            """
            Emit every pending candidate that can no longer be merged.

            A candidate is mergeable only with a candidate starting within
            `fragment_merge_max_gap_frames`. Once the current frame is further
            than that past the candidate's end, no future candidate can partner
            it and the answer is final. That bound is what makes this a one-pass
            streaming decision rather than a post-hoc pass over the whole video.
            """
            while pending:
                head = pending[0]
                if not force and (
                    now_frame - head.segment.end_frame
                    <= resolved.fragment_merge_max_gap_frames
                ):
                    return
                pending.pop(0)
                commit(_resolve_pending(head, pending))

        def _resolve_pending(
            head: _PendingEvent, rest: list[_PendingEvent]
        ) -> _PendingEvent:
            """
            Merge `head` with as many following candidates as the physics allows.

            Chained, because a single stroke is not always cut in two: #34+#35+#36
            held 2+7+2 detections of one event. Each additional fragment is
            admitted only on a full passing merge attempt against the running
            result, so a chain can never accumulate beyond what each step earns.
            """
            barriers = _barrier_snapshot(clips, signals, rest)
            rivals = [p.segment for p in rest]
            while rest:
                attempt = merger.try_merge(
                    head.segment, rest[0].segment, barriers, rivals
                )
                if not attempt.merged:
                    head.merges.append(attempt)
                    break
                result.fragment_merges.append(attempt)
                follower = rest.pop(0)
                merger.merge(head.segment, follower.segment, attempt)
                head.points = head.points + follower.points
                # The merged event inherits the LAST fragment's cap, because that
                # is the boundary in front of the combined span. A cap recorded at
                # the split we just overrode describes where this event used to
                # stop, which is no longer meaningful.
                head.hard_end_frame = follower.hard_end_frame
            return head

        def _barrier_snapshot(
            registry: ClipWriterRegistry, timeline: SignalTimeline, rest: list
        ) -> BarrierSet:
            """Barriers a merge may not cross, from this pass only."""
            spans = BarrierSet()
            for start, end in registry.exclusion_ranges():
                spans.add(start, end)
            for entry in rest:
                spans.add(entry.segment.start_frame, entry.segment.end_frame)
            # A candidate's own confirmed span is not a barrier against itself.
            return spans.merge()

        def register_horizon(segment: EventSegment, kind: str) -> None:
            """
            File an impermeable barrier for a rejected/untracked event.

            The zone keeps LATER clips out. It cannot rewrite frames already on
            disk, so where the rejection is discovered while a clip is still open
            the clip is also cut — but only before it has written into the zone.
            Truncating past written frames would leave bytes the shortened frame
            map does not account for, which is worse than a clip that ends a little
            early; in that case the zone alone protects the next clip.
            """
            start = int(segment.start_frame)
            end = int(
                segment.ball_lost_frame
                or segment.closed_at_frame
                or segment.end_frame
            )
            end = max(end, int(segment.end_frame))
            clips.register_exclusion_zone(start, end, kind)
            result.event_horizons.append({
                "kind": kind,
                "start_frame": start,
                "end_frame": end,
                "confirmed_count": segment.confirmed_count,
                "event_index": segment.index,
            })
            clips.close_before(
                start,
                reason=f"rejected_candidate:{kind}",
            )

        def register_untracked_window(window) -> None:
            """
            A motion cluster with no confirmed ball detection at all.

            This is the ZERO-DETECTION EVENT case. Nothing is invented: no
            DeliveryRecord, no trajectory, no clip is produced for it. But it
            IS a temporal barrier, so a later clip's pre-roll can never reach
            back through a delivery the detector missed.
            """
            clips.register_exclusion_zone(
                window.start, window.end, "UNTRACKED_ACTIVITY_WINDOW"
            )
            result.event_horizons.append({
                "kind": "UNTRACKED_ACTIVITY_WINDOW",
                "start_frame": window.start,
                "end_frame": window.end,
                "peak_motion": round(float(window.peak_motion), 3),
            })
            result.warnings.append(
                f"Untracked activity window {window.start}-{window.end} "
                f"(peak motion {window.peak_motion:.1f}) has no confirmed ball "
                f"detection; kept as an exclusion zone, not emitted as a "
                f"delivery."
            )
            # If a clip is open and has not yet written into this window, stop
            # it before the window so the untracked delivery is never swallowed.
            clips.close_before(
                window.start, reason="untracked_activity"
            )

        def reject_delivery(
            segment: EventSegment, verdict, window: tuple[int, int]
        ) -> None:
            """
            File a delivery the pre-persistence purity gate refused.

            Nothing has been written and nothing has been persisted, so this costs
            only a warning and a horizon. The horizon matters: the reason the
            window was unsafe is that the footage belongs to something else, and
            the next delivery's pre-roll has to be kept out of exactly that span.
            """
            result.warnings.append(
                f"Refused delivery for event {segment.index} at frames "
                f"{window[0]}-{window[1]}: {verdict.classification} — "
                f"{verdict.reason}"
            )
            result.purity_rejections.append({
                "event_index": segment.index,
                "delivery_id_candidate": delivery_id,
                "window": [int(window[0]), int(window[1])],
                **verdict.as_dict(),
            })
            register_horizon(segment, "PURITY_REJECTED_EVENT")
            log.error(
                "Purity gate refused event %d window %d-%d: %s",
                segment.index, window[0], window[1], verdict.reason,
            )

        purity = ClipValidator(
            fps=fps, width=width, height=height,
            config=cfg.segmentation, resolved=resolved,
        )

        def emit_event(
            segment: Optional[EventSegment],
            points: Optional[list] = None,
            hard_end_frame: Optional[int] = None,
        ) -> None:
            """
            Turn one finished batting event into a delivery, or drop it.

            Called only by the segmenter, so a delivery exists if and only if an
            event was found. Nothing here can create an event from nothing, and
            nothing here can merge two: a merge would require the segmenter to
            report one event covering two balls, which is the case the
            validator exists to catch.

            `points` must be the trajectory the CANDIDATE collected, carried on
            the `_PendingEvent`. It cannot be read from the loop's accumulator
            here, because emission is deferred: a candidate held for a merge
            decision is committed several events later, by which time
            `cur_points` belongs to a different ball entirely. Reading it would
            file one delivery's trajectory under another's boundaries and then
            clear the accumulator, destroying the event still being tracked.

            `hard_end_frame` caps the clip window at a frame already known to
            belong to something else. It is set only when the close was caused
            by a new ball being visible at this very frame; capping then means
            the clip is never even opened over the next event's footage, rather
            than relying on a later truncation to remove it.
            """
            nonlocal delivery_id

            if segment is None:
                return
            if not segment.confirmed_frames:
                # CASE F: no ball was ever confirmed. Nothing to deliver, and
                # nothing may be invented to fill the gap.
                result.warnings.append(
                    f"Discarded event candidate at frame {segment.start_frame}: "
                    f"no confirmed ball detection."
                )
                register_horizon(segment, "NO_CONFIRMED_DETECTIONS")
                return
            if segment.confirmed_count < cfg.min_delivery_frames:
                # Too few ACCEPTED detections for the standard bar. There is
                # exactly one way through, and the segmenter is the only thing
                # that can grant it: a coherent weak promotion, measured from the
                # gate-refused measurements this event collected. The bar itself
                # is untouched — the event still had to carry MIN_DELIVERY_FRAMES
                # real measurements in total, and several independent signals had
                # to agree that they were all the same ball. A promoted event is
                # permanently labelled as such downstream.
                if not segment.is_weak_confirmation:
                    result.warnings.append(
                        f"Discarded event at frames {segment.start_frame}-"
                        f"{segment.end_frame}: only {segment.confirmed_count} "
                        f"confirmed detections, below the minimum of "
                        f"{cfg.min_delivery_frames}. Kept in the segmentation "
                        f"evidence as a rejected event, not emitted as a delivery."
                    )
                    register_horizon(segment, "REJECTED_EVENT_CANDIDATE")
                    log.info(
                        "Rejected thin event %d: %d confirmed frames at %d-%d",
                        segment.index, segment.confirmed_count,
                        segment.start_frame, segment.end_frame,
                    )
                    return

            delivery = self._finalise(
                delivery_id=delivery_id,
                segment=segment,
                points=points or [],
                homography=homography,
                fps=fps,
                width=width,
                height=height,
                pre_roll=pre_roll,
                post_roll=post_roll,
                hard_end_frame=hard_end_frame,
                paths=paths,
                clips=clips,
                cfg=cfg,
                speeds_seen=speeds_seen,
                resolved=resolved,
                signals=signals,
                available_frame=frame_number,
                purity=purity,
                segments=segmenter.events,
                on_purity_reject=reject_delivery,
            )
            delivery_id += 1

            if delivery is None:
                # Refused by the pre-persistence purity gate. No frame was
                # written, no record was appended; the evidence is filed as a
                # horizon so the neighbourhood can still be segmented correctly.
                return

            result.deliveries.append(delivery)
            if delivery.bowling.speed_kmh is not None and not delivery.bowling.speed_imputed:
                speeds_seen.append(delivery.bowling.speed_kmh)

            # Per-delivery analytics and the clip window opened here, in the
            # middle of the single pass. Reported as a substage rather than as a
            # top-level phase: no stage boundary was crossed, and pretending
            # otherwise would describe a pipeline this codebase does not have.
            emit(
                "delivery_finalise", delivery.delivery_id, delivery.delivery_id,
                f"delivery {delivery.delivery_id}: analytics + clip window",
            )

            # Persist after every delivery so a crash still leaves every
            # completed delivery recoverable.
            self._append_to_tracking(tracking_path, delivery, cfg, result)

        tracking_path = paths.tracking_json
        atomic_write_json(
            tracking_path,
            TrackingResult(
                analysis_id=paths.analysis_id,
                fps=fps, width=width, height=height,
                total_frames=reported_frames,
                homography=homography,
                source_caps=result.source_caps,
            ).tracking_document(cfg),
        )

        last_emit = time.perf_counter()
        emit("tracking", 0, reported_frames or 0, "starting pass")

        last_gray_small = None
        consecutive_black_frames = 0
        no_detection_run = 0
        post_loss_action = False

        while True:
            t_tick = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            t_decode = time.perf_counter()
            result.time_stage("decode", t_decode - t_tick)

            # 1-based, from the decode counter. Authoritative timebase.
            frame_number += 1

            # Lightweight per-frame vision signals on downsampled 80x45 grayscale
            gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 45), interpolation=cv2.INTER_AREA)
            mean_lum = float(np.mean(gray_small))
            is_black = (mean_lum < cfg.segmentation.black_frame_luminance_threshold)
            
            if is_black:
                consecutive_black_frames += 1
            else:
                consecutive_black_frames = 0

            # Scene cut and motion
            is_scene_cut = False
            motion_energy = 0.0
            pitch_motion = 0.0
            if last_gray_small is not None:
                diff = cv2.absdiff(gray_small, last_gray_small)
                motion_energy = float(np.mean(diff))
                if motion_energy > cfg.segmentation.scene_cut_diff_threshold:
                    is_scene_cut = True
                # Pitch corridor: center 40% horizontally, middle 60% vertically
                pitch_roi = diff[9:36, 24:56]
                pitch_motion = float(np.mean(pitch_roi))
            last_gray_small = gray_small

            # 2. Retain the frame so a clip can start before this event.
            ring.push(frame_number, frame)
            t_ring = time.perf_counter()
            result.time_stage("ring_encode", t_ring - t_decode)

            # 3. Detection — the only YOLO call in the system.
            cx, cy, conf = detector.detect_with_confidence(frame)
            detection = None if cx is None else (cx, cy)
            # Detection silence: frames in a row where the detector saw no ball at
            # all, before the gate ever got a say. A tracker coasting on Kalman
            # prediction still produces `track.position` here, so this counts real
            # blindness, not interpolation.
            if detection is None:
                no_detection_run += 1
            else:
                no_detection_run = 0
            t_detect = time.perf_counter()
            result.time_stage("detect", t_detect - t_ring)

            # 4. Track. This is a short-range smoother; whether it continues the
            #    current event or ends it is the segmenter's decision, not the
            #    filter's.
            track = tracker.update(detection, frame_number)
            t_track = time.perf_counter()
            result.time_stage("kalman", t_track - t_detect)

            # 5. Segment. Everything the boundary decision needs comes from this
            #    one call: provenance, tracker state, and the raw detection when
            #    the gate refused it.
            #
            #    Whether the BALL lies in the pitch corridor is recorded per
            #    measurement, so a candidate can be judged on its own corridor
            #    context during fragment merging: a stroke stays on the pitch, a
            #    fielder's boot does not. Same box as the pitch-motion ROI above,
            #    stated in fractions so it means the same thing at any resolution.
            in_corridor = False
            if detection is not None:
                cx0, cx1, cy0, cy1 = PITCH_CORRIDOR
                in_corridor = (
                    cx0 * width <= detection[0] <= cx1 * width
                    and cy0 * height <= detection[1] <= cy1 * height
                )
            evidence = FrameEvidence(
                frame=frame_number,
                confirmed=(track.source == "detected"),
                candidate=track.gated_out,
                position=detection,
                confidence=conf,
                tracker_position=track.position,
                tracker_state=_tracker_state(track, tracker),
                is_black=is_black,
                is_scene_cut=is_scene_cut,
                motion_energy=motion_energy,
                pitch_motion=pitch_motion,
                in_pitch_corridor=in_corridor,
            )
            decision = segmenter.observe(evidence)

            # 5b. Event-horizon signals for the clip-window search. Recorded for
            #     EVERY frame from the same pass, so the boundary search never
            #     needs a second decode. Cheap: a few scalars per frame.
            signals.record(FrameSignal(
                frame=frame_number,
                is_black=is_black,
                is_scene_cut=is_scene_cut,
                motion=motion_energy,
                pitch_motion=pitch_motion,
                confirmed=(track.source == "detected"),
                candidate=bool(track.gated_out),
            ))
            closed_window = activity.update(
                frame_number, motion_energy,
                had_confirmed=(track.source == "detected"),
                had_candidate=bool(track.gated_out),
            )
            if closed_window is not None and closed_window.is_untracked:
                register_untracked_window(closed_window)

            # Snapshot the filter's state BEFORE any reseed below rewrites it.
            #
            #     A reseed replaces the filter with the box the filter had just
            #     REFUSED, so a trajectory point recorded afterwards would file
            #     that box as a measurement this event accepted — on a frame the
            #     segmenter had simultaneously classified as gate-refused
            #     evidence. The result was a point labelled `detected`, carrying
            #     another ball's position, at a frame absent from this event's
            #     confirmed detections. Hence the snapshot.
            track_position = track.position
            track_confirmed = evidence.confirmed

            # A short dropout inside ONE event: the segmenter has decided the
            # ball was never lost, so restart the filter on the ball that is
            # actually visible instead of letting MAX_MISSED end the delivery.
            # See `_needs_reseed` for why this no longer also demands a null
            # position — that conjunction was unreachable, so the reseed path
            # this comment describes had never actually run.
            if _needs_reseed(decision, track, tracker, detection):
                track = tracker.reseed(detection, frame_number)
                result.reseeds_applied += 1

            # 6. Close the finished event BEFORE touching the accumulator.
            #    The closed candidate is QUEUED, not emitted: it cannot become a
            #    delivery until we know the next candidate is not the rest of it.
            #    The split cap travels with it, so a candidate emitted later is
            #    still bounded by the frame before the next event.
            #
            #    `cur_points` is handed over with the candidate before being reset,
            #    so deferring the emission never loses tracking data.
            if decision.closed is not None:
                split_here = decision.opened_new_event
                pending.append(_PendingEvent(
                    segment=decision.closed,
                    points=cur_points,
                    hard_end_frame=(frame_number - 1) if split_here else None,
                ))
                tracker.reset_delivery()

            if decision.opened_new_event:
                # New event, new accumulator. The previous event's points just
                # went into the queue above.
                cur_points = []

            # Anything that can no longer be merged is emitted now. Bounded by
            # the merge horizon, so the queue can never grow without limit and the
            # pre-roll these clips need is still inside the ring buffer (the
            # buffer was sized for exactly this, above).
            flush_pending(frame_number)

            # Trajectory point, from the PRE-reseed snapshot above.
            #
            # Provenance is the SEGMENTER's verdict for this frame, taken from the
            # pre-reseed track:
            #   confirmed -> the filter absorbed a real box, so this point is that
            #                measurement and carries its confidence.
            #   refused   -> the gate rejected the box. The measurement stays
            #                EVIDENCE and never becomes a trajectory point
            #                (core/constants.py), so the frame carries the
            #                filter's own extrapolation instead, labelled as one.
            if track_position is not None and segmenter.open_event is not None:
                cur_points.append(
                    TrajectoryPoint(
                        original_frame=frame_number,
                        x=int(track_position[0]),
                        y=int(track_position[1]),
                        source="detected" if track_confirmed else "predicted",
                        confidence=conf if track_confirmed else None,
                    )
                )

            # 7. Clip-window bounding. THIS MUST RUN BEFORE tick().
            #    A barrier reached on this frame means the open clip must stop
            #    accepting frames here, or the clip contains footage that belongs
            #    to something else. Running it after tick() would already have
            #    written this frame, which is the bug this ordering prevents.
            #
            #    The "new event" barrier is deliberately NOT here: it ran above,
            #    before `flush_pending()`, so that a clip could never be opened on
            #    top of a still-open previous one.
            #
            #    Every cut below passes `min_end` = this delivery's ball-loss
            #    frame. That floor is what keeps a barrier discovered mid-stroke
            #    from cutting the stroke off: a barrier that lies BEFORE the ball
            #    was lost is declined rather than obeyed.
            active_lost = clips.active_ball_lost_frame
            if consecutive_black_frames >= cfg.segmentation.black_frame_run_barrier:
                # Halt any open clip immediately when entering a black gap
                clips.close_before(
                    frame_number - consecutive_black_frames + 1,
                    next_delivery_id=None, reason="black_gap",
                    min_end=active_lost,
                )
            elif is_scene_cut:
                # Halt any open clip immediately on camera cut
                clips.close_before(
                    frame_number, next_delivery_id=None, reason="scene_cut",
                    min_end=active_lost,
                )
            elif (
                active_lost is not None
                and no_detection_run >= resolved.hard_split_frames
            ):
                # DETECTION SILENCE, as a stream-time horizon. Every other barrier
                # is evidence about the PICTURE; this one is evidence about the
                # TRACKING, and it is the only barrier available for the common
                # case where footage simply goes quiet with no black frame, no cut
                # and no motion change: the ball has been unseen for longer than
                # the hard-split ceiling, so a different ball — or nothing at all —
                # owns everything from here on.
                #
                # The ceiling is `hard_split_frames` and not something shorter on
                # purpose. A shorter silence is ordinary during a stroke, which is
                # precisely when there is no ball on screen; cutting there would
                # delete the follow-through. By the time `hard_split_frames` has
                # passed, the post-roll cap has long since run out, so this can
                # only ever fire on a window something else extended.
                clips.close_before(
                    frame_number - no_detection_run + 1,
                    next_delivery_id=None, reason="detection_silence",
                    min_end=active_lost,
                )
            elif active_lost is not None:
                # Post-roll valley: after this delivery's own follow-through
                # (motion above the active threshold once the ball is lost) a
                # sustained quiet run is the dead-ball reset, and the window
                # ends there rather than running on toward the next event. The
                # action guard stops a flat or quiet frame right after ball-loss
                # from cutting the stroke before it happens.
                if frame_number > active_lost:
                    if motion_energy >= resolved.pitch_activity_active_threshold:
                        post_loss_action = True
                    elif (
                        post_loss_action
                        and motion_energy <= resolved.pitch_activity_quiet_threshold
                    ):
                        clips.close_before(
                            frame_number, reason="inactivity_valley",
                            min_end=active_lost,
                        )
                        post_loss_action = False
                else:
                    post_loss_action = False

            # 8. Stream into the open clip, if any. No-op when none is open.
            clips.tick(frame_number, frame)
            t_clip = time.perf_counter()
            result.time_stage("clip_write", t_clip - t_track)

            # Global trace (cheap, kept for the sparkline view).
            if track.position is not None:
                result.trajectory_global.append(
                    {"frame": frame_number, "x": int(track.position[0]),
                     "y": int(track.position[1]), "source": track.source}
                )

            # ── Progress ──────────────────────────────────────────────────────
            now = time.perf_counter()
            # Everything from the Kalman update to here: per-point record
            # building, event segmentation, analytics, clip opening and the
            # incremental tracking.json rewrite.
            result.time_stage("record_and_finalise", now - t_track)
            if now - last_emit >= 0.5:
                last_emit = now
                pct = (
                    f" ({frame_number / reported_frames * 100:.0f}%)"
                    if reported_frames
                    else ""
                )
                emit(
                    "tracking", frame_number, reported_frames,
                    f"frame {frame_number}{pct}, {len(result.deliveries)} deliveries",
                )

        # ── End of video: an event may still be open ──────────────────────────
        # Closed, not deleted, and queued like every other candidate so it gets
        # the same merge opportunity. If it holds too few confirmed detections to
        # be a delivery, `emit_event` records why and drops it.
        final_segment = segmenter.close_open_event(at_frame=frame_number)
        if final_segment is not None:
            pending.append(_PendingEvent(
                segment=final_segment,
                points=cur_points,
                hard_end_frame=frame_number,
            ))
        cur_points = []
        # Force: nothing more can arrive, so every queued candidate is final.
        flush_pending(frame_number, force=True)
        tracker.reset_delivery()

        # A clip whose window runs past the last decoded frame is closed here.
        # ClipFrameMap is the same object the delivery already holds a reference
        # to, so finalising it here updates the record in place.
        clips.flush_and_close()

        result.segments = list(segmenter.events)
        result.segmentation_summary = segmenter.summary()

        if not result.deliveries:
            result.warnings.append(
                "No deliveries were detected. The model found no sustained ball "
                "trajectory in this video."
            )

        # ── Independent single-event validation ───────────────────────────────
        # Runs on stored tracking data only. It re-derives how many events each
        # clip holds rather than trusting the segmentation that produced it, so
        # a merge inside the segmenter is still caught.
        emit(
            "purity_validation", len(result.deliveries), len(result.deliveries),
            "re-segmenting stored clips to check one-clip-one-event",
        )
        validator = ClipValidator(
            fps=fps, width=width, height=height,
            config=cfg.segmentation, resolved=segmenter.resolved,
        )
        report = validator.validate(result.deliveries, result.segments)
        result.validation = report
        for record, delivery in zip(report.records, result.deliveries):
            delivery.validation = record.as_dict()

        result.frames_decoded = frame_number
        result.wall_seconds = time.perf_counter() - t_start
        result.ring_stats = ring.stats()
        result.clip_results = dict(clips.results)
        result.clip_stats = {
            "clips_written": len(clips.results),
            "codecs_used": sorted(
                {r.codec_used for r in clips.results.values() if r.codec_used}
            ),
            "frame_maps_valid": sum(
                1 for r in clips.results.values() if r.frame_map.frame_map_valid
            ),
            "frame_maps_invalid": sum(
                1 for r in clips.results.values() if not r.frame_map.frame_map_valid
            ),
            "errors": dict(clips.errors),
        }

        # Final, complete rewrite of the tracking document.
        atomic_write_json(tracking_path, result.tracking_document(cfg))

        emit(
            "tracking_complete", frame_number, frame_number or 1,
            f"{len(result.deliveries)} deliveries in {result.wall_seconds:.1f}s",
        )
        return result

    # ── Delivery finalisation ────────────────────────────────────────────────
    def _finalise(
        self,
        delivery_id: int,
        segment: EventSegment,
        points: list[TrajectoryPoint],
        homography: np.ndarray,
        fps: float,
        width: int,
        height: int,
        pre_roll: int,
        post_roll: int,
        paths: Paths,
        clips: ClipWriterRegistry,
        cfg: PipelineConfig,
        speeds_seen: list[float],
        hard_end_frame: Optional[int] = None,
        resolved=None,
        signals=None,
        available_frame: Optional[int] = None,
        purity: Optional[ClipValidator] = None,
        segments: Optional[list] = None,
        on_purity_reject=None,
    ) -> Optional[Delivery]:
        """
        Build the delivery record from one finished batting event.

        Returns None if the pre-persistence purity gate refuses the window, in
        which case nothing has been written and `on_purity_reject` has been told.

        The event — not a pixel gap, and not a detection cluster — defines the
        boundaries. Everything below derives from `segment`.

        CLIP WINDOW RULE
        ----------------
            desired_start = event_start - pre_roll
            desired_end   = ball_lost   + post_roll
            actual_start  = max(desired_start, previous_clip_end + 1)
            actual_end    = min(desired_end,   hard_end)

        `hard_end` is supplied only when a new ball is already visible on the
        frame this event closed, in which case it is that frame minus one.
        Otherwise it is None and the end is bounded later, by `close_before()`,
        the moment the next event begins. Both paths implement the same rule;
        the eager cap is preferred when it is available, because then the window
        is never opened over the next event's footage at all.
        """
        trajectory = Trajectory(points=points)

        # `geometry_calibrated` is passed EXPLICITLY. It is the flag that decides
        # whether speed/line/length/swing are real measurements or must be null,
        # and relying on the default here is what previously let a template
        # homography publish 33 fabricated speeds on real_cricket.mp4 while the
        # analysis-level calibration block simultaneously claimed the geometry was
        # uncalibrated.
        release_frame, bounce, bowling, samples, flags = analyse_delivery(
            trajectory,
            homography=homography,
            fps=fps,
            width=width,
            height=height,
            impute_missing_speeds=cfg.impute_missing_speeds,
            speeds_from=speeds_seen,
            geometry_calibrated=cfg.geometry_calibrated,
        )

        # Trajectory quality is a judgement about the DATA, made from the data.
        quality = Quality(
            trajectory_valid=segment.confirmed_count >= 3, flags=list(flags)
        )
        if not quality.trajectory_valid:
            quality.analytics_complete = False
        if trajectory.gaps:
            quality.flags.append(
                f"trajectory_has_{len(trajectory.gaps)}_gap(s)"
            )

        first_detected_frame = segment.confirmed_frames[0]
        last_detected_frame = segment.confirmed_frames[-1]
        ball_lost_frame = segment.ball_lost_frame or last_detected_frame

        if segment.ambiguous:
            quality.flags.append("event_boundary_ambiguous")
            for reason in segment.ambiguous_reasons:
                quality.flags.append(f"ambiguous: {reason}")

        # A weak promotion is a real delivery and gets analysed like one, but it
        # is never allowed to look like an ordinary one: the flag travels with
        # the record so any consumer, human or downstream, can see that this
        # event's boundaries rest on gate-refused measurements agreeing with each
        # other rather than on detections passing the normal bar.
        if segment.is_weak_confirmation:
            quality.flags.append("weak_confirmation")
            quality.flags.append(
                f"weak_confirmation:{segment.weak_evidence_count}_"
                f"measurements_{segment.confirmation_method}"
            )

        # ── Dynamic, barrier-bounded window ──────────────────────────────────
        # Pre-roll reaches back before the first detection (the run-up); post-roll
        # extends past the ball being lost, because the STROKE happens after the
        # ball disappears behind the bat. Both are BOUNDED SEARCHES, not static
        # additions: they start from the delivery's own reliable anchors and walk
        # outward only as far as the evidence allows. A black gap, a camera cut,
        # a reset valley, a rejected candidate or an untracked activity window
        # all stop the search early. The roll lengths are CAPS, not targets — a
        # shorter safe clip always wins over a longer one that might contain
        # another event.
        excluded = clips.exclusion_ranges()
        # Resolved caps are a ceiling on the CONFIGURED roll, applied here rather
        # than trusted to `signals`, so nothing downstream can be talked into a
        # longer window than policy allows. `min` in both directions: policy can
        # only shorten a roll, never lengthen one.
        max_back = pre_roll
        max_forward = post_roll
        if resolved is not None:
            max_back = min(max_back, resolved.max_pre_roll_frames)
            max_forward = min(max_forward, resolved.max_post_roll_frames)
        if signals is not None:
            safe_start, start_reason = signals.safe_pre_roll_start(
                first_detected_frame, max_back, excluded
            )
        else:
            safe_start, start_reason = max(1, first_detected_frame - pre_roll), "static"
        desired_start = max(1, int(safe_start))

        desired_end = ball_lost_frame + max_forward
        end_reason = "static"
        if signals is not None:
            # Scan forward from the LAST CONFIRMED DETECTION, not from
            # ball_lost_frame: an event may coast for up to the prediction
            # ceiling after its last detection, and that coasting span routinely
            # crosses a black gap or a camera cut. Anchoring the forward search
            # on the last real measurement stops the window at that barrier
            # instead of dragging it through the transition.
            safe_end, end_reason = signals.safe_post_roll_end(
                last_detected_frame, desired_end, excluded,
                known_frame=available_frame,
                stroke_from=ball_lost_frame,
            )
            desired_end = int(safe_end)
        # The cap is a hard ceiling on the result too, not just on the input: a
        # barrier search that ends up longer than policy allows is truncated back,
        # because "the evidence permitted this long" is not an argument against
        # the policy.
        if desired_end > ball_lost_frame + max_forward:
            desired_end = ball_lost_frame + max_forward
            end_reason = f"{end_reason}+capped" if end_reason != "static" else "capped"
        if hard_end_frame is not None and desired_end > hard_end_frame:
            desired_end = min(desired_end, hard_end_frame)
            end_reason = "next_event"

        # Record the boundary reasoning on the event BEFORE snapshotting it onto
        # the delivery, so `delivery.event` actually carries it.
        if segment.evidence is not None:
            segment.evidence["clip_window"] = {
                "start_reason": start_reason,
                "end_reason": end_reason,
                "desired_start": desired_start,
                "desired_end": desired_end,
                "max_pre_roll_frames": max_back,
                "max_post_roll_frames": max_forward,
                "configured_pre_roll_frames": pre_roll,
                "configured_post_roll_frames": post_roll,
            }

        # ── Pre-persistence purity gate ──────────────────────────────────────
        # Runs BEFORE the window is handed to the writer, so an unsafe window
        # never becomes bytes. It runs after the cap and barrier work above,
        # because those decide what the window actually IS — judging purity
        # against a window that is about to change would be judging the wrong
        # thing.
        #
        # Repair first, refuse second. A window that merely reaches too far is
        # narrowed to the largest honest sub-range; a window that cannot be made
        # honest is refused outright. Either way nothing is persisted until this
        # returns SINGLE_EVENT.
        purity_verdict = None
        if purity is not None:
            verdict = purity.window_purity(
                desired_start, desired_end, segment, segments or [],
                min_end=ball_lost_frame,
            )
            if verdict.classification != SINGLE_EVENT:
                if verdict.repairable:
                    desired_start, desired_end = verdict.repaired_window
                    verdict = purity.window_purity(
                        desired_start, desired_end, segment, segments or [],
                        min_end=ball_lost_frame,
                    )
                if segment.evidence is not None:
                    segment.evidence["purity"] = verdict.as_dict()
                if verdict.classification != SINGLE_EVENT:
                    log.error(
                        "Purity gate refused event %d window %d-%d: %s",
                        segment.index, desired_start, desired_end, verdict.reason,
                    )
                    if on_purity_reject is not None:
                        on_purity_reject(
                            segment, verdict, (desired_start, desired_end)
                        )
                    return None
                # Repaired. The reasoning must show the original request too, or
                # a repaired window looks arbitrary to anyone reading it later.
                if segment.evidence is not None:
                    segment.evidence["clip_window"]["purity_repair"] = {
                        "from": [verdict.start, verdict.end],
                        "reason": verdict.reason,
                    }
            purity_verdict = verdict
            if segment.evidence is not None:
                segment.evidence["clip_window"]["purity"] = verdict.as_dict()

        delivery = Delivery(
            delivery_id=delivery_id,
            detection_start_frame=first_detected_frame,
            detection_end_frame=last_detected_frame,
            ball_lost_frame=ball_lost_frame,
            release_frame=release_frame,
            trajectory=trajectory,
            speed_samples=samples,
            bounce=bounce,
            bowling=bowling,
            quality=quality,
            # Always the single pass. "clip_fallback" is set only by the
            # documented fallback path in pipeline.py, never here.
            tracking_source="single_pass",
            # The evidence behind these boundaries travels with the delivery, so
            # a wrong boundary can be diagnosed without re-running the pass.
            event=segment.as_dict(),
        )

        try:
            frame_map = clips.open_for(
                delivery_id=delivery_id,
                start_frame=desired_start,
                end_frame=desired_end,
                fps=fps,
                size=(width, height),
                out_path=paths.clip_path(delivery_id),
                post_roll_frames=post_roll,
                ball_lost_frame=ball_lost_frame,
            )
            # Pre-roll actually granted, which is less than requested whenever
            # the start had to be pushed past the previous clip.
            frame_map.pre_roll_actual_sec = (
                (first_detected_frame - frame_map.clip_start_frame) / fps
                if fps > 0 else 0.0
            )
            # Stamp clip_frame NOW, from the effective window, so the mapping is
            # known before any byte is written.
            frame_map.assign_clip_frames(trajectory)
            delivery.clip = frame_map
            delivery.clip_path = str(paths.clip_path(delivery_id))
            delivery.clip_url = f"{paths.public_prefix}/clips/delivery_{delivery_id:03d}.mp4"
        except Exception as exc:
            clips.errors[delivery_id] = str(exc)
            delivery.quality.flags.append("clip_write_failed")
            log.error("Clip write failed for delivery %d: %s", delivery_id, exc)

        log.info(
            "Delivery %d: frames %d-%d (lost %d), clip %d-%d, "
            "%d detected / %d total points, speed=%s",
            delivery_id, first_detected_frame, last_detected_frame, ball_lost_frame,
            delivery.clip.clip_start_frame if delivery.clip else -1,
            delivery.clip.clip_end_frame if delivery.clip else -1,
            len(segment.confirmed_frames), len(points),
            bowling.speed_kmh if bowling.speed_kmh is not None else "N/A",
        )
        return delivery

    # ── Incremental persistence ──────────────────────────────────────────────
    @staticmethod
    def _append_to_tracking(
        tracking_path, delivery: Delivery, cfg: PipelineConfig, result: TrackingResult
    ) -> None:
        """Rewrite tracking.json with the newly finished delivery appended."""
        doc = result.tracking_document(cfg)
        doc["deliveries"] = [d.as_dict() for d in result.deliveries]
        doc["ring_buffer"] = result.ring_stats
        atomic_write_json(tracking_path, doc)