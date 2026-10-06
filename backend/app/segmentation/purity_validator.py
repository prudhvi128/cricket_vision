"""
clip_validator.py — Does every generated clip contain exactly ONE batting event?

THE QUESTION THIS ANSWERS
-------------------------
Segmentation can be wrong in two directions, and only one of them is visible by
counting things:

  * it SPLITS one event into two clips (a fragment), and
  * it MERGES two events into one clip (the defect that motivated this work).

A merge is invisible to any check that trusts the segmenter, because the segment
that merged them looks like one tidy event. So this validator does not take the
segmenter's word for it. For every written clip it re-derives, from the stored
tracking evidence alone, how many distinct ball events the clip's frames contain
— then reports what it found.

THE FOUR VERDICTS
-----------------
  SINGLE_EVENT     exactly one event, comfortably inside the clip.
  MULTIPLE_EVENTS  two or more events' confirmed evidence inside the clip.
                   This is the failure the requirement names explicitly, and it
                   is always reported with the separate frame ranges.
  NO_EVENT         the clip contains no confirmed ball detection at all, so it
                   is not a delivery of anything.
  AMBIGUOUS        one event, but the evidence does not settle whether the
                   boundary is right.

AMBIGUOUS IS NOT A SOFT PASS
----------------------------
Uncertainty is never resolved in favour of SINGLE_EVENT. A boundary that sits
close to a clip edge, an event the segmenter itself flagged, an event that ended
because the video ended rather than because the ball went quiet — each of these
is reported as AMBIGUOUS with its reason. The instruction is explicit: an
uncertain case must not be silently classified as SINGLE_EVENT.

NO VIDEO IS READ
----------------
Everything here comes from stored tracking data: the trajectories with their
per-point provenance, the clip frame maps, and the event segments the single pass
recorded. Opening the source video again would break the single-pass guarantee,
and opening the clip would prove nothing the frame map does not already say.

THE SHOT CLASSIFIER IS NOT CONSULTED
------------------------------------
It runs after segmentation and may be used for diagnostics elsewhere. A label
cannot establish how many deliveries there are — by the time a shot label
exists, the boundaries it might have informed are already fixed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..core.config import ResolvedSegmentation, SegmentationConfig
from .candidate_generation import (
    BoundaryReason,
    EventSegment,
    EventSegmenter,
    FrameEvidence,
)

log = logging.getLogger(__name__)

SINGLE_EVENT = "SINGLE_EVENT"
MULTIPLE_EVENTS = "MULTIPLE_EVENTS"
NO_EVENT = "NO_EVENT"
AMBIGUOUS = "AMBIGUOUS"

# Precedence when more than one finding applies. A clip with no ball in it is
# NO_EVENT even if its window also overlaps a neighbour; a clip with two balls in
# it is MULTIPLE_EVENTS even if one of them is itself uncertain.
_SEVERITY = {NO_EVENT: 3, MULTIPLE_EVENTS: 2, AMBIGUOUS: 1, SINGLE_EVENT: 0}


@dataclass
class PurityVerdict:
    """
    Whether a PROPOSED clip window holds exactly one batting event, judged before
    anything is written or persisted.

    This is the gate the end-of-pass validator cannot be. `validate()` runs after
    the pass, when a bad window has already been cut to disk and its delivery
    already appended to `tracking.json`. Its job is to report what happened. This
    one runs at the moment the window is known and its job is to stop a bad window
    from happening, or to narrow it until it is honest.

    `classification` uses the same three verdicts as the end-of-pass report, so a
    consumer never has to learn two vocabularies. `AMBIGUOUS` is deliberately
    absent: this gate is not asked to express doubt, only to answer "does this
    window contain somebody else's ball, or none at all".
    """

    classification: str
    reason: str
    start: int
    end: int
    own_confirmed_frames: list[int] = field(default_factory=list)
    # Per foreign event: how many of its confirmed detections fall inside the
    # proposed window. An event whose span merely overlaps contributes zero here,
    # and is reported as a weaker overlap rather than as a second ball.
    foreign_event_indices: list[int] = field(default_factory=list)
    foreign_confirmed_counts: dict = field(default_factory=dict)
    foreign_ranges: list[list[int]] = field(default_factory=list)
    # Set when the window can be narrowed to become honest. `repair_start` /
    # `repair_end` are the proposed replacement bounds; None means "leave as is".
    repairable: bool = False
    repair_start: Optional[int] = None
    repair_end: Optional[int] = None
    repair_reason: str = ""

    @property
    def repaired_window(self) -> Optional[tuple[int, int]]:
        if not self.repairable:
            return None
        return (int(self.repair_start), int(self.repair_end))

    def as_dict(self) -> dict:
        return {
            "classification": self.classification,
            "reason": self.reason,
            "start": self.start,
            "end": self.end,
            "own_confirmed_frames": list(self.own_confirmed_frames),
            "foreign_event_indices": list(self.foreign_event_indices),
            "foreign_confirmed_counts": dict(self.foreign_confirmed_counts),
            "foreign_ranges": [list(r) for r in self.foreign_ranges],
            "repairable": self.repairable,
            "repair_start": self.repair_start,
            "repair_end": self.repair_end,
            "repair_reason": self.repair_reason,
        }


@dataclass
class ValidationRecord:
    """
    One clip's verdict, in the shape the requirement asks for.

    `reason` is written to be read by a person deciding whether to trust the
    clip, so it states the evidence rather than the conclusion alone.
    """

    delivery_id: int
    clip_start: Optional[int]
    clip_end: Optional[int]
    classification: str
    event_count_estimate: int
    event_ranges: list[list[int]]
    confidence: float
    reason: str
    flags: list[str] = field(default_factory=list)
    own_event_index: Optional[int] = None
    start_margin_frames: Optional[int] = None
    end_margin_frames: Optional[int] = None
    requires_human_review: bool = False

    def as_dict(self) -> dict:
        return {
            "delivery_id": self.delivery_id,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "classification": self.classification,
            "event_count_estimate": self.event_count_estimate,
            "event_ranges": self.event_ranges,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "flags": list(self.flags),
            "own_event_index": self.own_event_index,
            "start_margin_frames": self.start_margin_frames,
            "end_margin_frames": self.end_margin_frames,
            "requires_human_review": self.requires_human_review,
        }


@dataclass
class ValidationReport:
    records: list[ValidationRecord] = field(default_factory=list)

    def by_classification(self, name: str) -> list[ValidationRecord]:
        return [r for r in self.records if r.classification == name]

    @property
    def counts(self) -> dict[str, int]:
        return {
            name: sum(1 for r in self.records if r.classification == name)
            for name in (SINGLE_EVENT, MULTIPLE_EVENTS, NO_EVENT, AMBIGUOUS)
        }

    @property
    def needs_review(self) -> list[ValidationRecord]:
        """Only what a human must look at. Everything else passes."""
        return [r for r in self.records if r.requires_human_review]

    def as_dict(self) -> dict:
        return {
            "summary": {
                "clips_validated": len(self.records),
                **self.counts,
                "requires_human_review": len(self.needs_review),
                "mean_confidence": (
                    round(
                        sum(r.confidence for r in self.records) / len(self.records),
                        4,
                    )
                    if self.records else None
                ),
            },
            "requires_human_review": [r.delivery_id for r in self.needs_review],
            "records": [r.as_dict() for r in self.records],
        }


class ClipValidator:
    """
    Checks every clip against the stored tracking evidence.

    Constructed with the video's own units (fps, resolution) so its thresholds
    mean the same thing on any footage. Holds no state between clips, so a run
    can be validated again later from `tracking.json` alone.
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

    # ── Public API ───────────────────────────────────────────────────────────

    def validate(
        self,
        deliveries: Iterable,
        segments: list[EventSegment],
    ) -> ValidationReport:
        """
        Validate each delivery's clip.

        `deliveries` are the `Delivery` records; `segments` the events the single
        pass found. Both come from the same run, so their frame numbers share a
        timebase.
        """
        report = ValidationReport()
        segs = list(segments)
        for delivery in deliveries:
            report.records.append(self._validate_one(delivery, segs))
        return report

    # ── One clip ─────────────────────────────────────────────────────────────

    def _validate_one(self, delivery, segments: list[EventSegment]) -> ValidationRecord:
        did = getattr(delivery, "delivery_id", None)
        clip = getattr(delivery, "clip", None)
        start = clip.clip_start_frame if clip else None
        end = clip.clip_end_frame if clip else None

        confirmed = self._confirmed_frames(delivery)

        # ── NO_EVENT: a clip with no measurement in it is not a delivery ─────
        if start is None or end is None:
            return self._record(
                did, None, None, NO_EVENT, 0, [], 1.0,
                "no clip frame map, so the clip cannot be shown to contain "
                "any tracked event",
                flags=["clip_map_missing"],
            )
        if not confirmed:
            return self._record(
                did, start, end, NO_EVENT, 0, [], 0.0,
                f"clip {start}-{end} contains no confirmed ball detection "
                f"(the delivery's own trajectory has none inside the window)",
                flags=["no_confirmed_detection_in_clip"],
                requires_review=True,
            )

        findings: list[tuple[str, str, list[str], float]] = []

        # ── Finding 1: other events contribute evidence inside this clip ─────
        # The direct test for the defect: a second event's CONFIRMED ball is
        # inside this clip's window, so this clip contains two batting events.
        contributors = [
            s for s in segments
            if s.confirmed_frames
            and not (
                s.end_frame < start or s.start_frame > end
            )
        ]
        if len(contributors) > 1:
            ranges = [[s.start_frame, s.end_frame] for s in contributors]
            findings.append((
                MULTIPLE_EVENTS,
                f"{len(contributors)} distinct tracked events have confirmed "
                f"detections inside clip {start}-{end}: "
                + "; ".join(f"event {s.index} at {s.start_frame}-{s.end_frame}"
                            for s in contributors),
                ["multiple_event_cores_in_clip"],
                0.0,
            ))

        # ── Finding 2: independent re-segmentation of the clip's own evidence ─
        # Catches the case finding 1 cannot: the segmenter itself merged two
        # events, so `segments` contains one entry and the clip looks clean.
        # Re-running the segmenter over ONLY the confirmed detections inside this
        # window asks the question again, without trusting the original answer.
        internal = self._resegment(confirmed, start, end)
        if len(internal) > 1:
            ranges = [[a, b] for a, b in internal]
            findings.append((
                MULTIPLE_EVENTS,
                f"re-segmenting the confirmed detections inside clip {start}-{end} "
                f"independently finds {len(internal)} events: "
                + "; ".join(f"{a}-{b}" for a, b in internal)
                + ". The stored segmentation merged them, so this clip holds "
                "more than one batting event",
                ["independent_resegment_found_multiple"],
                0.0,
            ))

        # ── Finding 3: the clip's window swallows a neighbour's event span ───
        # Even if a neighbour contributed no confirmed frames inside the window,
        # a window that reaches deep into the neighbouring event's span may be
        # showing that event's footage.
        if len(contributors) == 1:
            own = contributors[0]
            reachers = [
                s for s in segments
                if s.index != own.index
                and not (
                    s.end_frame < start or s.start_frame > end
                )
            ]
            if reachers:
                findings.append((
                    MULTIPLE_EVENTS,
                    f"clip {start}-{end} overlaps the span of event(s) "
                    + ", ".join(
                        f"{s.index} ({s.start_frame}-{s.end_frame})"
                        for s in reachers
                    )
                    + " even where they contributed no confirmed detection",
                    ["clip_window_covers_neighbouring_event"],
                    0.25,
                ))

        own_event = self._own_event(delivery, segments)
        start_margin = (own_event.start_frame - start) if own_event else None
        end_margin = (end - own_event.end_frame) if own_event else None

        # ── Finding 4: the event itself was flagged uncertain ────────────────
        if own_event is not None and own_event.ambiguous:
            findings.append((
                AMBIGUOUS,
                "the event this clip was built from is itself uncertain: "
                + "; ".join(own_event.ambiguous_reasons or ["no detail recorded"]),
                ["own_event_ambiguous"],
                min(own_event.confidence, self.resolved.ambiguous_confidence),
            ))

        # ── Finding 5: the boundary sits on the clip edge ───────────────────
        # A clip that starts within a hair of its own event, or ends within a
        # hair of the ball being lost, may have been cut on the wrong side. No
        # amount of frame-map arithmetic can rule that out, so it is reported
        # rather than asserted.
        margin = self.resolved.clip_margin_frames
        edge_flags: list[str] = []
        # A deliberately SHORT window is not ambiguous when its edge is a
        # confirmed hard barrier. If the search stopped at a black gap, a scene
        # cut, a neighbouring clip or an exclusion zone, starting on the
        # event's own first detection is the SAFE outcome, not a doubtful one.
        event = getattr(delivery, "event", None) or {}
        clip_window = (event.get("evidence") or {}).get("clip_window") or {}
        hard_start = (
            clip_window.get("start_reason")
            in ("black_gap", "scene_cut", "exclusion_zone", "start_of_video")
            or bool(clip is not None and clip.start_clamped_by_previous)
        )
        hard_end = clip_window.get("end_reason") in (
            "black_gap", "scene_cut", "exclusion_zone", "inactivity_valley"
        )
        if start_margin is not None and start_margin < margin and not hard_start:
            edge_flags.append("clip_starts_at_its_own_event")
            findings.append((
                AMBIGUOUS,
                f"clip starts {start_margin} frame(s) before its event's first "
                f"confirmed detection, within the {margin}-frame margin; the "
                f"clip may open inside the previous event",
                ["start_margin_too_small"],
                0.45,
            ))
        if end_margin is not None and end_margin < margin and not hard_end:
            edge_flags.append("clip_ends_at_its_own_event")
            findings.append((
                AMBIGUOUS,
                f"clip ends {end_margin} frame(s) after its event's last "
                f"confirmed detection, within the {margin}-frame margin; the "
                f"clip may stop before the stroke has happened",
                ["end_margin_too_small"],
                0.45,
            ))

        # ── Finding 6: post-roll was cut short by the next event ────────────
        # Only a truncation caused by ANOTHER CRICKET EVENT is ambiguous: the
        # stroke may fall on the far side of the cut. A cut caused by a black
        # gap, a camera cut or a reset valley is a deliberate, safe stop at a
        # non-cricket boundary, and must not be reported as uncertainty —
        # otherwise every clip in an edited highlights package looks ambiguous.
        if clip is not None and clip.truncated_before_frame is not None:
            reason = getattr(clip, "truncated_reason", None) or "unknown"
            if reason in ("next_event", "unknown"):
                findings.append((
                    AMBIGUOUS,
                    f"the window was cut at {clip.truncated_before_frame} because "
                    f"another event begins there, so the stroke may fall outside "
                    f"this clip",
                    ["post_roll_truncated_by_next_event"],
                    0.55,
                ))
            else:
                edge_flags.append(f"post_roll_stopped_at_{reason}")

        # ── Verdict ─────────────────────────────────────────────────────────
        if not findings:
            return self._record(
                did, start, end, SINGLE_EVENT, 1,
                [[own_event.start_frame, own_event.end_frame]] if own_event
                else [[confirmed[0], confirmed[-1]]],
                own_event.confidence if own_event else 0.5,
                f"clip {start}-{end} contains one tracked event "
                f"({confirmed[0]}-{confirmed[-1]} confirmed) and no other "
                f"event's evidence",
                start_margin=start_margin,
                end_margin=end_margin,
                own_event_index=own_event.index if own_event else None,
            )

        worst = max(findings, key=lambda f: _SEVERITY[f[0]])
        classification = worst[0]
        flags: list[str] = []
        for _, _, f, _ in findings:
            flags.extend(f)
        # Confidence is the least confident finding, not the most severe one: a
        # clip that is definitely broken is confidently broken.
        confidence = min(f[3] for f in findings)
        ranges = (
            [[own_event.start_frame, own_event.end_frame]] if own_event else []
        ) or [[confirmed[0], confirmed[-1]]]
        if classification == MULTIPLE_EVENTS:
            # Report the separate events, which is what the requirement asks for
            # on every multiple-event clip.
            if len(contributors) > 1:
                ranges = [[s.start_frame, s.end_frame] for s in contributors]
            elif len(internal) > 1:
                ranges = [[a, b] for a, b in internal]

        return self._record(
            did, start, end, classification, max(len(contributors), len(internal), 1),
            ranges, confidence,
            "; ".join(f[1] for f in findings),
            flags=sorted(set(flags)),
            start_margin=start_margin,
            end_margin=end_margin,
            own_event_index=own_event.index if own_event else None,
            requires_review=True,
        )

    # ── Pre-persistence purity gate ─────────────────────────────────────────

    def window_purity(
        self,
        start: int,
        end: int,
        own: Optional[EventSegment],
        segments: Iterable[EventSegment],
        min_end: Optional[int] = None,
    ) -> PurityVerdict:
        """
        Decide whether the window `start`-`end` may be persisted as one delivery.

        Called from inside the single pass, the moment the window is known and
        before a single frame is written or a single byte of `tracking.json` is
        appended. Two questions, both answerable from evidence this pass already
        recorded:

            NO_EVENT         does the window contain any of THIS delivery's own
                             confirmed detections? If not, the clip would show
                             footage of nothing this pipeline measured.
            MULTIPLE_EVENTS  does any OTHER event put a confirmed detection
                             inside the window? If so, the clip would show two
                             batting events.

        When the answer is MULTIPLE_EVENTS the window is usually repairable
        without losing anything: a neighbour entirely ahead of this delivery can
        be cut off at the frame before its first detection, and a neighbour
        entirely behind it can be stepped over. The repair is returned, never
        applied, so the caller decides — and it is refused outright if it would
        end the clip before `min_end`, normally the ball-loss frame, because a
        clip that stops before the ball was lost cannot contain the stroke.

        Events this one ABSORBED are not foreign. A merged delivery legitimately
        covers the frames of the candidates it merged, so their indices are
        excluded.

        A neighbour whose SPAN overlaps the window but which contributed no
        confirmed detection inside it is not enough to condemn the window: that is
        footage the segmenter already judged not to be a ball there, and the
        bounded-roll search plus the exclusion zones are what handle it. It is
        still counted and reported.
        """
        start, end = int(start), int(end)
        own_frames = sorted(
            f for f in (own.confirmed_frames if own else []) if start <= f <= end
        )

        own_ids = {own.index} if own is not None else set()
        if own is not None:
            own_ids |= set(own.merged_from or [])

        foreign_frames: dict[int, list[int]] = {}
        foreign_spans: list[list[int]] = []
        overlapping_spans: list[list[int]] = []
        for s in segments:
            if s.index in own_ids or not s.confirmed_frames:
                continue
            inside = [f for f in s.confirmed_frames if start <= f <= end]
            span_overlaps = not (s.end_frame < start or s.start_frame > end)
            if inside:
                foreign_frames[s.index] = sorted(inside)
                foreign_spans.append([s.start_frame, s.end_frame])
            elif span_overlaps:
                overlapping_spans.append([s.start_frame, s.end_frame])

        indices = sorted(foreign_frames)
        counts = {str(i): len(foreign_frames[i]) for i in indices}

        # ── NO_EVENT ────────────────────────────────────────────────────────
        if not own_frames:
            first = own.confirmed_frames[0] if (own and own.confirmed_frames) else None
            last = own.confirmed_frames[-1] if (own and own.confirmed_frames) else None
            if own is None:
                reason = "no event was supplied for this window, so nothing can vouch for it"
            elif first is None:
                reason = (
                    f"event {own.index} has no confirmed detection at all, so the "
                    f"window {start}-{end} cannot be shown to contain one"
                )
            else:
                reason = (
                    f"window {start}-{end} contains none of event {own.index}'s "
                    f"confirmed detections ({first}-{last})"
                )
            # The one repair that can be safe: the window was cut short of the
            # event's own first detection. Extending the END forward to that
            # detection cannot remove this delivery's footage, and cannot remove
            # the stroke either.
            repair_start, repair_end, repair_reason = None, None, ""
            if first is not None and first > end:
                repair_start, repair_end = start, first
                repair_reason = (
                    f"the window ended at {end}, before this delivery's first "
                    f"confirmed detection at {first}; extending the end forward to "
                    f"{first} restores the event without discarding anything"
                )
            return PurityVerdict(
                classification=NO_EVENT,
                reason=reason,
                start=start,
                end=end,
                own_confirmed_frames=[],
                foreign_event_indices=indices,
                foreign_confirmed_counts=counts,
                foreign_ranges=foreign_spans,
                repairable=repair_reason != "",
                repair_start=repair_start,
                repair_end=repair_end,
                repair_reason=repair_reason,
            )

        # ── MULTIPLE_EVENTS ─────────────────────────────────────────────────
        if indices:
            detail = "; ".join(
                f"event {i} contributes {len(foreign_frames[i])} confirmed "
                f"detection(s) inside {start}-{end}"
                for i in indices
            )
            floor = int(min_end) if min_end is not None else None
            first_own, last_own = own_frames[0], own_frames[-1]
            repair_start, repair_end, repair_reason = None, None, ""

            earliest_foreign = min(foreign_frames[i][0] for i in indices)
            latest_foreign = max(foreign_frames[i][-1] for i in indices)

            if earliest_foreign > last_own:
                # Every foreign detection is ahead of this delivery. Stopping the
                # clip just before the earliest of them removes the second event
                # and costs only post-roll.
                stop = earliest_foreign - 1
                if floor is not None and stop < floor:
                    repair_reason = (
                        f"event(s) {indices} begin at {earliest_foreign}, which is "
                        f"before this delivery's ball-loss floor of {floor}; cutting "
                        f"there would discard the stroke, so no safe narrower "
                        f"window exists"
                    )
                else:
                    repair_start, repair_end = start, stop
                    repair_reason = (
                        f"every foreign detection is after this delivery's last "
                        f"confirmation at {last_own}; ending at {stop} removes them"
                    )
            elif latest_foreign < first_own:
                # Every foreign detection is behind this delivery. Stepping the
                # start past the latest of them removes it and costs only
                # pre-roll.
                begin = latest_foreign + 1
                if begin > first_own:
                    repair_reason = (
                        f"event(s) {indices} run to {latest_foreign}, past this "
                        f"delivery's first confirmed detection at {first_own}; "
                        f"starting at {begin} would leave this event's own opening "
                        f"frame outside the clip"
                    )
                else:
                    repair_start, repair_end = begin, end
                    repair_reason = (
                        f"every foreign detection is before this delivery's first "
                        f"confirmation at {first_own}; starting at {begin} removes "
                        f"them"
                    )
            else:
                repair_reason = (
                    f"event(s) {indices} interleave with this delivery's own "
                    f"confirmed detections ({first_own}-{last_own}), so no sub-range "
                    f"of this window holds exactly one batting event"
                )

            return PurityVerdict(
                classification=MULTIPLE_EVENTS,
                reason=(
                    f"window {start}-{end} would contain more than one batting "
                    f"event: {detail}"
                ),
                start=start,
                end=end,
                own_confirmed_frames=own_frames,
                foreign_event_indices=indices,
                foreign_confirmed_counts=counts,
                foreign_ranges=foreign_spans,
                repairable=repair_start is not None and repair_end is not None,
                repair_start=repair_start,
                repair_end=repair_end,
                repair_reason=repair_reason,
            )

        # ── SINGLE_EVENT ────────────────────────────────────────────────────
        return PurityVerdict(
            classification=SINGLE_EVENT,
            reason=(
                f"window {start}-{end} contains {len(own_frames)} of event "
                f"{own.index}'s own confirmed detections and no other event's"
            ),
            start=start,
            end=end,
            own_confirmed_frames=own_frames,
            foreign_event_indices=[],
            foreign_confirmed_counts={},
            foreign_ranges=overlapping_spans,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _confirmed_frames(delivery) -> list[int]:
        """Frames inside the clip window where the ball was actually detected."""
        traj = getattr(delivery, "trajectory", None)
        clip = getattr(delivery, "clip", None)
        if traj is None or clip is None:
            return []
        return sorted(
            p.original_frame for p in traj.points
            if p.source == "detected" and clip.contains_original_frame(p.original_frame)
        )

    @staticmethod
    def _own_event(delivery, segments: list[EventSegment]) -> Optional[EventSegment]:
        """The event this delivery was built from: the one covering its detections."""
        d = delivery
        best = None
        for s in segments:
            if s.start_frame <= d.detection_start_frame <= s.end_frame:
                if best is None or s.index > best.index:
                    best = s
        if best is not None:
            return best
        # Fall back to the nearest event if the delivery's start sits exactly on
        # a boundary the segmenter closed at the previous frame.
        return min(
            segments,
            key=lambda s: min(
                abs(s.start_frame - d.detection_start_frame),
                abs(s.end_frame - d.detection_start_frame),
            ),
            default=None,
        )

    def _resegment(
        self, confirmed: list[int], clip_start: int, clip_end: int
    ) -> list[tuple[int, int]]:
        """
        Re-run the segmenter over a clip's confirmed detections alone.

        The points are fed as an uninterrupted run of confirmations, because
        that is the question being asked: taken on its own, does this footage
        contain one ball event or more than one? Gaps between the points carry
        their real spacing, since the segmenter works in frame numbers.

        Returns the (start, end) of each event found.
        """
        if not confirmed:
            return []
        # Positions are not available here in general (only frames were kept), so
        # this pass sees confirmations with no motion history. A segmenter with
        # no kinematics can still split on SILENCE, which is exactly the check
        # that matters: a clip whose detections are separated by more than the
        # prediction ceiling contains more than one ball event.
        seg = EventSegmenter(
            fps=self.resolved.fps,
            width=self.resolved.width,
            height=self.resolved.height,
            resolved=self.resolved,
        )
        for f in confirmed:
            seg.observe(
                FrameEvidence(
                    frame=f,
                    confirmed=True,
                    # A constant placeholder: the kinematics tests are skipped
                    # when there is no motion history, which is the honest
                    # outcome of having only frames here.
                    position=(self.resolved.width / 2.0, self.resolved.height / 2.0),
                    tracker_state="tracking",
                )
            )
        seg.close_open_event(at_frame=clip_end)
        out: list[tuple[int, int]] = []
        for s in seg.events:
            if s.confirmed_frames:
                out.append((s.confirmed_frames[0], s.confirmed_frames[-1]))
        return out

    def _record(
        self,
        delivery_id,
        start,
        end,
        classification,
        count,
        ranges,
        confidence,
        reason,
        flags: Optional[list[str]] = None,
        start_margin: Optional[int] = None,
        end_margin: Optional[int] = None,
        own_event_index: Optional[int] = None,
        requires_review: Optional[bool] = None,
    ) -> ValidationRecord:
        if requires_review is None:
            requires_review = (
                classification in (MULTIPLE_EVENTS, NO_EVENT, AMBIGUOUS)
                or confidence < self.resolved.ambiguous_confidence
            )
        return ValidationRecord(
            delivery_id=delivery_id,
            clip_start=start,
            clip_end=end,
            classification=classification,
            event_count_estimate=count,
            event_ranges=ranges,
            confidence=confidence,
            reason=reason,
            flags=list(flags or []),
            own_event_index=own_event_index,
            start_margin_frames=start_margin,
            end_margin_frames=end_margin,
            requires_human_review=requires_review,
        )