"""
Physically validated merging of shredded event fragments.

WHY THIS EXISTS
---------------
One batting event is sometimes cut into two or three candidates. The mechanism is
known and measured: `EventSegmenter` fits its reference velocity over a sliding
window of the OPEN event, and across a bounce off the bat that fit can be badly
wrong. The reachability test then judges the returning ball physically impossible
and splits — splitting one stroke into pieces none of which is a delivery.

Measured on real_cricket.mp4 (docs/STAGE4_NEAR_MISS_DIAGNOSIS.md):

    #34 + #35 + #36   2 + 7 + 2  = 11 confirmed detections, ONE stroke
    #43 + #44         5 + 1      =  6, split by `unreachable_displacement`
    #77 + #78 + #79   8 + 6 + 4  = 18, where #78 re-detects a delivered ball

Each is either too thin to be a delivery on its own, or a duplicate of footage
already delivered. Both are defects. Neither is repairable by lowering a
threshold: `MIN_DELIVERY_FRAMES` stays where it is, and this module asserts at
import that the combined-evidence floor equals it.

WHY THE SPLIT IS RE-TESTED RATHER THAN TRUSTED
----------------------------------------------
The split was decided by ONE kinematic opinion — the outgoing velocity of the open
event, fitted over a window that may straddle the bounce. That opinion has no
knowledge of what the returning ball was doing just before it was cut off. This
module re-tests continuity from BOTH sides, using each fragment's own entry and
exit velocities and its own confirmed endpoints, and only then combines them.

MERGING IS A REPAIR, NOT A RE-SEGMENTATION
------------------------------------------
Every one of these gates must hold:

  1. ADJACENT      — the two candidates are neighbours in time, and the silence
                     between them is bounded and short.
  2. REPAIRABLE    — at least one of them is too thin to be a delivery on its
                     own. Two candidates that both already qualify are NEVER
                     merged: they are two deliveries, and merging them would
                     create exactly the one-clip-two-events defect this whole
                     module exists to prevent.
  3. NO BARRIER    — no black gap, camera cut, or exclusion zone lies between
                     them. A barrier is a physical discontinuity, not an
                     oversight to be smoothed over.
  4. NO RIVAL      — no third candidate has confirmed detections in the gap.
  5. CONTINUOUS    — the displacement from the first fragment's last confirmed
                     position to the second's first lies inside a reachable radius
                     built from BOTH fragments' own speeds.
  6. SAME DIRECTION— neither fragment's velocity is reversed relative to the
                     other across the gap.
  7. SAME CORRIDOR — both fragments agree about the pitch corridor. A stroke on
                     the pitch and a boot in the field are not one ball.
  8. WORTH IT      — the combined confirmed detections reach the delivery floor.

Provenance is mandatory. `merge()` returns the merged segment with the absorbed
candidate indices, the gap, the reason, and the full measurement behind every
gate, so a delivered clip can always be decomposed back into what it was built
from. Nothing is hidden: the fragments are recorded, not erased.

It holds no video and runs no detector. It decides only about frame ranges and
positions that the single authoritative pass already produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..core.config import ResolvedSegmentation, SegmentationConfig
from ..core.constants import FRAGMENT_MERGE_MIN_COMBINED_FRAMES
from .candidate_generation import EventSegment

# Merging must not become a back door around the delivery threshold. Checked at
# import so the two can never drift apart silently.
assert (
    FRAGMENT_MERGE_MIN_COMBINED_FRAMES
    == SegmentationConfig().fragment_merge_min_combined_frames
), "fragment merge floor has drifted from MIN_DELIVERY_FRAMES"


class MergeGate:
    """The eight independent gates. Every one must pass."""

    ADJACENT = "adjacent_within_bounded_gap"
    REPAIRABLE = "at_least_one_fragment_too_thin_to_deliver"
    NO_BARRIER = "no_event_barrier_in_gap"
    NO_RIVAL = "no_competing_candidate_in_gap"
    CONTINUOUS = "endpoints_physically_connected"
    SAME_DIRECTION = "direction_not_reversed"
    SAME_CORRIDOR = "pitch_corridor_context_agrees"
    WORTH_IT = "combined_evidence_reaches_delivery_floor"

    ALL = (
        ADJACENT,
        REPAIRABLE,
        NO_BARRIER,
        NO_RIVAL,
        CONTINUOUS,
        SAME_DIRECTION,
        SAME_CORRIDOR,
        WORTH_IT,
    )


@dataclass
class MergeAttempt:
    """The complete, auditable outcome of testing one adjacent pair."""

    merged: bool
    reason: str
    checks: dict = field(default_factory=dict)
    gap_frames: int = 0
    gap_seconds: float = 0.0
    displacement_px: Optional[float] = None
    reachable_radius_px: Optional[float] = None
    direction_change_deg: Optional[float] = None
    combined_confirmed: int = 0
    absorbed: list[int] = field(default_factory=list)

    @property
    def failed_gates(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]


@dataclass
class BarrierSet:
    """
    Frames a clip or a merge may not cross.

    Collected from the same pass: black-gap runs, camera cuts, and the exclusion
    zones filed for rejected candidates and untracked activity windows. Held as a
    sorted list of inclusive spans so a gap query stays O(log n).
    """

    spans: list[tuple[int, int]] = field(default_factory=list)

    def add(self, start: int, end: int) -> "BarrierSet":
        """Record one span. Returns self, so spans can be chained."""
        start, end = int(start), int(end)
        if end < start:
            start, end = end, start
        self.spans.append((start, end))
        self.spans.sort()
        return self

    def merge(self) -> "BarrierSet":
        """Collapse overlapping and touching spans into their union."""
        merged = BarrierSet()
        for start, end in sorted(self.spans):
            if merged.spans and start <= merged.spans[-1][1] + 1:
                prev_s, prev_e = merged.spans[-1]
                merged.spans[-1] = (prev_s, max(prev_e, end))
            else:
                merged.spans.append((start, end))
        return merged

    def intersects(self, start: int, end: int) -> bool:
        """Does any barrier touch the inclusive frame range?"""
        lo, hi = 0, len(self.spans) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s, e = self.spans[mid]
            if end < s:
                hi = mid - 1
            elif start > e:
                lo = mid + 1
            else:
                return True
        return False

    def in_range(self, start: int, end: int) -> list[tuple[int, int]]:
        return [(s, e) for s, e in self.spans if not (e < start or s > end)]


def _angle_between_deg(v1: tuple, v2: tuple) -> Optional[float]:
    a = math.hypot(v1[0], v1[1])
    b = math.hypot(v2[0], v2[1])
    if a < 1e-6 or b < 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (a * b)))
    return math.degrees(math.acos(cosine))


class FragmentMerger:
    """
    Decides whether two adjacent event candidates are one ball.

    Stateless apart from configuration: `merge()` is a pure decision over two
    segments, so it can be unit-tested directly and re-run over stored events
    long after the pass that produced them.
    """

    def __init__(
        self,
        resolved: Optional[ResolvedSegmentation] = None,
        config: Optional[SegmentationConfig] = None,
        fps: float = 25.0,
        width: int = 320,
        height: int = 180,
    ) -> None:
        self.resolved = resolved or (config or SegmentationConfig()).resolve(
            fps, width, height
        )

    # ── The decision ──────────────────────────────────────────────────────────

    def try_merge(
        self,
        first: EventSegment,
        second: EventSegment,
        barriers: Optional[BarrierSet] = None,
        rivals: Optional[list[EventSegment]] = None,
    ) -> MergeAttempt:
        """Test one adjacent pair. Never mutates either argument."""
        r = self.resolved
        barriers = barriers or BarrierSet()
        gap = second.start_frame - first.end_frame
        checks: dict[str, bool] = {}

        # ── 1. ADJACENT ────────────────────────────────────────────────────
        # Strictly later, with a bounded silence. A negative gap means the two
        # candidates overlap in time, which is a segmenter inconsistency, not a
        # merge opportunity.
        adjacent = 1 <= gap <= r.fragment_merge_max_gap_frames
        checks[MergeGate.ADJACENT] = adjacent

        # ── 2. REPAIRABLE ──────────────────────────────────────────────────
        # Merging exists to rescue a thin fragment. Two candidates that already
        # qualify as deliveries are left strictly alone — combining them would
        # manufacture the one-clip-two-events failure this module prevents.
        thin_enough = min(
            first.confirmed_count, second.confirmed_count
        ) < r.fragment_merge_min_combined_frames
        checks[MergeGate.REPAIRABLE] = thin_enough

        # ── 3. NO BARRIER ──────────────────────────────────────────────────
        # The gap itself, exclusive of the two fragments' own confirmed frames:
        # a barrier sitting under a real detection is not what separates them.
        barrier_free = not barriers.intersects(
            first.end_frame + 1, second.start_frame - 1
        )
        checks[MergeGate.NO_BARRIER] = barrier_free

        # ── 4. NO RIVAL ────────────────────────────────────────────────────
        # A third candidate with real detections in the gap means the fragments
        # are not neighbours at all.
        rivals_in_gap = [
            s for s in (rivals or [])
            if s.index not in (first.index, second.index)
            and s.confirmed_frames
            and not (s.end_frame < first.end_frame or s.start_frame > second.start_frame)
        ]
        no_rival = not rivals_in_gap
        checks[MergeGate.NO_RIVAL] = no_rival

        # ── 5. CONTINUOUS ──────────────────────────────────────────────────
        # Reachability rebuilt from BOTH fragments' own speeds, not from the
        # outgoing velocity whose misfit may have caused the split. The faster
        # of the two governs, so a merge is never made harder by a fragment that
        # was moving quickly before it was cut off.
        reach_radius = None
        displacement = None
        continuous = False
        if first.last_position and second.first_position:
            span = max(1, gap)
            displacement = math.hypot(
                second.first_position[0] - first.last_position[0],
                second.first_position[1] - first.last_position[1],
            )
            speed = max(
                _speed(first.exit_velocity),
                _speed(second.entry_velocity),
                1e-6,
            )
            reach_radius = (
                speed * span
                + r.fragment_merge_reach_base_px
                + r.fragment_merge_reach_rate_px * span
            )
            continuous = displacement <= reach_radius
        checks[MergeGate.CONTINUOUS] = continuous

        # ── 6. SAME DIRECTION ──────────────────────────────────────────────
        # Neither side may be reversed with respect to the other. Checked twice:
        # against the gap displacement, and the two fragments against each
        # other, because two fragments can agree with each other while both run
        # backwards relative to the line between them.
        direction_change = None
        reversed_hard = False
        if displacement is not None and displacement > 1e-6:
            gap_vector = (
                second.first_position[0] - first.last_position[0],
                second.first_position[1] - first.last_position[1],
            )
            # Each fragment's own velocity, measured against the line it should
            # have continued along across the gap.
            observed = [
                a
                for a in (
                    _angle_between_deg(v, gap_vector)
                    for v in (first.exit_velocity, second.entry_velocity)
                    if _speed(v) > 1e-6
                )
                if a is not None
            ]
            if observed:
                direction_change = max(observed)
                reversed_hard = direction_change >= r.reversal_degrees
            if first.exit_velocity and second.entry_velocity:
                between = _angle_between_deg(
                    first.exit_velocity, second.entry_velocity
                )
                if between is not None and between >= r.reversal_degrees:
                    reversed_hard = True
                    direction_change = max(
                        direction_change or 0.0, between
                    )
        checks[MergeGate.SAME_DIRECTION] = not reversed_hard

        # ── 7. SAME CORRIDOR ───────────────────────────────────────────────
        # Agreement about the pitch corridor. A stroke and a fielder's boot both
        # produce detections, but only the stroke stays on the pitch.
        ratio = r.fragment_merge_corridor_ratio
        a_in = first.corridor_ratio >= ratio
        b_in = second.corridor_ratio >= ratio
        corridor_ok = (a_in == b_in) and (first.in_corridor_frames > 0 or
                                          second.in_corridor_frames > 0)
        checks[MergeGate.SAME_CORRIDOR] = corridor_ok

        # ── 8. WORTH IT ────────────────────────────────────────────────────
        combined = first.confirmed_count + second.confirmed_count
        worth = combined >= r.fragment_merge_min_combined_frames
        checks[MergeGate.WORTH_IT] = worth

        passed = [g for g in MergeGate.ALL if checks.get(g)]
        failed = [g for g in MergeGate.ALL if not checks.get(g)]

        return MergeAttempt(
            merged=not failed,
            reason=(
                "physically_continuous_fragments"
                if not failed
                else f"blocked_by:{','.join(failed)}"
            ),
            checks={g: checks.get(g, False) for g in MergeGate.ALL},
            gap_frames=gap,
            gap_seconds=gap / r.fps if r.fps else 0.0,
            displacement_px=displacement,
            reachable_radius_px=reach_radius,
            direction_change_deg=direction_change,
            combined_confirmed=combined,
            absorbed=[second.index] if not failed else [],
        )

    # ── The action ────────────────────────────────────────────────────────────

    def merge(
        self,
        first: EventSegment,
        second: EventSegment,
        attempt: MergeAttempt,
    ) -> EventSegment:
        """
        Fold `second` into `first`, in place, and record the provenance.

        Only ever called after `try_merge` returned `merged=True`. The absorbed
        candidate's identity, frame list and measurements are kept: the merge
        hides nothing, it records what it combined.
        """
        # Captured BEFORE any mutation. Recording these afterwards would report
        # the combined total against both indices, which is exactly the kind of
        # quietly-wrong audit trail this provenance exists to prevent.
        head_confirmed = first.confirmed_count
        follower_confirmed = second.confirmed_count

        first.confirmed_frames = sorted(
            set(first.confirmed_frames) | set(second.confirmed_frames)
        )
        first.predicted_frames = sorted(
            set(first.predicted_frames) | set(second.predicted_frames)
        )
        first.reacquired_frames = sorted(
            set(first.reacquired_frames) | set(second.reacquired_frames)
        )
        first.gated_continuation_frames = sorted(
            set(first.gated_continuation_frames)
            | set(second.gated_continuation_frames)
        )
        first.in_corridor_frames += second.in_corridor_frames

        # The event now spans both fragments' footage.
        first.end_frame = max(first.end_frame, second.end_frame)
        first.closed_at_frame = max(first.closed_at_frame, second.closed_at_frame)
        first.ball_lost_frame = max(
            first.ball_lost_frame or first.end_frame,
            second.ball_lost_frame or second.end_frame,
        )
        if second.first_position is not None:
            first.last_position = second.last_position
        if second.entry_velocity is not None:
            first.entry_velocity = first.entry_velocity or second.entry_velocity
        if second.exit_velocity is not None:
            first.exit_velocity = second.exit_velocity

        # Provenance: which candidates were absorbed, across what silence, on what
        # evidence, and which gates agreed.
        for idx in second.merged_from or [second.index]:
            if idx not in first.merged_from:
                first.merged_from.append(idx)
        first.merged_gap_frames = attempt.gap_frames
        first.merge_reason = attempt.reason
        first.merge_checks = {
            "gates": dict(attempt.checks),
            "gap_frames": attempt.gap_frames,
            "gap_seconds": round(attempt.gap_seconds, 4),
            "displacement_px": (
                round(attempt.displacement_px, 2)
                if attempt.displacement_px is not None else None
            ),
            "reachable_radius_px": (
                round(attempt.reachable_radius_px, 2)
                if attempt.reachable_radius_px is not None else None
            ),
            "direction_change_deg": (
                round(attempt.direction_change_deg, 1)
                if attempt.direction_change_deg is not None else None
            ),
            "combined_confirmed": attempt.combined_confirmed,
            "absorbed": list(attempt.absorbed),
            "fragment_confirmed_counts": {
                str(first.index): head_confirmed,
                str(second.index): follower_confirmed,
            },
        }

        # A merged event is at least as trustworthy as its parts: it inherits the
        # weaker confidence rather than averaging, so merging can never make a
        # doubtful event look certain.
        first.confidence = min(first.confidence, second.confidence)
        if second.ambiguous:
            first.ambiguous = True
            for reason in second.ambiguous_reasons:
                if reason not in first.ambiguous_reasons:
                    first.ambiguous_reasons.append(reason)
        first.close_reason = (
            second.close_reason if second.close_reason else first.close_reason
        )
        return first


def _speed(velocity) -> float:
    if not velocity:
        return 0.0
    return math.hypot(velocity[0], velocity[1])