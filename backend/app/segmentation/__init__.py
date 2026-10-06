"""
segmentation — Deciding which frames belong to which delivery.

This stage owns every judgement about *where a delivery starts and ends* and
*whether a clip is honest enough to serve*. It never sees pixels as pixels, only
frame indices, positions, and detector evidence produced upstream by `tracking`.

  candidate_generation.py  `EventSegmenter`: proposes candidate deliveries,
                           accumulating evidence frame by frame and committing a
                           segment when the activity signal collapses.
  barriers.py              `ActivityWindowTracker`, `SignalTimeline`, `FrameSignal`:
                           the per-frame activity/barrier signal the segmenter
                           reads, and the horizon/barrier bookkeeping.
  fragment_merger.py       `FragmentMerger`, `BarrierSet`, `MergeGate`: joins
                           fragments of one delivery that were split by a brief
                           detection gap, without merging two real deliveries.
  purity_validator.py      `ClipValidator`: the last gate. A segment that is not
                           `SINGLE_EVENT`, or that is flagged `AMBIGUOUS`, is
                           served with its purity recorded; `MULTIPLE_EVENTS`,
                           `NO_EVENT` and missing verdicts are quarantined.

Delivery IDs are assigned here and are never renumbered downstream, so a clip URL
means the same thing for the life of the analysis.
"""

from .candidate_generation import EventSegment, EventSegmenter, FrameEvidence
from .fragment_merger import BarrierSet, FragmentMerger
from .purity_validator import (
    AMBIGUOUS,
    MULTIPLE_EVENTS,
    NO_EVENT,
    SINGLE_EVENT,
    ClipValidator,
)

__all__ = [
    "AMBIGUOUS",
    "MULTIPLE_EVENTS",
    "NO_EVENT",
    "SINGLE_EVENT",
    "BarrierSet",
    "ClipValidator",
    "EventSegment",
    "EventSegmenter",
    "FrameEvidence",
    "FragmentMerger",
]