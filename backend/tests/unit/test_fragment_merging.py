"""
Physically validated merging of shredded event fragments.

`FragmentMerger` is the safety-critical half of the one-clip-one-event guarantee.
Getting it wrong in the permissive direction produces a clip containing two
deliveries — the exact defect the whole segmentation rewrite exists to remove —
so most of this suite is about the gates that must REFUSE a merge.

The scenarios are the ones measured on real_cricket.mp4 and recorded in
docs/STAGE4_NEAR_MISS_DIAGNOSIS.md:

    #34 + #35 + #36   2 + 7 + 2 confirmed detections of ONE stroke
    #43 + #44         5 + 1, split by `unreachable_displacement`
    #77 + #78 + #79   #78 re-detects a ball already delivered as #77
    #3                a flat horizontal streak, 0% inside the pitch corridor
    #68               5.2 px of total movement at 1.0 px/frame

#68 is the reason corridor context and displacement are gates at all: it is
noise, it is thin, and it is adjacent to real footage, so only its geometry
keeps it out of a delivery.
"""

from __future__ import annotations

import unittest

from app.core.config import SegmentationConfig
from app.core.constants import MIN_DELIVERY_FRAMES, PITCH_CORRIDOR
from app.segmentation.candidate_generation import EventSegment
from app.segmentation.fragment_merger import BarrierSet, FragmentMerger, MergeGate

FPS = 25.0
WIDTH = 1280
HEIGHT = 720


def segment(
    index: int,
    frames: list[int],
    first_position: tuple,
    last_position: tuple,
    exit_velocity: tuple | None = None,
    entry_velocity: tuple | None = None,
    in_corridor: bool = True,
    corridor_frames: int | None = None,
    confidence: float = 1.0,
    ambiguous: bool = False,
) -> EventSegment:
    """A closed EventSegment with the physical context a merge test reads."""
    seg = EventSegment(
        index=index,
        start_frame=frames[0],
        end_frame=frames[-1],
        closed_at_frame=frames[-1],
        confirmed_frames=list(frames),
        close_reason="coast_ceiling",
        confidence=confidence,
        ambiguous=ambiguous,
    )
    seg.first_position = first_position
    seg.last_position = last_position
    seg.exit_velocity = exit_velocity
    seg.entry_velocity = entry_velocity
    seg.in_corridor_frames = (
        len(frames) if corridor_frames is None else corridor_frames
    ) if in_corridor else 0
    return seg


def merger() -> FragmentMerger:
    return FragmentMerger(
        resolved=SegmentationConfig().resolve(FPS, WIDTH, HEIGHT)
    )


def corridor_point(x: float, y: float) -> tuple:
    """A point inside the pitch corridor, for clean fixtures."""
    cx = (PITCH_CORRIDOR[0] + PITCH_CORRIDOR[1]) / 2 * WIDTH
    cy = (PITCH_CORRIDOR[2] + PITCH_CORRIDOR[3]) / 2 * HEIGHT
    return (cx + x, cy + y)


def rejoinable_pair() -> tuple[EventSegment, EventSegment]:
    """
    Two thin, physically continuous fragments of one stroke.

    7 + 2 = 9 confirmed detections, comfortably over MIN_DELIVERY_FRAMES, across
    a 6-frame silence. Both agree about the corridor and both travel the same way,
    so every geometry gate passes and the pair is a legitimate repair.
    """
    first = segment(
        1, list(range(100, 107)), corridor_point(0, 0), corridor_point(70, 0),
        exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
    )
    second = segment(
        2, [112, 113], corridor_point(80, 0), corridor_point(100, 0),
        exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
    )
    return first, second


class TestOneShreddedFragmentsRejoin(unittest.TestCase):
    """A. The repair the module exists for."""

    def test_two_thin_fragments_of_one_stroke_merge(self):
        """The #43 + #44 shape, at a combined count that clears the floor."""
        first, second = rejoinable_pair()
        attempt = merger().try_merge(first, second)
        self.assertTrue(
            attempt.merged,
            f"expected a merge, blocked by {attempt.failed_gates}",
        )
        self.assertEqual(attempt.checks[MergeGate.ADJACENT], True)
        self.assertEqual(attempt.combined_confirmed, 9)

    def test_a_pair_that_stays_below_the_floor_is_still_not_a_delivery(self):
        """
        The real #43 + #44 held 5 + 1 = 6 confirmed detections. Merging them is
        physically fine but USELESS: 6 is under MIN_DELIVERY_FRAMES, and merging
        must not become a back door around that threshold. The gate refuses, and
        the pair stays a rejected candidate rather than a thin delivery.
        """
        first = segment(
            43, list(range(5928, 5933)), corridor_point(0, 0), corridor_point(50, 0),
            exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
        )
        second = segment(
            44, [5934], corridor_point(55, 5), corridor_point(55, 5),
            exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
        )
        attempt = merger().try_merge(first, second)
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.failed_gates, [MergeGate.WORTH_IT])
        self.assertLess(attempt.combined_confirmed, MIN_DELIVERY_FRAMES)

    def test_three_way_chain_merges_when_each_step_earns_it(self):
        """#34 + #35 + #36: 2 + 7 + 2 of one stroke."""
        m = merger()
        head = segment(
            34, [4922, 4923], corridor_point(0, 0), corridor_point(20, 0),
            exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
        )
        middle = segment(
            35, list(range(4924, 4931)), corridor_point(30, 0), corridor_point(100, 0),
            exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
        )
        tail = segment(
            36, [4931, 4932], corridor_point(110, 0), corridor_point(130, 0),
            exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
        )

        first = m.try_merge(head, middle)
        self.assertTrue(first.merged, first.failed_gates)
        m.merge(head, middle, first)
        second = m.try_merge(head, tail)
        self.assertTrue(second.merged, second.failed_gates)
        m.merge(head, tail, second)

        self.assertEqual(len(head.confirmed_frames), 11)
        self.assertEqual(head.merged_from, [35, 36])
        self.assertEqual(head.start_frame, 4922)
        self.assertEqual(head.end_frame, 4932)

    def test_merge_records_full_provenance(self):
        m = merger()
        first, second = rejoinable_pair()
        attempt = m.try_merge(first, second)
        m.merge(first, second, attempt)

        self.assertEqual(first.merged_from, [2])
        self.assertEqual(first.merged_gap_frames, 6)
        self.assertEqual(first.merge_reason, "physically_continuous_fragments")
        checks = first.merge_checks
        self.assertTrue(all(checks["gates"].values()))
        self.assertEqual(checks["combined_confirmed"], 9)
        self.assertEqual(checks["absorbed"], [2])
        # Nothing is hidden: both fragments' evidence survives the merge.
        self.assertEqual(checks["fragment_confirmed_counts"], {"1": 7, "2": 2})
        # And it survives serialisation, so a delivered clip is decomposable.
        record = first.as_dict()
        self.assertEqual(record["merged_from"], [2])
        self.assertEqual(record["merge_reason"], "physically_continuous_fragments")
        self.assertIsNotNone(record["merge_checks"])

    def test_duplicate_fragment_of_a_delivered_ball_is_absorbed(self):
        """
        #78 re-detects a ball already delivered as #77. Folding it in leaves one
        clip for one event instead of a redundant overlapping clip.
        """
        m = merger()
        delivered = segment(
            77, list(range(10120, 10128)), corridor_point(0, 0), corridor_point(80, 0),
            exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
        )
        duplicate = segment(
            78, list(range(10129, 10135)), corridor_point(90, 0), corridor_point(150, 0),
            exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
        )
        attempt = m.try_merge(delivered, duplicate)
        self.assertTrue(attempt.merged, attempt.failed_gates)
        m.merge(delivered, duplicate, attempt)
        self.assertEqual(len(delivered.confirmed_frames), 14)
        self.assertEqual(delivered.merged_from, [78])


class TestTwoGatesThatMustRefuse(unittest.TestCase):
    """B. Over-merging would create a clip with two events. These refuse."""

    def _pair(self):
        """A rejoinable pair: every gate passes unless a test breaks one."""
        return rejoinable_pair()

    def test_two_qualifying_deliveries_are_never_merged(self):
        """
        The single most important refusal. Both candidates already clear
        MIN_DELIVERY_FRAMES, so they are two deliveries and combining them would
        manufacture the one-clip-two-events defect this module prevents.
        """
        first, second = self._pair()
        first.confirmed_frames = list(range(100, 110))
        first.end_frame = 109
        second.confirmed_frames = list(range(112, 122))
        second.end_frame = 121
        attempt = merger().try_merge(first, second)
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.checks[MergeGate.REPAIRABLE], False)
        # The other gates pass, so this refusal is specifically about repairability.
        self.assertEqual(attempt.checks[MergeGate.CONTINUOUS], True)

    def test_black_gap_blocks_the_merge(self):
        first, second = self._pair()
        # A black run inside the 107-111 silence between the fragments.
        barriers = BarrierSet().add(108, 110).merge()
        attempt = merger().try_merge(first, second, barriers)
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.checks[MergeGate.NO_BARRIER], False)

    def test_exclusion_zone_blocks_the_merge(self):
        """An untracked activity window or rejected candidate is impermeable."""
        first, second = self._pair()
        barriers = BarrierSet().add(109, 109).merge()
        attempt = merger().try_merge(first, second, barriers)
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.checks[MergeGate.NO_BARRIER], False)

    def test_barrier_under_a_real_detection_is_not_a_merge_barrier(self):
        """
        The barrier query is scoped to the silence BETWEEN the fragments. A
        barrier sitting under one fragment's own confirmed detection does not
        separate the two, so it must not block the repair.
        """
        first, second = self._pair()
        barriers = BarrierSet().add(100, 105).merge()
        attempt = merger().try_merge(first, second, barriers)
        self.assertTrue(attempt.merged, attempt.failed_gates)

    def test_unreachable_displacement_blocks_the_merge(self):
        """
        The fragments are a long way apart relative to how fast either was
        travelling, so they cannot be one ball.
        """
        first, second = self._pair()
        second.first_position = corridor_point(900, 0)
        second.last_position = corridor_point(900, 0)
        attempt = merger().try_merge(first, second)
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.checks[MergeGate.CONTINUOUS], False)

    def test_direction_reversal_blocks_the_merge(self):
        """The second fragment runs back the way the first was going."""
        first, second = self._pair()
        first.exit_velocity = (10.0, 0.0)
        second.entry_velocity = (-10.0, 0.0)
        attempt = merger().try_merge(first, second)
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.checks[MergeGate.SAME_DIRECTION], False)

    def test_corridor_disagreement_blocks_the_merge(self):
        """
        A stroke on the pitch and a boot in the field are not one ball. This is
        the #3 case: a flat streak that never entered the corridor.
        """
        first, second = self._pair()
        second.in_corridor_frames = 0
        attempt = merger().try_merge(first, second)
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.checks[MergeGate.SAME_CORRIDOR], False)

    def test_gap_beyond_the_bound_blocks_the_merge(self):
        """A merge is bounded in time; long silences are new balls."""
        first, second = self._pair()
        second.start_frame = 200          # ~93 frames of silence
        attempt = merger().try_merge(first, second)
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.checks[MergeGate.ADJACENT], False)

    def test_competing_candidate_in_the_gap_blocks_the_merge(self):
        """Two candidates are not neighbours if a third has detections between."""
        first, second = self._pair()
        rival = segment(
            99, [109], corridor_point(75, 0), corridor_point(75, 0),
        )
        attempt = merger().try_merge(first, second, rivals=[rival])
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.checks[MergeGate.NO_RIVAL], False)

    def test_overlapping_candidates_do_not_merge(self):
        """A negative gap is a segmenter inconsistency, not a merge opportunity."""
        first, second = self._pair()
        second.start_frame = 105
        attempt = merger().try_merge(first, second)
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.checks[MergeGate.ADJACENT], False)

    def test_merge_never_lowers_the_delivery_threshold(self):
        """Two fragments totalling less than MIN_DELIVERY_FRAMES stay rejected."""
        first = segment(
            1, [100, 101], corridor_point(0, 0), corridor_point(20, 0),
            exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
        )
        second = segment(
            2, [102, 103], corridor_point(30, 0), corridor_point(50, 0),
            exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
        )
        attempt = merger().try_merge(first, second)
        self.assertFalse(attempt.merged)
        self.assertEqual(attempt.checks[MergeGate.WORTH_IT], False)
        self.assertLess(attempt.combined_confirmed, MIN_DELIVERY_FRAMES)

    def test_static_noise_never_merges_into_a_delivery(self):
        """
        #68 moved 5.2 px in total at 1.0 px/frame. Even though it is thin, even
        though it is adjacent, and even though the combined count would clear the
        floor, its geometry and corridor context both refuse it.
        """
        delivery = segment(
            67, list(range(3000, 3012)), corridor_point(0, 0), corridor_point(110, 0),
            exit_velocity=(10.0, 0.0), entry_velocity=(10.0, 0.0),
        )
        noise = segment(
            68, [3013], corridor_point(111, 0), corridor_point(113, 0),
            exit_velocity=(1.0, 0.0), entry_velocity=(1.0, 0.0),
            in_corridor=False,
        )
        attempt = merger().try_merge(delivery, noise)
        self.assertFalse(attempt.merged)


class TestThreeBarrierSet(unittest.TestCase):
    """C. The barrier query the gates rely on."""

    def test_intersects_touching_spans(self):
        barriers = BarrierSet([(100, 110), (120, 130)]).merge()
        self.assertTrue(barriers.intersects(105, 106))
        self.assertTrue(barriers.intersects(110, 120))
        self.assertFalse(barriers.intersects(111, 119))

    def test_overlapping_spans_collapse(self):
        barriers = BarrierSet([(100, 110), (105, 130), (200, 210)]).merge()
        self.assertEqual(barriers.spans, [(100, 130), (200, 210)])

    def test_reverse_spans_are_normalised(self):
        barriers = BarrierSet().add(110, 100).merge()
        self.assertEqual(barriers.spans, [(100, 110)])

    def test_empty_set_intersects_nothing(self):
        self.assertFalse(BarrierSet().intersects(1, 1000))


class TestFourMergedEventQuality(unittest.TestCase):
    """D. Merging must never make a doubtful event look certain."""

    def test_confidence_is_the_weaker_of_the_two(self):
        m = merger()
        confident, doubtful = rejoinable_pair()
        doubtful.confidence = 0.4
        attempt = m.try_merge(confident, doubtful)
        self.assertTrue(attempt.merged, attempt.failed_gates)
        m.merge(confident, doubtful, attempt)
        self.assertEqual(confident.confidence, 0.4)

    def test_ambiguity_is_inherited(self):
        m = merger()
        clean, doubtful = rejoinable_pair()
        doubtful.ambiguous = True
        doubtful.ambiguous_reasons.append("only 2 confirmed detections")
        attempt = m.try_merge(clean, doubtful)
        self.assertTrue(attempt.merged, attempt.failed_gates)
        m.merge(clean, doubtful, attempt)
        self.assertTrue(clean.ambiguous)
        self.assertIn("only 2 confirmed detections", clean.ambiguous_reasons)


if __name__ == "__main__":
    unittest.main()