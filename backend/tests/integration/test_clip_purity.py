"""
test_clip_purity.py — a bad clip window must never become a persisted delivery.

The defect these tests exist for
-------------------------------
`ClipValidator.validate()` runs AFTER the pass. By then a window that contains
two batting events has already been written to disk and its DeliveryRecord has
already been appended to `tracking.json`. Its only power is to report. Reporting
a bad artefact is not the same as not making one.

So there is a SECOND, earlier gate. `window_purity()` is called from inside the
single tracking loop, the moment the window is known and BEFORE the writer is
handed anything, and it answers two questions that the pass can already answer
without a second decode:

    NO_EVENT         does this window contain any of THIS delivery's own
                     confirmed detections? If not, the clip is footage of
                     nothing this pipeline measured.
    MULTIPLE_EVENTS  does some OTHER event put a confirmed detection inside the
                     window? If so, the clip is two batting events, which is
                     exactly what makes a clip useless for shot classification.

Neither verdict may reach a DeliveryRecord. Repair first, refuse second: a window
that merely reaches too far is narrowed to the largest honest sub-range, and only
a window that CANNOT be made honest is refused outright — and a narrowing is
refused too if it would end the clip before the ball was lost, because a clip
that stops before the stroke cannot contain the stroke.

HOW THESE TESTS PROVE IT
------------------------
Two levels:

  1. The gate itself, on hand-built events, so every verdict and every repair
     rule is pinned exactly — including the ones a normal run never triggers,
     because the barrier search upstream usually prevents them from arising.
  2. Real single-pass runs over tagged video, asserting the persisted document
     contains no MULTIPLE_EVENTS or NO_EVENT delivery at all.

What is deliberately NOT here: any re-tracking. A purity verdict is computed from
frames this pass already looked at. Opening the source video again to re-detect
would be the second pass this architecture exists to prevent.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.core.config import Paths, PipelineConfig
from app.segmentation.purity_validator import (
    AMBIGUOUS,
    MULTIPLE_EVENTS,
    NO_EVENT,
    SINGLE_EVENT,
    ClipValidator,
)
from app.segmentation.candidate_generation import EventSegment
from app.tracking.tracking_service import TrackingService

from tests.helpers import StubDetector, make_tagged_video

W, H, FPS = 320, 180, 25.0
BG = 30


def make_event(index: int, confirmed: range | list[int], **kw) -> EventSegment:
    """An event whose ONLY claim is the frames it confirmed."""
    frames = list(confirmed)
    return EventSegment(
        index=index,
        start_frame=min(frames),
        end_frame=max(frames),
        closed_at_frame=max(frames) + 5,
        confirmed_frames=frames,
        **kw,
    )


def frames(a: int, b: int) -> list[int]:
    return list(range(a, b + 1))


class PurityCase(unittest.TestCase):
    def setUp(self):
        self.validator = ClipValidator(fps=FPS, width=W, height=H)
        # Three real events, laid out as a broadcast would: one delivery, then a
        # gap, then two deliveries close together.
        self.a = make_event(1, frames(40, 52))
        self.b = make_event(2, frames(130, 145))
        self.c = make_event(3, frames(150, 165))
        self.segments = [self.a, self.b, self.c]


# ═══════════════════════════════════════════════════════════════════════════
# 1. SINGLE_EVENT — the only verdict that may be persisted
# ═══════════════════════════════════════════════════════════════════════════
class TestSingleEventIsTheOnlyPass(PurityCase):
    def test_a_window_around_one_event_passes(self):
        v = self.validator.window_purity(20, 90, self.a, self.segments)
        self.assertEqual(v.classification, SINGLE_EVENT)
        self.assertEqual(v.own_confirmed_frames, frames(40, 52))
        self.assertEqual(v.foreign_event_indices, [])
        self.assertFalse(v.repairable)

    def test_the_verdict_names_what_it_saw(self):
        v = self.validator.window_purity(20, 90, self.a, self.segments)
        self.assertIn("13", v.reason)          # its own confirmed count
        self.assertIn("event 1", v.reason)
        self.assertIn("no other event's", v.reason)

    def test_the_verdict_is_serialisable_for_the_record(self):
        d = self.validator.window_purity(20, 90, self.a, self.segments).as_dict()
        self.assertEqual(d["classification"], SINGLE_EVENT)
        self.assertEqual(d["start"], 20)
        self.assertEqual(d["end"], 90)
        self.assertEqual(d["own_confirmed_frames"], frames(40, 52))
        self.assertFalse(d["repairable"])


# ═══════════════════════════════════════════════════════════════════════════
# 2. NO_EVENT — a clip of footage this pipeline never measured
# ═══════════════════════════════════════════════════════════════════════════
class TestNoEventIsRefused(PurityCase):
    def test_a_window_with_none_of_the_event_s_own_detections_is_refused(self):
        v = self.validator.window_purity(200, 260, self.a, self.segments)
        self.assertEqual(v.classification, NO_EVENT)
        self.assertEqual(v.own_confirmed_frames, [])
        self.assertIn("none of event 1's confirmed detections", v.reason)

    def test_the_reason_states_the_range_it_missed(self):
        v = self.validator.window_purity(200, 260, self.a, self.segments)
        self.assertIn("40-52", v.reason)
        self.assertIn("200-260", v.reason)

    def test_a_window_supplied_with_no_event_at_all_is_refused(self):
        v = self.validator.window_purity(200, 260, None, self.segments)
        self.assertEqual(v.classification, NO_EVENT)
        self.assertIn("no event was supplied", v.reason)
        self.assertFalse(v.repairable)

    def test_an_event_with_no_confirmed_detection_cannot_vouch_for_a_window(self):
        empty = make_event(4, frames(300, 302))
        empty.confirmed_frames = []
        v = self.validator.window_purity(290, 340, empty, self.segments)
        self.assertEqual(v.classification, NO_EVENT)
        self.assertIn("no confirmed detection at all", v.reason)

    def test_the_one_safe_repair_extends_the_end_to_the_first_detection(self):
        """The window stopped short of its own delivery; extending forward can
        only add this event's footage, never remove the stroke."""
        v = self.validator.window_purity(20, 30, self.a, self.segments)
        self.assertEqual(v.classification, NO_EVENT)
        self.assertTrue(v.repairable)
        self.assertEqual(v.repaired_window, (20, 40))
        self.assertIn("restores the event", v.repair_reason)

    def test_a_window_that_opened_after_the_event_ended_cannot_be_repaired(self):
        """There is no honest narrower window: the footage is simply elsewhere."""
        v = self.validator.window_purity(60, 90, self.a, self.segments)
        self.assertEqual(v.classification, NO_EVENT)
        self.assertFalse(v.repairable)
        self.assertIsNone(v.repaired_window)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MULTIPLE_EVENTS — two batting events in one clip
# ═══════════════════════════════════════════════════════════════════════════
class TestMultipleEventsIsRefused(PurityCase):
    def test_a_window_reaching_the_next_event_is_refused(self):
        v = self.validator.window_purity(125, 200, self.b, self.segments)
        self.assertEqual(v.classification, MULTIPLE_EVENTS)
        self.assertEqual(v.foreign_event_indices, [3])
        self.assertEqual(v.foreign_confirmed_counts, {"3": len(frames(150, 165))})
        self.assertIn("more than one batting event", v.reason)
        self.assertIn("event 3 contributes", v.reason)

    def test_a_neighbour_ahead_is_cut_off_one_frame_before_it_starts(self):
        v = self.validator.window_purity(120, 200, self.b, self.segments)
        self.assertEqual(v.classification, MULTIPLE_EVENTS)
        self.assertTrue(v.repairable)
        self.assertEqual(v.repaired_window, (120, 149))
        self.assertIn("ending at 149 removes them", v.repair_reason)

    def test_a_neighbour_behind_is_stepped_over(self):
        v = self.validator.window_purity(40, 175, self.c, self.segments)
        self.assertEqual(v.classification, MULTIPLE_EVENTS)
        self.assertTrue(v.repairable)
        self.assertEqual(v.repaired_window, (146, 175))
        self.assertIn("starting at 146 removes them", v.repair_reason)

    def test_interleaved_events_cannot_be_repaired_at_all(self):
        """No sub-range of this window holds exactly one batting event: each
        event's confirmations sit inside the other's."""
        interleaved = [
            make_event(1, [40, 42, 44, 46]),
            make_event(2, [41, 43, 45, 47]),
        ]
        v = self.validator.window_purity(35, 55, interleaved[0], interleaved)
        self.assertEqual(v.classification, MULTIPLE_EVENTS)
        self.assertFalse(v.repairable)
        self.assertIn("interleave", v.repair_reason)

    def test_a_repair_that_would_cut_before_the_stroke_is_refused(self):
        """Ending the clip before ball-loss would discard the stroke, so the
        honest answer is to refuse the delivery rather than ship half an event."""
        v = self.validator.window_purity(
            120, 200, self.b, self.segments, min_end=155,
        )
        self.assertEqual(v.classification, MULTIPLE_EVENTS)
        self.assertFalse(v.repairable)
        self.assertIn("would discard the stroke", v.repair_reason)
        self.assertIn("155", v.repair_reason)

    def test_the_same_window_repairs_when_the_stroke_is_inside_it(self):
        v = self.validator.window_purity(
            120, 200, self.b, self.segments, min_end=145,
        )
        self.assertTrue(v.repairable)
        self.assertEqual(v.repaired_window, (120, 149))

    def test_a_neighbour_behind_is_stepped_over_without_losing_the_stroke(self):
        v = self.validator.window_purity(40, 175, self.c, self.segments)
        self.assertEqual(v.classification, MULTIPLE_EVENTS)
        self.assertEqual(v.foreign_event_indices, [1, 2])
        self.assertTrue(v.repairable)
        self.assertEqual(v.repaired_window, (146, 175))
        self.assertIn("146", v.repair_reason)

    def test_two_foreign_events_are_both_named(self):
        wide = self.validator.window_purity(20, 200, self.a, self.segments)
        self.assertEqual(wide.foreign_event_indices, [2, 3])
        self.assertEqual(sorted(wide.foreign_confirmed_counts), ["2", "3"])
        self.assertEqual(len(wide.foreign_ranges), 2)


# ═══════════════════════════════════════════════════════════════════════════
# 4. What does NOT condemn a window
# ═══════════════════════════════════════════════════════════════════════════
class TestOverlappingSpanAloneIsNotCondemning(PurityCase):
    def test_a_neighbour_whose_span_overlaps_but_measured_nothing_here_passes(self):
        """Footage the segmenter already judged not to be a ball. The bounded-roll
        search and the exclusion zones exist to handle it; it is not evidence that
        a second event is in the window.

        The neighbour's span reaches back over this window while every one of its
        CONFIRMED detections is still ahead of it — which is exactly what happens
        when an event coasts past its last sighting.
        """
        straddler = make_event(4, frames(150, 155))
        straddler.start_frame = 100          # span widened by coasting
        v = self.validator.window_purity(20, 139, self.a, [self.a, straddler])
        self.assertEqual(v.classification, SINGLE_EVENT)
        self.assertEqual(v.foreign_event_indices, [])
        self.assertEqual(v.foreign_ranges, [[100, 155]])
        self.assertIn("no other event's", v.reason)


class TestMergedEventsAreNotForeign(PurityCase):
    def test_events_this_one_absorbed_are_not_second_batting_events(self):
        """A merged delivery legitimately covers the frames of the candidates it
        merged; treating those as foreign would make merging self-defeating."""
        a = make_event(1, frames(40, 70), merged_from=[2])
        b = make_event(2, frames(52, 70))
        v = self.validator.window_purity(30, 95, a, [a, b])
        self.assertEqual(v.classification, SINGLE_EVENT)
        self.assertEqual(v.foreign_event_indices, [])


class TestThisGateNeverExpressesDoubt(PurityCase):
    """`AMBIGUOUS` belongs to the end-of-pass report, which is asked to grade.
    This gate is asked only "does this window hold somebody else's ball, or none
    at all", so it must never answer with doubt — otherwise a doubt would be
    persisted as a verdict."""

    def test_every_window_gets_one_of_exactly_three_verdicts(self):
        for start in range(20, 180, 7):
            for end in range(start, 200, 11):
                for own in (self.a, self.b, self.c, None):
                    v = self.validator.window_purity(start, end, own, self.segments)
                    self.assertIn(
                        v.classification,
                        {SINGLE_EVENT, MULTIPLE_EVENTS, NO_EVENT},
                        f"{start}-{end} produced {v.classification}",
                    )
                    self.assertNotEqual(v.classification, AMBIGUOUS)


# ═══════════════════════════════════════════════════════════════════════════
# 5. End to end: nothing unsafe is persisted
# ═══════════════════════════════════════════════════════════════════════════
class TestNoUnsafeDeliveryReachesTheDocument(unittest.TestCase):
    """
    Real single-pass runs over real tagged video. The barrier search upstream
    usually prevents an unsafe window from being proposed at all, so these runs
    are here to prove the invariant HOLDS on the whole document rather than to
    trigger the gate: not one persisted delivery may be MULTIPLE_EVENTS or
    NO_EVENT, by either the gate's own verdict or the end-of-pass report.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def analyse(self, windows, n_frames, tags=None):
        video = make_tagged_video(
            self.dir / "source.mp4", n_frames=n_frames,
            tags=tags or [(w[0], w[1], 200 - 60 * i)
                          for i, w in enumerate(windows)],
            width=W, height=H, fps=FPS, background=BG,
        )
        self.video = video
        paths = Paths(analysis_id="test_clip_purity").ensure()
        self.delivered = TrackingService(PipelineConfig()).analyse(
            str(video), paths, detector=StubDetector(windows)
        )
        return self.delivered

    def _assert_clean(self, result):
        for d in result.deliveries:
            gate = (d.event or {}).get("evidence", {}).get("clip_window", {}).get("purity")
            self.assertIsNotNone(
                gate, f"delivery {d.delivery_id} carries no purity verdict"
            )
            self.assertEqual(
                gate["classification"], SINGLE_EVENT,
                f"delivery {d.delivery_id} was persisted on a "
                f"{gate['classification']} verdict: {gate['reason']}",
            )
            self.assertEqual(gate["foreign_event_indices"], [])
            validation = d.validation or {}
            self.assertNotEqual(validation.get("classification"), MULTIPLE_EVENTS)
            self.assertNotEqual(validation.get("classification"), NO_EVENT)

    def test_widely_spaced_deliveries(self):
        result = self.analyse([(31, 45), (200, 215), (340, 355)], n_frames=420)
        self.assertEqual(len(result.deliveries), 3)
        self._assert_clean(result)

    def test_back_to_back_deliveries(self):
        """The overlap case: post-roll and pre-roll reach into each other."""
        result = self.analyse([(31, 45), (50, 64), (70, 84)], n_frames=200)
        self.assertGreaterEqual(len(result.deliveries), 2)
        self._assert_clean(result)

    def test_a_delivery_the_detector_missed_entirely(self):
        """A zero-detection delivery leaves no event to collide with, so the
        neighbours' windows must not reach across the hole either."""
        result = self.analyse([(31, 45), (200, 215)], n_frames=300)
        self.assertEqual(len(result.deliveries), 2)
        self._assert_clean(result)

    def test_a_single_delivery_video(self):
        result = self.analyse([(31, 45)], n_frames=120)
        self.assertEqual(len(result.deliveries), 1)
        self._assert_clean(result)

    def test_no_refused_window_became_a_delivery(self):
        """Whatever the gate turned away is recorded as a rejection with its
        verdict — it is never silently dropped and never persisted."""
        result = self.analyse([(31, 45), (50, 64), (70, 84)], n_frames=200)
        for rejection in result.purity_rejections:
            self.assertIn(rejection["classification"], {MULTIPLE_EVENTS, NO_EVENT})
            self.assertIn("reason", rejection)
        doc = result.tracking_document(PipelineConfig())["segmentation"]
        self.assertEqual(
            doc["purity_rejected_count"], len(result.purity_rejections)
        )
        self.assertEqual(doc["purity_rejections"], result.purity_rejections)


if __name__ == "__main__":
    unittest.main()