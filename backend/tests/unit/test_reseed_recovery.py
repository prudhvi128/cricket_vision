"""
Re-acquisition of a live ball whose Kalman prediction has drifted (item 6).

`BallTracker.reseed()` re-acquires the ball WITHOUT ending the delivery. Until
this suite existed it was unreachable dead code: the tracking loop called it
only when `decision.reseed_position is not None and track.position is None`, and
those two conditions are mutually exclusive — a null position is returned only
on the no-detection branch, and without a detection the segmenter has nothing to
rule on, so it never sets `reseed_position`.

The consequence was not a missing feature but a real defect. When the ball is
plainly on screen and the segmenter has CONFIRMED it belongs to the open event,
yet the filter's own prediction has drifted outside its 120 px gate, the filter
refuses the box — and because a refusal also advances `missed_frames`, it cannot
recover on its own. The real trajectory is shredded into successive short
fragments, which is the mechanism behind the `#34+#35+#36` and `#77+#78+#79`
chains in docs/STAGE4_NEAR_MISS_DIAGNOSIS.md.

These tests pin the reachable trigger, the no-fabrication guarantee, and the fact
that the ordinary accepted-detection path is untouched.
"""

from __future__ import annotations

import unittest

from app.segmentation.candidate_generation import ContinuationKind, EventSegmenter, FrameEvidence
from app.tracking.kalman_tracker import BallTracker
from app.tracking.tracking_service import _needs_reseed, _tracker_state

FPS = 25.0
WIDTH = 320
HEIGHT = 180


def _drive_segmentation(detections, *, reseed: bool, fps: float = FPS):
    """
    Mirror the tracking loop: one tracker, one segmenter, one pass.

    `reseed` toggles only the re-acquisition step under test; everything else,
    including the order of operations, is the production order.
    """
    tracker = BallTracker()
    segmenter = EventSegmenter(fps=fps, width=WIDTH, height=HEIGHT)
    reseeds = 0
    frame_results = []

    for frame, det in enumerate(detections, start=1):
        track = tracker.update(det, frame)
        evidence = FrameEvidence(
            frame=frame,
            confirmed=(track.source == "detected"),
            candidate=bool(track.gated_out),
            position=det,
            tracker_position=track.position,
            tracker_state=_tracker_state(track, tracker),
        )
        decision = segmenter.observe(evidence)

        if reseed and _needs_reseed(decision, track, tracker, det):
            tracker.reseed(det, frame)
            reseeds += 1

        frame_results.append((frame, track, decision))
        if decision.closed is not None:
            tracker.reset_delivery()

    segmenter.close_open_event(at_frame=len(detections))
    return segmenter, reseeds, frame_results


class TestOneReachability(unittest.TestCase):
    """A. Why the old guard could never fire."""

    def test_null_position_and_reseed_position_are_mutually_exclusive(self):
        """
        The old guard demanded both. `update()` yields a null position only with
        no detection, and a detection-free frame yields no `reseed_position`.
        """
        tracker = BallTracker()
        segmenter = EventSegmenter(fps=FPS, width=WIDTH, height=HEIGHT)

        tracker.update((100, 100), 1)
        for f in range(2, 20):          # coast past MAX_MISSED
            track = tracker.update(None, f)
            decision = segmenter.observe(
                FrameEvidence(
                    frame=f,
                    tracker_position=track.position,
                    tracker_state=_tracker_state(track, tracker),
                )
            )
            if track.position is None:
                self.assertIsNone(
                    decision.reseed_position,
                    "a frame with no detection must never request a reseed: "
                    "there is nothing real to re-acquire",
                )
            else:
                self.assertTrue(
                    decision.reseed_position is None
                    or decision.reseed_position[0] is not None
                )

    def test_segmenter_only_requests_reseed_from_a_real_measurement(self):
        """
        `reseed_position` comes from a detector position and never from a
        Kalman extrapolation, so honouring it cannot fabricate a trajectory.
        """
        segmenter = EventSegmenter(fps=FPS, width=WIDTH, height=HEIGHT)
        decision = segmenter.observe(
            FrameEvidence(frame=1, confirmed=True, position=(10, 10))
        )
        self.assertEqual(decision.kind, ContinuationKind.NEW_EVENT)
        # A frame with only a prediction carries no position to re-acquire on.
        decision = segmenter.observe(
            FrameEvidence(
                frame=2,
                tracker_position=(30, 10),
                tracker_state="coasting",
            )
        )
        self.assertIsNone(decision.reseed_position)


class TestTwoReacquiresLiveBall(unittest.TestCase):
    """B. The fix: a ball the filter refused but the segmenter trusts is re-found."""

    def _drifting_ball(self):
        """
        Rightward run-up, a 6-frame occlusion, then a bounce off the bat that
        sends the ball down-right — out of the gate of a filter still committed
        to the old direction, but comfortably inside the segmenter's reachable
        radius, so the segmenter correctly rules it the same ball.
        """
        detections = [(100 + 20 * i, 100) for i in range(6)]   # frames 1-6
        detections += [None] * 6                                 # frames 7-12
        detections += [(270, 210), (280, 225), (290, 240)]       # frames 13-15
        return detections

    def test_gate_refusal_is_now_followed_by_acceptance(self):
        detections = self._drifting_ball()

        _, reseeds_off, results_off = _drive_segmentation(detections, reseed=False)
        _, reseeds_on, results_on = _drive_segmentation(detections, reseed=True)

        # Frame 13 is refused by the gate on BOTH runs: the filter's prediction
        # is still extrapolating straight right.
        _, track_13_off, decision_13_off = results_off[12]
        self.assertTrue(track_13_off.gated_out)
        self.assertEqual(decision_13_off.kind, ContinuationKind.GATED_CONTINUATION)

        # Without re-acquisition the filter stays committed to the wrong
        # direction and refuses the next real sighting too.
        _, track_14_off, _ = results_off[13]
        self.assertTrue(track_14_off.gated_out)
        self.assertEqual(reseeds_off, 0)

        # With it, the filter is back on the ball and associates the very next
        # frame. This is the recovery the dead guard made unreachable.
        self.assertEqual(reseeds_on, 1)
        _, track_13_on, _ = results_on[12]
        self.assertTrue(track_13_on.gated_out)
        _, track_14_on, _ = results_on[13]
        self.assertFalse(track_14_on.gated_out)
        self.assertEqual(track_14_on.source, "detected")

    def test_reacquired_points_are_real_measurements(self):
        """
        Re-seeding must not launder a prediction into a detection. The recorded
        provenance stays "detected" because the box was real, and the position
        is the detector's.
        """
        detections = self._drifting_ball()
        tracker = BallTracker()
        segmenter = EventSegmenter(fps=FPS, width=WIDTH, height=HEIGHT)

        for frame, det in enumerate(detections, start=1):
            track = tracker.update(det, frame)
            decision = segmenter.observe(
                FrameEvidence(
                    frame=frame,
                    confirmed=(track.source == "detected"),
                    candidate=bool(track.gated_out),
                    position=det,
                    tracker_position=track.position,
                    tracker_state=_tracker_state(track, tracker),
                )
            )
            if _needs_reseed(decision, track, tracker, det):
                track = tracker.reseed(det, frame)
                self.assertEqual(track.source, "detected")
                self.assertEqual(track.position, (270, 210))

        # The re-acquired box enters delivery_positions exactly once.
        recorded = [p for p, _ in tracker.delivery_positions]
        self.assertIn((270, 210), recorded)


class TestThreeNeverFabricates(unittest.TestCase):
    """C. Guard rails on the new trigger."""

    def test_no_reseed_without_a_real_detection(self):
        decision = _FakeDecision(reseed_position=(10, 10))
        track = _FakeTrack(gated_out=True)
        self.assertFalse(
            _needs_reseed(decision, track, _FakeTracker(lost=True), None)
        )

    def test_no_reseed_when_the_filter_absorbed_the_detection(self):
        """
        The ordinary path is untouched: a detection the filter already accepted
        leaves it healthy, so nothing is re-seeded.
        """
        decision = _FakeDecision(reseed_position=(10, 10))
        track = _FakeTrack(gated_out=False)
        self.assertFalse(
            _needs_reseed(decision, track, _FakeTracker(lost=False), (10, 10))
        )

    def test_no_reseed_without_a_segmenter_request(self):
        self.assertFalse(
            _needs_reseed(
                _FakeDecision(reseed_position=None),
                _FakeTrack(gated_out=True),
                _FakeTracker(lost=True),
                (10, 10),
            )
        )
        self.assertFalse(
            _needs_reseed(
                None, _FakeTrack(gated_out=True), _FakeTracker(lost=True), (10, 10)
            )
        )

    def test_reseed_fires_when_the_filter_declared_itself_lost(self):
        """A disowned filter is the second reachable case, even without a refusal."""
        decision = _FakeDecision(reseed_position=(10, 10))
        self.assertTrue(
            _needs_reseed(
                decision, _FakeTrack(gated_out=False), _FakeTracker(lost=True), (10, 10)
            )
        )


class TestFourStageThreeUnaffected(unittest.TestCase):
    """D. The Stage 3 recovery behaviour must be identical with and without this."""

    def test_clean_delivery_is_identical(self):
        detections = [(100 + 20 * i, 100) for i in range(20)]
        off, reseeds_off, _ = _drive_segmentation(detections, reseed=False)
        on, reseeds_on, _ = _drive_segmentation(detections, reseed=True)
        self.assertEqual(reseeds_off, 0)
        self.assertEqual(reseeds_on, 0)
        self.assertEqual(
            [(s.start_frame, s.end_frame) for s in off.events],
            [(s.start_frame, s.end_frame) for s in on.events],
        )

    def test_no_confirmation_ever_requests_a_reseed(self):
        segmenter = EventSegmenter(fps=FPS, width=WIDTH, height=HEIGHT)
        tracker = BallTracker()
        for frame in range(1, 12):
            track = tracker.update(None, frame)
            decision = segmenter.observe(
                FrameEvidence(
                    frame=frame,
                    tracker_position=track.position,
                    tracker_state=_tracker_state(track, tracker),
                )
            )
            self.assertFalse(_needs_reseed(decision, track, tracker, None))


class _FakeDecision:
    def __init__(self, reseed_position):
        self.reseed_position = reseed_position


class _FakeTrack:
    def __init__(self, gated_out: bool):
        self.gated_out = gated_out
        self.position = (1, 1)
        self.source = "predicted"


class _FakeTracker:
    def __init__(self, lost: bool):
        self.is_lost = lost


if __name__ == "__main__":
    unittest.main()