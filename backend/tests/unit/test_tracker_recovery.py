"""
Stage 3 regression tests: Kalman recovery after a track becomes lost.

THE DEFECT THESE COVER
----------------------
`BallTracker.update()` declared a track lost once `missed_frames` exceeded
`MAX_MISSED_FRAMES` — it then returned `(None, None)` for every empty frame,
meaning "I have no ball". But it left `kf.initialized` True, so the NEXT
detection was still compared against that disowned prediction and refused by the
gate.

Because a refusal also advances `missed_frames`, and `missed_frames` only resets
on an *accepted* detection, the refusal prevented its own escape: the counter
climbed 9 -> 40 (MAX_MISSED is 8) and could never fall back. Measured on
real_cricket.mp4: 123 of 195 refusals happened while the filter already reported
itself lost, and 0 of 116 gated frames were followed by an accepted one.

The fix gates only while the track is healthy. These tests pin that behaviour so
it cannot regress, and pin the guarantees that keep the fix honest:

  A. Healthy tracking — a detection near the prediction is still ASSOCIATED.
     This is the guarantee that the fix does not weaken the gate.
  B. Genuine loss — the track does become lost after enough missed frames.
  C. Recovery — a lost track re-initialises on a new valid detection.
  D. Stale prediction — a lost track does NOT resume tracking its old ghost,
     and a healthy track still refuses a distant impostor.
  E. Existing delivery behaviour — segmentation and provenance are unchanged.
"""

from __future__ import annotations

import unittest

from app.core.constants import GATE_RADIUS, MAX_MISSED, MIN_DELIVERY_FRAMES
from app.tracking.kalman_tracker import BallTracker


def drive_misses(tracker: BallTracker, start_frame: int, count: int) -> object:
    """Feed `count` empty frames, returning the last result."""
    result = None
    for i in range(count):
        result = tracker.update(None, start_frame + i)
    return result


class TestAHealthyTrackingIsUnchanged(unittest.TestCase):
    """A. While the track is healthy the gate must still associate normally."""

    def test_detection_near_prediction_is_associated(self):
        t = BallTracker()
        t.update((100, 100), 1)
        # 5 px away: well inside GATE_RADIUS, must be associated as "detected".
        result = t.update((105, 100), 2)
        self.assertEqual(result.source, "detected")
        self.assertFalse(result.gated_out)
        self.assertEqual(t.missed_frames, 0)

    def test_predict_interleave_still_associates(self):
        """detection -> prediction -> detection, the exact A-scenario."""
        t = BallTracker()
        t.update((100, 100), 1)
        pred = t.update(None, 2)
        self.assertEqual(pred.source, "predicted")
        # Just inside the gate of the predicted position.
        result = t.update((pred.position[0] + 10, pred.position[1] + 10), 3)
        self.assertEqual(result.source, "detected")
        self.assertFalse(result.gated_out)

    def test_healthy_track_still_refuses_distant_detection(self):
        """The fix must NOT weaken the gate while the track is healthy."""
        t = BallTracker()
        t.update((100, 100), 1)
        far = (100 + GATE_RADIUS + 50, 100)
        result = t.update(far, 2)
        self.assertEqual(result.source, "predicted")
        self.assertTrue(result.gated_out)
        # And it must NOT be recorded as a measurement.
        self.assertEqual(len(t.delivery_positions), 1)

    def test_association_holds_right_up_to_the_lost_boundary(self):
        """At missed_frames == MAX_MISSED the track is still healthy."""
        t = BallTracker()
        t.update((100, 100), 1)
        # MAX_MISSED misses leaves missed_frames == MAX_MISSED: still alive,
        # because `is_lost` is strictly greater-than.
        drive_misses(t, 2, MAX_MISSED)
        self.assertEqual(t.missed_frames, MAX_MISSED)
        self.assertFalse(t.is_lost)
        result = t.update((110, 100), 100)
        self.assertEqual(result.source, "detected")
        self.assertFalse(result.gated_out)


class TestBGenuineLoss(unittest.TestCase):
    """B. The track must genuinely become lost, not coast forever."""

    def test_track_becomes_lost_after_max_missed(self):
        t = BallTracker()
        t.update((100, 100), 1)
        result = drive_misses(t, 2, MAX_MISSED + 1)
        self.assertTrue(t.is_lost)
        self.assertIsNone(result.position)
        self.assertIsNone(result.source)

    def test_track_still_predicts_while_within_coast(self):
        t = BallTracker()
        t.update((100, 100), 1)
        result = drive_misses(t, 2, MAX_MISSED)
        self.assertFalse(t.is_lost)
        self.assertEqual(result.source, "predicted")

    def test_refusals_alone_also_drive_the_track_lost(self):
        """Gated detections are non-observations and must count toward loss."""
        t = BallTracker()
        t.update((100, 100), 1)
        for i in range(2, 2 + MAX_MISSED + 1):
            result = t.update((100 + GATE_RADIUS + 200, 100), i)
            self.assertTrue(result.gated_out)
        self.assertTrue(t.is_lost)


class TestCRecoveryAfterLoss(unittest.TestCase):
    """C. The core fix: a lost track must re-initialise on a new detection."""

    def test_lost_track_reinitialises_on_new_detection(self):
        t = BallTracker()
        t.update((100, 100), 1)
        drive_misses(t, 2, MAX_MISSED + 2)
        self.assertTrue(t.is_lost)

        # New ball, far from the stale prediction — the case that used to be
        # refused forever.
        result = t.update((900, 600), 100)
        self.assertEqual(result.source, "detected")
        self.assertFalse(result.gated_out)
        self.assertEqual(result.position, (900, 600))
        self.assertEqual(t.missed_frames, 0)
        self.assertFalse(t.is_lost)

    def test_recovered_detection_resets_the_missed_counter(self):
        """The ratchet that drove missed_frames to 40 must be broken."""
        t = BallTracker()
        t.update((100, 100), 1)
        drive_misses(t, 2, MAX_MISSED + 20)
        self.assertGreater(t.missed_frames, MAX_MISSED)
        t.update((900, 600), 200)
        self.assertEqual(t.missed_frames, 0)

    def test_recovery_is_immediate_not_deferred(self):
        """Recovery happens on the very next frame, not after N more misses.

        This is the exact livelock signature: 0 of 116 gated frames in the real
        video were followed by an accepted one.
        """
        t = BallTracker()
        t.update((100, 100), 1)
        drive_misses(t, 2, MAX_MISSED + 1)
        self.assertTrue(t.is_lost)
        first = t.update((900, 600), 100)
        self.assertEqual(first.source, "detected")
        second = t.update((904, 602), 101)
        self.assertEqual(second.source, "detected")
        self.assertEqual(t.missed_frames, 0)

    def test_recovered_point_is_a_real_measurement_not_fabricated(self):
        """Provenance must survive recovery: real box in, real point out."""
        t = BallTracker()
        t.update((100, 100), 1)
        drive_misses(t, 2, MAX_MISSED + 2)
        t.update((900, 600), 100)
        last = t.delivery_positions[-1]
        self.assertEqual(last[0], (900, 600))
        self.assertEqual(last[1], 100)

    def test_recovery_does_not_fabricate_trajectory_points(self):
        """The gap stays empty in the trail — nothing is invented to bridge it."""
        t = BallTracker()
        t.update((100, 100), 1)
        drive_misses(t, 2, MAX_MISSED + 2)
        t.update((900, 600), 100)
        trail = t.traj_points
        # No interpolated/extrapolated filler between the two detections.
        for point in trail:
            self.assertIn(point, [(100, 100), (900, 600)])

    def test_repeated_loss_and_recovery_cycles(self):
        t = BallTracker()
        t.update((100, 100), 1)
        for cycle, x in enumerate((900, 400, 1200, 250)):
            drive_misses(t, 2 + cycle * 50, MAX_MISSED + 2)
            self.assertTrue(t.is_lost)
            result = t.update((x, 300), 3 + cycle * 50)
            self.assertEqual(result.source, "detected")
            self.assertEqual(t.missed_frames, 0)


class TestDNoAssociationWithStalePrediction(unittest.TestCase):
    """D. Recovery must not become blind trust of anything that appears."""

    def test_lost_track_does_not_resume_the_old_ghost(self):
        """After recovery the track follows the NEW ball, not the stale state."""
        t = BallTracker()
        t.update((100, 100), 1)
        drive_misses(t, 2, MAX_MISSED + 2)
        t.update((900, 600), 100)          # recover on the new ball
        # The next frame belongs to the new ball; it must be associated with it.
        result = t.update((905, 603), 101)
        self.assertEqual(result.source, "detected")
        self.assertFalse(result.gated_out)
        # And it must sit near the NEW position, nowhere near the old ghost.
        self.assertGreater(abs(result.position[0] - 100), GATE_RADIUS)

    def test_gate_still_refuses_impostor_while_healthy(self):
        """A lone far-away detection must NOT silently open a delivery."""
        t = BallTracker()
        t.update((100, 100), 1)
        for i in range(2, 6):
            t.update((1000, 1000), i)
        self.assertFalse(t.is_lost)          # 4 refusals < MAX_MISSED
        self.assertEqual(len(t.delivery_positions), 1)
        self.assertFalse(t.in_delivery is False and len(t.delivery_positions) > 1)

    def test_spurious_seed_cannot_reach_delivery_threshold_alone(self):
        """One false positive after loss must not become a delivery.

        MIN_DELIVERY_FRAMES still has to be met by corroborated detections.
        """
        t = BallTracker()
        t.update((100, 100), 1)
        drive_misses(t, 2, MAX_MISSED + 2)
        t.update((900, 600), 100)           # spurious seed, now tracked
        self.assertEqual(t.missed_frames, 0)
        # It dies again within the coast window, having produced 1 detection.
        self.assertLess(len(t.delivery_positions), MIN_DELIVERY_FRAMES)

    def test_spurious_seed_track_dies_and_recovers_again(self):
        """A false positive must not permanently poison the filter."""
        t = BallTracker()
        t.update((100, 100), 1)
        drive_misses(t, 2, MAX_MISSED + 2)
        t.update((900, 600), 100)           # spurious
        drive_misses(t, 101, MAX_MISSED + 2)
        self.assertTrue(t.is_lost)
        result = t.update((300, 200), 200)  # real ball
        self.assertEqual(result.source, "detected")
        self.assertEqual(result.position, (300, 200))


class TestEDeliveryAndSegmentationBehaviourUnchanged(unittest.TestCase):
    """E. The fix must not disturb delivery accumulation or provenance."""

    def test_delivery_accumulates_only_real_detections(self):
        t = BallTracker()
        t.update((100, 100), 1)
        t.update((102, 101), 2)
        t.update(None, 3)
        t.update(None, 4)
        t.update((106, 103), 5)
        self.assertEqual([f for _, f in t.delivery_positions], [1, 2, 5])

    def test_reset_delivery_still_clears_everything(self):
        t = BallTracker()
        t.update((100, 100), 1)
        t.reset_delivery()
        self.assertFalse(t.kf.initialized)
        self.assertEqual(t.missed_frames, 0)
        self.assertEqual(t.delivery_positions, [])
        self.assertEqual(list(t.traj_points), [])

    def test_cold_start_unaffected(self):
        """First detection with no history still seeds cleanly."""
        t = BallTracker()
        result = t.update((640, 360), 1)
        self.assertEqual(result.source, "detected")
        self.assertEqual(result.position, (640, 360))

    def test_no_delivery_invented_from_pure_silence(self):
        t = BallTracker()
        drive_misses(t, 1, 30)
        self.assertFalse(t.in_delivery)
        self.assertEqual(t.delivery_positions, [])

    def test_gate_radius_constant_unchanged(self):
        """Guard against the fix being 'solved' by widening the gate."""
        from app.tracking import kalman_tracker as tracker_mod

        self.assertEqual(tracker_mod.GATE_RADIUS_PX, GATE_RADIUS)
        self.assertEqual(tracker_mod.GATE_RADIUS_PX, 120)

    def test_min_delivery_frames_constant_unchanged(self):
        """Guard against the fix being 'solved' by lowering the threshold."""
        self.assertEqual(MIN_DELIVERY_FRAMES, 8)


if __name__ == "__main__":
    unittest.main()