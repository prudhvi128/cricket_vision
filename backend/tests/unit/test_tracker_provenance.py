"""
Tracker provenance: detected vs predicted must be truthful.

Decision #8 requires the detected/predicted distinction to survive into
storage. Two things are asserted here:

  1. `update()` reports which it produced, including the subtle case where a
     real detection exists but the gate REJECTS it — the returned position is
     then an extrapolation of an ignored measurement, and calling it "detected"
     would be a lie.
  2. `delivery_positions` still records detections ONLY, because the speed fit
     must never be fitted to a Kalman output.
"""

from __future__ import annotations

import unittest

from app.tracking.kalman_tracker import BallTracker


class TestTrackResultProvenance(unittest.TestCase):
    def test_detection_yields_detected_source(self):
        t = BallTracker()
        result = t.update((100, 100), 1)
        self.assertEqual(result.source, "detected")
        self.assertEqual(result.position, (100, 100))

    def test_missed_frame_yields_predicted_source(self):
        t = BallTracker()
        t.update((100, 100), 1)
        result = t.update(None, 2)
        self.assertEqual(result.source, "predicted")
        self.assertIsNotNone(result.position)

    def test_ball_lost_yields_no_position_and_no_source(self):
        t = BallTracker()
        t.update((100, 100), 1)
        # MAX_MISSED is 8, so is_lost becomes true on the 9th missed frame.
        for i in range(2, 11):
            result = t.update(None, i)
        self.assertTrue(t.is_lost)
        self.assertIsNone(result.position)
        self.assertIsNone(result.source)

    def test_gate_rejected_detection_is_reported_as_predicted(self):
        """A detection beyond GATE_RADIUS is ignored, so the output is not measured."""
        t = BallTracker()
        t.update((100, 100), 1)
        # GATE_RADIUS is 120 px; 900 px away is far beyond it.
        result = t.update((1000, 1000), 2)
        self.assertEqual(result.source, "predicted")
        # The bogus detection must not have been recorded as a measurement.
        self.assertEqual(len(t.delivery_positions), 1)
        self.assertEqual(t.delivery_positions[0][1], 1)

    def test_detection_within_gate_is_recorded(self):
        t = BallTracker()
        t.update((100, 100), 1)
        result = t.update((110, 104), 2)
        self.assertEqual(result.source, "detected")
        self.assertEqual(len(t.delivery_positions), 2)


class TestDeliveryPositionsSemantics(unittest.TestCase):
    def test_only_detections_accumulate(self):
        t = BallTracker()
        t.update((100, 100), 1)
        t.update((102, 101), 2)
        t.update(None, 3)          # predicted
        t.update(None, 4)          # predicted
        t.update((106, 103), 5)
        frames = [f for _, f in t.delivery_positions]
        self.assertEqual(frames, [1, 2, 5])

    def test_delivery_list_resets_on_new_delivery(self):
        t = BallTracker()
        t.update((100, 100), 1)
        t.update((105, 102), 2)
        self.assertEqual(len(t.delivery_positions), 2)
        t.reset_delivery()
        self.assertFalse(t.in_delivery)
        self.assertEqual(t.delivery_positions, [])
        self.assertEqual(list(t.traj_points), [])

    def test_delivery_starts_on_first_detection(self):
        t = BallTracker()
        self.assertFalse(t.in_delivery)
        t.update((50, 50), 1)
        self.assertTrue(t.in_delivery)

    def test_predictions_before_any_detection_do_not_open_a_delivery(self):
        """The tracker must not invent a delivery out of nothing."""
        t = BallTracker()
        for i in range(1, 6):
            result = t.update(None, i)
        self.assertFalse(t.in_delivery)
        self.assertEqual(t.delivery_positions, [])


class TestTrackerStateHygiene(unittest.TestCase):
    def test_reset_clears_the_kalman_filter(self):
        t = BallTracker()
        t.update((100, 100), 1)
        t.reset_delivery()
        self.assertFalse(t.kf.initialized)
        self.assertEqual(t.missed_frames, 0)

    def test_trail_is_bounded(self):
        from app.core.constants import MAX_TRAIL

        t = BallTracker()
        for i in range(1, MAX_TRAIL + 50):
            t.update((100 + i, 100), i)
        self.assertLessEqual(len(t.trajectory), MAX_TRAIL)


if __name__ == "__main__":
    unittest.main()