"""
Phase 3 decision D1: missing speed is NEVER filled in.
Phase 3 decision D2: the speed number is an estimate, and says so.

These are the two decisions most likely to regress silently, because the
upstream behaviour (median imputation, unlabelled fudge factor) is what the code
used to do. Both are asserted directly.
"""

from __future__ import annotations

import unittest

import numpy as np

from app.core.constants import (
    SPEED_CALIBRATION_OFFSET_KMH,
    SPEED_CALIBRATION_SCALE,
)
from app.schemas.tracking import Trajectory, TrajectoryPoint
from app.analytics.bowling import analyse_delivery
from app.analytics.calibration import calibration_metadata, make_homography
from app.analytics.trajectory import compute_release_speed

W, H, FPS = 640, 360, 25.0
HOMOGRAPHY = make_homography(W, H)


def straight_trajectory(n: int = 12, detected: bool = True) -> Trajectory:
    """A plausible release-to-bounce path in pixel space."""
    pts = []
    for i in range(n):
        x = 250 + i * 8
        y = 150 + i * 6
        pts.append(
            TrajectoryPoint(
                original_frame=100 + i,
                x=x, y=y,
                source="detected" if detected else "predicted",
                confidence=0.9 if detected else None,
            )
        )
    return Trajectory(points=pts)


def too_short_trajectory(n: int = 2) -> Trajectory:
    return Trajectory(points=[
        TrajectoryPoint(original_frame=100 + i, x=200 + i, y=150 + i,
                        source="detected", confidence=0.9)
        for i in range(n)
    ])


class TestD1NoImputation(unittest.TestCase):
    def test_default_config_does_not_impute(self):
        from app.core.config import PipelineConfig

        self.assertFalse(
            PipelineConfig().impute_missing_speeds,
            "D1 requires imputation to be OFF by default",
        )

    def test_short_trajectory_reports_null_speed_not_a_median(self):
        traj = too_short_trajectory(2)
        _, _, bowling, _, _ = analyse_delivery(
            traj, HOMOGRAPHY, FPS, W, H,
            impute_missing_speeds=False,
            speeds_from=[120.0, 130.0, 140.0],
        )
        self.assertIsNone(bowling.speed_kmh)
        self.assertFalse(bowling.speed_imputed)
        self.assertIsNone(bowling.speed_method)

    def test_empty_trajectory_reports_null_speed(self):
        _, _, bowling, _, flags = analyse_delivery(
            Trajectory(points=[]), HOMOGRAPHY, FPS, W, H,
            speeds_from=[120.0],
        )
        self.assertIsNone(bowling.speed_kmh)
        self.assertFalse(bowling.speed_imputed)

    def test_prediction_only_trajectory_yields_no_speed(self):
        """A Kalman-only path must not produce a speed at all."""
        traj = straight_trajectory(12, detected=False)
        _, _, bowling, _, _ = analyse_delivery(traj, HOMOGRAPHY, FPS, W, H)
        self.assertIsNone(bowling.speed_kmh)
        self.assertFalse(bowling.speed_imputed)

    def test_null_speed_is_flagged_for_consumers(self):
        _, _, _, _, flags = analyse_delivery(
            too_short_trajectory(2), HOMOGRAPHY, FPS, W, H,
        )
        self.assertIn("speed_unavailable", flags)

    def test_imputation_when_explicitly_enabled_is_loudly_marked(self):
        """Opt-in behaviour is retained for offline study, but never silent."""
        traj = too_short_trajectory(2)
        _, _, bowling, _, flags = analyse_delivery(
            traj, HOMOGRAPHY, FPS, W, H,
            impute_missing_speeds=True,
            speeds_from=[120.0, 130.0, 140.0],
        )
        self.assertEqual(bowling.speed_kmh, 130.0)   # median
        self.assertTrue(bowling.speed_imputed)
        self.assertEqual(bowling.speed_method, "imputed_median")
        self.assertIn("speed_imputed", flags)


class TestD2CalibrationIsLabelled(unittest.TestCase):
    def test_calibration_block_states_it_is_an_estimate(self):
        meta = calibration_metadata()
        self.assertTrue(meta["speed_is_estimate"])
        self.assertFalse(meta["speed_is_calibrated"])
        self.assertEqual(meta["speed_method"], "homography_least_squares")

    def test_calibration_persists_the_actual_scale_and_offset(self):
        meta = calibration_metadata()
        self.assertEqual(meta["speed_scale"], SPEED_CALIBRATION_SCALE)
        self.assertEqual(meta["speed_offset_kmh"], SPEED_CALIBRATION_OFFSET_KMH)
        self.assertIn(str(SPEED_CALIBRATION_SCALE), meta["speed_formula"])
        self.assertIn(str(SPEED_CALIBRATION_OFFSET_KMH), meta["speed_formula"])

    def test_calibration_records_the_homography_it_used(self):
        meta = calibration_metadata()
        fracs = meta["homography_corner_fractions"]
        self.assertEqual(set(fracs), {"TL", "TR", "BL", "BR"})
        self.assertIn("broadcast", meta["homography_note"].lower())

    def test_a_computed_speed_is_not_marked_calibrated(self):
        traj = straight_trajectory(12)
        _, _, bowling, samples, _ = analyse_delivery(traj, HOMOGRAPHY, FPS, W, H)
        if bowling.speed_kmh is not None:
            self.assertFalse(bowling.speed_calibrated)
            self.assertFalse(bowling.speed_imputed)
            self.assertEqual(bowling.speed_method, "homography_least_squares")
            # The evidence behind the number must be recorded alongside it.
            self.assertGreater(len(samples), 0)

    def test_upstream_formula_is_preserved_exactly(self):
        """The arithmetic must match upstream, even though it is mislabelled."""
        raw = compute_release_speed(
            [((250 + i * 8, 150 + i * 6), 100 + i) for i in range(12)],
            FPS, HOMOGRAPHY,
        )
        self.assertIsNotNone(raw)
        # compute_release_speed applies the calibration internally; verify the
        # same result by projecting manually.
        from app.analytics.calibration import pixel_to_ground

        pts = [((250 + i * 8, 150 + i * 6), 100 + i) for i in range(12)]
        ts = np.array([f / FPS for _, f in pts])
        gxs = np.array([pixel_to_ground(p, HOMOGRAPHY)[0] for p, _ in pts])
        gys = np.array([pixel_to_ground(p, HOMOGRAPHY)[1] for p, _ in pts])
        t_rel = ts - ts[0]
        vx = np.polyfit(t_rel, gxs, 1)[0]
        vy = np.polyfit(t_rel, gys, 1)[0]
        expected = float(np.hypot(vx, vy) * 3.6 * 2.0 - 20.0)
        self.assertAlmostEqual(raw, round(expected, 1), delta=1.5)


class TestAnalyticsUsesPixelCoordinatesForZones(unittest.TestCase):
    """classify_length/classify_line divide by the frame dimension themselves.

    Passing ground-plane metres there would produce meaningless zone labels, so
    this locks in that the correct units are used.
    """

    def test_length_and_line_are_produced_when_calibrated(self):
        """
        Calibrated explicitly.

        `analyse_delivery` defaults to `geometry_calibrated=False`, so a caller
        that has not supplied pitch corners gets `null` for every ground-plane
        quantity. These assertions are about the calibrated path, so they say so.
        """
        traj = straight_trajectory(14)
        _, bounce, bowling, _, _ = analyse_delivery(
            traj, HOMOGRAPHY, FPS, W, H, geometry_calibrated=True
        )
        self.assertIsNotNone(bowling.length)
        self.assertIsNotNone(bowling.line)
        self.assertTrue(bowling.geometry_calibrated)
        # D2: even calibrated, the speed is a least-squares estimate over an
        # assumed homography, not a calibrated reading. `speed_calibrated` must
        # never become True while the number comes from this fit.
        self.assertFalse(bowling.speed_calibrated)

    def test_uncalibrated_default_nulls_ground_plane_but_keeps_pixels(self):
        """
        The safe default: no corners measured on this video means no metres.

        This is the behaviour that was missing before. `analyse_delivery` used to
        inherit `geometry_calibrated=True`, so every delivery carried a speed,
        length and line derived from corner fractions measured on a different
        broadcast template — fabricated values that no test could have caught
        because the tests called the function with a homography and assumed that
        was enough.
        """
        traj = straight_trajectory(14)
        _, bounce, bowling, samples, _ = analyse_delivery(
            traj, HOMOGRAPHY, FPS, W, H
        )
        # Homography was passed, and it is still not trusted: a homography without
        # corners measured on THIS video does not calibrate anything.
        self.assertFalse(bowling.speed_calibrated)
        self.assertIsNone(bowling.speed_kmh)
        self.assertIsNone(bowling.length)
        self.assertIsNone(bowling.line)
        self.assertIsNone(bowling.swing)
        self.assertIsNone(bowling.bounce_angle)
        self.assertFalse(bowling.geometry_calibrated)
        self.assertFalse(bowling.speed_imputed)
        # Nothing is invented: no speed samples from an uncalibrated fit.
        self.assertEqual([s for s in samples if s.speed_kmh is not None], [])
        # Pixel-space information is unaffected and still usable.
        self.assertIsNotNone(bounce.x_px)
        self.assertIsNotNone(bounce.y_px)
        self.assertIsNotNone(bounce.x_norm)
        self.assertIsNotNone(bounce.original_frame)

    def test_bounce_is_either_scored_or_labelled_as_fallback(self):
        traj = straight_trajectory(14)
        _, bounce, _, _, _ = analyse_delivery(traj, HOMOGRAPHY, FPS, W, H)
        self.assertIn(bounce.method, ("score", "fallback_max_y"))
        if bounce.method == "fallback_max_y":
            self.assertFalse(bounce.detected)
        self.assertIsNotNone(bounce.original_frame)
        # Normalised coordinates are the ones the frontend plots.
        self.assertIsNotNone(bounce.x_norm)
        self.assertTrue(0.0 <= bounce.x_norm <= 1.0)
        self.assertTrue(0.0 <= bounce.y_norm <= 1.0)

    def test_bounce_point_is_a_real_tracked_frame(self):
        traj = straight_trajectory(14)
        _, bounce, _, _, _ = analyse_delivery(traj, HOMOGRAPHY, FPS, W, H)
        real_frames = {p.original_frame for p in traj.points}
        self.assertIn(bounce.original_frame, real_frames)
        self.assertIn((bounce.x_px, bounce.y_px),
                      {(p.x, p.y) for p in traj.points})

    def test_speed_samples_reference_real_tracked_points(self):
        traj = straight_trajectory(12)
        _, _, _, samples, _ = analyse_delivery(traj, HOMOGRAPHY, FPS, W, H)
        tracked = {(p.original_frame, p.x, p.y) for p in traj.points}
        for s in samples:
            self.assertIn((s.original_frame, s.x, s.y), tracked)


if __name__ == "__main__":
    unittest.main()