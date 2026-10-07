"""
test_pitch_geometry.py - a detected quad becomes metres on the ground.

Unit tests over `app.analytics.pitch_geometry`, the module that joins the two
calibrations this project already had (WHERE is the pitch in this frame /
WHERE is it in metres) and refuses to do so whenever the evidence is thin.

What they pin down:

  * `flight_to_bounce` cuts the trajectory at its deepest image point, so the
    post-bounce rise cannot vote the batter to the bowler's end;
  * every refusal is a NAMED fact rather than a number: no calibration, no
    quad, too few detections, travel too short or too even to say which
    direction is down the pitch, a quad whose aspect contradicts the travel,
    a transform that does not keep the quad on the pitch;
  * the happy path is checked end to end: the four image corners land inside
    0..2.64 m by 0..20.12 m, the bounce lands a plausible distance from the
    batting crease, and `classify_length_m` on that distance gives the label
    the API publishes;
  * `map_homography` normalises the same transform so the batting end is at
    y = 1 whatever the camera did, which is what a top-down pitch map needs;
  * the geometry's own convention (y = 0 non-batting end, y = PITCH_LENGTH_M
    batting end) is the same one `analytics/calibration.py` uses for operator
    corners, so one length formula serves both sources.
"""

from __future__ import annotations

import unittest

import numpy as np

from app.analytics.pitch_geometry import (
    AXIS_U,
    AXIS_V,
    SOURCE_PITCH_KEYPOINTS,
    as_metadata,
    build_pitch_geometry,
    flight_to_bounce,
)
from app.analytics.trajectory import classify_length_m
from app.core.config import PitchConfig
from app.core.constants import CREASE_WIDTH_M, PITCH_LENGTH_M
from app.pitch.calibration import PitchCalibration, build_calibration
from app.pitch.detector import Keypoint, PitchDetection

SIZE = (640, 360)
W, H = SIZE

# An end-on view: the quad runs far more down the frame than across it, which
# is what a 22-yard pitch looks like from behind the stumps, and slightly wider
# at the near edge the way perspective makes it.
TALL_QUAD = {
    "top_left": (250.0, 20.0),
    "top_right": (390.0, 20.0),
    "bottom_right": (430.0, 340.0),
    "bottom_left": (210.0, 340.0),
}

# A side-on view: the pitch lies ACROSS the frame, so its long axis is u.
WIDE_QUAD = {
    "top_left": (64.0, 120.0),
    "top_right": (576.0, 120.0),
    "bottom_right": (576.0, 240.0),
    "bottom_left": (64.0, 240.0),
}

# Three points of a rectangle: a calibration the validator accepts WITHOUT
# describing a rectangle, because three keypoints cannot define one.
TRIANGLE = [(64.0, 36.0), (576.0, 36.0), (320.0, 324.0)]

# Release high in the frame, bouncing deep in it - the shape of a delivery seen
# from behind the batter. On the real footage of this project 29 of the 34
# usable deliveries travel mostly downward like this.
FALLING = [(320.0, 60.0), (322.0, 150.0), (325.0, 240.0), (330.0, 330.0)]

# The same delivery seen side on: the ball crosses the frame from right to
# left with barely any drop, so the along-pitch axis is U and the batter is
# whichever end it was heading for.
CROSSING = [(500.0, 140.0), (400.0, 147.0), (300.0, 153.0), (200.0, 160.0)]


def calibration(quad, *, frame_number: int = 42, image_size=SIZE) -> PitchCalibration:
    kps = [Keypoint(name, xy[0], xy[1], 0.9) for name, xy in quad.items()]
    detection = PitchDetection(
        keypoints=kps, frame_number=frame_number, image_size=image_size, confidence=0.9
    )
    cal, reason = build_calibration(
        detection, cfg=PitchConfig(), image_size=image_size
    )
    if reason:
        raise AssertionError(f"fixture calibration refused: {reason}")
    return cal


def triangles():
    """Three named keypoints are not a rectangle - but they do build a calibration."""
    detection = PitchDetection(
        keypoints=[Keypoint(f"p{i}", x, y, 0.9) for i, (x, y) in enumerate(TRIANGLE, 1)],
        frame_number=1,
        image_size=SIZE,
        confidence=0.9,
    )
    cal, _ = build_calibration(detection, cfg=PitchConfig(), image_size=SIZE)
    return cal


def apply(matrix, x: float, y: float) -> tuple[float, float]:
    """Homogeneous transform, as a caller would do it with the matrix itself."""
    vec = np.array([x, y, 1.0])
    mapped = matrix @ vec
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def length_of(matrix, x: float, y: float) -> str | None:
    """The published label: metres from the batting crease, straight through."""
    _, y_m = apply(matrix, x, y)
    return classify_length_m(PITCH_LENGTH_M - y_m)


# ── flight_to_bounce ─────────────────────────────────────────────────────────
class TestFlightToBounce(unittest.TestCase):
    def test_an_empty_flight_stays_empty(self):
        self.assertEqual(flight_to_bounce([]), [])

    def test_the_flight_stops_at_its_deepest_point(self):
        flight = [(10.0, 20.0), (20.0, 80.0), (30.0, 60.0), (40.0, 40.0)]
        self.assertEqual(flight_to_bounce(flight), [(10.0, 20.0), (20.0, 80.0)])

    def test_the_bounce_point_itself_is_kept(self):
        flight = [(10.0, 20.0), (20.0, 80.0)]
        self.assertEqual(flight_to_bounce(flight), flight)

    def test_a_trajectory_that_never_descends_keeps_only_its_first_point(self):
        """The honest failure mode of this rule, recorded here on purpose.

        A ball whose detections only rise in the image (a camera behind the
        bowler, say) has its deepest point first, so the cut leaves one point
        and the caller refuses instead of voting on a direction it cannot see.
        """
        flight = [(10.0, 80.0), (20.0, 60.0), (30.0, 40.0)]
        self.assertEqual(flight_to_bounce(flight), [(10.0, 80.0)])


# ── Refusals: a reason, never a guess ────────────────────────────────────────
class TestRefusals(unittest.TestCase):
    def test_no_calibration_at_all(self):
        geometry, reason = build_pitch_geometry(None, FALLING)
        self.assertIsNone(geometry)
        self.assertEqual(reason, "no_pitch_calibration")

    def test_a_calibration_without_a_quad_is_refused(self):
        geometry, reason = build_pitch_geometry(triangles(), FALLING)
        self.assertIsNone(geometry)
        self.assertEqual(reason, "pitch_quad_unavailable")

    def test_too_few_detected_points(self):
        geometry, reason = build_pitch_geometry(calibration(TALL_QUAD), FALLING[:2])
        self.assertIsNone(geometry)
        self.assertEqual(reason, "too_few_detected_points")

    def test_travel_too_even_to_name_a_direction(self):
        """Diagonal travel: neither axis is 1.2x the other, so nothing is picked."""
        diagonal = [(300.0, 80.0), (330.0, 130.0), (360.0, 180.0), (390.0, 230.0)]
        geometry, reason = build_pitch_geometry(calibration(TALL_QUAD), diagonal)
        self.assertIsNone(geometry)
        self.assertEqual(reason, "ball_travel_axis_ambiguous")

    def test_a_quad_that_contradicts_the_travel_is_refused(self):
        """Vertical travel across a wide, side-on quad: the axes disagree."""
        falling = [(320.0, 130.0), (321.0, 160.0), (322.0, 190.0), (323.0, 220.0)]
        geometry, reason = build_pitch_geometry(calibration(WIDE_QUAD), falling)
        self.assertIsNone(geometry)
        self.assertEqual(reason, "pitch_quad_aspect_disagrees_with_travel")

    def test_travel_too_short_to_say_which_end_is_the_batter(self):
        nudge = [(320.0, 60.0), (321.0, 66.0), (322.0, 72.0), (323.0, 80.0)]
        geometry, reason = build_pitch_geometry(calibration(TALL_QUAD), nudge)
        self.assertIsNone(geometry)
        self.assertEqual(reason, "pitch_orientation_ambiguous")


# ── The happy path, checked end to end ───────────────────────────────────────
class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.cal = calibration(TALL_QUAD, frame_number=7)
        self.geometry, reason = build_pitch_geometry(self.cal, flight_to_bounce(FALLING))
        self.assertEqual(reason, "")
        assert self.geometry is not None
        self.geom = self.geometry

    def test_the_axis_is_the_one_the_quad_is_elongated_along(self):
        self.assertEqual(self.geom.axis, AXIS_V)
        self.assertGreaterEqual(self.geom.travel_delta, 0.12)

    def test_every_corner_of_the_quad_stays_on_the_measured_pitch(self):
        for name, (x, y) in TALL_QUAD.items():
            x_m, y_m = apply(self.geom.homography, x, y)
            self.assertGreaterEqual(x_m, -1e-3, name)
            self.assertLessEqual(x_m, CREASE_WIDTH_M + 1e-3, name)
            self.assertGreaterEqual(y_m, -1e-3, name)
            self.assertLessEqual(y_m, PITCH_LENGTH_M + 1e-3, name)

    def test_the_bounce_lands_a_plausible_distance_from_the_crease(self):
        x_m, y_m = apply(self.geom.homography, *FALLING[-1])
        self.assertGreater(x_m, 0.0)
        self.assertLess(x_m, CREASE_WIDTH_M)
        distance = PITCH_LENGTH_M - y_m
        self.assertGreater(distance, 0.0)
        self.assertLess(distance, PITCH_LENGTH_M)
        self.assertEqual(classify_length_m(distance), "Yorker")

    def test_the_map_normalises_to_a_pitch_with_the_batting_end_at_one(self):
        self.assertEqual(self.geom.axis, AXIS_V)
        self.assertEqual(self.geom.batting_end, 1.0)
        for name, (x, y) in TALL_QUAD.items():
            _, map_y = apply(self.geom.map_homography, x, y)
            self.assertGreaterEqual(map_y, -1e-3, name)
            self.assertLessEqual(map_y, 1.0 + 1e-3, name)
        _, bounce_map_y = apply(self.geom.map_homography, *FALLING[-1])
        self.assertAlmostEqual(bounce_map_y, 1.0, delta=0.1)

    def test_the_frame_it_was_measured_on_is_recorded(self):
        self.assertEqual(self.geom.calibration_frame, 7)
        self.assertEqual(self.geom.source, SOURCE_PITCH_KEYPOINTS)

    def test_the_batting_end_property_only_answers_for_a_lengthwise_axis(self):
        self.assertEqual(self.geom.batting_end_v, 1.0)


# ── The other camera: travel along the frame's width ─────────────────────────
class TestCrossingTravel(unittest.TestCase):
    def test_the_batter_is_the_end_the_ball_was_heading_for(self):
        cal = calibration(WIDE_QUAD, frame_number=3)
        geometry, reason = build_pitch_geometry(cal, flight_to_bounce(CROSSING))
        self.assertEqual(reason, "")
        assert geometry is not None
        self.assertEqual(geometry.axis, AXIS_U)
        self.assertEqual(geometry.batting_end, 0.0)
        self.assertLess(geometry.travel_delta, 0.0)

        # The bounce is still measured from the SAME crease, in metres, and the
        # label it produces is the one the API publishes.
        _, y_m = apply(geometry.homography, *CROSSING[-1])
        distance = PITCH_LENGTH_M - y_m
        self.assertEqual(classify_length_m(distance), "Good Length")
        # ...and on the normalised map the batting end is still y = 1.
        _, map_y = apply(geometry.map_homography, *CROSSING[-1])
        self.assertAlmostEqual(map_y, y_m / PITCH_LENGTH_M, places=6)

    def test_the_lengthwise_axis_follows_the_quad(self):
        cal = calibration(WIDE_QUAD)
        geometry, _ = build_pitch_geometry(cal, CROSSING)
        assert geometry is not None
        self.assertEqual(geometry.axis, AXIS_U)
        self.assertIsNone(
            geometry.batting_end_v,
            "a batter at an END of a widthwise quad has no single v to report",
        )


# ── What gets written into the analysis record ───────────────────────────────
class TestMetadata(unittest.TestCase):
    def test_no_geometry_says_so_in_every_field(self):
        self.assertEqual(
            as_metadata(None),
            {
                "source": None,
                "axis": None,
                "batting_end": None,
                "travel_delta": None,
                "calibration_frame": None,
            },
        )

    def test_geometry_records_the_evidence_behind_its_choice(self):
        geometry, reason = build_pitch_geometry(
            calibration(TALL_QUAD, frame_number=7), FALLING
        )
        self.assertEqual(reason, "")
        meta = as_metadata(geometry)
        self.assertEqual(meta["source"], SOURCE_PITCH_KEYPOINTS)
        self.assertEqual(meta["axis"], AXIS_V)
        self.assertEqual(meta["batting_end"], 1.0)
        self.assertEqual(meta["calibration_frame"], 7)
        self.assertIsInstance(meta["travel_delta"], float)


# ── Length in metres ─────────────────────────────────────────────────────────
class TestLengthInMetres(unittest.TestCase):
    """The published length label, from a distance measured off THIS video."""

    def test_the_coaching_thresholds(self):
        cases = [
            (0.0, "Yorker"),
            (1.49, "Yorker"),
            (1.5, "Full"),
            (3.99, "Full"),
            (4.0, "Good Length"),
            (6.99, "Good Length"),
            (7.0, "Short"),
            (9.99, "Short"),
            (10.0, "Very Short"),
            (PITCH_LENGTH_M, "Very Short"),
        ]
        for metres, label in cases:
            with self.subTest(metres=metres):
                self.assertEqual(classify_length_m(metres), label)

    def test_something_off_the_pitch_has_no_length(self):
        for metres in (None, -0.01, PITCH_LENGTH_M + 0.01, float("nan")):
            with self.subTest(metres=metres):
                self.assertIsNone(classify_length_m(metres))


if __name__ == "__main__":
    unittest.main()
