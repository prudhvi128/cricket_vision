"""
test_pitch_calibration.py - raw keypoints to a calibration that may be trusted.

Unit tests over `app.pitch.calibration`. What they pin down:

  * four usable keypoints buy corners AND an image->unit-square homography,
    and `image_to_pitch` round-trips the corners it was built from;
  * three usable keypoints buy a calibration WITHOUT a rectangle: no corners,
    no homography, `image_to_pitch` -> None (never a number from a quad the
    model never reported);
  * every refusal is a named fact - too few keypoints, a keypoint outside the
    frame, a hull too small to be a pitch, an unreadable frame size;
  * confidences are only a reason to reject when the response carried some;
  * the hull test is a SHARE OF THE FRAME, not a raw area;
  * the smoother averages what it has and forgets what disappeared.

These feed `pitch/service.py`, which never lets an unvalidated detection reach
a client, so a refusal here is a feature: it is the honest answer to "there is
no readable pitch in this picture".
"""

from __future__ import annotations

import unittest

import numpy as np

from app.core.config import PitchConfig
from app.pitch.calibration import (
    PitchCalibration,
    PitchSmoother,
    build_calibration,
    build_homography,
    derive_corners,
)
from app.pitch.detector import Keypoint, PitchDetection

SIZE = (640, 360)  # (width, height)
W, H = SIZE

# A rectangle covering 64% of the frame: comfortably a pitch.
QUAD = {
    "top_left": (64.0, 36.0),
    "top_right": (576.0, 36.0),
    "bottom_right": (576.0, 324.0),
    "bottom_left": (64.0, 324.0),
}
# Three points of the same rectangle plus one inside it - still a hull.
TRIANGLE = [(64.0, 36.0), (576.0, 36.0), (320.0, 324.0)]


def detection(
    points,
    *,
    frame_number: int = 12,
    confidence=0.9,
    image_size=SIZE,
) -> PitchDetection:
    """A parsed model response, as `detector.parse_pitch_response` produces."""
    if isinstance(points, dict):
        kps = [Keypoint(name, xy[0], xy[1], 0.9) for name, xy in points.items()]
    else:
        kps = [
            Keypoint(f"point_{i}", xy[0], xy[1], 0.9) for i, xy in enumerate(points, 1)
        ]
    return PitchDetection(
        keypoints=kps,
        frame_number=frame_number,
        image_size=image_size,
        confidence=confidence,
    )


def build(points, *, cfg=None, image_size=SIZE, **kwargs):
    return build_calibration(
        detection(points, **kwargs), cfg=cfg or PitchConfig(), image_size=image_size
    )


# ── Four keypoints: a real calibration ───────────────────────────────────────
class TestFourKeypointCalibration(unittest.TestCase):
    def test_corners_are_derived_and_ordered(self):
        cal, reason = build(list(QUAD.values()))
        self.assertEqual(reason, "")
        self.assertIsInstance(cal, PitchCalibration)
        self.assertEqual(set(cal.corners), set(QUAD))
        for name, expected in QUAD.items():
            self.assertEqual(cal.corners[name], expected, name)

    def test_homography_maps_the_corners_to_the_unit_square(self):
        cal, _ = build(list(QUAD.values()))
        self.assertIsNotNone(cal.homography)
        self.assertEqual(cal.homography.shape, (3, 3))
        for name, (x, y) in QUAD.items():
            u, v = cal.image_to_pitch(x, y)
            self.assertAlmostEqual(u, 0.0 if "left" in name else 1.0, places=3, msg=name)
            self.assertAlmostEqual(v, 0.0 if "top" in name else 1.0, places=3, msg=name)

    def test_center_and_inside_test(self):
        cal, _ = build(list(QUAD.values()))
        self.assertEqual(cal.center, (320.0, 180.0))
        self.assertAlmostEqual(cal.image_to_pitch(320.0, 180.0)[0], 0.5, places=3)
        self.assertAlmostEqual(cal.image_to_pitch(320.0, 180.0)[1], 0.5, places=3)
        self.assertTrue(cal.inside_pitch(320.0, 180.0))
        self.assertFalse(cal.inside_pitch(0.0, 0.0))  # outside the quad
        self.assertFalse(cal.inside_pitch(64.0 - 40.0, 36.0 - 40.0))

    def test_width_and_length_are_pixel_spans(self):
        cal, _ = build(list(QUAD.values()))
        self.assertAlmostEqual(cal.width, 512.0)
        self.assertAlmostEqual(cal.length, 288.0)

    def test_hull_area_fraction_is_a_share_of_the_frame(self):
        cal, _ = build(list(QUAD.values()))
        self.assertAlmostEqual(cal.hull_area_fraction, (512 * 288) / (W * H), places=6)

    def test_interior_keypoints_do_not_become_corners(self):
        points = list(QUAD.values()) + [(320.0, 180.0), (200.0, 100.0)]
        cal, reason = build(points)
        self.assertEqual(reason, "")
        self.assertEqual(set(cal.corners), set(QUAD))

    def test_as_dict_is_complete_and_serialisable(self):
        cal, _ = build(list(QUAD.values()))
        d = cal.as_dict()
        self.assertEqual(
            set(d),
            {
                "frame_number",
                "image_size",
                "keypoints",
                "corners",
                "center",
                "confidence",
                "width_px",
                "length_px",
                "homography",
            },
        )
        self.assertEqual(d["frame_number"], 12)
        self.assertEqual(d["image_size"], [W, H])
        self.assertEqual(d["width_px"], 512.0)
        self.assertEqual(d["length_px"], 288.0)
        self.assertEqual(len(d["homography"]), 3)
        self.assertTrue(all(len(row) == 3 for row in d["homography"]))
        # JSON-safe: floats, not numpy scalars.
        json_ok = True
        for row in d["homography"]:
            for value in row:
                json_ok = json_ok and isinstance(value, float)
        self.assertTrue(json_ok)

    def test_confidence_falls_back_to_the_keypoint_mean(self):
        kps = [
            Keypoint(f"point_{i}", xy[0], xy[1], conf)
            for i, (xy, conf) in enumerate(
                zip(QUAD.values(), [0.6, 0.8, 0.7, 0.9]), start=1
            )
        ]
        cal, reason = build_calibration(
            PitchDetection(kps, 1, SIZE, confidence=None),
            cfg=PitchConfig(),
            image_size=SIZE,
        )
        self.assertEqual(reason, "")
        self.assertAlmostEqual(cal.confidence, 0.75, places=4)


# ── Three keypoints: a calibration WITHOUT a rectangle ───────────────────────
class TestThreeKeypointCalibration(unittest.TestCase):
    def test_no_corners_and_no_homography_are_invented(self):
        cal, reason = build(TRIANGLE)
        self.assertEqual(reason, "")
        self.assertIsInstance(cal, PitchCalibration)
        self.assertEqual(cal.corners, {})
        self.assertIsNone(cal.homography)

    def test_positions_are_unavailable_rather_than_guessed(self):
        cal, _ = build(TRIANGLE)
        self.assertIsNone(cal.image_to_pitch(320.0, 180.0))
        self.assertFalse(cal.inside_pitch(320.0, 180.0))

    def test_center_falls_back_to_the_keypoints(self):
        cal, _ = build(TRIANGLE)
        expected = tuple(
            sum(p[i] for p in TRIANGLE) / 3.0 for i in (0, 1)
        )
        self.assertAlmostEqual(cal.center[0], expected[0], places=6)
        self.assertAlmostEqual(cal.center[1], expected[1], places=6)

    def test_spans_are_zero_without_corners(self):
        cal, _ = build(TRIANGLE)
        self.assertEqual(cal.width, 0.0)
        self.assertEqual(cal.length, 0.0)

    def test_as_dict_stays_serialisable(self):
        cal, _ = build(TRIANGLE)
        d = cal.as_dict()
        self.assertEqual(d["corners"], {})
        self.assertIsNone(d["homography"])
        self.assertIsNone(d["width_px"])
        self.assertEqual(d["center"], [round(cal.center[0], 2), round(cal.center[1], 2)])
        self.assertEqual(len(d["keypoints"]), 3)


# ── Refusals ─────────────────────────────────────────────────────────────────
class TestRefusals(unittest.TestCase):
    def test_unknown_frame_size(self):
        cal, reason = build(list(QUAD.values()), image_size=(0, 0))
        self.assertIsNone(cal)
        self.assertEqual(reason, "unknown_frame_size")

    def test_too_few_keypoints(self):
        cal, reason = build([(64.0, 36.0), (576.0, 36.0)])
        self.assertIsNone(cal)
        self.assertEqual(reason, "only_2_usable_keypoints")

    def test_low_confidence_keypoints_are_refused(self):
        kps = [
            Keypoint(f"p{i}", xy[0], xy[1], 0.2)
            for i, xy in enumerate(QUAD.values(), 1)
        ]
        cal, reason = build_calibration(
            PitchDetection(kps, 1, SIZE, confidence=0.2),
            cfg=PitchConfig(),
            image_size=SIZE,
        )
        self.assertIsNone(cal)
        self.assertEqual(reason, "only_0_usable_keypoints")

    def test_unscored_keypoints_are_not_treated_as_zero_confidence(self):
        kps = [Keypoint(f"p{i}", xy[0], xy[1], None) for i, xy in enumerate(QUAD.values(), 1)]
        cal, reason = build_calibration(
            PitchDetection(kps, 1, SIZE, confidence=None),
            cfg=PitchConfig(),
            image_size=SIZE,
        )
        self.assertEqual(reason, "")
        self.assertIsNone(cal.confidence)

    def test_mixed_scores_keep_the_unscored_points(self):
        kps = [
            Keypoint("a", 64.0, 36.0, None),
            Keypoint("b", 576.0, 36.0, 0.9),
            Keypoint("c", 576.0, 324.0, 0.1),  # below the floor: dropped
            Keypoint("d", 64.0, 324.0, None),
        ]
        cal, reason = build_calibration(
            PitchDetection(kps, 1, SIZE, confidence=0.9),
            cfg=PitchConfig(),
            image_size=SIZE,
        )
        # Three keypoints survive - a calibration, but no rectangle.
        self.assertEqual(reason, "")
        self.assertEqual(set(cal.keypoints), {"a", "b", "d"})
        self.assertEqual(cal.corners, {})

    def test_keypoint_far_outside_the_frame_is_dropped(self):
        kps = [
            Keypoint("a", 64.0, 36.0, None),
            Keypoint("b", W + 0.05 * W, 36.0, None),  # 5% past the edge
            Keypoint("c", 576.0, 324.0, None),
            Keypoint("d", 64.0, 324.0, None),
        ]
        cal, reason = build_calibration(
            PitchDetection(kps, 1, SIZE, confidence=0.9),
            cfg=PitchConfig(),
            image_size=SIZE,
        )
        # Only three survive: still a calibration, but no rectangle.
        self.assertEqual(reason, "")
        self.assertEqual(set(cal.keypoints), {"a", "c", "d"})
        self.assertEqual(cal.corners, {})

    def test_keypoint_a_little_past_the_edge_is_kept(self):
        """Models land a corner a few pixels out; 2% of slack is allowed."""
        kps = [Keypoint(name, xy[0], xy[1], None) for name, xy in QUAD.items()]
        kps[1] = Keypoint("top_right", W + 0.01 * W, 36.0, None)  # 1% past
        cal, reason = build_calibration(
            PitchDetection(kps, 1, SIZE, confidence=0.9),
            cfg=PitchConfig(),
            image_size=SIZE,
        )
        self.assertEqual(reason, "")
        self.assertIn("top_right", cal.corners)

    def test_a_hull_smaller_than_the_configured_share_is_refused(self):
        """A postage-stamp cluster of keypoints is not a pitch."""
        tiny = [(100.0, 100.0), (110.0, 100.0), (110.0, 110.0), (100.0, 110.0)]
        cal, reason = build(tiny)
        self.assertIsNone(cal)
        self.assertEqual(reason, "keypoint_hull_too_small")

    def test_a_hull_that_is_big_enough_is_accepted(self):
        cal, reason = build(list(QUAD.values()))
        self.assertEqual(reason, "")
        self.assertIsNotNone(cal)


# ── Corner derivation ────────────────────────────────────────────────────────
class TestDeriveCorners(unittest.TestCase):
    def test_input_order_does_not_matter(self):
        shuffled = [
            QUAD["bottom_right"],
            QUAD["top_left"],
            QUAD["bottom_left"],
            QUAD["top_right"],
        ]
        self.assertEqual(derive_corners(shuffled), QUAD)

    def test_three_points_are_not_a_rectangle(self):
        self.assertIsNone(derive_corners(TRIANGLE))

    def test_collinear_points_are_rejected(self):
        line = [(0.0, 0.0), (100.0, 100.0), (200.0, 200.0), (300.0, 300.0)]
        self.assertIsNone(derive_corners(line))

    def test_an_interior_point_never_becomes_a_corner(self):
        with_interior = list(QUAD.values()) + [(320.0, 180.0)]
        self.assertEqual(derive_corners(with_interior), QUAD)

    def test_duplicate_extremes_are_rejected(self):
        # Three distinct points where two coincide: the hull cannot have four.
        degenerate = [(64.0, 36.0), (64.0, 36.0), (576.0, 36.0), (64.0, 324.0)]
        self.assertIsNone(derive_corners(degenerate))


# ── Homography ───────────────────────────────────────────────────────────────
class TestBuildHomography(unittest.TestCase):
    def test_a_skewed_quad_still_round_trips(self):
        corners = {
            "top_left": (50.0, 40.0),
            "top_right": (600.0, 20.0),
            "bottom_right": (620.0, 340.0),
            "bottom_left": (30.0, 330.0),
        }
        matrix = build_homography(corners, SIZE)
        self.assertIsNotNone(matrix)
        unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        for expected, (x, y) in zip(unit, corners.values()):
            vec = np.array([x, y, 1.0])
            mapped = matrix @ vec
            got = mapped[:2] / mapped[2]
            self.assertTrue(np.allclose(got, expected, atol=1e-3))

    def test_a_missing_corner_is_refused(self):
        incomplete = {k: v for k, v in QUAD.items() if k != "bottom_left"}
        self.assertIsNone(build_homography(incomplete, SIZE))

    def test_a_degenerate_quad_is_refused(self):
        collinear = {
            "top_left": (0.0, 0.0),
            "top_right": (100.0, 0.0),
            "bottom_right": (200.0, 0.0),
            "bottom_left": (300.0, 0.0),
        }
        self.assertIsNone(build_homography(collinear, SIZE))

    def test_non_finite_coordinates_are_refused(self):
        broken = dict(QUAD)
        broken["top_left"] = (float("nan"), 36.0)
        self.assertIsNone(build_homography(broken, SIZE))


# ── Smoothing ────────────────────────────────────────────────────────────────
class TestPitchSmoother(unittest.TestCase):
    def test_the_first_set_seeds_the_average(self):
        smoother = PitchSmoother(0.4)
        points, confidence = smoother.apply({"a": (10.0, 10.0)}, 0.5)
        self.assertEqual(points, {"a": (10.0, 10.0)})
        self.assertEqual(confidence, 0.5)

    def test_the_second_set_is_blended(self):
        smoother = PitchSmoother(0.4)
        smoother.apply({"a": (10.0, 10.0)}, 0.5)
        points, confidence = smoother.apply({"a": (20.0, 20.0)}, 0.7)
        self.assertAlmostEqual(points["a"][0], 14.0, places=6)
        self.assertAlmostEqual(points["a"][1], 14.0, places=6)
        self.assertAlmostEqual(confidence, 0.58, places=6)

    def test_disappeared_keypoints_are_forgotten(self):
        smoother = PitchSmoother(0.4)
        smoother.apply({"a": (10.0, 10.0)}, 0.5)
        points, _ = smoother.apply({"b": (5.0, 5.0)}, 0.6)
        self.assertEqual(set(points), {"b"})
        self.assertEqual(points["b"], (5.0, 5.0))

    def test_a_new_keypoint_is_taken_as_is(self):
        smoother = PitchSmoother(0.4)
        smoother.apply({"a": (10.0, 10.0)}, 0.5)
        points, _ = smoother.apply({"a": (10.0, 10.0), "b": (90.0, 90.0)}, 0.5)
        self.assertEqual(points["b"], (90.0, 90.0))

    def test_alpha_one_passes_through(self):
        smoother = PitchSmoother(1.0)
        smoother.apply({"a": (0.0, 0.0)}, 0.1)
        points, confidence = smoother.apply({"a": (50.0, 60.0)}, 0.9)
        self.assertEqual(points["a"], (50.0, 60.0))
        self.assertEqual(confidence, 0.9)

    def test_alpha_zero_freezes_the_average(self):
        smoother = PitchSmoother(0.0)
        smoother.apply({"a": (10.0, 10.0)}, 0.5)
        points, confidence = smoother.apply({"a": (999.0, 999.0)}, 0.9)
        self.assertEqual(points["a"], (10.0, 10.0))
        self.assertEqual(confidence, 0.5)

    def test_alpha_is_clamped(self):
        self.assertEqual(PitchSmoother(5.0).alpha, 1.0)
        self.assertEqual(PitchSmoother(-2.0).alpha, 0.0)

    def test_a_none_confidence_clears_the_average(self):
        smoother = PitchSmoother(0.4)
        smoother.apply({"a": (10.0, 10.0)}, 0.5)
        _, confidence = smoother.apply({"a": (10.0, 10.0)}, None)
        self.assertIsNone(confidence)

    def test_reset_starts_over(self):
        smoother = PitchSmoother(0.4)
        smoother.apply({"a": (10.0, 10.0)}, 0.5)
        smoother.reset()
        points, confidence = smoother.apply({"z": (1.0, 1.0)}, 0.3)
        self.assertEqual(points, {"z": (1.0, 1.0)})
        self.assertEqual(confidence, 0.3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
