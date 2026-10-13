"""
test_pitch_mask_geometry.py — pure mask→quad derivation.

No model, no I/O: polygons in, four corners (or a refusal) out. The corners
must obey the SAME extremes convention pitch/calibration.py `derive_corners`
uses, because the EMA smoother and the homography read them by name.
"""

from __future__ import annotations

import unittest

from app.pitch.mask_geometry import (
    corners_as_named,
    mask_centroid,
    mask_in_bounds,
    mask_polygon_area_fraction,
    quad_aspect_ratio,
    quad_from_mask,
    validate_mask,
)

W, H = 1280, 720

# A trapezoidal pitch seen from behind the bowler's arm: narrow at the top
# (far end), wide at the bottom (near end), centred in the frame. Vertices
# in clockwise order, as a mask boundary would be.
TRAPEZOID = [
    (540.0, 120.0),
    (740.0, 120.0),
    (900.0, 640.0),
    (380.0, 640.0),
]

# The same trapezoid with a noisy boundary: extra vertices along each edge
# and a dent on one side, as a real mask's polygon has.
NOISY_TRAPEZOID = [
    (540.0, 120.0), (590.0, 120.0), (640.0, 120.0), (690.0, 120.0), (740.0, 120.0),
    (780.0, 250.0), (820.0, 400.0), (860.0, 520.0), (900.0, 640.0),
    (760.0, 640.0), (700.0, 600.0), (640.0, 640.0),  # the dent
    (500.0, 640.0), (380.0, 640.0),
    (420.0, 520.0), (470.0, 400.0), (505.0, 250.0),
]

TRIANGLE = [(400.0, 100.0), (900.0, 100.0), (640.0, 620.0)]

# A crowd-shaped blob: covers 38% of the frame (the measured false positive).
CROWD = [
    (0.0, 0.0), (1280.0, 0.0), (1280.0, 720.0), (0.0, 720.0),
    (0.0, 400.0), (200.0, 380.0), (400.0, 420.0), (640.0, 390.0),
    (900.0, 430.0), (1100.0, 400.0),
]


class TestQuadFromMask(unittest.TestCase):
    def test_a_clean_quad_mask_yields_its_own_four_corners(self):
        quad, method = quad_from_mask(TRAPEZOID)
        self.assertIsNotNone(quad)
        self.assertEqual(method, "direct_hull")
        self.assertEqual(len(quad), 4)
        # The four extremes of the trapezoid are its four corners.
        named = corners_as_named(quad)
        self.assertEqual(set(named), {
            "top_left", "top_right", "bottom_right", "bottom_left",
        })
        self.assertEqual(named["top_left"], (540.0, 120.0))
        self.assertEqual(named["top_right"], (740.0, 120.0))
        self.assertEqual(named["bottom_right"], (900.0, 640.0))
        self.assertEqual(named["bottom_left"], (380.0, 640.0))

    def test_a_noisy_boundary_still_yields_four_corners(self):
        quad, method = quad_from_mask(NOISY_TRAPEZOID)
        self.assertIsNotNone(quad)
        self.assertEqual(len(quad), 4)
        # The dent is an interior feature of the boundary: the simplified
        # quad still lands on the outer extremes.
        named = corners_as_named(quad)
        self.assertAlmostEqual(named["top_left"][1], 120.0, delta=15.0)
        self.assertAlmostEqual(named["bottom_right"][1], 640.0, delta=15.0)
        self.assertAlmostEqual(named["bottom_left"][0], 380.0, delta=15.0)
        self.assertAlmostEqual(named["top_right"][0], 740.0, delta=15.0)

    def test_a_triangle_is_refused(self):
        quad, method = quad_from_mask(TRIANGLE)
        self.assertIsNone(quad)
        self.assertIn(method, {"degenerate_hull", "quad_unavailable", "too_few_vertices"})

    def test_three_vertices_are_refused(self):
        quad, method = quad_from_mask(TRAPEZOID[:3])
        self.assertIsNone(quad)
        self.assertEqual(method, "too_few_vertices")

    def test_nan_vertices_are_dropped_not_repaired(self):
        dirty = TRAPEZOID + [(float("nan"), 100.0), (1.0, float("inf"))]
        quad, _ = quad_from_mask(dirty)
        self.assertIsNotNone(quad)
        # The clean four still produce the clean quad.
        named = corners_as_named(quad)
        self.assertEqual(named["top_left"], (540.0, 120.0))

    def test_all_nan_is_refused(self):
        quad, method = quad_from_mask([(float("nan"), float("nan"))] * 6)
        self.assertIsNone(quad)
        self.assertEqual(method, "too_few_vertices")

    def test_corners_are_ordered_for_a_perspective_transform(self):
        quad, _ = quad_from_mask(TRAPEZOID)
        # top_left → top_right → bottom_right → bottom_left winding.
        tl, tr, br, bl = quad
        self.assertLess(tl[1], bl[1])   # top above bottom, left side
        self.assertLess(tr[1], br[1])   # top above bottom, right side
        self.assertLess(tl[0], tr[0])   # left before right, top
        self.assertLess(bl[0], br[0])   # left before right, bottom

    def test_a_degenerate_line_is_refused(self):
        line = [(100.0, 100.0), (200.0, 200.0), (300.0, 300.0), (400.0, 400.0)]
        quad, _ = quad_from_mask(line)
        self.assertIsNone(quad)


class TestValidateMask(unittest.TestCase):
    KW = dict(
        min_area_fraction=0.01,
        max_area_fraction=0.25,
        min_aspect_ratio=1.0,
        max_aspect_ratio=6.0,
    )

    def test_a_real_sized_pitch_mask_passes(self):
        ok, reason = validate_mask(TRAPEZOID, (W, H), **self.KW)
        self.assertTrue(ok, reason)

    def test_a_crowd_mask_is_refused_by_area(self):
        ok, reason = validate_mask(CROWD, (W, H), **self.KW)
        self.assertFalse(ok)
        self.assertEqual(reason, "mask_area_too_large")

    def test_a_postage_stamp_mask_is_refused_by_area(self):
        tiny = [(600.0, 350.0), (610.0, 350.0), (610.0, 360.0), (600.0, 360.0)]
        ok, reason = validate_mask(tiny, (W, H), **self.KW)
        self.assertFalse(ok)
        self.assertEqual(reason, "mask_area_too_small")

    def test_a_mask_far_outside_the_frame_is_refused(self):
        outside = [(x + 2000.0, y + 2000.0) for x, y in TRAPEZOID]
        ok, reason = validate_mask(outside, (W, H), **self.KW)
        self.assertFalse(ok)
        self.assertEqual(reason, "mask_outside_frame")

    def test_a_few_vertices_are_refused(self):
        ok, reason = validate_mask(TRAPEZOID[:2], (W, H), **self.KW)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("mask_polygon_too_small"))

    def test_an_absurdly_elongated_mask_is_refused(self):
        # 1080 x 100: clears the area floor (0.117 of the frame) so the
        # aspect ceiling is what refuses it.
        sliver = [(100.0, 300.0), (1180.0, 300.0), (1180.0, 400.0), (100.0, 400.0)]
        ok, reason = validate_mask(sliver, (W, H), **self.KW)
        self.assertFalse(ok)
        self.assertEqual(reason, "mask_aspect_too_elongated")

    def test_an_unknown_frame_size_is_refused(self):
        ok, reason = validate_mask(TRAPEZOID, (0, 0), **self.KW)
        self.assertFalse(ok)
        self.assertEqual(reason, "unknown_frame_size")


class TestAreaAndAspect(unittest.TestCase):
    def test_area_fraction_is_a_share_of_the_frame(self):
        fraction = mask_polygon_area_fraction(TRAPEZOID, (W, H))
        self.assertGreater(fraction, 0.05)
        self.assertLess(fraction, 0.25)

    def test_a_full_frame_mask_is_about_one(self):
        frame = [(0.0, 0.0), (float(W), 0.0), (float(W), float(H)), (0.0, float(H))]
        self.assertAlmostEqual(mask_polygon_area_fraction(frame, (W, H)), 1.0, places=3)

    def test_aspect_of_a_square_is_one(self):
        square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        self.assertAlmostEqual(quad_aspect_ratio(square), 1.0, places=5)

    def test_aspect_of_a_2_to_1_rect_is_two(self):
        rect = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]
        self.assertAlmostEqual(quad_aspect_ratio(rect), 2.0, places=5)

    def test_degenerate_aspect_is_infinite(self):
        line = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]
        self.assertEqual(quad_aspect_ratio(line), float("inf"))

    def test_centroid_is_the_mean_of_the_vertices(self):
        center = mask_centroid([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        self.assertEqual(center, (5.0, 5.0))

    def test_bounds_accepts_a_mask_bleeding_two_pixels(self):
        bleed = [(-5.0, -5.0), (W + 5.0, -5.0), (W + 5.0, H + 5.0), (-5.0, H + 5.0)]
        self.assertTrue(mask_in_bounds(bleed, (W, H)))

    def test_bounds_refuses_a_mask_halfway_to_the_next_frame(self):
        far = [(x + W, y) for x, y in TRAPEZOID]
        self.assertFalse(mask_in_bounds(far, (W, H)))


if __name__ == "__main__":
    unittest.main()
