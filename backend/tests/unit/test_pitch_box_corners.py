"""
test_pitch_box_corners.py — the box is a rectangle, not a pitch.

`app.pitch.box_corners` turns the optional local detector's BOX into four
APPROXIMATE corners. These tests pin down that conversion and, more
importantly, the refusal rules: which rectangles are still sane enough to
record as an estimate and which must be recorded as invalid with a named
reason. Every number below is a guess about where a pitch might be, never a
measurement of one — the assertions are about geometry, not accuracy.

Covered here:

  * the normal case: a box becomes P1 near-left, P2 near-right, P3 far-right,
    P4 far-left, with the far edge pulled in by `top_inset * width` so the
    trapezoid narrows away from the camera;
  * `top_inset=0` degenerates the trapezoid to a rectangle — still valid, but
    it says nothing about perspective;
  * a tiny box collapses (near width, far width or height at/below threshold)
    and is refused as "too small or flipped";
  * an inset so large the far edge crosses itself is refused the same way;
  * an inset with the wrong sign (far edge WIDER than the near edge) is
    refused as "far edge wider than near edge";
  * `corners_as_dict` rounds to 2 dp in the same style as the pitch block.

Nothing here touches ultralytics, the network, or the rest of the project —
this module deliberately has no project imports, which is what makes these
tests instant.
"""

from __future__ import annotations

import unittest

from app.pitch.box_corners import (
    CORNER_NAMES,
    box_to_corners,
    corners_as_dict,
    corners_valid,
)


class TestBoxToCorners(unittest.TestCase):
    """Rectangle in, perspective-shaped quad out."""

    def test_normal_box(self) -> None:
        # A 200x350 box sitting at (100, 50)..(300, 400) with the default
        # 8% inset: the near edge keeps its corners exactly, the far edge
        # moves 16 px (8% of 200) in from each side and drops to y1.
        c = box_to_corners(100, 50, 300, 400, top_inset=0.08)
        self.assertEqual(len(c), 4)
        p1, p2, p3, p4 = c
        self.assertEqual(p1, (100, 400))   # near-left
        self.assertEqual(p2, (300, 400))   # near-right
        self.assertEqual(p3, (284, 50))    # far-right, pulled in
        self.assertEqual(p4, (116, 50))    # far-left, pulled in
        # The near edge is exactly the box width; the far edge is narrower.
        self.assertAlmostEqual(p2[0] - p1[0], 200)
        self.assertAlmostEqual(p3[0] - p4[0], 168)
        # Far edge sits at the TOP of the box, near edge at the bottom.
        self.assertEqual(p4[1], 50)
        self.assertEqual(p1[1], 400)

    def test_near_edge_is_the_box_bottom(self) -> None:
        """P1/P2 are the box's bottom corners, never its top."""
        c = box_to_corners(0, 0, 100, 100, top_inset=0.1)
        self.assertEqual(c[0], (0, 100))
        self.assertEqual(c[1], (100, 100))

    def test_order_is_p1_p2_p3_p4(self) -> None:
        """The order every consumer downstream depends on."""
        self.assertEqual(CORNER_NAMES, ("P1", "P2", "P3", "P4"))
        c = box_to_corners(10, 20, 110, 220, top_inset=0.05)
        self.assertEqual([p[1] for p in c[:2]], [220, 220])   # both near (bottom)
        self.assertEqual([p[1] for p in c[2:]], [20, 20])      # both far (top)
        self.assertLess(c[3][0], c[0][0] + 6)                  # P4 left of P1
        self.assertGreater(c[2][0], c[1][0] - 6)               # P3 right of P2

    def test_inset_zero_is_a_rectangle(self) -> None:
        """No inset, no perspective: the far edge equals the near edge."""
        c = box_to_corners(100, 50, 300, 400, top_inset=0.0)
        p1, p2, p3, p4 = c
        self.assertEqual(p3, (300, 50))
        self.assertEqual(p4, (100, 50))
        self.assertEqual(p2[0] - p1[0], p3[0] - p4[0])
        valid, reason = corners_valid(c)
        # A rectangle passes the sanity check — it is merely uninformative
        # about where the far end of the pitch is.
        self.assertTrue(valid)
        self.assertEqual(reason, "ok")

    def test_default_inset_is_eight_percent(self) -> None:
        """The shipped default, exercised through the module's own constant."""
        c = box_to_corners(0, 0, 1000, 500)   # top_inset defaults to 0.08
        self.assertEqual(c[3], (80.0, 0))
        self.assertEqual(c[2], (920.0, 0))

    def test_top_drop_moves_the_far_edge_down(self) -> None:
        """`top_drop` is a fraction of the box height (0 = top of the box)."""
        c = box_to_corners(0, 100, 200, 300, top_inset=0.1, top_drop=0.5)
        self.assertEqual(c[3][1], 200)   # 100 + 0.5 * 200
        self.assertEqual(c[2][1], 200)
        self.assertEqual(c[1][1], 300)   # near edge untouched

    def test_inset_scales_with_width_not_absolute_pixels(self) -> None:
        """A box twice as wide pulls the far edge twice as far in."""
        narrow = box_to_corners(0, 0, 100, 100, top_inset=0.1)
        wide = box_to_corners(0, 0, 200, 100, top_inset=0.1)
        self.assertAlmostEqual(narrow[3][0], 10.0)    # 10% of 100, from x = 0
        self.assertAlmostEqual(wide[3][0], 20.0)      # 10% of 200, from x = 0


class TestCornersValid(unittest.TestCase):
    """The refusal rules, one named reason at a time."""

    def test_normal_box_is_valid(self) -> None:
        valid, reason = corners_valid(box_to_corners(100, 50, 300, 400, 0.08))
        self.assertTrue(valid)
        self.assertEqual(reason, "ok")

    def test_tiny_box_is_refused(self) -> None:
        # 4 px wide: the near edge is under the 5 px floor.
        valid, reason = corners_valid(box_to_corners(0, 0, 4, 400, 0.08))
        self.assertFalse(valid)
        self.assertEqual(reason, "too small or flipped")

    def test_short_box_is_refused(self) -> None:
        # 3 px tall: a pitch needs more vertical extent than that.
        valid, reason = corners_valid(box_to_corners(0, 0, 400, 3, 0.08))
        self.assertFalse(valid)
        self.assertEqual(reason, "too small or flipped")

    def test_box_with_no_height_is_refused(self) -> None:
        valid, reason = corners_valid(box_to_corners(0, 100, 400, 100, 0.08))
        self.assertFalse(valid)
        self.assertEqual(reason, "too small or flipped")

    def test_huge_inset_collapses_the_far_edge(self) -> None:
        # inset 0.5 * 200 = 100 px each side ⇒ far width exactly 0 ⇒ refused.
        valid, reason = corners_valid(box_to_corners(100, 50, 300, 400, 0.5))
        self.assertFalse(valid)
        self.assertEqual(reason, "too small or flipped")

    def test_inset_beyond_half_flips_the_far_edge(self) -> None:
        # inset 0.6 * 200 = 122.4 px each side ⇒ the far corners cross over
        # (far width < 0), i.e. the quad is inside out.
        c = box_to_corners(100, 50, 300, 400, 0.6)
        self.assertLess(c[2][0], c[3][0])
        valid, reason = corners_valid(c)
        self.assertFalse(valid)
        self.assertEqual(reason, "too small or flipped")

    def test_far_edge_wider_than_near_edge_is_refused(self) -> None:
        # A negative inset (the "wrong sign") flares the far edge out beyond
        # the near edge by more than the 2% tolerance: exactly what an
        # upside-down or nonsensical box looks like once squared away.
        c = box_to_corners(100, 50, 300, 400, -0.05)
        self.assertGreater(c[2][0] - c[3][0], (c[1][0] - c[0][0]) * 1.02)
        valid, reason = corners_valid(c)
        self.assertFalse(valid)
        self.assertEqual(reason, "far edge wider than near edge")

    def test_far_edge_slightly_narrower_is_ok(self) -> None:
        """The 2% tolerance exists so a hand-tuned inset is not refused."""
        c = box_to_corners(0, 0, 1000, 500, 0.004)
        valid, reason = corners_valid(c)
        self.assertTrue(valid)
        self.assertEqual(reason, "ok")

    def test_validity_does_not_mutate_the_input(self) -> None:
        c = box_to_corners(100, 50, 300, 400, 0.08)
        before = list(c)
        corners_valid(c)
        self.assertEqual(c, before)


class TestCornersAsDict(unittest.TestCase):
    """The stored shape: named, rounded, JSON-serialisable."""

    def test_names_and_rounding(self) -> None:
        # x1=10.006, x2=300.004 ⇒ width 289.998 ⇒ inset 23.19984 px
        d = corners_as_dict(box_to_corners(10.006, 20.5, 300.004, 400.0, 0.08))
        self.assertEqual(
            set(d), {"P1", "P2", "P3", "P4"}
        )
        self.assertEqual(d["P1"], [10.01, 400.0])
        self.assertEqual(d["P2"], [300.0, 400.0])
        self.assertEqual(d["P3"], [276.8, 20.5])
        self.assertEqual(d["P4"], [33.21, 20.5])

    def test_every_value_is_a_two_element_list_of_numbers(self) -> None:
        d = corners_as_dict(box_to_corners(0, 0, 100, 100))
        for name in CORNER_NAMES:
            self.assertEqual(len(d[name]), 2)
            self.assertIsInstance(d[name][0], float)
            self.assertIsInstance(d[name][1], float)

    def test_order_matches_the_tuple_form(self) -> None:
        c = box_to_corners(0, 0, 200, 200, 0.05)
        d = corners_as_dict(c)
        for name, point in zip(CORNER_NAMES, c):
            self.assertEqual(d[name], [round(point[0], 2), round(point[1], 2)])


if __name__ == "__main__":
    unittest.main()
