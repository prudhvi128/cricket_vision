"""
box_corners.py — Pure geometry: a detector box to four APPROXIMATE pitch corners.

Nothing in this module imports from the rest of the project, so it can be tested
and reasoned about on its own. It owns no measurement and makes no claim to one:
the four points below are GUESSES derived from a rectangle, not detected pitch
corners, and every consumer labels them as such (`corners_source="box_estimate"`).

Corner order (the same one everywhere downstream):

    P3 far-left ──── P4 far-right      top of the box, pulled inward by
         │                │            `top_inset * width` so the far edge of
         │     pitch      │            the resulting trapezoid is narrower than
         │                │            the near edge, the way a pitch seen in
    P1 near-left ── P2 near-right      perspective is.

`corners_valid` is the sanity check on the result: a near edge that collapsed,
a far edge wider than the near edge (the inset went the wrong way), or a box too
short to hold a pitch all read as invalid, with the reason spelled out. An
invalid set is still returned — the caller records it rather than dropping it —
it is simply never usable as geometry.
"""

from __future__ import annotations


def box_to_corners(x1, y1, x2, y2, top_inset=0.08, top_drop=0.0):
    """Box -> 4 corners. Order: P1 near-left, P2 near-right, P3 far-right, P4 far-left."""
    w = x2 - x1
    h = y2 - y1
    inset = top_inset * w
    top_y = y1 + top_drop * h
    return [(x1, y2), (x2, y2), (x2 - inset, top_y), (x1 + inset, top_y)]


def corners_valid(c):
    p1, p2, p3, p4 = c
    near_w = p2[0] - p1[0]
    far_w = p3[0] - p4[0]
    height = p1[1] - p4[1]
    if near_w <= 5 or far_w <= 0 or height <= 5:
        return False, "too small or flipped"
    if far_w > near_w * 1.02:
        return False, "far edge wider than near edge"
    return True, "ok"


CORNER_NAMES = ("P1", "P2", "P3", "P4")


def corners_as_dict(corners) -> dict:
    """`{"P1": [x, y], ...}` rounded to 2 dp, matching the pitch block's style."""
    return {
        name: [round(float(p[0]), 2), round(float(p[1]), 2)]
        for name, p in zip(CORNER_NAMES, corners)
    }


__all__ = ["CORNER_NAMES", "box_to_corners", "corners_as_dict", "corners_valid"]
