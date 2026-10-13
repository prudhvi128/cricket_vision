"""
mask_geometry.py — from a segmentation mask polygon to a four-corner quad.

Pure functions, no model, no I/O: given one mask's boundary (as returned by
ultralytics in ORIGINAL frame pixels) they derive the four-corner quad the
calibration consumes, or say why they cannot.

THE QUAD COMES FROM THE MASK BOUNDARY, NEVER FROM THE FRAME
-----------------------------------------------------------
The trained model answers with a mask over the complete wicket-to-wicket
pitch. Four corners are then taken from that boundary: convex hull,
`approxPolyDP` at a short list of simplification fractions until exactly four
corners survive, and — only if that fails — a greedy reduction of the hull's
flattest corner until four remain. No fixed inset, no axis-aligned bounding
box: the mask IS the boundary and the corners are what simplifying it
produces.

CORNERS ARE NAMED BY THE SAME EXTREMES CONVENTION AS `derive_corners`
--------------------------------------------------------------------
`top_left` = min(x+y), `bottom_right` = max(x+y), `top_right` = max(x−y),
`bottom_left` = min(x−y) — the exact rule pitch/calibration.py uses. Two
modules deriving corners independently and disagreeing would make the EMA
smoother blend two different namings into a bow-tie; deriving them the same
way is what keeps smoothing and the homography stable.

EVERY CHECK HERE CAN REFUSE
---------------------------
A triangle, a degenerate hull, a mask covering 38% of the frame (a crowd),
a mask with NaN coordinates: all refused with a reason. A refusal is a fact
worth recording; a fabricated quad drawn over the video is not.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ..core.constants import PITCH_SEG_APPROX_POLYDP_FRACS

Size = tuple[int, int]

# How far outside the frame a mask vertex may sit before the mask is refused.
# Ultralytics lands vertices on the mask boundary, which can bleed a pixel or
# two past the edge; a vertex halfway to the next frame is a different object.
_BOUND_SLACK_FRACTION = 0.02

# The greedy hull reduction's loop ceiling. A hull with more vertices than
# this has a boundary too wiggly to trust, and reducing it to four would be
# fabrication rather than simplification.
_MAX_HULL_VERTICES = 64

# Ordered so the unit square reads top-left → top-right → bottom-right →
# bottom-left, i.e. the winding a perspective transform expects.
_CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


def mask_polygon_area_fraction(
    points: Sequence[tuple[float, float]], image_size: Size
) -> float:
    """
    Share of the frame the mask's CONVEX HULL covers.

    The hull, not the raw polygon: a mask with a dent along one edge has the
    same area as the pitch it covers, and a share expressed against the hull
    is what `min_area_fraction`/`max_area_fraction` were measured against.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return 0.0
    pts = _finite_points(points)
    if len(pts) < 3:
        return 0.0
    arr = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(arr)
    return float(cv2.contourArea(hull)) / float(width * height)


def mask_centroid(
    points: Sequence[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    """Mean of the mask vertices — the consistency anchor between estimates."""
    pts = _finite_points(points)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(pts))


def quad_aspect_ratio(quad: Sequence[tuple[float, float]]) -> float:
    """
    min-area-rect aspect ratio of the quad: max(w, h) / min(w, h).

    Reported for the RAW MASK too (not only the quad) so a crowd-shaped blob
    can be refused before simplification even runs. A degenerate quad (zero
    width or height) is infinite: the caller's ceiling refuses it.
    """
    pts = _finite_points(quad)
    if len(pts) < 3:
        return float("inf")
    arr = np.array(pts, dtype=np.float32)
    (_, _), (rw, rh), _angle = cv2.minAreaRect(arr)
    if rw <= 1e-6 or rh <= 1e-6:
        return float("inf")
    return float(max(rw, rh) / min(rw, rh))


def mask_in_bounds(
    points: Sequence[tuple[float, float]], image_size: Size
) -> bool:
    """Every finite vertex lies inside the frame (plus a couple of pixels)."""
    width, height = image_size
    if width <= 0 or height <= 0:
        return False
    pts = _finite_points(points)
    if not pts:
        return False
    slack_x = _BOUND_SLACK_FRACTION * width
    slack_y = _BOUND_SLACK_FRACTION * height
    return all(
        -slack_x <= x <= width + slack_x and -slack_y <= y <= height + slack_y
        for x, y in pts
    )


def validate_mask(
    points: Sequence[tuple[float, float]],
    image_size: Size,
    *,
    min_area_fraction: float,
    max_area_fraction: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> tuple[bool, str]:
    """
    Shape-plausibility gate on the raw mask polygon, before simplification.

    Returns `(True, "ok")` or `(False, reason)`. Area and aspect bounds are
    the ones measured off real pitch masks versus the crowd false positive
    (see core/constants.py); nothing here is fitted to one video's framing.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return False, "unknown_frame_size"
    pts = _finite_points(points)
    if len(pts) < 3:
        return False, f"mask_polygon_too_small_{len(pts)}"
    if not mask_in_bounds(pts, image_size):
        return False, "mask_outside_frame"
    area = mask_polygon_area_fraction(pts, image_size)
    if area < min_area_fraction:
        return False, "mask_area_too_small"
    if area > max_area_fraction:
        return False, "mask_area_too_large"
    aspect = quad_aspect_ratio(pts)
    if aspect < min_aspect_ratio:
        return False, "mask_aspect_too_square"
    if aspect > max_aspect_ratio:
        return False, "mask_aspect_too_elongated"
    return True, "ok"


def quad_from_mask(
    points: Sequence[tuple[float, float]],
) -> tuple[Optional[list[tuple[float, float]]], str]:
    """
    Four corners from the mask boundary, ordered top_left → bottom_left,
    plus which simplification produced them (`direct_hull`, `approx_polydp`
    or `hull_reduce`).

    `(None, reason)` when no honest quad exists: fewer than four finite
    vertices, a triangle hull, a degenerate reduction. The caller must
    record the refusal — inventing the missing corner would draw a boundary
    the model never reported.
    """
    pts = _finite_points(points)
    if len(pts) < 4:
        return None, "too_few_vertices"
    arr = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(arr).reshape(-1, 2)
    hull_pts = [(float(p[0]), float(p[1])) for p in hull]
    if len(hull_pts) < 4:
        return None, "degenerate_hull"
    if len(hull_pts) > _MAX_HULL_VERTICES:
        return None, "hull_too_complex"

    # Exactly four hull vertices already: the mask is a quad.
    if len(hull_pts) == 4:
        corners = _order_extremes(hull_pts)
        if corners is not None:
            return [corners[name] for name in _CORNER_ORDER], "direct_hull"

    # approxPolyDP at a short list of fractions until exactly four corners.
    for frac in PITCH_SEG_APPROX_POLYDP_FRACS:
        epsilon = frac * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
        if len(approx) != 4:
            continue
        candidate = [(float(p[0]), float(p[1])) for p in approx]
        if len(set(candidate)) != 4:
            continue
        corners = _order_extremes(candidate)
        if corners is not None:
            return [corners[name] for name in _CORNER_ORDER], "approx_polydp"

    # Fallback: greedily drop the flattest hull corner until four remain.
    reduced = _greedy_hull_to_quad(hull_pts)
    corners = _order_extremes(reduced) if reduced else None
    if corners is None:
        return None, "quad_unavailable"
    return [corners[name] for name in _CORNER_ORDER], "hull_reduce"


def corners_as_named(
    quad: Sequence[tuple[float, float]],
) -> Optional[dict[str, tuple[float, float]]]:
    """`{top_left: (x, y), ...}` for a four-corner list, or None."""
    pts = _finite_points(quad)
    if len(pts) != 4:
        return None
    return _order_extremes(pts)


# ── internals ────────────────────────────────────────────────────────────────
def _finite_points(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Drop NaN/inf/non-numeric vertices. A malformed vertex is not repaired."""
    out: list[tuple[float, float]] = []
    for p in points:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if x != x or y != y:  # NaN
            continue
        if abs(x) == float("inf") or abs(y) == float("inf"):
            continue
        out.append((x, y))
    return out


def _order_extremes(
    points: Sequence[tuple[float, float]],
) -> Optional[dict[str, tuple[float, float]]]:
    """
    The same extremes convention pitch/calibration.py `derive_corners` uses.

    Four extremes must be four DISTINCT points: a triangle, a line or a
    collapsed pair is a degenerate quad and is refused, never repaired.
    """
    if len(points) < 4:
        return None
    pts = list(points)
    top_left = min(pts, key=lambda p: p[0] + p[1])
    bottom_right = max(pts, key=lambda p: p[0] + p[1])
    top_right = max(pts, key=lambda p: p[0] - p[1])
    bottom_left = min(pts, key=lambda p: p[0] - p[1])
    corners = {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_right": bottom_right,
        "bottom_left": bottom_left,
    }
    if len(set(corners.values())) != 4:
        return None
    return corners


def _greedy_hull_to_quad(
    hull: list[tuple[float, float]],
) -> Optional[list[tuple[float, float]]]:
    """
    Reduce a hull to four vertices by repeatedly dropping the flattest one.

    "Flattest" = the vertex whose two neighbours are closest to collinear
    with it, measured as the area the triangle (prev, vertex, next) covers.
    The smallest triangle is the corner the mask boundary bends the least at,
    so dropping it changes the covered region the least. Four vertices are
    left exactly; a hull that reaches three is degenerate and refused.
    """
    pts = list(hull)
    if len(pts) < 4:
        return None
    while len(pts) > 4:
        worst_i = -1
        worst_area = float("inf")
        n = len(pts)
        for i in range(n):
            a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            area = abs(
                (b[0] - a[0]) * (c[1] - a[1])
                - (c[0] - a[0]) * (b[1] - a[1])
            ) / 2.0
            if area < worst_area:
                worst_area = area
                worst_i = i
        if worst_i < 0:
            return None
        pts.pop(worst_i)
    if len(pts) != 4 or len(set(pts)) != 4:
        return None
    return pts


__all__ = [
    "corners_as_named",
    "mask_centroid",
    "mask_in_bounds",
    "mask_polygon_area_fraction",
    "quad_aspect_ratio",
    "quad_from_mask",
    "validate_mask",
]
