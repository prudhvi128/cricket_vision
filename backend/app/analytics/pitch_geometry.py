"""
pitch_geometry.py — From a detected pitch quad to metres on the ground.

Two calibration concepts already exist in this project and this module joins
them:

    app/pitch/                 "WHERE is the pitch in this frame?" (model, pixels)
    analytics/calibration.py   "WHERE is the pitch in METRES?"     (operator corners)

`PitchService` gives the first. This module takes one validated quad — the
image→unit-square homography `PitchCalibration` already computes — plus the
ball's own tracked points, and produces a pixel→metres homography together with
WHICH END of that quad the batter stands at. Without that orientation a metre
scale is meaningless: the same quad read from either end gives the same numbers
with the batter at the opposite end of them.

THREE ASSUMPTIONS, ALL STATED
-----------------------------
1. The quad's LONG axis runs along the pitch and its SHORT axis across it. This
   is checked, not assumed, and the check ships as three separate refusals:
     * CROP — a quad whose bottom edge sits in the broadcast scorecard band has
       run out of VISIBLE pitch: the model cannot segment through the overlay,
       so that quad is only part of the wicket-to-wicket rectangle. It is
       refused with its own reason, never stretched to 20.12 m.
     * ELONGATION — measured by min-area-rect aspect (`pitch/
       mask_geometry.quad_aspect_ratio`), rotation-invariantly, because the
       axis-aligned span check this replaces refused every quad the moment the
       camera tilted: the same rectangle rotated 30° halves its axis-aligned
       ratio while its true shape is unchanged.
     * DIRECTION — the ball's image-space travel must lie within
       PITCH_GEOMETRY_TRAVEL_AXIS_MAX_DEG of that long axis, so a wide
       side-on quad cannot claim a vertically travelling ball however the
       homography stretches the unit square.
   A roughly square quad, or one that disagrees with the ball's travel, is
   refused — never guessed.
2. The long axis is scaled to PITCH_LENGTH_M (20.12 m, Law 6) and the short axis
   to CREASE_WIDTH_M (2.64 m, the return creases). The detected quad is treated
   as that rectangle, and its edge nearest the batter is treated as the batting
   crease — which is what makes a "metres from the crease" reading possible at
   all. If the model's quad is drawn somewhere else, every length shifts with it;
   the assumption is repeated in the analysis metadata so it cannot be missed.
3. The batter's end is whichever way the ball was TRAVELLING, taken from the
   release-to-bounce part of the trajectory (the image-space lowest point, so
   this needs no geometry at all). A ball that barely moves in the detections,
   or one whose travel is ambiguous, yields no geometry and therefore no metres.

The output coordinates follow `analytics/calibration.py` exactly: y = 0 at the
NON-batting end, y = PITCH_LENGTH_M at the batting end, x across the pitch.
That is what lets one length formula serve both geometry sources.

Everything here fails closed: `(None, reason)`, never a plausible number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..core.constants import (
    CREASE_WIDTH_M,
    PITCH_GEOMETRY_ASPECT_MIN_RATIO,
    PITCH_GEOMETRY_CROP_BAND_FRACTION,
    PITCH_GEOMETRY_CROPPED_REASON,
    PITCH_GEOMETRY_MIN_TRAVEL_POINTS,
    PITCH_GEOMETRY_NON_METRIC_SOURCE,
    PITCH_GEOMETRY_TRAVEL_AXIS_MAX_DEG,
    PITCH_GEOMETRY_TRAVEL_MIN_DELTA,
    PITCH_DETECTION_SOURCE_KEYPOINTS,
    PITCH_DETECTION_SOURCE_SEGMENTATION,
    PITCH_LENGTH_M,
)
from ..pitch.calibration import PitchCalibration
from ..pitch.mask_geometry import quad_aspect_ratio

# Which unit-square axis a geometry reads along.
AXIS_U = "u"
AXIS_V = "v"

SOURCE_PITCH_KEYPOINTS = PITCH_DETECTION_SOURCE_KEYPOINTS
SOURCE_PITCH_SEGMENTATION = PITCH_DETECTION_SOURCE_SEGMENTATION

# Recorded as the rejection reason when a calibration's source is not allowed
# to become metric geometry. One constant, in core/constants.py, so the
# analysis record and the run report use the same words for the same refusal.
NON_METRIC_SOURCE_REASON = PITCH_GEOMETRY_NON_METRIC_SOURCE


@dataclass(frozen=True)
class PitchGeometry:
    """
    A pixel→metres homography for one delivery, plus the orientation that makes
    its metres mean something.

    `homography` uses the project's ground convention (y = 0 at the non-batting
    end, y = PITCH_LENGTH_M at the batting end; x = 0..CREASE_WIDTH_M across).
    `map_homography` is the same transform normalised to 0..1 on both axes, so a
    client drawing a top-down pitch always has the batting end at y = 1.
    """

    homography: np.ndarray
    map_homography: np.ndarray
    axis: str                       # AXIS_U or AXIS_V: which axis runs along the pitch
    batting_end: float              # 0.0 or 1.0, in unit-square units along `axis`
    travel_delta: float             # how far along the axis the ball travelled
    calibration_frame: int
    source: str = SOURCE_PITCH_KEYPOINTS

    @property
    def batting_end_v(self) -> float | None:
        """
        The batting end's position in unit-square V, for a client that draws the
        detected quad itself. None when the along-pitch axis is U, because then
        the batting end is a whole edge of the quad, not one value of V.
        """
        if self.axis == AXIS_V:
            return float(self.batting_end)
        return None


def flight_to_bounce(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    The release-to-bounce part of a detected trajectory: everything up to and
    including the lowest detection in the image.

    It needs no geometry — the ball falls, so the lowest tracked point is the
    bounce — and it exists because the part AFTER the bounce is not evidence of
    direction: a struck ball comes back up the image, and voting on that would
    flip the batter to the bowler's end.
    """
    if not points:
        return []
    low = max(range(len(points)), key=lambda i: points[i][1])
    return list(points[: low + 1])


# The corner order the geometry checks read the quad in: top-left → top-right
# → bottom-right → bottom-left, the same order pitch/calibration.py names them.
_CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


def _quad_points(
    calibration: PitchCalibration,
) -> Optional[list[tuple[float, float]]]:
    """The quad's four corners in reading order, or None when a name is missing."""
    corners = calibration.corners
    if any(name not in corners for name in _CORNER_ORDER):
        return None
    return [
        (float(corners[name][0]), float(corners[name][1]))
        for name in _CORNER_ORDER
    ]


def _quad_long_axis(
    quad: Sequence[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    """
    Unit vector along the quad's LONGER edge pair, or None when degenerate.

    The two edge pairs are averaged (top+bottom, right+left) so a trapezoid's
    perspective does not tilt the answer to one of its four edges: the axis of
    a trapezoid seen end-on is the mean of its foreshortened and stretched
    sides, which for a symmetric trapezoid is exactly the vertical.
    """
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
    # across = top edge + bottom edge, both pointing the same way (left→right).
    across = ((x1 - x0) + (x2 - x3), (y1 - y0) + (y2 - y3))
    across_len = (math.dist((x0, y0), (x1, y1)) + math.dist((x3, y3), (x2, y2))) / 2.0
    # along = right edge + left edge, both pointing the same way (top→bottom).
    along = ((x2 - x1) + (x3 - x0), (y2 - y1) + (y3 - y0))
    along_len = (math.dist((x1, y1), (x2, y2)) + math.dist((x0, y0), (x3, y3))) / 2.0
    vec = across if across_len >= along_len else along
    norm = math.hypot(vec[0], vec[1])
    if norm <= 1e-9:
        return None
    return (vec[0] / norm, vec[1] / norm)


def _travel_direction(
    points: Sequence[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    """
    Unit vector of the ball's image-space travel, release end → bounce end.

    First-third to last-third centroids, the same robust pairing the
    unit-square axis vote uses, so one noisy detection at either end cannot
    swing the angle. None when the detections do not move at all — though the
    axis vote upstream has already refused that trajectory.
    """
    if len(points) < 2:
        return None
    k = max(1, len(points) // 3)
    first = points[:k]
    last = points[-k:]
    fx = sum(float(p[0]) for p in first) / len(first)
    fy = sum(float(p[1]) for p in first) / len(first)
    lx = sum(float(p[0]) for p in last) / len(last)
    ly = sum(float(p[1]) for p in last) / len(last)
    dx, dy = lx - fx, ly - fy
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    return (dx / norm, dy / norm)


def _axis_divergence_deg(
    a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Acute angle between two UNDIRECTED axes, in degrees (0..90).

    Undirected because an axis has no head: the quad's long axis points
    "either way along the pitch" and the ball's travel likewise; only their
    agreement in direction matters, not which end is which.
    """
    dot = abs(a[0] * b[0] + a[1] * b[1])
    return math.degrees(math.acos(min(1.0, dot)))


def build_pitch_geometry(
    calibration: Optional[PitchCalibration],
    flight_points: Sequence[tuple[float, float]],
) -> tuple[Optional[PitchGeometry], str]:
    """
    Derive ground-plane geometry from one detected quad, or say why it cannot.

    `flight_points` are detected ball positions in TIME ORDER, from release up
    to the bounce — the part of the trajectory whose direction of travel is the
    ball heading down the pitch rather than the ball coming back off a bat.

    Returns `(geometry, "")` or `(None, reason)`. The reason is for the analysis
    record and the run report; a client never sees it (it sees `pitch.state` and
    null measurements).
    """
    if calibration is None:
        return None, "no_pitch_calibration"
    # ── The metric gate ──────────────────────────────────────────────────────
    # Two measured sources may become metres: the keypoint quad and the
    # segmentation quad. Both are refused the moment their GEOMETRY fails the
    # checks below — a quad that is near-square, that disagrees with the
    # ball's travel, that maps a corner outside the pitch, or whose homography
    # is degenerate yields no number at all, never a plausible one. An UNKNOWN
    # source (one this project does not measure) is refused outright, before
    # any number is computed, because its axis convention cannot be audited.
    # The reason is recorded per delivery, so "why is there no length" is
    # answerable from the artefact.
    source = getattr(calibration, "source", SOURCE_PITCH_KEYPOINTS)
    if source not in (SOURCE_PITCH_KEYPOINTS, SOURCE_PITCH_SEGMENTATION):
        return None, NON_METRIC_SOURCE_REASON
    quad = _quad_points(calibration)
    if calibration.homography is None or quad is None:
        return None, "pitch_quad_unavailable"
    if len(flight_points) < PITCH_GEOMETRY_MIN_TRAVEL_POINTS:
        return None, "too_few_detected_points"

    # ── Unit-square coordinates of the flight ────────────────────────────────
    uv: list[tuple[float, float]] = []
    for x, y in flight_points:
        mapped = calibration.image_to_pitch(float(x), float(y))
        if mapped is not None:
            uv.append(mapped)
    if len(uv) < PITCH_GEOMETRY_MIN_TRAVEL_POINTS:
        return None, "too_few_points_mapped_into_pitch"

    # ── The quad itself must be a whole, elongated pitch rectangle ──────────
    # Three checks on the QUAD, each of which can refuse on its own fact. They
    # run before the ball's axis vote below: a cropped or square quad is wrong
    # no matter which way the ball travelled, and "cropped by the overlay" is a
    # more useful thing to record about a delivery than "the travel was even".

    # 1. CROP: a quad whose bottom edge sits in the broadcast scorecard band
    #    has run out of VISIBLE pitch — the model cannot segment through the
    #    overlay, so that quad is only part of the wicket-to-wicket rectangle
    #    and must never be stretched to 20.12 m. Its own reason, never the
    #    aspect-mismatch one below: "cropped" and "square" are different facts
    #    and a client explaining a missing length should say which happened.
    _frame_w, frame_h = calibration.image_size
    crop_band = PITCH_GEOMETRY_CROP_BAND_FRACTION * frame_h
    bottom_edge_y = max(y for _x, y in quad)
    if frame_h - bottom_edge_y <= crop_band:
        return None, PITCH_GEOMETRY_CROPPED_REASON

    # 2. ELONGATION, rotation-invariantly: the min-area-rect aspect of the
    #    quad must reach the same ratio the axis vote below trusts of the
    #    BALL. The check this replaced compared the quad's axis-ALIGNED spans,
    #    which punished every rotated or strongly perspective quad — on the
    #    Oct-11 footage that refused all 37 calibrations outright. A quad that
    #    is roughly square cannot say which axis is which, so it still goes.
    ratio = PITCH_GEOMETRY_ASPECT_MIN_RATIO
    if quad_aspect_ratio(quad) < ratio:
        return None, "pitch_quad_aspect_disagrees_with_travel"

    us = [p[0] for p in uv]
    vs = [p[1] for p in uv]
    du = max(us) - min(us)
    dv = max(vs) - min(vs)

    # ── Which axis is along the pitch ────────────────────────────────────────
    if dv >= du * ratio:
        axis = AXIS_V
    elif du >= dv * ratio:
        axis = AXIS_U
    else:
        # The ball moved about as far one way as the other: this trajectory
        # cannot say which direction is down the pitch, so no axis, no metres.
        return None, "ball_travel_axis_ambiguous"

    # 3. DIRECTION: the ball's image-space travel must lie within
    #    PITCH_GEOMETRY_TRAVEL_AXIS_MAX_DEG of the quad's own long axis. The
    #    unit-square vote above picks the axis INSIDE the quad; this says the
    #    quad is elongated the same way IN THE IMAGE, so a wide side-on quad
    #    cannot claim a vertically travelling ball however the homography
    #    stretches the unit square.
    quad_axis = _quad_long_axis(quad)
    if quad_axis is None:
        return None, "pitch_quad_unavailable"
    travel_axis = _travel_direction(flight_points)
    if travel_axis is None:
        return None, "pitch_orientation_ambiguous"
    if _axis_divergence_deg(quad_axis, travel_axis) > PITCH_GEOMETRY_TRAVEL_AXIS_MAX_DEG:
        return None, "pitch_quad_aspect_disagrees_with_travel"

    # ── Which end is the batter's ────────────────────────────────────────────
    along = [p[1] if axis == AXIS_V else p[0] for p in uv]
    k = max(1, len(along) // 3)
    delta = (sum(along[-k:]) / k) - (sum(along[:k]) / k)
    if abs(delta) < PITCH_GEOMETRY_TRAVEL_MIN_DELTA:
        return None, "pitch_orientation_ambiguous"
    batting_end = 1.0 if delta > 0 else 0.0

    # ── pixel → metres ───────────────────────────────────────────────────────
    # y = 0 at the non-batting end, y = PITCH_LENGTH_M at the batting end,
    # x = 0..CREASE_WIDTH_M across — the convention build_homography() uses.
    w = float(CREASE_WIDTH_M)
    length = float(PITCH_LENGTH_M)
    if axis == AXIS_V:
        # across = u, along = v
        scale = (
            np.array([[w, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, 1.0]])
            if batting_end == 1.0
            else np.array([[w, 0.0, 0.0], [0.0, -length, length], [0.0, 0.0, 1.0]])
        )
    else:
        # across = v, along = u
        scale = (
            np.array([[0.0, w, 0.0], [length, 0.0, 0.0], [0.0, 0.0, 1.0]])
            if batting_end == 1.0
            else np.array([[0.0, w, 0.0], [-length, 0.0, length], [0.0, 0.0, 1.0]])
        )

    homography = scale @ np.asarray(calibration.homography, dtype=np.float64)
    if not np.isfinite(homography).all():
        return None, "pitch_homography_not_finite"

    normalise = np.array(
        [[1.0 / w, 0.0, 0.0], [0.0, 1.0 / length, 0.0], [0.0, 0.0, 1.0]]
    )
    map_homography = normalise @ homography

    # The quad's own corners must land inside the box they define. They do by
    # construction, but a degenerate quad would not, and the caller must never
    # receive a transform that maps the pitch somewhere else entirely.
    for corner in calibration.corners.values():
        vec = np.array([corner[0], corner[1], 1.0], dtype=np.float64)
        mapped = homography @ vec
        if abs(mapped[2]) < 1e-12:
            return None, "pitch_homography_degenerate"
        x_m = float(mapped[0] / mapped[2])
        y_m = float(mapped[1] / mapped[2])
        if not (-1e-3 <= x_m <= w + 1e-3 and -1e-3 <= y_m <= length + 1e-3):
            return None, "pitch_quad_maps_outside_pitch"

    return (
        PitchGeometry(
            homography=homography,
            map_homography=map_homography,
            axis=axis,
            batting_end=batting_end,
            travel_delta=float(delta),
            calibration_frame=int(calibration.frame_number),
            source=source,
        ),
        "",
    )


def as_metadata(geometry: Optional[PitchGeometry]) -> dict:
    """The orientation, persisted where a reader of the result can audit it."""
    if geometry is None:
        return {
            "source": None,
            "axis": None,
            "batting_end": None,
            "travel_delta": None,
            "calibration_frame": None,
        }
    return {
        "source": geometry.source,
        "axis": geometry.axis,
        "batting_end": geometry.batting_end,
        "travel_delta": round(geometry.travel_delta, 4),
        "calibration_frame": geometry.calibration_frame,
    }


__all__ = [
    "AXIS_U",
    "AXIS_V",
    "SOURCE_PITCH_KEYPOINTS",
    "PitchGeometry",
    "as_metadata",
    "build_pitch_geometry",
    "flight_to_bounce",
]
