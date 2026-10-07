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
   is checked, not assumed: the ball must travel further along the chosen axis
   than across it (it cannot cover 20 m sideways), and the quad itself must be
   elongated the same way. A roughly square quad is refused, never guessed.
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

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..core.constants import (
    CREASE_WIDTH_M,
    PITCH_GEOMETRY_ASPECT_MIN_RATIO,
    PITCH_GEOMETRY_MIN_TRAVEL_POINTS,
    PITCH_GEOMETRY_TRAVEL_MIN_DELTA,
    PITCH_LENGTH_M,
)
from ..pitch.calibration import PitchCalibration

# Which unit-square axis a geometry reads along.
AXIS_U = "u"
AXIS_V = "v"

SOURCE_PITCH_KEYPOINTS = "pitch_keypoints"


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
    if calibration.homography is None or len(calibration.corners) < 4:
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

    us = [p[0] for p in uv]
    vs = [p[1] for p in uv]
    du = max(us) - min(us)
    dv = max(vs) - min(vs)

    # ── Which axis is along the pitch ────────────────────────────────────────
    ratio = PITCH_GEOMETRY_ASPECT_MIN_RATIO
    if dv >= du * ratio:
        axis = AXIS_V
    elif du >= dv * ratio:
        axis = AXIS_U
    else:
        # The ball moved about as far one way as the other: this trajectory
        # cannot say which direction is down the pitch, so no axis, no metres.
        return None, "ball_travel_axis_ambiguous"

    # The quad must be elongated the same way. `PitchCalibration.width` is the
    # horizontal span of the quad, `.length` the vertical one.
    quad_along = float(calibration.length) if axis == AXIS_V else float(calibration.width)
    quad_across = float(calibration.width) if axis == AXIS_V else float(calibration.length)
    if quad_across <= 0.0 or quad_along < quad_across * ratio:
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
