"""
calibration.py — Pixel↔ground geometry and the honesty gate around it.

This module owns ONE thing: the pixel→ground-plane mapping, and the decision of
whether the result of that mapping may be shown to anyone as a measurement.

PORTED FROM UPSTREAM, DELIBERATELY DIVERGENT
--------------------------------------------
`build_homography` / `pixel_to_ground` are ported from CricketTracker
`backend/tracker_modules/utils.py`. The single divergence is that the corner
fractions are read from `core.constants.HOMOGRAPHY_CORNER_FRACTIONS` instead of
being baked into this file, so calibration is inspectable and overridable without
touching the math. `build_homography` also accepts `corners_px`, which does not
exist upstream, so a real per-video pitch-corner measurement can replace the
template fractions.

WHY THE HONESTY GATE EXISTS
---------------------------
The template homography is measured off one specific side-on broadcast feed. On
a different framing it still returns plausible-looking numbers — the Phase 3
fixture produced 42-199 km/h and bounces off the end of a 20.12 m pitch. So
`geometry_status` refuses to describe ground-plane output as measurement unless
pitch corners were measured on *this* video; `analyse_delivery` then reports null
rather than a plausible wrong number.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from ..core.constants import (
    CREASE_WIDTH_M,
    HOMOGRAPHY_CORNER_FRACTIONS,
    MAX_EARLY_POSITIONS,
    PITCH_LENGTH_M,
    SPEED_CALIBRATION_OFFSET_KMH,
    SPEED_CALIBRATION_SCALE,
    SPEED_MAX_KMH,
    SPEED_MIN_KMH,
)

log = logging.getLogger(__name__)


# ── Homography ────────────────────────────────────────────────────────────────
def build_homography(
    width: int, height: int, corners_px: Optional[dict] = None
) -> np.ndarray:
    """
    Build a pixel → ground-plane homography for the standard side-on
    broadcast feed used by this app (TATA IPL broadcast template).

    Four pitch-corner pixel fractions are mapped to real-world (metres):
      TL → (0, 0)                 (bowling end, leg side)
      TR → (CREASE_WIDTH_M, 0)    (bowling end, off side)
      BL → (0, PITCH_LENGTH_M)    (batting end, leg side)
      BR → (CREASE_WIDTH_M, PITCH_LENGTH_M)

    The default fractions are `HOMOGRAPHY_CORNER_FRACTIONS`, measured directly
    off a real reference frame from this broadcast feed (far crease ≈27% down
    the frame spanning x 42-56%; near crease ≈66% down spanning x 43-57%). They
    are NOT a generic property of cricket video: the previous values assumed the
    crease spanned ~40-50% of the frame width when it actually spans ~14%, which
    threw every speed calculation off by roughly 3x.

    Pass `corners_px` to use a real per-video pitch-corner measurement instead.
    Re-measure and update the constants if camera framing changes.
    """
    frac = corners_px or HOMOGRAPHY_CORNER_FRACTIONS
    src = np.float32(
        [
            [float(frac["TL"][0]) * width, float(frac["TL"][1]) * height],
            [float(frac["TR"][0]) * width, float(frac["TR"][1]) * height],
            [float(frac["BL"][0]) * width, float(frac["BL"][1]) * height],
            [float(frac["BR"][0]) * width, float(frac["BR"][1]) * height],
        ]
    )
    dst = np.float32(
        [
            [0, 0],
            [CREASE_WIDTH_M, 0],
            [0, PITCH_LENGTH_M],
            [CREASE_WIDTH_M, PITCH_LENGTH_M],
        ]
    )
    H, _ = cv2.findHomography(src, dst)
    return H


def pixel_to_ground(pt: tuple, H: np.ndarray) -> tuple:
    """Map a single pixel (x, y) to ground-plane coordinates (metres) via H."""
    p = np.float32([[pt]]).reshape(-1, 1, 2)
    gp = cv2.perspectiveTransform(p, H)
    return float(gp[0][0][0]), float(gp[0][0][1])


def make_homography(
    width: int, height: int, corners_px: Optional[dict] = None
) -> np.ndarray:
    """
    Build the pixel→ground homography for a frame size.

    A thin wrapper over `build_homography` so callers do not have to know that the
    corner fractions live in `core.constants`. `corners_px` carries a real
    per-video calibration when one exists.
    """
    return build_homography(int(width), int(height), corners_px)


# The single reason ground-plane analytics are withheld, phrased for a consumer.
GEOMETRY_INVALID_REASON = (
    "Pitch-corner calibration is required for this video. The available "
    "homography is a fixed set of corner fractions measured from a different "
    "broadcast template, so its ground-plane coordinates, and every speed, "
    "length, line, swing and bounce angle derived from them, do not describe "
    "this footage."
)


def geometry_status(corners_px: Optional[dict]) -> dict:
    """
    Whether the ground-plane geometry may be presented as physical measurement.

    There are exactly two states:

      * calibrated  — pitch corners were measured on THIS video's frames.
        Ground-plane metres are meaningful.
      * uncalibrated — no measurement exists. Metres are withheld and every
        quantity derived from them is reported null rather than estimated.

    This exists because a fixed homography produces plausible-looking numbers
    from wrong geometry. Measured on the Phase 3 fixture, those numbers ranged
    42-199 km/h and placed bounces off the end of a 20.12 m pitch. Returning
    null is not a loss of function; it is the difference between an honest
    answer and a fabricated one.
    """
    calibrated = bool(corners_px) and {"TL", "TR", "BL", "BR"} <= set(corners_px)
    return {
        "geometry_calibrated": calibrated,
        "geometry_source": "per_video_pitch_corners" if calibrated
                           else "template_corner_fractions",
        "ground_plane_analytics_available": calibrated,
        "geometry_note": (
            "Pitch corners were measured on this video; ground-plane metres "
            "are meaningful."
            if calibrated else GEOMETRY_INVALID_REASON
        ),
        "suppressed_when_uncalibrated": [
            "speed_kmh", "length", "line", "swing", "bounce_angle",
            "bounce.ground_x_m", "bounce.ground_y_m",
            "speed_samples[].ground_x_m", "speed_samples[].ground_y_m",
        ],
        "pitch_corners_px": (
            {k: [float(v[0]), float(v[1])] for k, v in corners_px.items()}
            if calibrated else None
        ),
    }


def calibration_metadata(corners_px: Optional[dict] = None) -> dict:
    """
    Provenance block persisted at analysis level and served by /api/reference.

    Nothing about the speed number is hidden from a consumer.
    """
    geo = geometry_status(corners_px)
    return {
        # Explicitly NOT a calibrated reading. D2 forbids presenting the
        # 2x/-20 heuristic as a physical measurement.
        "speed_is_estimate": True,
        "speed_is_calibrated": False,
        "speed_method": "homography_least_squares",
        "speed_scale": SPEED_CALIBRATION_SCALE,
        "speed_offset_kmh": SPEED_CALIBRATION_OFFSET_KMH,
        "speed_formula": (
            f"speed_kmh = speed_mps * 3.6 * {SPEED_CALIBRATION_SCALE}"
            f" + ({SPEED_CALIBRATION_OFFSET_KMH})"
        ),
        "speed_note": (
            "Homography-derived ESTIMATE, not a physical measurement. A single "
            "fixed-camera homography cannot absorb broadcast foreshortening, so "
            "a hand-tuned scale and offset are applied. Per-video pitch-corner "
            "calibration is required for real accuracy."
        ),
        "speed_imputation_enabled_by_default": False,
        "speed_sanity_bounds_kmh": [SPEED_MIN_KMH, SPEED_MAX_KMH],
        "speed_fit_max_detections": MAX_EARLY_POSITIONS,
        "pitch_length_m": PITCH_LENGTH_M,
        "crease_width_m": CREASE_WIDTH_M,
        "homography_corner_fractions": {
            k: list(v) for k, v in HOMOGRAPHY_CORNER_FRACTIONS.items()
        },
        "homography_note": (
            "Corner fractions are measured off one specific side-on broadcast "
            "template and are NOT a general property of cricket video. "
            "Different framing invalidates every speed and length/line result."
        ),
        **geo,
    }


__all__ = [
    "GEOMETRY_INVALID_REASON",
    "build_homography",
    "calibration_metadata",
    "geometry_status",
    "make_homography",
    "pixel_to_ground",
]