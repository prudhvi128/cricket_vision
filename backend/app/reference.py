"""
reference.py — Enum and calibration data served to clients.

Both merged frontends hardcoded these strings independently and both got them
wrong: `PitchMap.tsx` renders "Leg"/"Off" while the backend emits
"Leg Side"/"Off Side", and the line-length axis is labelled inconsistently
between the two apps.

Rather than repeat that, the backend publishes the authoritative values once and
the frontends fetch them. Values come from `app.core.constants`, so there is a
single definition in the system.
"""

from __future__ import annotations

from .core.constants import (
    CREASE_WIDTH_M,
    HOMOGRAPHY_CORNER_FRACTIONS,
    LENGTH_ZONES,
    LINE_ZONES,
    PITCH_LENGTH_M,
    SCHEMA_VERSION,
    SHOT_CLASSES,
    SHOT_N_FRAMES,
)
from .analytics.calibration import calibration_metadata


def reference_document() -> dict:
    """The full /api/reference payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "shot_classes": list(SHOT_CLASSES),
        "shot_input_frames": SHOT_N_FRAMES,
        "length_zones": [
            {"label": label, "from": lo, "to": hi} for label, lo, hi in LENGTH_ZONES
        ],
        "line_zones": [
            {"label": label, "from": lo, "to": hi} for label, lo, hi in LINE_ZONES
        ],
        "line_labels": [label for label, _, _ in LINE_ZONES],
        "length_labels": [label for label, _, _ in LENGTH_ZONES],
        "pitch": {
            "length_m": PITCH_LENGTH_M,
            "crease_width_m": CREASE_WIDTH_M,
        },
        "calibration": calibration_metadata(),
        "trajectory_sources": {
            "detected": "Position came from a YOLO detection on that exact frame.",
            "predicted": (
                "Position is a Kalman extrapolation; there was no detection on "
                "that frame. Excluded from all physics fits."
            ),
        },
        "tracking_sources": {
            "single_pass": (
                "Produced by the one YOLO + Kalman pass over the source video."
            ),
            "clip_fallback": (
                "Produced by re-running detection on the delivery clip because "
                "the original tracking record was missing, corrupt, or had no "
                "usable trajectory. The source video is never re-read for this."
            ),
        },
        "speed_notes": [
            "speed_kmh is an ESTIMATE derived from a fixed-camera homography, "
            "not a physical measurement. See calibration.speed_formula.",
            "speed_kmh is null when it could not be computed. It is never "
            "replaced with a median or another delivery's speed.",
            "speed_imputed is false unless imputation was explicitly enabled, "
            "and the analysis config records whether it was.",
        ],
        "homography_corner_fractions": {
            k: list(v) for k, v in HOMOGRAPHY_CORNER_FRACTIONS.items()
        },
    }