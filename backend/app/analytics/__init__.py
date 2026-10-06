"""
analytics — What can be concluded from a tracked trajectory.

Ownership split inside this package:

  calibration.py  pixel↔ground geometry, and the gate that decides whether that
                  geometry may be shown as physical measurement. Owns
                  `build_homography`, `pixel_to_ground`, `geometry_status`,
                  `calibration_metadata`.
  trajectory.py   Trajectory maths: speed, bounce, swing, release/bounce angle,
                  and length/line classification from PIXEL coordinates.
  bowling.py      Assembles the above into one per-delivery `Bowling` result.
  pitch_map.py    SVG pitch map rendering for the overlay and API response.

The distinction that matters: `trajectory.py` works in pixel space and metres
along the path; `bowling.py` decides what may be *claimed*. Neither re-runs
detection or tracking, and neither re-reads the source video.
"""

from .bowling import analyse_delivery
from .calibration import (
    build_homography,
    calibration_metadata,
    geometry_status,
    make_homography,
    pixel_to_ground,
)

__all__ = [
    "analyse_delivery",
    "build_homography",
    "calibration_metadata",
    "geometry_status",
    "make_homography",
    "pixel_to_ground",
]