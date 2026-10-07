"""
pitch — Optional pitch keypoint detection and calibration.

    detector.py    one frame → one hosted model call → keypoints (or an error)
    calibration.py keypoints → a validated quad, an optional homography, smoothing
    service.py     the non-blocking, interval-gated state machine around both

Nothing in here may touch the ball detector, the tracker, or any analytic
measurement: speed, length, line and swing keep coming from the operator's
measured corners (`analytics/calibration.py`), which are a different thing
entirely. This package only answers "where is the pitch in this frame", so a
client can draw it and place the ball on it.
"""

from .calibration import PitchCalibration, PitchSmoother, build_calibration
from .detector import Keypoint, PitchDetection, RoboflowPitchDetector, parse_pitch_response
from .service import PitchService

__all__ = [
    "Keypoint",
    "PitchCalibration",
    "PitchDetection",
    "PitchService",
    "PitchSmoother",
    "RoboflowPitchDetector",
    "build_calibration",
    "parse_pitch_response",
]
