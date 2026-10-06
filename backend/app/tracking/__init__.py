"""
tracking — Detection, Kalman tracking, and the single decode pass that joins them.

`tracking_service.py` is the only place the source video is decoded. Per
UNIFIED_ARCHITECTURE.md §1/C2, YOLO detection and Kalman tracking each run
EXACTLY once per uploaded video; nothing downstream may reopen the source. The
two upstream-parity components it drives:

  detector.py        `BallDetector` — YOLO11, ported from CricketTracker
                     `tracker_modules/detector.py`. Divergence: reads
                     config/constants and exposes `detect_with_confidence()`.
  kalman_tracker.py  `BallTracker`, `KalmanBallTracker`, `TrackResult` — ported
                     from CricketTracker `tracker_modules/tracker.py`.
                     Divergence: `update()` returns `TrackResult` carrying
                     detected-vs-predicted provenance (§7.1).
                     `delivery_positions` semantics are unchanged.

The tracking service also owns the ring buffer, the clip writer, and the
per-delivery analytics call, because all three are only meaningful while the
decoded frame is still in hand.
"""

from .detector import BallDetector
from .kalman_tracker import BallTracker, KalmanBallTracker, TrackResult
from .tracking_service import TrackingResult, TrackingService

__all__ = [
    "BallDetector",
    "BallTracker",
    "KalmanBallTracker",
    "TrackResult",
    "TrackingResult",
    "TrackingService",
]