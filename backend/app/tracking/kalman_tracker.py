"""
tracker.py — Kalman-filter ball tracker with occlusion handling.

THE ONLY BALL TRACKER IN THE SYSTEM (UNIFIED_ARCHITECTURE.md §1 / C2).

Features:
  - Constant-velocity Kalman Filter (4-state: x, y, vx, vy)
  - Graceful handling of missed detections (predict-only up to MAX_MISSED)
  - False-positive rejection via gating (Mahalanobis-like distance)
  - Automatic reset when the ball is lost for too long

SCOPE: THIS IS A SMOOTHER, NOT THE SEGMENTER
--------------------------------------------
This filter answers "where is the ball right now". It does NOT decide where one
delivery ends and the next begins — `event_segmenter` does, from the provenance
recorded here. The separation is deliberate:

  * `MAX_MISSED_FRAMES` bounds how long this filter will extrapolate. It is a
    statement about how stale a Kalman estimate may get, NOT a rule about how
    long a delivery lasts.
  * A ball hidden behind the bat for longer than MAX_MISSED_FRAMES reappears via
    `reseed()`, which restarts the filter on the visible ball while the
    DELIVERY continues. Without that, one stroke gets cut in half.

CHANGES FROM UPSTREAM
---------------------
1. `update()` returns a TrackResult carrying `source`, so callers can persist
   whether each point is a real detection or a Kalman extrapolation, and
   `gated_out`, so a refused measurement is distinguishable from an empty frame.

2. A gate-rejected detection now advances `missed_frames`. Upstream left it
   frozen, which let a stream of detections belonging to the NEXT ball keep the
   previous delivery alive indefinitely — a direct cause of two batting events
   being merged into one delivery clip.

`delivery_positions` semantics are deliberately UNCHANGED: it still records
     detection-confirmed frames only, because compute_release_speed must never be
     fitted to extrapolated points. The fuller per-delivery point list including
     extrapolations is accumulated by the caller, not by this class.

3. The association gate is now applied only while the track is still healthy —
   `self.kf.initialized and self.missed_frames <= MAX_MISSED_FRAMES`. Once the
   filter has declared the track lost it no longer refuses measurements on the
   strength of a prediction it has disowned; a new detection re-initialises the
   filter through the pre-existing cold-start branch. This was the single change
   behind 195 gate refusals on real_cricket.mp4, 63% of which occurred while the
   filter already reported itself lost. GATE_RADIUS is untouched, no threshold
   was relaxed, and healthy-track behaviour is bit-identical.
"""

from typing import NamedTuple, Optional

import numpy as np
from collections import deque

from ..core.constants import GATE_RADIUS, MAX_MISSED, MAX_TRAIL

# ── Config ────────────────────────────────────────────────────────────────────
MAX_TRAIL_POINTS   = MAX_TRAIL       # maximum trajectory points to keep
MAX_MISSED_FRAMES  = MAX_MISSED      # frames without detection before declaring ball lost
GATE_RADIUS_PX     = GATE_RADIUS     # pixel radius — detections further away are rejected


class TrackResult(NamedTuple):
    """
    Outcome of one tracker update.

    position : (x, y) in full-resolution source pixel space, or None if lost.
    source   : "detected"  — position came from a YOLO detection this frame.
               "predicted" — position is a Kalman extrapolation (no detection).
               None        — ball is lost, no position.
    gated_out: True when a real detection existed this frame but fell outside
               GATE_RADIUS and was therefore refused. The position returned is
               then the previous estimate, and `source` is "predicted" — the
               measurement was seen and deliberately not used. Reported so the
               segmenter can tell "nothing in frame" apart from "something in
               frame that is not where I expected", which are different events.
    """
    position: Optional[tuple]
    source: Optional[str]
    gated_out: bool = False


class KalmanBallTracker:
    """
    4-state Kalman Filter: [x, y, vx, vy].
    Designed for a cricket ball moving under near-constant velocity between
    frames (ignoring gravity within a single-frame step).
    """

    def __init__(self):
        # State vector: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float32)

        # State transition (constant-velocity model)
        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = 1.0   # x += vx
        self.F[1, 3] = 1.0   # y += vy

        # Measurement matrix (we observe x, y only)
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance — larger = trust measurement more
        self.Q = np.diag([5.0, 5.0, 20.0, 20.0]).astype(np.float32)

        # Measurement noise covariance
        self.R = np.diag([10.0, 10.0]).astype(np.float32)

        # Estimate covariance
        self.P = np.eye(4, dtype=np.float32) * 500.0

        self.initialized = False

    def init(self, pt: tuple):
        """Seed the filter with the first observed position."""
        self.state = np.array([[pt[0]], [pt[1]], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 500.0
        self.initialized = True

    def predict(self) -> tuple:
        """Time-update step. Returns predicted (x, y)."""
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q
        return int(self.state[0, 0]), int(self.state[1, 0])

    def update(self, pt: tuple):
        """Measurement-update step."""
        z = np.array([[pt[0]], [pt[1]]], dtype=np.float32)
        y = z - self.H @ self.state                  # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.state = self.state + K @ y
        self.P     = (np.eye(4) - K @ self.H) @ self.P

    def position(self) -> tuple:
        """Current estimated position (x, y)."""
        return int(self.state[0, 0]), int(self.state[1, 0])

    def velocity(self) -> tuple:
        """Current estimated velocity (vx, vy) in pixels/frame."""
        return float(self.state[2, 0]), float(self.state[3, 0])


class BallTracker:
    """
    High-level tracker that wraps KalmanBallTracker and manages:
      - trajectory buffer
      - missed-frame counter
      - per-delivery position accumulator
    """

    def __init__(self):
        self.kf             = KalmanBallTracker()
        self.trajectory     = deque(maxlen=MAX_TRAIL_POINTS)
        self.missed_frames  = 0
        self.in_delivery    = False
        self.delivery_positions: list[tuple] = []   # [(pixel_pt, frame_no)] detections only

    # ── Per-frame update ──────────────────────────────────────────────────────
    def update(self, detection: tuple | None, frame_no: int) -> TrackResult:
        """
        Feed one frame's detection result.

        Returns a TrackResult(position, source).

        `source` is "detected" when the position derives from this frame's YOLO
        box and "predicted" when it is a Kalman extrapolation. Callers persist
        this so a trajectory never silently presents an estimate as a
        measurement.

        NOTE: when a detection exists but is rejected by the gate, the returned
        position is the previous Kalman estimate — that is an extrapolation of
        an ignored measurement, and is reported as "predicted".
        """
        if detection is not None:
            # Gate: ignore detections far from the current estimate.
            #
            # The gate may only be applied while this filter still stands behind
            # its prediction. `update()` already declares the track lost once
            # `missed_frames` exceeds MAX_MISSED_FRAMES — it then returns
            # (None, None) for every empty frame, i.e. "I have no ball". Holding
            # `kf.initialized` true anyway meant the NEXT detection was compared
            # against that disowned prediction and refused as a false positive.
            #
            # That was a livelock, not a rejection policy. Because a refusal also
            # advances `missed_frames`, and `missed_frames` only resets on an
            # ACCEPTED detection, the refusal actively prevented its own escape:
            # the counter climbed 9 -> 40 (MAX_MISSED is 8) and could never come
            # back down. Measured over real_cricket.mp4: 123 of 195 refusals
            # happened while the filter already reported itself lost, and 0 of
            # 116 gated frames were followed by an accepted one. Recovery was
            # possible only when the coast ceiling closed the event and
            # `reset_delivery()` ran.
            #
            # So: once the track is lost, fall through to the ordinary cold-start
            # path below and re-initialise on the new detection. `kf.init()`
            # seeds from a real measured box, so provenance stays truthful
            # (`source` is "detected" and the point enters `delivery_positions`).
            # Nothing is fabricated, and a lone spurious seed still cannot become
            # a delivery — it must be corroborated by MIN_DELIVERY_FRAMES.
            if self.kf.initialized and self.missed_frames <= MAX_MISSED_FRAMES:
                px, py = self.kf.predict()
                dist = ((detection[0] - px) ** 2 + (detection[1] - py) ** 2) ** 0.5
                if dist > GATE_RADIUS_PX:
                    # Likely a false positive — skip update but keep prediction.
                    #
                    # `missed_frames` MUST still advance here. It previously did
                    # not, and that was a live path to merging two batting events
                    # into one delivery: a stream of detections belonging to the
                    # NEXT ball all sit beyond the gate, so each one left
                    # `missed_frames` frozen at whatever it was, the track never
                    # died, `in_delivery` never cleared, and no boundary was ever
                    # offered to the clip writer. A refused measurement is
                    # evidence AGAINST the current track, not evidence for it.
                    self.missed_frames += 1
                    smoothed = self.kf.position()
                    self.trajectory.append(smoothed)
                    return TrackResult(smoothed, "predicted", gated_out=True)
                self.kf.update(detection)
            else:
                self.kf.init(detection)

            self.missed_frames = 0
            pos = self.kf.position()
            self.trajectory.append(pos)

            # Delivery accumulation — DETECTED FRAMES ONLY.
            # Kalman extrapolations deliberately never enter this list.
            if not self.in_delivery:
                self.in_delivery = True
                self.delivery_positions = []
            self.delivery_positions.append((pos, frame_no))

            return TrackResult(pos, "detected")

        else:
            # No detection — predict-only
            self.missed_frames += 1
            if self.kf.initialized and self.missed_frames <= MAX_MISSED_FRAMES:
                pos = self.kf.predict()
                self.trajectory.append(pos)
                return TrackResult(pos, "predicted")
            else:
                # Ball truly lost
                return TrackResult(None, None)

    def reseed(self, pt: tuple, frame_no: int) -> TrackResult:
        """
        Re-acquire the ball WITHOUT ending the delivery.

        WHY THIS EXISTS
        ---------------
        `MAX_MISSED_FRAMES` bounds how long the filter will extrapolate, and once
        it expires `update()` returns no position at all. That is correct for the
        filter but wrong for the DELIVERY: a ball passing behind the batter's
        bat, or behind a fielder's arm, routinely disappears for longer than
        that and then comes straight back. Treated as a lost ball, that ends one
        delivery and starts another, so a single stroke is cut in half.

        Event identity is therefore owned by `event_segmenter`, not by this
        filter. When the segmenter decides the ball was never lost — the gap was
        short and the returning position was reachable — it calls `reseed` to
        restart the filter on the ball that is actually visible, and the event
        continues uninterrupted.

        Provenance is unaffected: this IS a real detection, so the returned
        source is "detected" and the position goes into `delivery_positions`.
        The filter's velocity is re-seeded at zero and re-learns over the next
        few frames, which is why the segmenter fits its reference velocity over
        a window rather than trusting the filter.
        """
        self.kf.init(pt)
        self.missed_frames = 0
        pos = self.kf.position()
        self.trajectory.append(pos)
        if not self.in_delivery:
            self.in_delivery = True
            self.delivery_positions = []
        self.delivery_positions.append((pos, frame_no))
        return TrackResult(pos, "detected")

    def reset_delivery(self):
        """Call after a delivery is finalised to reset delivery state."""
        self.in_delivery = False
        self.delivery_positions = []
        self.trajectory.clear()
        self.kf.initialized = False
        self.missed_frames  = 0

    @property
    def is_lost(self) -> bool:
        return self.missed_frames > MAX_MISSED_FRAMES

    @property
    def traj_points(self) -> list:
        return list(self.trajectory)
