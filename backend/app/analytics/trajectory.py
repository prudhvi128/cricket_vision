"""
trajectory.py — Trajectory analysis, physics, and delivery classification.

Speed calculation:
- Every detected ball position is projected from pixel-space to the real
     ground plane (metres) using the homography built in
     `analytics.calibration.build_homography`.
  - Velocity is recovered with a least-squares linear fit of ground position
    vs. real elapsed time (frame_no / fps) across the early-delivery samples,
    instead of averaging noisy frame-to-frame pixel differences. A single
    outlier-rejection pass (residuals > 2.5σ are dropped and the line is
    re-fit) keeps one bad detection from skewing the result.
  - The result is only sanity-bounded (rejecting physically-impossible
    readings caused by a bad detection), never forced into a fixed
    "realistic" window. If the data isn't good enough to trust, the function
    returns None rather than inventing a number — the caller/UI should show
    "—" for that ball instead of a fabricated speed.

Other fixes carried over from v1:
  - Bounce: improved scoring — requires local minimum in RAW Y pixel space
    AND in ground-plane Y, plus parabolic fit residual check
  - All 6 length zones, all 5 line zones
"""

import math
import numpy as np

from .calibration import pixel_to_ground

from ..core.constants import (
    LENGTH_ZONES as _LENGTH_ZONES,
    LENGTH_ZONES_M as _LENGTH_ZONES_M,
    LINE_ZONES as _LINE_ZONES,
    MAX_EARLY_POSITIONS,
    PITCH_LENGTH_M,
    CREASE_WIDTH_M,
    SPEED_CALIBRATION_OFFSET_KMH,
    SPEED_CALIBRATION_SCALE,
    SPEED_MAX_KMH,
    SPEED_MIN_KMH,
)


# ── Ported helpers (from CricketTracker tracker_modules/utils.py) ─────────────
# These two lived in a shared `utils.py` alongside the homography helpers. They
# are used only by this module, so they are defined here rather than in a
# grab-bag module: `calibration.py` owns geometry, this module owns the
# trajectory maths that consumes it.
def savgol_smooth(values: list, window: int = 7, poly: int = 2) -> list:
    """
    Smooth a 1-D list with a Savitzky-Golay-style polynomial fit.
    Falls back to the raw values if there are too few points.
    """
    n = len(values)
    if n < window:
        return list(values)
    half = window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = values[lo:hi]
        x = list(range(len(seg)))
        try:
            coeffs = np.polyfit(x, seg, poly)
            center = i - lo
            val = sum(coeffs[-(k + 1)] * (center ** k) for k in range(poly + 1))
        except Exception:
            val = values[i]
        smoothed.append(float(val))
    return smoothed


def compute_angle_deg(p1: tuple, p2: tuple) -> float:
    """Angle (degrees) of the vector from p1 to p2 relative to horizontal."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

# ── Speed config ──────────────────────────────────────────────────────────────
# MAX_EARLY_POSITIONS   use the first N real detections (never Kalman-extrapolated).
# SPEED_MIN/MAX_KMH     sanity bounds — discard implausible readings caused by a bad
#                       detection or a homography glitch. NOT used to force-fit a
#                       "nice" number.
# SPEED_CALIBRATION_*   the heuristic scale/offset applied to homography-derived
#                       speed. Persisted into analysis metadata and exposed via the
#                       API. See UNIFIED_ARCHITECTURE.md §10 decision D2.
#
# Values live in app.core.constants and are re-exported here because this module's
# public surface refers to them by these names.
__all__ = [
    "compute_release_speed",
    "detect_bounce",
    "detect_release_frame",
    "estimate_swing",
    "compute_release_angle",
    "compute_bounce_angle",
    "classify_length",
    "classify_line",
    "predict_trajectory",
    "savgol_smooth",
    "compute_angle_deg",
    "pixel_to_ground",
    "LENGTH_ZONES",
    "LINE_ZONES",
    "MAX_EARLY_POSITIONS",
    "SPEED_MIN_KMH",
    "SPEED_MAX_KMH",
    "SPEED_CALIBRATION_SCALE",
    "SPEED_CALIBRATION_OFFSET_KMH",
    "PITCH_LENGTH_M",
    "CREASE_WIDTH_M",
]


def compute_release_speed(
    early_positions: list,   # list of ((px, py), frame_no)
    fps: float,
    H: np.ndarray,
    debug: bool = False,
    apply_scale_offset: bool = True,
) -> float | None:
    """
    Compute bowling speed (km/h) from the early segment of the delivery.

    Method:
    1. Project every (px, py) to ground-plane (gx, gy) metres via the
       pixel→ground homography.
    2. Convert frame numbers to real elapsed time using the actual frame
       index deltas (handles dropped/duplicate frames correctly).
    3. Fit gx(t) and gy(t) independently with ordinary least squares —
       the slope of each fit *is* the velocity component in that axis.
       This uses every sample at once, so it is far less sensitive to a
       single noisy detection than averaging pairwise frame-to-frame speeds.
    4. Re-fit once after dropping residual outliers (>2.5σ) — guards
       against one mis-detected frame distorting the whole estimate.
    5. speed = hypot(vx, vy) * 3.6 → km/h.

    `apply_scale_offset` is the ONLY place the upstream 2x/-20 heuristic lives.
    It is a correction for a homography fitted to the wrong pitch (the fixed
    broadcast template), where it is applied by default. `analyse_delivery`
    passes False whenever the geometry was measured on THIS video, because a
    measured quad already has the right scale and applying a hand-tuned
    multiplier to it would corrupt a real number.

    Returns None if there isn't enough reliable data, or if the resulting
    speed is outside the physically-plausible sanity range — it never
    clamps/forces a result into a fixed window.
    """
    if len(early_positions) < 3:
        return None

    # ── Project to ground plane + build a real time axis ──────────────────────
    ts, gxs, gys = [], [], []
    for (px, py), f in early_positions:
        gx, gy = pixel_to_ground((px, py), H)
        ts.append(f / fps)
        gxs.append(gx)
        gys.append(gy)

    ts  = np.asarray(ts,  dtype=np.float64)
    gxs = np.asarray(gxs, dtype=np.float64)
    gys = np.asarray(gys, dtype=np.float64)

    # Drop duplicate-timestamp samples (defensive — would break the fit)
    keep = np.concatenate(([True], np.diff(ts) > 1e-6))
    ts, gxs, gys = ts[keep], gxs[keep], gys[keep]
    if len(ts) < 3:
        return None

    t0 = ts[0]
    t_rel = ts - t0   # fit in relative time for numerical stability

    def _fit_slope(t: np.ndarray, vals: np.ndarray) -> float:
        """Least-squares slope of vals vs t, with one outlier-rejection pass."""
        A = np.vstack([t, np.ones_like(t)]).T
        slope, intercept = np.linalg.lstsq(A, vals, rcond=None)[0]
        resid = vals - (slope * t + intercept)
        std = resid.std()
        if std > 1e-9:
            mask = np.abs(resid) < 2.5 * std
            if 2 <= mask.sum() < len(t):
                slope, intercept = np.linalg.lstsq(A[mask], vals[mask], rcond=None)[0]
        return float(slope)

    vx = _fit_slope(t_rel, gxs)   # m/s
    vy = _fit_slope(t_rel, gys)   # m/s
    speed_mps = math.hypot(vx, vy)

    # ── CALIBRATION WARNING ──────────────────────────────────────────────────
    # Raw homography-derived speed reads ~60-80 km/h under broadcast perspective
    # when the homography is the fixed template: one calibrated homography cannot
    # fully absorb the camera's foreshortening. Upstream corrects this with a
    # fixed 2x scale and a -20 km/h offset, tuned by hand against broadcast
    # readings of 130-150 km/h.
    #
    # THAT HEURISTIC IS NOT A PHYSICAL MEASUREMENT and it is only correct for the
    # geometry it was tuned against. When `apply_scale_offset` is False the
    # caller has stated that the homography measures THIS video's pitch, so the
    # fit is reported as metres per second × 3.6 with nothing bolted on. Either
    # way the result is an ESTIMATE (the ball is in the air, not on the ground
    # plane the homography describes) and is never marked calibrated. Scale and
    # offset are persisted into every analysis's calibration block and surfaced
    # through the API so no consumer can mistake either form for a reading.
    # See UNIFIED_ARCHITECTURE.md §10 decision D2.
    # ──────────────────────────────────────────────────────────────────────────
    if apply_scale_offset:
        kmph = speed_mps * 3.6 * SPEED_CALIBRATION_SCALE + SPEED_CALIBRATION_OFFSET_KMH
    else:
        kmph = speed_mps * 3.6

    if debug:
        print(
            f"  [SPEED] n={len(ts)} samples  vx={vx:.2f} vy={vy:.2f} m/s  "
            f"→ {kmph:.1f} km/h"
        )

    if not (SPEED_MIN_KMH < kmph < SPEED_MAX_KMH):
        if debug:
            print(
                f"  [SPEED] {kmph:.1f} km/h outside plausible range "
                f"[{SPEED_MIN_KMH}-{SPEED_MAX_KMH}] — discarding "
                f"(check camera framing / homography calibration)"
            )
        return None

    return round(kmph, 1)


# ── Bounce detection ──────────────────────────────────────────────────────────
def detect_bounce(traj_list: list, H: np.ndarray) -> int | None:
    """
    Robust bounce detection combining:
      1. Local minimum in RAW image-space Y (simplest, most reliable)
      2. Local minimum in ground-plane Y (perspective corrected)
      3. Sign change in vertical velocity (falling → rising)
      4. Horizontal-speed drop (energy lost at bounce)
      5. Parabolic fit residual — the ball arc should fit a parabola
         with a clear minimum in the window

    Returns best index into traj_list, or None.
    """
    n = len(traj_list)
    if n < 9:
        return None

    # Raw pixel Y values
    py_raw = [pt[1] for pt in traj_list]
    px_raw = [pt[0] for pt in traj_list]

    # Ground-plane values
    gy_raw = [pixel_to_ground(pt, H)[1] for pt in traj_list]
    gx_raw = [pixel_to_ground(pt, H)[0] for pt in traj_list]

    # Smooth ground-Y
    gy_sm  = savgol_smooth(gy_raw, window=7, poly=2)
    py_sm  = savgol_smooth(py_raw, window=5, poly=2)

    vy_gnd = [gy_sm[i+1] - gy_sm[i] for i in range(len(gy_sm)-1)]

    best_idx, best_score = None, -1

    for i in range(3, n - 3):
        score = 0

        # (1) Local min in raw pixel Y — in cricket side-view, bounce = highest Y pixel
        if (py_raw[i] >= py_raw[i-1] and py_raw[i] >= py_raw[i-2] and
                py_raw[i] >= py_raw[i+1] and py_raw[i] >= py_raw[i+2]):
            score += 2   # weighted more — most reliable signal

        # (2) Local min in smoothed ground-Y
        if (gy_sm[i] <= gy_sm[i-1] and gy_sm[i] <= gy_sm[i-2] and
                gy_sm[i] <= gy_sm[i+1] and gy_sm[i] <= gy_sm[i+2]):
            score += 1

        # (3) Sign change in vertical ground velocity (falling → rising)
        if i < len(vy_gnd):
            if vy_gnd[i-1] < -0.01 and vy_gnd[i] > -0.005:
                score += 2

        # (4) Horizontal-speed drop
        if i >= 2 and i + 2 < len(gx_raw):
            hb = abs(gx_raw[i]   - gx_raw[i-2])
            ha = abs(gx_raw[i+2] - gx_raw[i])
            if ha < hb * 1.30:
                score += 1

        # (5) Parabolic fit: fit quadratic over window i-4:i+4, min should be near i
        win = 4
        lo_w, hi_w = max(0, i-win), min(n, i+win+1)
        if hi_w - lo_w >= 5:
            xs  = list(range(hi_w - lo_w))
            ys  = py_sm[lo_w:hi_w]
            try:
                coeffs = np.polyfit(xs, ys, 2)
                if coeffs[0] < 0:          # concave down in pixel Y = maximum = bounce
                    score += 1
            except Exception:
                pass

        if score >= 3 and score > best_score:
            best_score = score
            best_idx   = i

    return best_idx


# ── Release point detection ───────────────────────────────────────────────────
def detect_release_frame(delivery_positions: list) -> int:
    if len(delivery_positions) < 3:
        return 0
    quarter = max(1, len(delivery_positions) // 4)
    best_i, best_d = 0, 0.0
    for i in range(1, quarter):
        (px1, py1), _ = delivery_positions[i - 1]
        (px2, py2), _ = delivery_positions[i]
        d = math.hypot(px2 - px1, py2 - py1)
        if d > best_d:
            best_d = d
            best_i = i - 1
    return best_i


# ── Swing / seam estimation ───────────────────────────────────────────────────
def estimate_swing(traj_list: list, bounce_idx: int | None, H: np.ndarray) -> str:
    if bounce_idx is None or bounce_idx < 2 or bounce_idx >= len(traj_list) - 2:
        return "Straight"
    gx0, _ = pixel_to_ground(traj_list[0], H)
    gxb, _ = pixel_to_ground(traj_list[bounce_idx], H)
    drift_m = gxb - gx0
    angle   = round(abs(drift_m) / 0.15, 1)
    if angle < 0.4:
        return "Straight"
    return f"{angle}° outswing" if drift_m > 0 else f"{angle}° inswing"


# ── Release angle ─────────────────────────────────────────────────────────────
def compute_release_angle(delivery_positions: list) -> float | None:
    if len(delivery_positions) < 4:
        return None
    (px1, py1), _ = delivery_positions[0]
    (px2, py2), _ = delivery_positions[3]
    angle = -compute_angle_deg((px1, py1), (px2, py2))
    return round(angle, 1)


# ── Bounce angle ──────────────────────────────────────────────────────────────
def compute_bounce_angle(traj_list: list, bounce_idx: int | None, H: np.ndarray) -> float | None:
    if bounce_idx is None or bounce_idx < 2 or bounce_idx + 2 >= len(traj_list):
        return None
    g1x, g1y = pixel_to_ground(traj_list[bounce_idx - 2], H)
    gbx, gby = pixel_to_ground(traj_list[bounce_idx], H)
    g2x, g2y = pixel_to_ground(traj_list[bounce_idx + 2], H)
    v_in  = (gbx - g1x, gby - g1y)
    mag_in = math.hypot(*v_in) + 1e-9
    angle_in = math.degrees(math.asin(min(1.0, abs(v_in[1]) / mag_in)))
    return round(angle_in, 1)


# ── Delivery classification ───────────────────────────────────────────────────
# Zone tables live in app.core.constants so this module classifies without
# duplicating them (see UNIFIED_ARCHITECTURE.md §5.3 and §9.1). Re-exported here
# for this module's historical importers.
LENGTH_ZONES = _LENGTH_ZONES
LINE_ZONES = _LINE_ZONES


def classify_length(bounce_y: float, frame_height: int) -> str:
    """PORTED RULE, image space. Kept for the upstream contract; see
    `classify_length_m` for the rule the analytics actually publish."""
    y_frac = bounce_y / frame_height
    for label, lo, hi in LENGTH_ZONES:
        if lo <= y_frac < hi:
            return label
    return "Full"


def classify_length_m(distance_from_crease_m: float | None) -> str | None:
    """
    Zone label for a bounce measured in METRES from the batting crease.

    This is the classification the API publishes, and it only ever runs on
    geometry that was measured on this video — see `LENGTH_ZONES_M` in
    `core/constants.py` for the thresholds themselves.

    Returns None rather than the nearest zone when the distance is missing or
    falls outside the measured pitch (0..PITCH_LENGTH_M). A bounce that cannot
    be placed on the pitch has no length, and clamping it to an edge zone would
    turn "we do not know" into a confident wrong answer.
    """
    if distance_from_crease_m is None:
        return None
    d = float(distance_from_crease_m)
    if not math.isfinite(d) or d < 0.0 or d > PITCH_LENGTH_M:
        return None
    for label, lo, hi in _LENGTH_ZONES_M:
        if lo <= d < hi:
            return label
    # Exactly at the far edge of the pitch: the last zone is the closed one.
    return _LENGTH_ZONES_M[-1][0]


def classify_line(bounce_x: float, frame_width: int) -> str:
    x_frac = bounce_x / frame_width
    for label, lo, hi in LINE_ZONES:
        if lo <= x_frac < hi:
            return label
    return "Middle"


# ── Ballistic trajectory prediction ──────────────────────────────────────────
def predict_trajectory(
    current_pos: tuple,
    velocity_px: tuple,
    n_steps: int = 30,
    gravity_px: float = 0.5,
) -> list:
    x, y   = float(current_pos[0]), float(current_pos[1])
    vx, vy = float(velocity_px[0]), float(velocity_px[1])
    points = []
    for _ in range(n_steps):
        x  += vx
        y  += vy
        vy += gravity_px
        points.append((int(x), int(y)))
    return points
