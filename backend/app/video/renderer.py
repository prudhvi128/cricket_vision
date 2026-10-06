"""
renderer.py — Broadcast-style cricket ball trajectory renderer.

Design goals (matching IPL DRS / Hawk-Eye aesthetic):
  - Glowing cyan/blue trail with tapering thickness + bloom glow
  - Orange/yellow dashed PREDICTED arc
  - Glowing green BOUNCE marker with ripple rings
  - Blue RELEASE marker
  - Red IMPACT / danger-zone marker
  - Semi-transparent dark HUD bar (speed, length, line, swing)
  - Mini pitch-map inset (bottom-right corner)
  - Darker vignette so the trail stands out against the pitch
"""

import cv2
import numpy as np
import math


# ─── COLOUR PALETTE (BGR) ────────────────────────────────────────────────────
C_TRAIL_CORE   = (255, 220,  30)   # bright cyan-yellow — newest segment
C_TRAIL_MID    = (120, 200, 255)   # light blue
C_TRAIL_OLD    = ( 40, 100, 255)   # deeper blue — oldest
C_GLOW_OUTER   = (200,  80,  20)   # orange outer glow

C_RELEASE      = (255, 160,  20)   # blue-white release dot
C_BOUNCE       = ( 30, 255,  80)   # bright green bounce
C_IMPACT       = (  0,  40, 255)   # red impact / danger
C_PREDICT      = (  0, 220, 255)   # yellow predicted arc
C_BALL         = (220, 255, 255)   # ball highlight

# HUD colours
C_HUD_BG       = (  0,   0,   0)
C_HUD_FAST     = (  0,  50, 255)   # red for 145+
C_HUD_MED      = (  0, 160, 255)   # orange 130-145
C_HUD_SLOW     = (  0, 230, 255)   # yellow <130

# Length badge colours (BGR)
_LENGTH_COLORS_BGR = {
    "Beamer":       ( 50,  20, 200),
    "Bouncer":      ( 50,  50, 220),
    "Short":        (220,  80,   0),
    "Good Length":  (  0, 180,  50),
    "Full":         (  0, 150, 255),
    "Yorker":       (  0,   0, 255),
    "Unknown":      (140, 140, 140),
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _trail_color(frac: float) -> tuple:
    """
    0 = oldest (deep blue) → 0.5 = mid (cyan) → 1 = newest (bright yellow-cyan)
    """
    if frac < 0.5:
        return _lerp(C_TRAIL_OLD, C_TRAIL_MID, frac * 2)
    return _lerp(C_TRAIL_MID, C_TRAIL_CORE, (frac - 0.5) * 2)


def _glow_circle(frame, center, radius, color, intensity=0.9, blur_k=21):
    """
    Draw a single glowing filled circle.

    Performance note: the blur is applied to a small patch around the circle
    (not the whole frame) — the result is pixel-identical to blurring the
    full frame (the kernel never reaches outside the padded patch) but is
    dramatically cheaper on HD/4K frames, since Gaussian blur cost scales
    with image area.
    """
    h, w = frame.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    pad = radius + blur_k
    x0, x1 = max(0, cx - pad), min(w, cx + pad)
    y0, y1 = max(0, cy - pad), min(h, cy + pad)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    ov  = np.zeros_like(roi)
    cv2.circle(ov, (cx - x0, cy - y0), radius, color, -1, cv2.LINE_AA)
    blurred = cv2.GaussianBlur(ov, (blur_k | 1, blur_k | 1), 0)
    cv2.addWeighted(roi, 1.0, blurred, intensity, 0, roi)
    frame[y0:y1, x0:x1] = roi


def _glow_line(frame, p1, p2, color, thickness, blur_k=9, intensity=0.7):
    """Draw a single glowing line segment."""
    h, w = frame.shape[:2]
    ov   = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.line(ov, p1, p2, color, thickness, cv2.LINE_AA)
    blurred = cv2.GaussianBlur(ov, (blur_k | 1, blur_k | 1), 0)
    cv2.addWeighted(frame, 1.0, blurred, intensity, 0, frame)
    cv2.line(frame, p1, p2, color, max(1, thickness // 2), cv2.LINE_AA)


# ─── TRAJECTORY TRAIL ────────────────────────────────────────────────────────
# Hawk-Eye / DRS broadcast style: single uniform bright line with a soft
# outer glow — no colour gradient, no tapered thickness, no bloom explosion.
C_LINE_CORE  = (255, 255, 255)   # pure white core  (BGR)
C_LINE_GLOW  = (200, 220, 255)   # very pale blue glow  (BGR)

def draw_trajectory_trail(
    frame: np.ndarray,
    traj_points: list,
    max_thickness: int = 8,      # kept for API compatibility, not used
) -> np.ndarray:
    """
    Clean uniform tracking line — Hawk-Eye / DRS professional style.

    Two passes over a tight ROI (same bounding-box optimisation as before):
      Pass 1: soft pale-blue glow (wide, low opacity)
      Pass 2: sharp white 2-px core line

    No colour gradients, no tapered widths, no orange bloom.
    """
    if len(traj_points) < 2:
        return frame

    h, w = frame.shape[:2]

    # ── ROI bounding box around the trail ────────────────────────────────────
    xs  = [int(p[0]) for p in traj_points]
    ys  = [int(p[1]) for p in traj_points]
    pad = 24
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
    rw, rh = x1 - x0, y1 - y0
    if rw <= 0 or rh <= 0:
        return frame

    def _shift(pt):
        return (int(pt[0]) - x0, int(pt[1]) - y0)

    roi = frame[y0:y1, x0:x1]

    # ── Pass 1: soft outer glow (pale blue, blurred) ──────────────────────────
    glow_ov = np.zeros((rh, rw, 3), dtype=np.uint8)
    pts = [_shift(p) for p in traj_points]
    for i in range(1, len(pts)):
        cv2.line(glow_ov, pts[i-1], pts[i], C_LINE_GLOW, 6, cv2.LINE_AA)
    blurred = cv2.GaussianBlur(glow_ov, (11, 11), 0)
    cv2.addWeighted(roi, 1.0, blurred, 0.60, 0, roi)

    # ── Pass 2: sharp white core line (uniform 2 px) ──────────────────────────
    for i in range(1, len(pts)):
        cv2.line(roi, pts[i-1], pts[i], C_LINE_CORE, 2, cv2.LINE_AA)

    frame[y0:y1, x0:x1] = roi

    # ── Small white dot at current ball position (no big glow explosion) ──────
    cx, cy = int(traj_points[-1][0]), int(traj_points[-1][1])
    cv2.circle(frame, (cx, cy), 5, C_LINE_CORE, -1, cv2.LINE_AA)

    return frame


# ─── BOUNCE MARKER ───────────────────────────────────────────────────────────
def draw_bounce_marker(frame: np.ndarray, pt: tuple) -> np.ndarray:
    """
    Green glowing bounce marker with:
    - 3 expanding ripple rings (simulating a radar ping)
    - Solid bright centre dot
    - 'BOUNCE' label
    """
    x, y = int(pt[0]), int(pt[1])
    h, w = frame.shape[:2]

    # Ripple rings (outer → inner)
    for r, alpha in [(28, 0.20), (18, 0.45), (11, 0.80)]:
        ov = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(ov, (x, y), r, C_BOUNCE, 2, cv2.LINE_AA)
        blurred = cv2.GaussianBlur(ov, (15, 15), 0)
        cv2.addWeighted(frame, 1.0, blurred, alpha, 0, frame)

    # Filled glow core
    _glow_circle(frame, (x, y), 10, C_BOUNCE, intensity=0.9, blur_k=15)
    cv2.circle(frame, (x, y), 7,  C_BOUNCE,        -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 4,  (210, 255, 210),  -1, cv2.LINE_AA)

    # Label with shadow
    lx, ly = x + 12, y - 10
    cv2.putText(frame, "PITCH POINT", (lx+1, ly+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "PITCH POINT", (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BOUNCE, 2, cv2.LINE_AA)
    return frame


# ─── RELEASE MARKER ──────────────────────────────────────────────────────────
def draw_release_marker(frame: np.ndarray, pt: tuple) -> np.ndarray:
    """Blue-orange glowing release marker."""
    x, y = int(pt[0]), int(pt[1])
    _glow_circle(frame, (x, y), 14, C_RELEASE, intensity=0.70, blur_k=17)
    cv2.circle(frame, (x, y), 8,  C_RELEASE,        -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 4,  (230, 230, 255),   -1, cv2.LINE_AA)
    lx, ly = x + 12, y - 10
    cv2.putText(frame, "RELEASE", (lx+1, ly+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "RELEASE", (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_RELEASE, 2, cv2.LINE_AA)
    return frame


# ─── PREDICTED ARC ───────────────────────────────────────────────────────────
def draw_prediction_arc(frame: np.ndarray, predicted_pts: list) -> np.ndarray:
    """
    Dashed yellow/orange arc showing the projected ball path after bounce.
    Uses thinner line and lower opacity to clearly distinguish from actual trail.
    """
    if len(predicted_pts) < 2:
        return frame

    h, w = frame.shape[:2]
    ov   = np.zeros((h, w, 3), dtype=np.uint8)

    dash_len = 5
    gap_len  = 4
    idx      = 0
    draw     = True

    while idx < len(predicted_pts) - 1:
        block = dash_len if draw else gap_len
        end   = min(idx + block, len(predicted_pts) - 1)
        if draw and end > idx:
            frac  = idx / len(predicted_pts)
            color = _lerp(C_PREDICT, (0, 140, 255), frac)  # yellow → orange fade
            cv2.line(
                ov,
                (int(predicted_pts[idx][0]),  int(predicted_pts[idx][1])),
                (int(predicted_pts[end][0]),   int(predicted_pts[end][1])),
                color, 3, cv2.LINE_AA,
            )
        idx  += block
        draw  = not draw

    blurred = cv2.GaussianBlur(ov, (9, 9), 0)
    cv2.addWeighted(frame, 1.0, blurred, 0.55, 0, frame)
    cv2.addWeighted(frame, 1.0, ov,      0.50, 0, frame)
    return frame


# ─── HUD OVERLAY ─────────────────────────────────────────────────────────────
def draw_hud(
    frame: np.ndarray,
    frame_no: int,
    fps: float,
    delivery_info: dict | None = None,
) -> np.ndarray:
    """
    Broadcast-style HUD:
    - Semi-transparent dark bar at the bottom
    - Speed badge (colour-coded by pace)
    - Length / Line / Swing labels
    - Frame + FPS counter (dim, bottom right)
    - Release / Bounce angle (top right)
    """
    h, w = frame.shape[:2]
    bar_h = 70

    # Dark bar
    roi = frame[h - bar_h:h, :]
    dark = np.zeros_like(roi)
    cv2.addWeighted(roi, 0.25, dark, 0.75, 0, roi)
    frame[h - bar_h:h, :] = roi

    # Frame counter
    cv2.putText(
        frame, f"Frame {frame_no}  |  {fps:.1f} fps",
        (w - 250, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 130, 130), 1, cv2.LINE_AA,
    )

    if not delivery_info:
        return frame

    d     = delivery_info
    speed = d.get("speed", 0.0) or 0.0

    # Speed colour
    if speed >= 145:
        spd_col = C_HUD_FAST
    elif speed >= 130:
        spd_col = C_HUD_MED
    else:
        spd_col = C_HUD_SLOW

    # Speed badge (left side)
    badge_text = f"{speed:.1f}"
    badge_x, badge_y = 20, h - bar_h + 16
    # Glow behind badge
    for tk, col in [(8, (0,0,0)), (2, spd_col)]:
        cv2.putText(frame, badge_text, (badge_x, badge_y + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, col, tk, cv2.LINE_AA)
    cv2.putText(frame, "km/h", (badge_x, badge_y + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # Delivery meta (to the right of speed)
    meta_x = 120
    meta_items = [
        ("BALL",   str(d.get("ball", "?"))),
        ("LENGTH", str(d.get("length", "?"))),
        ("LINE",   str(d.get("line", "?"))),
    ]
    for k, (label, value) in enumerate(meta_items):
        mx = meta_x + k * 160
        cv2.putText(frame, label, (mx, h - bar_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (mx, h - bar_h + 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)

    # Swing label (top left corner)
    swing = d.get("swing", "")
    if swing and swing != "Straight":
        cv2.putText(frame, f"↔ {swing}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

    # Release / bounce angles (top right)
    details = []
    if d.get("release_angle") is not None:
        details.append(f"Release: {d['release_angle']}°")
    if d.get("bounce_angle") is not None:
        details.append(f"Bounce: {d['bounce_angle']}°")
    for k, txt in enumerate(details):
        cv2.putText(frame, txt, (w - 230, 30 + k * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


# ─── MINI PITCH MAP INSET ────────────────────────────────────────────────────
def draw_mini_pitchmap(
    frame: np.ndarray,
    deliveries: list,
    pos: str = "top-right",
) -> np.ndarray:
    """
    Small pitch-map inset showing all bounce points so far.
    Placed top-right so it doesn't clash with the HUD bar.
    """
    h, w = frame.shape[:2]
    PM_W, PM_H = 90, 175
    MARGIN = 14

    if pos == "top-right":
        ox, oy = w - PM_W - MARGIN, MARGIN + 10
    elif pos == "top-left":
        ox, oy = MARGIN, MARGIN + 10
    elif pos == "bottom-right":
        ox, oy = w - PM_W - MARGIN, h - PM_H - MARGIN - 80
    else:
        ox, oy = MARGIN, h - PM_H - MARGIN - 80

    # Background
    roi  = frame[oy:oy+PM_H, ox:ox+PM_W]
    bg   = np.full_like(roi, (30, 90, 30))
    cv2.addWeighted(roi, 0.20, bg, 0.80, 0, roi)
    frame[oy:oy+PM_H, ox:ox+PM_W] = roi

    # Border
    cv2.rectangle(frame, (ox, oy), (ox+PM_W-1, oy+PM_H-1), (255, 255, 255), 1)

    # Crease lines
    for frac in [0.09, 0.14, 0.86, 0.91]:
        ly = oy + int(PM_H * frac)
        cv2.line(frame, (ox, ly), (ox+PM_W, ly), (220, 220, 220), 1)

    # Centre line
    cx = ox + PM_W // 2
    cv2.line(frame, (cx, oy + int(PM_H*0.14)), (cx, oy + int(PM_H*0.86)),
             (180, 180, 180), 1, cv2.LINE_AA)

    # Stumps (top & bottom)
    for stump_y_frac in [0.09, 0.91]:
        sy = oy + int(PM_H * stump_y_frac)
        for sx_off in [-5, 0, 5]:
            cv2.line(frame, (cx + sx_off, sy - 4), (cx + sx_off, sy + 4),
                     (255, 255, 200), 1)

    # Bounce dots
    for d in deliveries:
        if d.get("bounce_x") is None:
            continue
        bx = ox + int(d["bounce_x"] * PM_W)
        by = oy + int(d["bounce_y"] * PM_H)
        col = _LENGTH_COLORS_BGR.get(d.get("length", "Unknown"), (160, 160, 160))
        cv2.circle(frame, (bx, by), 4, col,           -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 4, (255,255,255),  1, cv2.LINE_AA)

    # Label
    cv2.putText(frame, "PITCH MAP", (ox + 4, oy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
