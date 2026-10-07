"""
bowling.py — Per-delivery bowling physics derived from the stored trajectory.

INPUT CONTRACT
--------------
Every function here takes the trajectory persisted by the tracking pass. None of
them re-run detection or tracking, and none of them re-read the source video
(UNIFIED_ARCHITECTURE.md §5, C2/C3). The overlay renderer and the API response
are handed the SAME objects these functions produced, so a bounce marker on
screen and a bounce coordinate in JSON cannot disagree.

TWO PHASE-3 DECISIONS ARE ENFORCED HERE, NOT AT THE CALL SITE
--------------------------------------------------------------
D1  Missing speed is reported as null. It is never filled from a median, from a
    neighbouring delivery, or from a prior analysis. Upstream CricketTracker
    model.py:177-188 does exactly that, at analysis level, with no way for a
    consumer to opt out. `impute_missing_speeds` exists only so the behaviour can
    be studied offline; it defaults to False, and whenever it is used the result
    is stamped `speed_imputed=True` so an imputed value can never be mistaken
    for a computed one.

D2  The upstream formula `kmph = speed_mps * 3.6 * 2.0 - 20.0` is a hand-tuned
    heuristic, not a physical measurement. It is preserved for continuity but is
    labelled an estimate everywhere it surfaces, and the scale/offset are
    persisted in the analysis calibration block, where the delivered calibration and provenance live.

SIGNATURES ARE THE UPSTREAM ONES
--------------------------------
The ported helpers use upstream's exact conventions, which are easy to get wrong:
  * `compute_release_speed(positions, fps, H)` — positions are ((x, y), frame).
  * `detect_bounce(traj_list, H)` — traj_list is bare (x, y) tuples; the return
    value is an INDEX into that list, not a frame number and not a coordinate.
  * `detect_release_frame(positions)` — also returns an INDEX.
  * `classify_line(bounce_x_px, frame_width)` — takes a PIXEL coordinate plus the
    frame dimension and normalises internally; passing ground-plane metres here
    would silently produce meaningless zones.

LENGTH IS THE ONE THAT CHANGED
------------------------------
`length` used to come from `classify_length(bounce_y_px, frame_height)` — a
fraction of the frame height, i.e. how far down the PICTURE the ball landed. It
now comes from `classify_length_m(PITCH_LENGTH_M - bounce.ground_y_m)`: metres
along the pitch from the batting crease, computed only when the geometry was
measured on this video (operator corners or the pitch-keypoint quad). Same
formula for both sources because both put y = 0 at the non-batting end and
y = PITCH_LENGTH_M at the batting end.
"""

from __future__ import annotations

import logging
import statistics
from typing import Optional

import numpy as np

from ..core.constants import MAX_EARLY_POSITIONS, PITCH_LENGTH_M
from ..schemas.tracking import Bounce, Bowling, SpeedSample, Trajectory
from .calibration import pixel_to_ground
from .pitch_geometry import PitchGeometry
from .trajectory import (
    classify_length_m,
    classify_line,
    compute_bounce_angle,
    compute_release_angle,
    compute_release_speed,
    detect_bounce,
    detect_release_frame,
    estimate_swing,
)

log = logging.getLogger(__name__)


# ── Main entry point ──────────────────────────────────────────────────────────
def analyse_delivery(
    trajectory: Trajectory,
    homography: np.ndarray,
    fps: float,
    width: int,
    height: int,
    impute_missing_speeds: bool = False,
    speeds_from: Optional[list[float]] = None,
    geometry_calibrated: bool = False,
    pitch_geometry: Optional[PitchGeometry] = None,
) -> tuple[Optional[int], Bounce, Bowling, list[SpeedSample], list[str]]:
    """
    Derive release, bounce, speed, length, line, swing and angles for one delivery.

    Parameters
    ----------
    trajectory
        The persisted trajectory. Only its detection-confirmed points feed the
        physics fits; Kalman extrapolations are excluded because fitting a slope
        to a filter's own output would launder an estimate into a measurement.
        The predicted points remain in `trajectory` for rendering.
    impute_missing_speeds
        Phase 3 decision D1. Defaults False. When False, a delivery whose speed
        cannot be computed reports null.
    speeds_from
        Speeds already computed for other deliveries in this analysis. Consulted
        ONLY when `impute_missing_speeds` is explicitly True, and the result is
        always stamped as imputed.
    geometry_calibrated
        Whether the pitch corners were measured on THIS video. When False the
        ground-plane projection is meaningless, so every quantity derived from
        it — speed, length, line, swing, bounce angle, and all metre
        coordinates — is reported null. It is NOT reported as an estimate: an
        estimate implies the number is roughly right, and a homography fitted to
        the wrong corners is not roughly right, it is confidently wrong.
        Pixel-space results (release frame, bounce pixel, release angle) are
        unaffected and still computed.

        THE DEFAULT IS `False`, not `True`. The caller MUST state whether it has
        a real per-video calibration; defaulting to calibrated makes the
        suppression path unreachable, which is exactly how a fabricated
        42-199 km/h range reached `result.json` in the first place. Callers that
        genuinely have measured pitch corners pass `geometry_calibrated=True`.
    pitch_geometry
        The delivery's own pitch-keypoint geometry (`analytics/pitch_geometry`),
        when the quad was detected on this footage. It wins over `homography`,
        because it measures THIS video's pitch and carries the batter's end —
        and when it is present the 2x/-20 speed heuristic is NOT applied, since
        a measured quad needs no hand-tuned correction. Passing it implies the
        geometry is calibrated.

    Returns
    -------
    (release_frame, bounce, bowling, speed_samples, quality_flags)
    """
    flags: list[str] = []
    detected = trajectory.detected_points()
    bowling = Bowling()
    samples: list[SpeedSample] = []

    # Which ground plane, if any, may be trusted.
    ground_homography = homography
    if pitch_geometry is not None:
        ground_homography = pitch_geometry.homography
        geometry_calibrated = True
        flags.append("ground_plane_from_pitch_keypoints")
    bowling.geometry_calibrated = bool(geometry_calibrated)

    if not geometry_calibrated:
        flags.append("pitch_calibration_required")

    # Upstream's two-list convention: `positions` carries frame numbers and is
    # what the speed and release-angle helpers consume; `traj_list` is bare
    # (x, y) and is what the bounce/swing/angle helpers consume.
    positions = [((p.x, p.y), p.original_frame) for p in detected]
    traj_list = [(p.x, p.y) for p in detected]

    if len(detected) < 3:
        bounce = Bounce(detected=False)
        flags.append("insufficient_detections")
        # Still honour an explicit imputation request: a delivery too short to
        # fit is precisely the case imputation exists for. On the default path
        # (D1) the speed simply stays null.
        if impute_missing_speeds and speeds_from:
            median = round(statistics.median(speeds_from), 1)
            bowling.speed_kmh = float(median)
            bowling.speed_method = "imputed_median"
            bowling.speed_imputed = True
            flags.append("speed_imputed")
        else:
            flags.append("speed_unavailable")
        return None, bounce, bowling, samples, flags

    # ── Release frame (advisory) ────────────────────────────────────────────
    # detect_release_frame returns an INDEX into positions, not a frame number.
    # Upstream never called it, so this is additive, not a ported behaviour.
    release_frame: Optional[int] = None
    rel_idx = detect_release_frame(positions)
    if 0 <= rel_idx < len(positions):
        release_frame = positions[rel_idx][1]
        flags.append("release_frame_heuristic")

    # ── Bounce ──────────────────────────────────────────────────────────────
    bounce_idx = detect_bounce(traj_list, ground_homography)
    bounce_method = "score"

    if bounce_idx is not None:
        bx, by = traj_list[bounce_idx]
        bounce_detected = True
    else:
        # Fallback: lowest tracked point in the image. This is a REAL tracked
        # coordinate, not a fabricated one — recorded with an explicit method so
        # a consumer can distinguish it from a score-based bounce.
        bounce_idx = max(range(len(traj_list)), key=lambda i: traj_list[i][1])
        bx, by = traj_list[bounce_idx]
        bounce_detected = False
        bounce_method = "fallback_max_y"
        flags.append("bounce_score_unavailable")

    bounce_point = detected[bounce_idx]
    # Ground-plane metres are withheld unless the pitch corners were measured on
    # this video. The PIXEL position is real tracked data and is always kept.
    gbx = gby = None
    if geometry_calibrated:
        gbx, gby = pixel_to_ground((bx, by), ground_homography)
    bounce = Bounce(
        detected=bounce_detected,
        original_frame=bounce_point.original_frame,
        x_px=int(bx),
        y_px=int(by),
        # Normalised to 0..1 of the frame — matches upstream's
        # round(bx / width, 3) so existing frontends keep working.
        x_norm=round(bx / width, 5) if width else None,
        y_norm=round(by / height, 5) if height else None,
        ground_x_m=round(gbx, 4) if gbx is not None else None,
        ground_y_m=round(gby, 4) if gby is not None else None,
        method=bounce_method,
    )

    if not geometry_calibrated:
        # Everything below projects pixels onto an assumed pitch. With the wrong
        # homography that projection is not approximately right, it is wrong in
        # an unknown direction, so nothing derived from it is reported. The
        # speed/line/length that WOULD come out of here is precisely the
        # 42-199 km/h range measured in Phase 3.
        bowling.speed_kmh = None
        bowling.speed_method = None
        bowling.speed_calibrated = False
        bowling.speed_imputed = False
        bowling.length = None
        bowling.line = None
        bowling.swing = None
        bowling.bounce_angle = None
        # Pixel-space only, so still meaningful without a homography.
        bowling.release_angle = compute_release_angle(positions)
        flags.append("ground_plane_analytics_suppressed")
        log.info(
            "Pitch calibration absent: speed, length, line, swing and bounce "
            "angle suppressed rather than estimated from an assumed homography"
        )
        return release_frame, bounce, bowling, samples, flags

    # ── Speed ───────────────────────────────────────────────────────────────
    # Two attempts, both upstream's: the first MAX_EARLY_POSITIONS detections,
    # then a shorter 6-detection fit. compute_release_speed already returns None
    # for implausible results, so there is no clamping anywhere here.
    early = positions[:MAX_EARLY_POSITIONS]
    spd = compute_release_speed(early, fps, ground_homography, apply_scale_offset=False)
    used = early if spd is not None else None

    if spd is None and len(positions) >= 4:
        shorter = positions[:6]
        spd = compute_release_speed(shorter, fps, ground_homography, apply_scale_offset=False)
        if spd is not None:
            used = shorter

    # Record exactly which detections produced the number, so the API can show
    # the evidence rather than asking a reader to trust it.
    if used:
        t0 = used[0][1]
        for (px, py), frame_no in used:
            gx, gy = pixel_to_ground((px, py), ground_homography)
            samples.append(
                SpeedSample(
                    original_frame=int(frame_no),
                    x=int(px),
                    y=int(py),
                    ground_x_m=round(float(gx), 4),
                    ground_y_m=round(float(gy), 4),
                    t_sec=round((frame_no - t0) / fps, 4) if fps > 0 else 0.0,
                )
            )

    if spd is not None:
        bowling.speed_kmh = round(float(spd), 1)
        bowling.speed_method = "homography_least_squares"
        # D2: the number exists, but it is an estimate, not a calibrated reading.
        bowling.speed_calibrated = False
        bowling.speed_imputed = False
    else:
        bowling.speed_method = None
        # ── D1: no imputation on the default path ──────────────────────────
        if impute_missing_speeds and speeds_from:
            median = round(statistics.median(speeds_from), 1)
            bowling.speed_kmh = float(median)
            bowling.speed_method = "imputed_median"
            bowling.speed_imputed = True
            flags.append("speed_imputed")
            log.warning(
                "Delivery speed imputed as median %.1f km/h "
                "(impute_missing_speeds=True)", median,
            )
        else:
            # Stays null. Reported as unavailable rather than filled in.
            bowling.speed_kmh = None
            flags.append("speed_unavailable")

    # ── Length / line / swing / angles ─────────────────────────────────────
    # Length: METRES along the pitch, from the batting crease. `ground_y_m` runs
    # 0 at the non-batting end to PITCH_LENGTH_M at the batting end for every
    # measured geometry, so one subtraction gives the distance from the crease.
    # A bounce that projects off the measured pitch gets null, not a zone.
    length_m = (PITCH_LENGTH_M - gby) if gby is not None else None
    bowling.length = classify_length_m(length_m)
    if bowling.length is None:
        flags.append("length_unavailable")

    # Line: still the ported IMAGE rule. Leg and off cannot be derived from a
    # pitch quad — the quad says nothing about which way the batter faces — and
    # inventing an orientation here would be exactly the guess this codebase
    # refuses to make.
    if width:
        bowling.line = classify_line(bx, width)

    bowling.swing = estimate_swing(traj_list, bounce_idx, ground_homography)
    bowling.release_angle = compute_release_angle(positions)
    bowling.bounce_angle = compute_bounce_angle(traj_list, bounce_idx, ground_homography)

    return release_frame, bounce, bowling, samples, flags