"""
Why is there no metric geometry? Answer it with the artefact, twice.

    python scripts/verification/compare_geometry_gate.py [run_dir]

For every delivery in a run's tracking.json the metric gate
(`analytics/pitch_geometry.build_pitch_geometry`) is replayed twice:

  STORED — the quad the analysis actually used: the EMA-smoothed calibration
           persisted in `delivery.pitch.keypoints`.
  FRESH  — the seg detector's own answer for that same calibration frame,
           re-run now, unsmoothed.

Same gate, same flight points, same constants — only the quad differs. The
difference between the two tallies is exactly what temporal smoothing cost,
and the residual rejections in the FRESH column are what the model's quad
costs. Both columns are needed: a run whose FRESH column is green has a
smoothing problem, a run whose FRESH column is red has a detector problem.

No network, no pipeline run: reads one run directory and the shipped weights.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.analytics.pitch_geometry import (  # noqa: E402
    AXIS_U,
    AXIS_V,
    PITCH_GEOMETRY_ASPECT_MIN_RATIO,
    build_pitch_geometry,
    flight_to_bounce,
)
from app.core.config import PITCH_SEG_CONFIG, PitchConfig  # noqa: E402
from app.core.constants import (  # noqa: E402
    PITCH_GEOMETRY_CROP_BAND_FRACTION,
    PITCH_GEOMETRY_TRAVEL_AXIS_MAX_DEG,
)
from app.pitch.calibration import PitchCalibration, build_calibration, build_homography  # noqa: E402
from app.pitch.segmentation_detector import LocalPitchSegDetector  # noqa: E402

IMAGE_SIZE = (1280, 720)


def stored_calibration(pitch: dict) -> PitchCalibration | None:
    corners = pitch.get("keypoints") or pitch.get("corners")
    if not corners or len(corners) < 4:
        return None
    named = {k: (float(v[0]), float(v[1])) for k, v in corners.items()}
    homography = build_homography(named, IMAGE_SIZE)
    # The delivery records the detector as `model_id` ("pitch_seg:…"), not as a
    # source token, so the gate's own provenance is re-derived from it here.
    model_id = str(pitch.get("model_id") or "")
    source = str(pitch.get("source") or "") or (
        "pitch_segmentation" if model_id.startswith("pitch_seg") else "pitch_keypoints"
    )
    return PitchCalibration(
        frame_number=int(pitch.get("frame_number") or 0),
        image_size=IMAGE_SIZE,
        keypoints=named,
        corners=named,
        confidence=pitch.get("confidence"),
        homography=homography,
        source=source,
    )


def flight_for(delivery: dict) -> list[tuple[float, float]]:
    detected = [
        (float(p["x"]), float(p["y"]))
        for p in ((delivery.get("trajectory") or {}).get("points") or [])
        if p.get("source") == "detected"
    ]
    return flight_to_bounce(detected)


def diagnose(calibration: PitchCalibration | None, flight: list) -> dict:
    """
    The gate's own arithmetic, laid out instead of reduced to a reason.

    Mirrors `build_pitch_geometry` in the same order: crop band → min-area-rect
    elongation → unit-square axis vote → travel-vs-long-axis direction. `oriented`
    is the rotation-invariant minAreaRect aspect; `achieved` the old axis-ALIGNED
    span ratio, kept only so a rejection whose oriented aspect is healthy can be
    seen as a measurement artefact rather than a bad quad. `bottom_gap` is the
    distance from the quad's bottom edge to the frame bottom — inside the
    scorecard band means the pitch runs out from under the overlay.
    """
    if calibration is None:
        return {"reason": "no_pitch_calibration"}
    h = float(calibration.width)
    v = float(calibration.length)
    corners = list(calibration.corners.values())
    oriented = 0.0
    gap = None
    bottom_gap = None
    crop_band = PITCH_GEOMETRY_CROP_BAND_FRACTION * IMAGE_SIZE[1]
    long_axis = None
    if len(corners) >= 4:
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        gap = min(
            min(xs), min(ys),
            IMAGE_SIZE[0] - max(xs), IMAGE_SIZE[1] - max(ys),
        )
        bottom_gap = IMAGE_SIZE[1] - max(ys)
        (_, _), (rw, rh), _ = cv2.minAreaRect(
            np.array(corners, dtype=np.float32)
        )
        if min(rw, rh) > 1e-6:
            oriented = max(rw, rh) / min(rw, rh)
        # The quad's long axis, same edge-pair averaging the gate uses.
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners
        across = ((x1 - x0) + (x2 - x3), (y1 - y0) + (y2 - y3))
        across_len = (
            math.dist((x0, y0), (x1, y1)) + math.dist((x3, y3), (x2, y2))
        ) / 2.0
        along = ((x2 - x1) + (x3 - x0), (y2 - y1) + (y3 - y0))
        along_len = (
            math.dist((x1, y1), (x2, y2)) + math.dist((x0, y0), (x3, y3))
        ) / 2.0
        vec = across if across_len >= along_len else along
        norm = math.hypot(vec[0], vec[1])
        if norm > 1e-9:
            long_axis = (vec[0] / norm, vec[1] / norm)
    uv = []
    for x, y in flight:
        mapped = calibration.image_to_pitch(float(x), float(y))
        if mapped is not None:
            uv.append(mapped)
    if len(uv) < 2:
        return {"reason": "too_few_points", "h": h, "v": v, "oriented": oriented, "gap": gap}
    du = max(p[0] for p in uv) - min(p[0] for p in uv)
    dv = max(p[1] for p in uv) - min(p[1] for p in uv)
    ratio = PITCH_GEOMETRY_ASPECT_MIN_RATIO
    base = {"h": h, "v": v, "oriented": oriented, "du": du, "dv": dv,
            "gap": gap, "bottom_gap": bottom_gap}
    # 1. crop band, before anything else about the quad's shape.
    if bottom_gap is not None and bottom_gap <= crop_band:
        return {**base, "reason": "pitch_cropped_by_overlay"}
    # 2. rotation-invariant elongation.
    if oriented < ratio:
        return {**base, "reason": "pitch_quad_aspect_disagrees_with_travel",
                "achieved": oriented, "required": ratio}
    # 3. unit-square axis vote.
    if dv >= du * ratio:
        axis = AXIS_V
    elif du >= dv * ratio:
        axis = AXIS_U
    else:
        return {**base, "reason": "ball_travel_axis_ambiguous"}
    # 4. travel vs the quad's long axis, in image space.
    travel_axis = None
    if len(flight) >= 2:
        k = max(1, len(flight) // 3)
        first, last = flight[:k], flight[-k:]
        fx = sum(p[0] for p in first) / len(first)
        fy = sum(p[1] for p in first) / len(first)
        lx = sum(p[0] for p in last) / len(last)
        ly = sum(p[1] for p in last) / len(last)
        tdx, tdy = lx - fx, ly - fy
        tnorm = math.hypot(tdx, tdy)
        if tnorm > 1e-9:
            travel_axis = (tdx / tnorm, tdy / tnorm)
    divergence = None
    if long_axis is not None and travel_axis is not None:
        divergence = math.degrees(
            math.acos(min(1.0, abs(long_axis[0] * travel_axis[0]
                                   + long_axis[1] * travel_axis[1])))
        )
    along = v if axis == AXIS_V else h
    across = h if axis == AXIS_V else v
    achieved = along / across if across > 0 else 0.0
    reason = ""
    if divergence is None or divergence > PITCH_GEOMETRY_TRAVEL_AXIS_MAX_DEG:
        reason = "pitch_quad_aspect_disagrees_with_travel"
    return {
        **base, "reason": reason, "axis": axis,
        "achieved": achieved, "required": ratio,
        "divergence": divergence,
    }


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if run is None:
        candidates = sorted(
            (ROOT / "data" / "runs").glob("*/tracking.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            print("FATAL: no run directory with tracking.json found")
            return 2
        run = candidates[-1].parent
    tracking_path = run / "tracking.json"
    if not tracking_path.is_file():
        print(f"FATAL: {tracking_path} not found")
        return 2

    tracking = json.loads(tracking_path.read_text())
    detector = LocalPitchSegDetector.from_config(PITCH_SEG_CONFIG)
    capture = cv2.VideoCapture(str(run / "source.mp4"))
    pitch_cfg = PitchConfig()

    stored_reasons: Counter[str] = Counter()
    fresh_reasons: Counter[str] = Counter()
    rows: list[tuple] = []
    stored_diag: dict[int, dict] = {}
    fresh_diag: dict[int, dict] = {}

    for delivery in tracking.get("deliveries", []):
        pitch = delivery.get("pitch") or {}
        frame = pitch.get("frame_number")
        flight = flight_for(delivery)
        delivery_id = delivery.get("delivery_id") or delivery.get("id")

        stored_cal = stored_calibration(pitch)
        _, stored_reason = build_pitch_geometry(stored_cal, flight)
        stored_reasons[stored_reason or "accepted"] += 1

        fresh_reason = "no_fresh_detection"
        fresh_cal = None
        if frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame) - 1)
            ok, image = capture.read()
            if ok:
                try:
                    detection = detector.detect(image, int(frame) - 1, IMAGE_SIZE)
                except Exception as exc:  # noqa: BLE001 — a refusal, not a crash
                    fresh_reason = f"detect_failed:{type(exc).__name__}"
                else:
                    fresh_cal, why = build_calibration(
                        detection, cfg=pitch_cfg, image_size=IMAGE_SIZE
                    )
                    if fresh_cal is None:
                        fresh_reason = f"calibration_refused:{why}"
                    else:
                        _, fresh_reason = build_pitch_geometry(fresh_cal, flight)
        fresh_reasons[fresh_reason or "accepted"] += 1
        rows.append((delivery_id, frame, stored_reason or "accepted", fresh_reason or "accepted"))
        stored_diag[delivery_id] = diagnose(stored_cal, flight)
        fresh_diag[delivery_id] = diagnose(fresh_cal, flight)

    print(f"run: {run}")
    print(f"deliveries: {len(rows)}")
    print()
    print(f"{'id':>4} {'frame':>7} {'stored quad':<44} {'fresh quad'}")
    for delivery_id, frame, stored, fresh in rows:
        print(f"{delivery_id:>4} {str(frame):>7} {stored:<44} {fresh}")
    print()
    print(f"{'reason':<46} {'stored':>7} {'fresh':>7}")
    for reason in sorted(set(stored_reasons) | set(fresh_reasons)):
        print(f"{reason:<46} {stored_reasons.get(reason, 0):>7} {fresh_reasons.get(reason, 0):>7}")
    print()
    print(f"accepted: stored={stored_reasons.get('accepted', 0)} "
          f"fresh={fresh_reasons.get('accepted', 0)}")
    print(
        f"cropped by the scorecard band: stored={stored_reasons.get('pitch_cropped_by_overlay', 0)} "
        f"fresh={fresh_reasons.get('pitch_cropped_by_overlay', 0)} "
        f"(band = {PITCH_GEOMETRY_CROP_BAND_FRACTION:.2f} * {IMAGE_SIZE[1]} px)"
    )
    aspect_rows = [
        (i, fresh_diag.get(i, {}))
        for i, _f, _s, fresh in rows
        if fresh == "pitch_quad_aspect_disagrees_with_travel"
    ]
    print(
        f"fresh quads the elongation/direction check refused: {len(aspect_rows)} "
        f"(of these, {sum(1 for _i, d in aspect_rows if (d.get('divergence') or 0) > PITCH_GEOMETRY_TRAVEL_AXIS_MAX_DEG)} "
        f"disagree with the ball's travel direction)"
    )
    print()
    print("Span measurement behind each decision "
          f"(oriented aspect must reach {PITCH_GEOMETRY_ASPECT_MIN_RATIO}, "
          f"travel divergence must stay within {PITCH_GEOMETRY_TRAVEL_AXIS_MAX_DEG:.0f} deg):")
    print(
        f"{'id':>4} | {'stored h':>8} {'v':>6} {'oriented':>8} {'axis':>4} {'achieved':>8} {'div':>5} {'gap':>6} {'bGap':>6} "
        f"| {'fresh h':>8} {'v':>6} {'oriented':>8} {'axis':>4} {'achieved':>8} {'div':>5} {'gap':>6} {'bGap':>6}"
    )
    for delivery_id, _frame, _stored, _fresh in rows:
        s = stored_diag.get(delivery_id, {})
        f = fresh_diag.get(delivery_id, {})

        def cell(d: dict) -> str:
            if not d or "h" not in d:
                return f"{'-':>8} {'-':>6} {'-':>8} {'-':>4} {'-':>8} {'-':>5} {'-':>6} {'-':>6}"
            gap = d.get("gap")
            div = d.get("divergence")
            return (
                f"{d.get('h', 0):>8.0f} {d.get('v', 0):>6.0f} "
                f"{d.get('oriented', 0):>8.2f} {d.get('axis', '-'):>4} "
                f"{d.get('achieved', 0):>8.2f} "
                f"{('%.0f' % div) if div is not None else '-':>5} "
                f"{('%.0f' % gap) if gap is not None else '-':>6} "
                f"{('%.0f' % d.get('bottom_gap')) if d.get('bottom_gap') is not None else '-':>6}"
            )

        print(f"{delivery_id:>4} | {cell(s)} | {cell(f)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
