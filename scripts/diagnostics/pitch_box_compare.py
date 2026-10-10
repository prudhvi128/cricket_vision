"""
pitch_box_compare.py — Run the real pipeline on one video and record a
comparison fingerprint for the optional local pitch-box detector.

Purpose (docs/… pitch box feature):

    1. Run the WHOLE pipeline (tracking, purity, shot, overlay) on a video.
    2. Print deliveries / clips / frames / timing, plus a deterministic
       TRACKING FINGERPRINT over every delivery's boundaries, trajectory and
       bowling analytics. Two runs whose fingerprints match had identical ball
       detections and tracking, whatever else differed.
    3. Report the pitch-box detector's own counters when they exist
       (`tracking.json` -> `pitch_box_detection`), or state that the feature
       was absent from this build/run.
    4. Optionally save N sample frames with the detected box and its
       APPROXIMATE corners drawn, into a diagnostics directory.

The fingerprint deliberately EXCLUDES anything the optional Roboflow pitch
keypoint detector can change (per-delivery `pitch` block, geometry-derived
speeds): set ROBOFLOW_API_KEY= to hold that path inert when the only variable
under test is the local box detector.

Usage:
    python scripts/diagnostics/pitch_box_compare.py --video real_cricket.mp4 \\
        --analysis-id pitchbox_off_baseline --summary docs/diagnostics/pitchbox_off.json
    CRICKET_PITCH_BOX_MODEL=pitch_detect.pt python scripts/diagnostics/... --analysis-id pitchbox_on
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

DEFAULT_VIDEO = ROOT / "real_cricket.mp4"


# ── Fingerprint ───────────────────────────────────────────────────────────────
def tracking_fingerprint(deliveries: list) -> str:
    """
    sha256 over the parts of every delivery the ball detector + tracker own.

    Boundaries, every trajectory point (frame, x, y, provenance, confidence),
    the bounce and the bowling numbers EXCEPT the geometry-derived ones that a
    per-video pitch quad may legitimately fill in. Same fingerprint => the
    ball detections and the tracking were identical.
    """
    rows = []
    for d in deliveries:
        b = d.get("bowling") or {}
        rows.append({
            "id": d.get("delivery_id"),
            "start": d.get("detection_start_frame"),
            "end": d.get("detection_end_frame"),
            "lost": d.get("ball_lost_frame"),
            "release": d.get("release_frame"),
            "points": [
                [p.get("original_frame"), p.get("x"), p.get("y"), p.get("source"),
                 p.get("confidence")]
                for p in (d.get("trajectory") or {}).get("points") or []
            ],
            "bounce": [
                (d.get("bounce") or {}).get("original_frame"),
                (d.get("bounce") or {}).get("x_px"),
                (d.get("bounce") or {}).get("y_px"),
                (d.get("bounce") or {}).get("method"),
            ],
            # `speed_kmh`/`length`/… are omitted ON PURPOSE: a real pitch quad
            # detected by the OTHER (Roboflow) feature may fill them in, and
            # that is not what this comparison is about.
            "shot": (d.get("shot") or {}).get("type"),
            "validation": (d.get("validation") or {}).get("classification"),
        })
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── Sample frames ─────────────────────────────────────────────────────────────
def save_samples(video: Path, deliveries: list, out_dir: Path, limit: int) -> list:
    """
    Save up to `limit` source frames with the pitch box and its APPROXIMATE
    corners drawn. Only frames that carry a `pitch.box_estimate` are drawn.
    """
    import cv2

    wanted = []
    for d in deliveries:
        est = (d.get("pitch") or {}).get("box_estimate")
        if isinstance(est, dict) and est.get("box"):
            wanted.append((int(est["frame_index"]), est))
    wanted.sort(key=lambda t: t[0])
    # Spread the samples across the whole video rather than taking the first
    # N: ten frames from the opening over always look like ten frames from
    # anywhere, and the pitch is in view for only part of this footage.
    if len(wanted) > limit > 0:
        step = len(wanted) / limit
        wanted = [wanted[int(i * step)] for i in range(limit)]
    if not wanted:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    saved = []
    by_frame = {f: est for f, est in wanted}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        est = by_frame.get(index)
        if est is not None:
            path = out_dir / f"frame_{index:05d}.jpg"
            cv2.imwrite(str(path), draw_estimate(frame, est))
            saved.append(str(path))
    cap.release()
    return saved


def draw_estimate(frame, est: dict):
    """Box in green, the four APPROXIMATE corners in yellow, labelled P1..P4."""
    import cv2

    img = frame.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in est["box"]]
    colour = (0, 200, 0) if est.get("corners_valid") else (0, 0, 220)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    # High enough to clear the P4/P3 corner labels drawn just above the box.
    cv2.putText(img, f"pitch box conf={est.get('box_conf')}", (x1, max(14, y1 - 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    corners = est.get("corners") or {}
    for name in ("P1", "P2", "P3", "P4"):
        pt = corners.get(name)
        if not pt:
            continue
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img, (cx, cy), 4, (0, 220, 255), -1)
        cv2.putText(img, name, (cx + 5, cy - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 220, 255), 1, cv2.LINE_AA)
    corners_ok = est.get("corners_valid")
    cv2.putText(img, f"corners_valid={corners_ok} ({est.get('corners_reason')})",
                (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 220, 255), 1, cv2.LINE_AA)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--analysis-id", required=True)
    ap.add_argument("--summary", default=None, help="write the summary JSON here")
    ap.add_argument("--samples-dir", default=None)
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        print(f"video not found: {video}")
        return 1

    from app.core.config import (
        PITCH_BOX_CONFIG,
        Paths,
        PipelineConfig,
        analysis_root,
    )
    from app.pipeline.analysis_pipeline import UnifiedPipeline
    from app.tracking.detector import BallDetector

    paths = Paths(analysis_id=args.analysis_id)
    shutil.rmtree(analysis_root(args.analysis_id), ignore_errors=True)
    paths.ensure()

    import os
    box_model = os.environ.get("CRICKET_PITCH_BOX_MODEL", "")
    print("=" * 78)
    print("pitch-box comparison run")
    print("=" * 78)
    print(f"  video                {video}")
    print(f"  analysis             {args.analysis_id}")
    print(f"  CRICKET_PITCH_BOX_MODEL = {box_model!r}")
    print(f"  pitch box detector   "
          f"{'ON' if PITCH_BOX_CONFIG.enabled else 'OFF'}"
          f"  ({PITCH_BOX_CONFIG.model_path})")
    print(f"  ROBOFLOW key set     {'yes' if os.environ.get('ROBOFLOW_API_KEY') else 'no'}"
          f"  (remote pitch keypoints are opt-in and are off unless"
          f" CRICKET_ROBOFLOW_PITCH is set)")

    detector = BallDetector()
    pipeline = UnifiedPipeline(detector=detector, config=PipelineConfig())

    t0 = time.perf_counter()
    result = pipeline.run(str(video), paths, progress=None)
    wall = time.perf_counter() - t0

    tr = result.tracking
    doc = json.loads(paths.result_json.read_text(encoding="utf-8"))
    deliveries = doc["deliveries"]
    tracking_doc = json.loads(paths.tracking_json.read_text(encoding="utf-8"))

    total_points = sum(
        len((d.get("trajectory") or {}).get("points") or []) for d in deliveries
    )
    detected_points = sum(
        sum(1 for p in (d.get("trajectory") or {}).get("points") or []
            if p.get("source") == "detected")
        for d in deliveries
    )
    box_stats = tracking_doc.get("pitch_box_detection")
    with_box = sum(
        1 for d in deliveries
        if isinstance((d.get("pitch") or {}).get("box_estimate"), dict)
    )
    # Ground-plane metrics and the (abandoned) remote keypoint path.
    bowling = [d.get("bowling") or {} for d in deliveries]
    with_speed = sum(1 for b in bowling if b.get("speed_kmh") is not None)
    with_line = sum(1 for b in bowling if b.get("line") is not None)
    with_length = sum(1 for b in bowling if b.get("length") is not None)
    with_swing = sum(1 for b in bowling if b.get("swing") is not None)
    with_release = sum(1 for b in bowling if b.get("release_angle") is not None)
    calibrated = sum(1 for b in bowling if b.get("geometry_calibrated") is True)
    keypoint = doc.get("pitch_detection") or {}
    roboflow_requests = int(keypoint.get("submitted") or 0)

    print("\n" + "-" * 78)
    print(f"  deliveries (tracking pass)   {len(tr.deliveries)}")
    print(f"  deliveries (result.json)     {len(deliveries)}")
    print(f"  quarantined                  {len(doc.get('quarantine') or [])}")
    print(f"  clips written                {tr.clip_stats.get('clips_written')}")
    print(f"  frames decoded               {tr.frames_decoded}")
    print(f"  wall seconds                 {round(wall, 1)}")
    in_loop = sum(tr.stage_seconds.values())
    top_stages = sorted(tr.stage_seconds.items(), key=lambda kv: -kv[1])[:5]
    print(f"  in-loop stage seconds        {round(in_loop, 1)}"
          f"  (top: {[(k, round(v, 1)) for k, v in top_stages]})")
    print(f"  timings                      {doc['performance']['timings_sec']}")
    print(f"  trajectory points            {total_points} "
          f"({detected_points} detected)")
    print(f"  tracking fingerprint         {tracking_fingerprint(deliveries)}")
    print(f"  deliveries with box_estimate {with_box}")
    print(f"  pitch box stats              {box_stats}")
    print(f"  roboflow requests            {roboflow_requests}"
          f"  (pitch_detection: {keypoint.get('enabled')=},"
          f" {keypoint.get('failures')=})")
    print(f"  geometry_calibrated          {calibrated}/{len(deliveries)}")
    print(f"  with speed/line/length/swing {with_speed}/{with_line}"
          f"/{with_length}/{with_swing}   release_angle {with_release}")

    samples = []
    if args.samples_dir:
        samples = save_samples(video, deliveries, Path(args.samples_dir), args.samples)
        print(f"  sample frames saved          {len(samples)} -> {args.samples_dir}")

    summary = {
        "video": str(video),
        "analysis_id": args.analysis_id,
        "cricket_pitch_box_model": box_model or None,
        "pitch_box_config": PITCH_BOX_CONFIG.metadata(),
        "roboflow_key_set": bool(os.environ.get("ROBOFLOW_API_KEY")),
        "roboflow_requests": roboflow_requests,
        "roboflow_detection": {
            k: keypoint.get(k)
            for k in ("enabled", "submitted", "detections", "failures", "last_error")
        },
        "deliveries_tracking": len(tr.deliveries),
        "deliveries_result": len(deliveries),
        "quarantined": len(doc.get("quarantine") or []),
        "clips_written": tr.clip_stats.get("clips_written"),
        "frames_decoded": tr.frames_decoded,
        "wall_seconds": round(wall, 1),
        "timings_sec": doc["performance"]["timings_sec"],
        "trajectory_points": total_points,
        "detected_points": detected_points,
        "tracking_fingerprint": tracking_fingerprint(deliveries),
        "pitch_box_detection": box_stats,
        "deliveries_with_box_estimate": with_box,
        "geometry_calibrated": calibrated,
        "with_speed": with_speed,
        "with_line": with_line,
        "with_length": with_length,
        "with_swing": with_swing,
        "with_release_angle": with_release,
        "sample_frames": samples,
    }
    if args.summary:
        out = Path(args.summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  summary written              {out}")

    print(f"  artefacts                    {analysis_root(args.analysis_id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
