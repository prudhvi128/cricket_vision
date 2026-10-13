"""
run_seg_ab.py — Real-video A/B for the pitch-SEGMENTATION feature.

Runs TrackingService on the real footage once with the seg detector OFF and
once ON, and writes one JSON per arm: wall time, stage seconds, delivery
count, per-delivery trajectory digests, the pitch_seg summary, and the
per-delivery geometry verdicts. The two arms are separate processes (the
config is resolved from the environment at import), so invoke it twice:

    CRICKET_PITCH_SEG_MODEL=off       python scripts/verification/run_seg_ab.py --arm off
    CRICKET_PITCH_SEG_MODEL=pitch_seg_yolo11s.pt python scripts/verification/run_seg_ab.py --arm on

then compare with --compare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

from app.core.config import PipelineConfig, Paths, analysis_root  # noqa: E402
from app.tracking.detector import BallDetector  # noqa: E402
from app.tracking.tracking_service import TrackingService  # noqa: E402

DEFAULT_VIDEO = ROOT / "real_cricket.mp4"
OUT_DIR = ROOT / "docs" / "diagnostics"


def trajectory_digest(delivery) -> str:
    h = hashlib.sha256()
    for p in delivery.trajectory.points:
        h.update(f"{p.original_frame},{p.x},{p.y},{p.source},{p.confidence};".encode())
    return h.hexdigest()[:16]


def run_arm(arm: str, video: Path) -> dict:
    analysis_id = f"seg_ab_{arm}"
    import shutil
    shutil.rmtree(analysis_root(analysis_id), ignore_errors=True)
    paths = Paths(analysis_id=analysis_id).ensure()

    detector = BallDetector()
    service = TrackingService(PipelineConfig())
    wall0 = time.perf_counter()
    result = service.analyse(str(video), paths, detector=detector)
    wall = time.perf_counter() - wall0

    doc = json.loads(paths.tracking_json.read_text(encoding="utf-8"))
    return {
        "arm": arm,
        "seg_env": os.environ.get("CRICKET_PITCH_SEG_MODEL"),
        "wall_s": round(wall, 2),
        "frames_decoded": result.frames_decoded,
        "stage_seconds": {k: round(v, 2) for k, v in result.stage_seconds.items()},
        "deliveries": len(result.deliveries),
        "pitch_seg_summary": result.pitch_seg_summary,
        "pitch_geometry_stats": result.pitch_geometry_stats,
        "per_delivery": [
            {
                "id": d.delivery_id,
                "start": d.detection_start_frame,
                "end": d.detection_end_frame,
                "lost": d.ball_lost_frame,
                "points": d.trajectory.point_count,
                "detected": d.trajectory.detected_count,
                "traj": trajectory_digest(d),
                "pitch_state": d.pitch.get("state"),
                "pitch_calibrated": d.pitch.get("calibrated"),
                "map_homography": d.pitch.get("map_homography"),
                "geometry_calibrated": d.bowling.geometry_calibrated,
                "speed_kmh": d.bowling.speed_kmh,
                "length": d.bowling.length,
                "line": d.bowling.line,
            }
            for d in result.deliveries
        ],
        "tracking_document_keys": sorted(doc.keys()),
    }


def compare(off: dict, on: dict) -> int:
    print("\n" + "=" * 78)
    print("A/B COMPARE")
    print("=" * 78)
    failures = []
    counts = {"passed": 0, "total": 0}

    def check(name: str, ok: bool, detail: str = "") -> None:
        counts["total"] += 1
        if ok:
            counts["passed"] += 1
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"\n         {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    check("same frames decoded", off["frames_decoded"] == on["frames_decoded"],
          f"{off['frames_decoded']} vs {on['frames_decoded']}")
    check("same delivery count", off["deliveries"] == on["deliveries"],
          f"{off['deliveries']} vs {on['deliveries']}")
    check("OFF arm has no pitch_seg_summary", off["pitch_seg_summary"] is None)
    check("ON arm has a pitch_seg_summary",
          on["pitch_seg_summary"] is not None
          and on["pitch_seg_summary"].get("enabled") is True)

    off_doc_keys = set(off["tracking_document_keys"])
    on_doc_keys = set(on["tracking_document_keys"])
    check("tracking document adds only pitch_seg_detection",
          on_doc_keys - off_doc_keys == {"pitch_seg_detection"}
          and off_doc_keys - on_doc_keys == set(),
          f"added: {sorted(on_doc_keys - off_doc_keys)}, "
          f"removed: {sorted(off_doc_keys - on_doc_keys)}")

    traj_mismatch = [
        (a["id"], b["id"]) for a, b in zip(off["per_delivery"], on["per_delivery"])
        if a["traj"] != b["traj"]
    ]
    check("every trajectory byte-identical ON vs OFF", not traj_mismatch,
          str(traj_mismatch[:5]))

    window_mismatch = [
        a["id"] for a, b in zip(off["per_delivery"], on["per_delivery"])
        if (a["start"], a["end"], a["lost"], a["points"], a["detected"])
        != (b["start"], b["end"], b["lost"], b["points"], b["detected"])
    ]
    check("every delivery window/point-count identical", not window_mismatch,
          str(window_mismatch[:5]))

    contaminated = [
        d["id"] for d in on["per_delivery"]
        if d["geometry_calibrated"] or d["speed_kmh"] is not None
        or d["length"] is not None or d["line"] is not None
        or d["map_homography"] is not None
    ]
    check("ON arm: no delivery gained metric geometry", not contaminated,
          f"contaminated: {contaminated[:5]}")

    if on["pitch_seg_summary"]:
        svc = on["pitch_seg_summary"].get("service", {})
        stats = on["pitch_geometry_stats"]
        rejected = stats["rejections"]
        allowed = {
            "segmentation_estimate_is_not_metric_geometry",
            "no_pitch_calibration",
        }
        check("ON arm: zero deliveries gained metric geometry",
              stats["deliveries_with_geometry"] == 0,
              f"with geometry: {stats['deliveries_with_geometry']}")
        check("every rejection uses a documented reason",
              set(rejected) <= allowed,
              f"rejections: {rejected}, deliveries: {stats['deliveries']}")
        print(f"\n  ON arm seg tally: samples={on['pitch_seg_summary'].get('samples')} "
              f"accepted={on['pitch_seg_summary'].get('accepted')} "
              f"no_mask={on['pitch_seg_summary'].get('no_mask')} "
              f"errors={on['pitch_seg_summary'].get('errors')}")
        print(f"  ON arm service: submitted={svc.get('submitted')} "
              f"detections={svc.get('detections')} refusals={svc.get('refusals')} "
              f"scene_changes={svc.get('scene_changes')}")

    print("\n" + "-" * 78)
    print(f"  wall OFF {off['wall_s']}s   wall ON {on['wall_s']}s   "
          f"delta {on['wall_s'] - off['wall_s']:+.1f}s")
    print(f"  tracking OFF {off['stage_seconds'].get('tracking', 0)}s   "
          f"tracking ON {on['stage_seconds'].get('tracking', 0)}s")
    print("-" * 78)
    print(f"  {counts['passed']}/{counts['total']} checks passed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--arm", choices=["off", "on"], default=None)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        print(f"Video not found: {video}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        off = json.loads((OUT_DIR / "seg_ab_off.json").read_text())
        on = json.loads((OUT_DIR / "seg_ab_on.json").read_text())
        return compare(off, on)

    if args.arm is None:
        print("Specify --arm off|on (one arm per process) or --compare")
        return 2

    # One arm per process, and the env var must be set by the CALLER before
    # python starts: PITCH_SEG_CONFIG is frozen at import, so assigning
    # os.environ in here would not re-resolve it. The resolved model name is
    # printed so the arm cannot silently disagree with what actually ran.
    from app.core.config import PITCH_SEG_CONFIG
    resolved = PITCH_SEG_CONFIG.model
    expected = "off" if args.arm == "off" else "pitch_seg_yolo11s.pt"
    if str(resolved).strip().lower() != expected:
        print(f"env mismatch: --arm {args.arm} expects {expected!r} "
              f"but PITCH_SEG_CONFIG resolved {resolved!r}. "
              f"Set CRICKET_PITCH_SEG_MODEL before starting python.")
        return 2
    print(f"\n=== arm={args.arm}  resolved model={resolved!r} ===")
    data = run_arm(args.arm, video)
    out = OUT_DIR / f"seg_ab_{args.arm}.json"
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"  wall {data['wall_s']}s  deliveries {data['deliveries']}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
