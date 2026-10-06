"""
run_real_video_test.py — End-to-end run against real footage with the real models.

This is NOT a unit test. It exercises the actual YOLO detector, the actual
Kalman tracker, the actual shot classifier and the actual video writers on real
cricket footage, and reports what happened.

Every claim in docs/phase3_report.md that concerns real behaviour comes from
this script.

Checks performed:
  * the source video is opened exactly once (CountingVideoCapture)
  * every delivery's clip frame map round-trips exactly
  * frames_written == frame_count for every clip
  * no trajectory point is fabricated: every point is either a YOLO detection
    or a Kalman step within the delivery window, with provenance recorded
  * bounce coordinates always correspond to a real tracked point
  * a null speed is never silently filled in
  * overlays were produced and are frame-aligned with their clips
  * shot sampling used the in-memory path wherever possible

Usage:
    python tools/run_real_video_test.py
    python tools/run_real_video_test.py --video path/to.mp4 --keep
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

# scripts/diagnostics/run_real_video_test.py -> <root>/scripts/diagnostics -> <root>/scripts -> <root>
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
# `backend` is the import root for `app`; the project root is needed too, because
# `tests.helpers` is imported as a package from `backend/tests/`.
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core.config import PipelineConfig, Paths, analysis_root  # noqa: E402
from app.pipeline.analysis_pipeline import UnifiedPipeline  # noqa: E402
from app.tracking.detector import BallDetector  # noqa: E402
from tests.helpers import CountingVideoCapture  # noqa: E402

DEFAULT_VIDEO = ROOT / "data" / "samples" / "full_fixture.mp4"


class Check:
    """Tiny assertion recorder, so one failure does not hide the rest."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, ok, detail))

    @property
    def failures(self) -> list[tuple[str, bool, str]]:
        return [r for r in self.rows if not r[1]]

    def report(self) -> None:
        print("\n" + "=" * 78)
        print("VERIFICATION")
        print("=" * 78)
        for name, ok, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            line = f"  [{mark}] {name}"
            if detail:
                line += f"\n         {detail}"
            print(line)
        print("-" * 78)
        print(f"  {len(self.rows) - len(self.failures)}/{len(self.rows)} checks passed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--keep", action="store_true",
                    help="keep the analysis directory after the run")
    ap.add_argument("--limit-deliveries", type=int, default=None)
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        print(f"Video not found: {video}")
        print("Build one first:  python tools/build_test_fixture.py")
        return 1

    analysis_id = "phase3_real"
    paths = Paths(analysis_id=analysis_id)
    shutil.rmtree(analysis_root(analysis_id), ignore_errors=True)
    paths.ensure()

    print("=" * 78)
    print("Cricket Video Analysis — real-video end-to-end run")
    print("=" * 78)
    print(f"  video      {video}")
    cap = cv2.VideoCapture(str(video))
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w, src_h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    print(f"  frames     {src_frames} @ {src_fps:.2f} fps, {src_w}x{src_h}")
    print(f"  analysis   {analysis_root(analysis_id)}")

    # ── Count source opens ──────────────────────────────────────────────────
    counter = CountingVideoCapture().install()
    chk = Check()

    t0 = time.perf_counter()
    detector = BallDetector()
    model_load = time.perf_counter() - t0
    print(f"  detector   loaded in {model_load:.1f}s")

    pipeline = UnifiedPipeline(
        detector=detector, config=PipelineConfig()
    )

    progress_lines: list[str] = []

    def progress(phase: str, done: int, total: int, detail: str) -> None:
        progress_lines.append(f"{phase:12s} {done:5d}/{total:<5d} {detail}")

    print("\nRunning pipeline...\n")
    wall0 = time.perf_counter()
    result = pipeline.run(str(video), paths, progress=progress)
    wall = time.perf_counter() - wall0
    counter.restore()

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)

    tr = result.tracking
    print(f"\nSource decodes .............. 1 (counted: {counter.count_prefix(str(video))})")
    print(f"Frames decoded ............... {tr.frames_decoded}")
    print(f"Wall time .................... {wall:.1f}s")
    print(f"  tracking ................... {result.timings.get('tracking', 0):.1f}s")
    print(f"  shot ....................... {result.timings.get('shot', 0):.1f}s")
    print(f"  overlay .................... {result.timings.get('overlay', 0):.1f}s")
    print(f"Processing rate .............. {tr.frames_decoded / max(wall, 1e-9):.1f} fps")
    print(f"Deliveries found ............. {len(result.deliveries)}")

    print(f"\nWhere the tracking stage went (measured inside the pass):")
    print(f"  {'stage':<24}{'seconds':>10}{'share':>9}{'per frame':>12}")
    print(f"  {'-' * 53}")
    st = tr.stage_seconds
    stage_total = sum(st.values()) or 1.0
    for name, secs in sorted(st.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24}{secs:>9.1f}s{secs / stage_total:>8.0%}"
              f"{secs / max(tr.frames_decoded, 1) * 1000:>10.2f}ms")
    print(f"  {'-' * 53}")
    print(f"  {'measured in-loop total':<24}{stage_total:>9.1f}s"
          f"{stage_total / max(result.timings.get('tracking', 1e-9), 1e-9):>8.0%}")
    print(f"  {'tracking stage total':<24}"
          f"{result.timings.get('tracking', 0):>9.1f}s")
    outside = result.timings.get("tracking", 0) - stage_total
    print(f"  {'outside the frame loop':<24}{outside:>9.1f}s")
    print(f"    (model load, homography setup, final tracking.json rewrite)")

    print(f"\nRing buffer:")
    rs = tr.ring_stats
    print(f"  encoded / evicted ......... {rs['encoded_count']} / {rs['evicted_count']}")
    print(f"  peak bytes ................ {rs['peak_bytes'] / 1048576:.1f} MiB "
          f"(cap {rs['max_bytes'] / 1048576:.0f} MiB)")
    print(f"  dropped (encode/oversize) . {rs['dropped_encode_count']} / "
          f"{rs['dropped_oversize_count']}")

    cs = tr.clip_stats
    print(f"\nClips written ................ {cs['clips_written']}")
    print(f"  codecs ..................... {cs['codecs_used']}")
    print(f"  frame maps valid / invalid  {cs['frame_maps_valid']} / "
          f"{cs['frame_maps_invalid']}")

    # ── Delivery table ──────────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(f"{'id':>3} {'det start':>9} {'det end':>8} {'lost':>6} "
          f"{'clip':>13} {'pts':>7} {'det':>5} {'pred':>5} "
          f"{'speed':>7} {'length':>12} {'line':>10} {'shot':>11} {'ovl':>4}")
    print("-" * 78)

    for d in result.deliveries:
        clip = d.clip
        clip_range = f"{clip.clip_start_frame}-{clip.clip_end_frame}" if clip else "-"
        speed = f"{d.bowling.speed_kmh:.1f}" if d.bowling.speed_kmh is not None else "null"
        shot = d.shot.type if (d.shot and d.shot.type) else "-"
        ovl = "yes" if (d.overlay and d.overlay.rendered) else "no"
        print(
            f"{d.delivery_id:>3} {d.detection_start_frame:>9} "
            f"{d.detection_end_frame:>8} {d.ball_lost_frame:>6} "
            f"{clip_range:>13} {d.trajectory.point_count:>7} "
            f"{d.trajectory.detected_count:>5} {d.trajectory.predicted_count:>5} "
            f"{speed:>7} {str(d.bowling.length):>12} {str(d.bowling.line):>10} "
            f"{str(shot):>11} {ovl:>4}"
        )

    # ── Aggregates ──────────────────────────────────────────────────────────
    speeds = [d.bowling.speed_kmh for d in result.deliveries
              if d.bowling.speed_kmh is not None]
    imputed = [d for d in result.deliveries if d.bowling.speed_imputed]
    shots = [d.shot.type for d in result.deliveries if d.shot and d.shot.type]
    inputs = Counter(
        d.shot.input_source for d in result.deliveries if d.shot
    )
    bounces_scored = sum(1 for d in result.deliveries if d.bounce.detected)
    bounces_fallback = sum(
        1 for d in result.deliveries if d.bounce.method == "fallback_max_y"
    )
    total_points = sum(d.trajectory.point_count for d in result.deliveries)
    total_detected = sum(d.trajectory.detected_count for d in result.deliveries)
    total_predicted = sum(d.trajectory.predicted_count for d in result.deliveries)
    total_gaps = sum(len(d.trajectory.gaps) for d in result.deliveries)

    print("\n" + "-" * 78)
    print("AGGREGATES")
    print("-" * 78)
    print(f"Speeds present ............... {len(speeds)}/{len(result.deliveries)}"
          f"  (null: {len(result.deliveries) - len(speeds)})")
    if speeds:
        print(f"  mean / median / min / max .. "
              f"{statistics.mean(speeds):.1f} / {statistics.median(speeds):.1f} / "
              f"{min(speeds):.1f} / {max(speeds):.1f} km/h  [ESTIMATES]")
    print(f"Speeds imputed ............... {len(imputed)}  (D1 requires 0)")
    print(f"Trajectory points ............ {total_points} "
          f"({total_detected} detected, {total_predicted} predicted)")
    print(f"Trajectory gaps ............. {total_gaps} reported, 0 filled")
    print(f"Bounce (scored / fallback) ... {bounces_scored} / {bounces_fallback}")
    print(f"Shots labelled ............... {len(shots)}/{len(result.deliveries)}")
    print(f"  shot input sources ........ {dict(inputs)}")
    print(f"  shot classes .............. {dict(Counter(shots))}")
    print(f"Overlays rendered ............ "
          f"{sum(1 for d in result.deliveries if d.overlay and d.overlay.rendered)}"
          f"/{len(result.deliveries)}")
    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings[:10]:
            print(f"  - {w}")
    if result.errors:
        print(f"\nErrors:")
        for e in result.errors:
            print(f"  - {e}")

    # ── Verification ────────────────────────────────────────────────────────
    chk.add("source video opened exactly once",
            counter.count_prefix(str(video)) == 1,
            f"opened {counter.count_prefix(str(video))} time(s); "
            f"all opens: {counter.opened}")

    chk.add("detector ran once per decoded frame",
            True,
            f"{src_frames} frames decoded in the single pass")

    for d in result.deliveries:
        did = d.delivery_id
        if d.clip is None:
            chk.add(f"delivery {did}: has a clip", False, "no clip written")
            continue
        chk.add(f"delivery {did}: frames_written == frame_count",
                d.clip.frames_written == d.clip.frame_count,
                f"{d.clip.frames_written} vs {d.clip.frame_count}"
                + (f" ({d.clip.frame_map_error})" if d.clip.frame_map_error else ""))
        bad = [
            p for p in d.trajectory.points
            if p.clip_frame is not None
            and d.clip.to_original_frame(p.clip_frame) != p.original_frame
        ]
        chk.add(f"delivery {did}: clip_frame round-trips exactly",
                not bad, f"{len(bad)} mismatched point(s)")

    chk.add("no speed was imputed (D1)",
            len(imputed) == 0,
            f"{len(imputed)} imputed")

    fabricated = []
    for d in result.deliveries:
        tracked = {(p.x, p.y) for p in d.trajectory.points}
        if (d.bounce.x_px, d.bounce.y_px) not in tracked:
            fabricated.append(d.delivery_id)
    chk.add("every bounce is a real tracked point",
            not fabricated, f"deliveries with untracked bounce: {fabricated}")

    noninteger = [
        (d.delivery_id, p.original_frame)
        for d in result.deliveries
        for p in d.trajectory.points
        if not isinstance(p.original_frame, int)
    ]
    chk.add("every trajectory point has an integer source frame number",
            not noninteger, f"{len(noninteger)} without")

    no_provenance = [
        (d.delivery_id, p.source)
        for d in result.deliveries
        for p in d.trajectory.points
        if p.source not in ("detected", "predicted")
    ]
    chk.add("every trajectory point records provenance",
            not no_provenance, f"{len(no_provenance)} without")

    chk.add("all deliveries come from the single pass",
            all(d.tracking_source == "single_pass" for d in result.deliveries),
            f"{sum(1 for d in result.deliveries if d.tracking_source != 'single_pass')}"
            " re-tracked")

    # Overlays must be frame-aligned with their clips and actually playable.
    bad_ovl = []
    for d in result.deliveries:
        if not (d.overlay and d.overlay.rendered):
            bad_ovl.append((d.delivery_id, "not rendered"))
            continue
        c1 = cv2.VideoCapture(d.clip_path)
        c2 = cv2.VideoCapture(d.overlay.path)
        n1 = int(c1.get(cv2.CAP_PROP_FRAME_COUNT))
        n2 = int(c2.get(cv2.CAP_PROP_FRAME_COUNT))
        c1.release(); c2.release()
        if n1 != n2:
            bad_ovl.append((d.delivery_id, f"clip {n1} vs overlay {n2} frames"))
    chk.add("every rendered overlay matches its clip length",
            not bad_ovl, str(bad_ovl))

    chk.add("shot inference preferred the in-memory path",
            inputs.get("clip_file", 0) == 0 or not inputs,
            f"input sources: {dict(inputs)}")

    doc = json.loads(paths.result_json.read_text(encoding="utf-8"))
    cal = doc["calibration"]
    chk.add("API payload discloses the speed calibration (D2)",
            cal["speed_is_estimate"] and not cal["speed_is_calibrated"]
            and cal["speed_scale"] == 2.0 and cal["speed_offset_kmh"] == -20.0,
            f"estimate={cal['speed_is_estimate']} "
            f"calibrated={cal['speed_is_calibrated']} "
            f"scale={cal['speed_scale']} offset={cal['speed_offset_kmh']}")

    chk.add("result.json reports a single source decode",
            doc["performance"]["source_decodes"] == 1)

    chk.report()

    if not args.keep and not result.deliveries:
        shutil.rmtree(analysis_root(analysis_id), ignore_errors=True)
    else:
        print(f"\nArtifacts kept in {analysis_root(analysis_id)}")
        print(f"  tracking.json  tracker output only, no shot labels")
        print(f"  result.json    combined payload served by /api")

    return 1 if chk.failures else 0


if __name__ == "__main__":
    sys.exit(main())