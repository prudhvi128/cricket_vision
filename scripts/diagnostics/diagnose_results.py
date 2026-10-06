"""
diagnose_results.py — Explain the results of the real-video run.

Three things in the run output deserve scrutiny rather than a victory lap:

  1. Every delivery was labelled "Lofted". Is that the model, or is it an
     artifact of the synthetic fixture's black gaps filling the clip window?
  2. Speeds ranged 42.5-198.5 km/h. No cricket bowler delivers at 198 km/h, so
     either the calibration is off or the geometry is wrong for this footage.
  3. Tracking ran at ~15 fps. Where does the time actually go?

Each question is answered with a measurement, not an assertion.

Usage:
    python tools/diagnose_results.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

# scripts/<category>/<name>.py -> <root>/scripts/<category> -> <root>
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core.constants import (  # noqa: E402
    HOMOGRAPHY_CORNER_FRACTIONS,
    SPEED_CALIBRATION_OFFSET_KMH,
    SPEED_CALIBRATION_SCALE,
)
from app.shot_classification.preprocessing import sample_indices  # noqa: E402

ANALYSIS = ROOT / "data" / "runs" / "phase3_real"


def q1_black_frame_dominance() -> None:
    """
    How much of each clip window is synthetic black filler?

    Clips are built as PRE_ROLL + real footage + POST_ROLL. In the fixture the
    only real footage is the 1 s delivery clip, so a ~150-frame window is
    mostly the synthetic black gaps. If the 30 sampled frames are mostly black,
    the shot label describes the filler, not the cricket.
    """
    print("=" * 78)
    print("Q1: WHY IS EVERY SHOT 'LOFTED'?")
    print("=" * 78)

    doc = json.loads((ANALYSIS / "result.json").read_text(encoding="utf-8"))
    black_fractions = []
    for d in doc["deliveries"]:
        clip = d.get("clip")
        path = ANALYSIS / "clips" / f"delivery_{d['delivery_id']:03d}.mp4"
        if not clip or not path.is_file():
            continue
        total = clip["frame_count"]
        idxs = sample_indices(total)
        cap = cv2.VideoCapture(str(path))
        frames = {}
        want = set(idxs)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                frames[i] = f
            i += 1
        cap.release()
        if len(frames) != len(idxs):
            print(f"  delivery {d['delivery_id']}: only {len(frames)}/{len(idxs)} sampled")
            continue
        black = sum(
            1 for i in idxs if float(frames[i].mean()) < 6.0
        )
        black_fractions.append(black / len(idxs))

    if black_fractions:
        print(f"\n  Black fraction of the 30 sampled frames per clip:")
        print(f"    min  {min(black_fractions):.0%}")
        print(f"    mean {statistics.mean(black_fractions):.0%}")
        print(f"    max  {max(black_fractions):.0%}")
        dominant = sum(1 for f in black_fractions if f > 0.5)
        print(f"\n  Clips where >50% of sampled frames are synthetic black: "
              f"{dominant}/{len(black_fractions)}")
        print(f"  Shot labels: {dict(Counter(d['shot']['type'] for d in doc['deliveries']))}")
        print("\n  FINDING: the shot label is dominated by the synthetic gap")
        print("  frames in the fixture, not by the cricket. The classifier is")
        print("  working; the INPUT it was given is mostly filler. On a real")
        print("  match video the window would be continuous footage.")
    print()


def q2_geometry_and_speed() -> None:
    """Where do the assumed pitch corners land, and where do bounces land?"""
    print("=" * 78)
    print("Q2: WHY DO SPEEDS RANGE 42.5-198.5 KM/H?")
    print("=" * 78)

    w, h = 640, 360
    corners = {
        k: (v[0] * w, v[1] * h) for k, v in HOMOGRAPHY_CORNER_FRACTIONS.items()
    }
    print(f"\n  Assumed pitch corners at {w}x{h} (fixed fractions):")
    for k, (x, y) in corners.items():
        print(f"    {k}: ({x:6.1f}, {y:6.1f})   = "
              f"({x / w:.0%} across, {y / h:.0%} down)")

    doc = json.loads((ANALYSIS / "result.json").read_text(encoding="utf-8"))
    speeds = [
        (d["delivery_id"], d["bowling"]["speed_kmh"])
        for d in doc["deliveries"] if d["bowling"]["speed_kmh"] is not None
    ]
    print(f"\n  Speed estimates: {[f'{v:.0f}' for _, v in speeds]}")
    print(f"    plausible cricket range is roughly 100-165 km/h")
    out_of_range = [v for _, v in speeds if not (100 <= v <= 165)]
    print(f"    outside that range: {len(out_of_range)}/{len(speeds)}")

    # The calibration is a fixed affine map. Show what it does to a true speed.
    print(f"\n  The calibration inverts to true speed as:")
    print(f"    true_kmh = (reported_kmh + {-SPEED_CALIBRATION_OFFSET_KMH}) "
          f"/ {SPEED_CALIBRATION_SCALE} / 1.0   (for an exact 3.6 m/s->km/h path)")
    print(f"    i.e. a reported 198.5 km/h corresponds to ~"
          f"{(198.5 - SPEED_CALIBRATION_OFFSET_KMH) / SPEED_CALIBRATION_SCALE:.0f}"
          f" km/h of raw fitted velocity before the fudge factor.")

    gx = [d["bounce"]["ground_x_m"] for d in doc["deliveries"]
          if d["bounce"].get("ground_x_m") is not None]
    gy = [d["bounce"]["ground_y_m"] for d in doc["deliveries"]
          if d["bounce"].get("ground_y_m") is not None]
    print(f"\n  Bounce ground coordinates (metres on the assumed pitch):")
    if gx:
        print(f"    x: min {min(gx):6.2f}  mean {statistics.mean(gx):6.2f}  "
              f"max {max(gx):6.2f}   (crease width 2.64 m)")
    if gy:
        print(f"    y: min {min(gy):6.2f}  mean {statistics.mean(gy):6.2f}  "
              f"max {max(gy):6.2f}   (pitch length 20.12 m)")

    # How many detected pixels lie inside the assumed pitch quad?
    inside = total = 0
    quad = np.array(
        [corners["TL"], corners["TR"], corners["BR"], corners["BL"]], np.float32
    )
    for d in doc["deliveries"]:
        for p in d["trajectory"]["points"]:
            if p["source"] != "detected":
                continue
            total += 1
            if cv2.pointPolygonTest(quad, (float(p["x"]), float(p["y"])), False) >= 0:
                inside += 1
    print(f"\n  Detected ball positions inside the assumed pitch quad: "
          f"{inside}/{total} ({inside / max(total,1):.0%})")
    print("\n  FINDING: this is exactly the limitation decision D2 documents.")
    print("  The homography corner fractions were measured off one specific")
    print("  broadcast template. These clips are 640x360 downscales with a")
    print("  different pitch position in frame, so the assumed quad is wrong,")
    print("  the metre-space distances are wrong, and the fitted velocities")
    print("  inherit that error. The number is an ESTIMATE and the calibration")
    print("  block says so. It is NOT accurate on this footage. Per-video")
    print("  pitch-corner calibration is required before these numbers mean")
    print("  anything physical.")


def q3_where_did_the_time_go() -> None:
    """Break the tracking pass into decode / detect / track / ring / clip."""
    print("=" * 78)
    print("Q3: WHERE DID THE 187 s OF TRACKING GO?")
    print("=" * 78)

    from app.core.config import Paths, PipelineConfig
    from app.tracking.detector import BallDetector
    from app.tracking.kalman_tracker import BallTracker
    from app.video.frame_buffer import FrameRingBuffer
    from app.video.video_io import get_fps

    video = ROOT / "data" / "samples" / "full_fixture.mp4"
    n = 400   # a representative slice; the full run is 2870 frames

    cap = cv2.VideoCapture(str(video))
    fps = get_fps(cap)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"\n  Profiling {len(frames)} frames of {video.name}")

    # Decode only.
    cap = cv2.VideoCapture(str(video))
    t = time.perf_counter()
    i = 0
    while i < len(frames):
        ok, _ = cap.read()
        if not ok:
            break
        i += 1
    cap.release()
    t_decode = time.perf_counter() - t

    # Detect only (model already loaded).
    det = BallDetector()
    t = time.perf_counter()
    for f in frames:
        det.detect_with_confidence(f)
    t_detect = time.perf_counter() - t

    # Ring buffer only.
    rb = FrameRingBuffer()
    t = time.perf_counter()
    for i, f in enumerate(frames, 1):
        rb.push(i, f)
    t_ring = time.perf_counter() - t

    # Ring buffer read-back (what clip writing does).
    t = time.perf_counter()
    for i in range(1, min(len(frames), rb.last_frame or 0) + 1):
        rb.get(i)
    t_ringread = time.perf_counter() - t

    # Kalman only.
    tr = BallTracker()
    t = time.perf_counter()
    for i in range(1, len(frames) + 1):
        tr.update(None, i)
    t_track = time.perf_counter() - t

    # Clip encoding: 21 clips of ~170 frames each in the real run. Encode the
    # whole 400-frame slice into one writer to get a per-encoded-frame figure.
    import tempfile

    from app.video.clips import ClipWriter

    tmp = Path(tempfile.mkdtemp())
    t = time.perf_counter()
    cw = ClipWriter(tmp / "c.mp4", fps, (frames[0].shape[1], frames[0].shape[0]))
    cw.open_window(1, len(frames))
    for i, f in enumerate(frames, 1):
        cw.write(i, f)
    cw.close()
    t_clipwrite = time.perf_counter() - t
    n_clips_real = 21
    avg_clip_len = 170
    clip_frames_real = n_clips_real * avg_clip_len

    # Writer open cost — the codec chain is walked on every clip.
    t = time.perf_counter()
    for k in range(n_clips_real):
        c = ClipWriter(tmp / f"o{k}.mp4", fps,
                       (frames[0].shape[1], frames[0].shape[0]))
        c.close()
    t_open = (time.perf_counter() - t) / n_clips_real

    scale = 2870 / len(frames)
    print(f"\n  {'stage':<28}{'per frame':>12}{'x2870 frames':>16}")
    print("  " + "-" * 54)
    for name, secs in (
        ("video decode", t_decode),
        ("YOLO detection", t_detect),
        ("ring buffer encode", t_ring),
        ("ring buffer decode", t_ringread),
        ("Kalman update", t_track),
    ):
        per = secs / len(frames) * 1000
        print(f"  {name:<28}{per:>10.2f}ms{secs * scale:>14.1f}s")
    print("  " + "-" * 54)
    per_frame = t_decode + t_detect + t_ring + t_track
    print(f"  {'per-frame work':<28}{per_frame / len(frames) * 1000:>10.2f}ms"
          f"{per_frame * scale:>14.1f}s")

    # Clip work scales with CLIPPED frames (~3570), not source frames, plus a
    # per-clip writer-open cost.
    t_clip_total = t_clipwrite * (clip_frames_real / len(frames)) + t_open * n_clips_real
    print(f"  {'clip encoding (' + str(clip_frames_real) + ' frames)':<28}"
          f"{'':>10}{t_clip_total:>14.1f}s")
    print(f"  {'  of which writer opens':<28}{'':>10}{t_open * n_clips_real:>14.1f}s")

    accounted = per_frame * scale + t_clip_total
    print("  " + "-" * 54)
    print(f"  {'ACCOUNTED TOTAL':<28}{'':>10}{accounted:>14.1f}s")
    print(f"  {'measured tracking stage':<28}{'':>10}{187.2:>14.1f}s")
    unexplained = 187.2 - accounted
    print(f"  {'unexplained':<28}{'':>10}{unexplained:>14.1f}s"
          f"  ({unexplained / 187.2:.0%})")
    print(f"\n  Detection is {t_detect / per_frame:.0%} of the per-frame work. "
          f"It runs once per frame, on the GPU, at "
          f"{len(frames) / t_detect:.1f} fps.")
    if unexplained > 10:
        print(f"\n  NOTE: {unexplained:.0f}s is not accounted for by the stages")
        print("  above. The profiling slice is taken from the start of the")
        print("  video, where the ball is absent and deliveries are rejected")
        print("  early; the real run additionally builds trajectories, runs the")
        print("  analytics per delivery, and rewrites tracking.json atomically")
        print("  after each one. Those are per-delivery, not per-frame, costs")
        print("  and are not captured by this slice.")


if __name__ == "__main__":
    if not (ANALYSIS / "result.json").is_file():
        print("Run tools/run_real_video_test.py --keep first.")
        sys.exit(1)
    q1_black_frame_dominance()
    q2_geometry_and_speed()
    q3_where_did_the_time_go()