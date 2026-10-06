"""
diagnose_real_deliveries.py — WHERE do the real deliveries disappear?

DIAGNOSTIC ONLY. Nothing in app/ is modified or re-tuned here. Thresholds are
never changed, segmentation is never rewritten, and no second YOLO pass and no
per-clip tracking are introduced.

PHASE 1 runs the EXISTING single-pass pipeline, unchanged:
    one source decode -> one YOLO11 pass -> one Kalman pass ->
    current event segmentation -> current event horizons -> current ClipValidator

The only instrumentation is a DETECTOR WRAPPER that counts. It delegates every
call to the real BallDetector, so model behaviour is identical; it records on
which frames a raw ball candidate came back. That is what lets us separate
"YOLO never saw it" from "YOLO saw it but the pipeline dropped it".

PHASE 2 is a separate, clearly-labelled diagnostic measurement: a lightweight
motion-only scan (no YOLO, no Kalman, not part of the pipeline) that builds an
independent ACTIVITY TIMELINE. It exists because the pipeline registers an
activity window ONLY when YOLO produced zero detections for it — so deliveries
that YOLO missed entirely leave no trace in tracking.json. To say "this real
delivery was completely undetected, and here is its timeline evidence", we need
a view of the video's activity that does not depend on YOLO.

CORRELATION then matches every observed activity burst to the pipeline stage
that consumed it (or to none, i.e. invisible to all signals), producing the
A..G attribution the request asks for.

Usage:
    python tools/diagnose_real_deliveries.py
    python tools/diagnose_real_deliveries.py --video path/to.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# scripts/<category>/<name>.py -> <root>/scripts/<category> -> <root>
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core.config import PipelineConfig, Paths, analysis_root  # noqa: E402
from app.tracking.detector import BallDetector  # noqa: E402
from app.tracking.tracking_service import TrackingService  # noqa: E402

DEFAULT_VIDEO = ROOT / "real_cricket.mp4"
ANALYSIS_ID = "real_diag"

# Same thresholds the pipeline uses (from SegmentationConfig defaults), so the
# diagnostic activity timeline lines up with what the pipeline saw.
QUIET = 1.5
ACTIVE = 4.0
BLACK_LUM = 3.0
MIN_WINDOW_FRAMES = 4
QUIET_HOLD_FRAMES = 4


class CountingDetector:
    """
    Delegating wrapper that counts raw YOLO hits without changing behaviour.

    `detect_with_confidence` is called exactly once per decoded frame by the
    tracking loop, and the loop's frame counter starts at 1 and increments by
    one per read, so the call index IS the source frame number.
    """

    def __init__(self, inner):
        self.inner = inner
        self.model = inner.model
        self.model_path = inner.model_path
        self.calls = 0
        self.raw_frames: list[int] = []          # frames with a raw ball candidate
        self.conf: list[float] = []              # their confidences

    def detect_with_confidence(self, frame):
        self.calls += 1
        cx, cy, conf = self.inner.detect_with_confidence(frame)
        if cx is not None:
            self.raw_frames.append(self.calls)
            if conf is not None:
                self.conf.append(float(conf))
        return cx, cy, conf

    # Any other attribute the loop might touch is delegated untouched.
    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)


def fmt_ts(frame: int, fps: float) -> str:
    if not fps:
        return f"f{frame}"
    s = (frame - 1) / fps
    return f"{int(s // 60):02d}:{s % 60:06.3f} (f{frame})"


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — independent motion/activity timeline (diagnostic, no YOLO)
# ═══════════════════════════════════════════════════════════════════════════
def activity_timeline(video_path: str, fps: float) -> list[dict]:
    """
    Streaming activity bursts using the SAME quiet/active thresholds as the
    pipeline. Returns every contiguous active cluster, whether or not YOLO saw
    anything inside it.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    bursts: list[dict] = []
    last_gray = None
    frame = 0
    active_start: int | None = None
    quiet_run = 0
    peak = 0.0
    n_black = 0

    while True:
        ok, img = cap.read()
        if not ok:
            break
        frame += 1
        gray = cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (80, 45),
            interpolation=cv2.INTER_AREA,
        )
        lum = float(np.mean(gray))
        motion = 0.0
        if last_gray is not None:
            motion = float(np.mean(cv2.absdiff(gray, last_gray)))
        last_gray = gray
        if lum < BLACK_LUM:
            n_black += 1

        if active_start is None:
            if motion >= ACTIVE:
                active_start = frame
                peak = motion
                quiet_run = 0
        else:
            peak = max(peak, motion)
            if motion <= QUIET:
                quiet_run += 1
                if quiet_run >= QUIET_HOLD_FRAMES:
                    end = frame - quiet_run
                    if end - active_start + 1 >= MIN_WINDOW_FRAMES:
                        bursts.append(
                            {
                                "start": active_start,
                                "end": end,
                                "peak_motion": round(peak, 3),
                            }
                        )
                    active_start = None
                    quiet_run = 0
            else:
                quiet_run = 0

    if active_start is not None:
        end = frame
        if end - active_start + 1 >= MIN_WINDOW_FRAMES:
            bursts.append(
                {"start": active_start, "end": end, "peak_motion": round(peak, 3)}
            )
    cap.release()

    for b in bursts:
        b["duration"] = b["end"] - b["start"] + 1
        b["ts_start"] = fmt_ts(b["start"], fps)
        b["ts_end"] = fmt_ts(b["end"], fps)
    return bursts


def overlaps(a0, a1, b0, b1) -> bool:
    return not (a1 < b0 or b0 > a1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    args = ap.parse_args()
    video = Path(args.video)
    if not video.is_file():
        print(f"Video not found: {video}")
        return 1

    shutil.rmtree(analysis_root(ANALYSIS_ID), ignore_errors=True)
    paths = Paths(analysis_id=ANALYSIS_ID).ensure()

    print("=" * 80)
    print("PHASE 1 — existing single-pass pipeline (unchanged)")
    print("=" * 80)

    inner = BallDetector()
    det = CountingDetector(inner)

    t0 = time.perf_counter()
    result = TrackingService(PipelineConfig()).analyse(str(video), paths, detector=det)
    wall = time.perf_counter() - t0

    fps = result.fps
    seg = result.segmentation_summary
    events = result.segments
    val = result.validation

    # ── Pipeline facts ──────────────────────────────────────────────────────
    confirmed_frames = sum(e.confirmed_count for e in events)
    predicted_frames = sum(e.predicted_count for e in events)
    primary = [e for e in events if e.confirmed_count >= seg["resolved_thresholds"]["min_event_confirmed_frames"]]
    rejected = [e for e in events if e.confirmed_count < seg["resolved_thresholds"]["min_event_confirmed_frames"]]

    horizons = result.event_horizons
    untracked = [h for h in horizons if h["kind"] == "UNTRACKED_ACTIVITY_WINDOW"]
    rejected_h = [h for h in horizons if h["kind"] == "REJECTED_EVENT_CANDIDATE"]
    noconf_h = [h for h in horizons if h["kind"] == "NO_CONFIRMED_DETECTIONS"]

    # raw detections that fall outside any tracked event span
    event_spans = [
        (e.start_frame, e.ball_lost_frame or e.end_frame) for e in events
    ]
    raw_outside = [
        f for f in det.raw_frames
        if not any(s <= f <= e for s, e in event_spans)
    ]

    print(f"\nwall={wall:.1f}s  frames_decoded={result.frames_decoded} "
          f"raw_detect_calls={det.calls}")

    counts = val.counts if val else {}
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"accepted deliveries ............ {len(result.deliveries)}")
    print(f"segmenter events (all) ......... {len(events)} "
          f"(primary {len(primary)}, sub-threshold {len(rejected)})")
    print(f"zero-detection activity windows  {len(untracked)}")
    print(f"validation: {counts}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2 — independent activity timeline
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 2 — independent motion timeline (diagnostic, NO YOLO)")
    print("=" * 80)
    t1 = time.perf_counter()
    bursts = activity_timeline(str(video), fps)
    print(f"{len(bursts)} activity bursts found in {time.perf_counter()-t1:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # CORRELATE: each burst -> the pipeline stage that consumed it
    # ══════════════════════════════════════════════════════════════════════
    buckets = {"accepted": [], "sub_threshold": [], "zero_detection": [],
               "invisible": [], "split_across": []}

    for b in bursts:
        bs, be = b["start"], b["end"]
        acc = [e for e in primary
               if overlaps(bs, be, e.start_frame, e.ball_lost_frame or e.end_frame)]
        rej = [e for e in rejected
               if overlaps(bs, be, e.start_frame, e.ball_lost_frame or e.end_frame)]
        win = [h for h in untracked
               if overlaps(bs, be, h["start_frame"], h["end_frame"])]
        entry = dict(b)
        if acc and rej:
            entry["stage"] = "E/D: one burst holds accepted + sub-threshold events"
            entry["events"] = [e.index for e in acc + rej]
            buckets["split_across"].append(entry)
        elif acc:
            entry["stage"] = "accepted"
            entry["event_index"] = acc[0].index
            buckets["accepted"].append(entry)
        elif rej:
            entry["stage"] = "C: rejected sub-threshold"
            entry["event_index"] = rej[0].index
            entry["confirmed"] = rej[0].confirmed_count
            buckets["sub_threshold"].append(entry)
        elif win:
            entry["stage"] = "A: YOLO zero detections (untracked activity window)"
            buckets["zero_detection"].append(entry)
        else:
            entry["stage"] = "A/none: activity with NO YOLO and NO pipeline window"
            buckets["invisible"].append(entry)

    print("\n" + "-" * 80)
    print("ATTRIBUTION (each independent activity burst -> pipeline stage)")
    print("-" * 80)
    for k in ("accepted", "sub_threshold", "zero_detection",
              "split_across", "invisible"):
        print(f"  {k:16s} {len(buckets[k])}")

    # ══════════════════════════════════════════════════════════════════════
    # TIMELINE — every observed candidate with its stage
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TIMELINE — every observed batting-activity candidate")
    print("=" * 80)
    all_entries = []
    for k in ("accepted", "sub_threshold", "zero_detection",
              "split_across", "invisible"):
        for e in buckets[k]:
            all_entries.append((e["start"], k, e))
    all_entries.sort()

    for i, (start, k, e) in enumerate(all_entries, 1):
        marker = {"accepted": "OK ", "sub_threshold": "SUB",
                  "zero_detection": "A-ZERO",
                  "split_across": "MERGE",
                  "invisible": "INVISIBLE"}[k]
        extra = ""
        if "event_index" in e:
            extra = f"  event#{e['event_index']}"
            if "confirmed" in e:
                extra += f" confirmed={e['confirmed']}"
        if "events" in e:
            extra = f"  events={e['events']}"
        print(f"  [{i:3d}] {marker:9s} {e['ts_start']:24s} -> "
              f"{e['ts_end']:24s} frames {e['start']:6d}-{e['end']:6d}"
              f" dur={e['duration']:4d}{extra}")

    # ══════════════════════════════════════════════════════════════════════
    # ACCEPTED DELIVERY TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ACCEPTED DELIVERIES")
    print("=" * 80)
    hdr = (f"{'id':>3} {'start':>7} {'end':>7} {'clip_s':>7} {'clip_e':>7} "
           f"{'raw':>5} {'conf':>5} {'class':<16} reason")
    print(hdr)
    print("-" * len(hdr))
    raw_sorted = det.raw_frames
    accepted_rows = []
    for d in result.deliveries:
        ev = d.event or {}
        idx = ev.get("event_index")
        span0 = d.detection_start_frame
        span1 = d.ball_lost_frame or d.detection_end_frame
        raw = sum(1 for f in raw_sorted if span0 <= f <= span1)
        conf = ev.get("confirmed_count", len(
            [p for p in d.trajectory.points if p.source == "detected"]))
        v = d.validation or {}
        clip = d.clip
        row = {
            "delivery_id": d.delivery_id,
            "source_start_frame": d.detection_start_frame,
            "source_end_frame": d.detection_end_frame,
            "clip_start_frame": clip.clip_start_frame if clip else None,
            "clip_end_frame": clip.clip_end_frame if clip else None,
            "raw_detection_count": raw,
            "confirmed_tracking_count": conf,
            "validation_classification": v.get("classification"),
            "validation_reason": v.get("reason"),
            "event_index": idx,
        }
        accepted_rows.append(row)
        print(f"{d.delivery_id:>3} {row['source_start_frame']:>7} "
              f"{row['source_end_frame']:>7} {str(row['clip_start_frame']):>7} "
              f"{row['clip_end_frame']:>7} {raw:>5} {conf:>5} "
              f"{str(v.get('classification')):<16} {str(v.get('reason'))[:60]}")

    # ══════════════════════════════════════════════════════════════════════
    # MULTI-EVENT CLIPS / UNDETECTED — explicit
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CLIPS CONTAINING MULTIPLE REAL DELIVERIES")
    print("=" * 80)
    multi = val.by_classification("MULTIPLE_EVENTS") if val else []
    if not multi:
        print("  (none reported by ClipValidator)")
    for r in multi:
        print(f"  delivery {r.delivery_id} clip {r.clip_start}-{r.clip_end} "
              f"ranges={r.event_ranges} :: {r.reason}")

    print("\n" + "=" * 80)
    print("COMPLETELY UNDETECTED (zero-detection + invisible activity)")
    print("=" * 80)
    undet = buckets["zero_detection"] + buckets["invisible"]
    if not undet:
        print("  (none)")
    for e in sorted(undet, key=lambda x: x["start"]):
        print(f"  {e['stage']:46s} {e['ts_start']} -> {e['ts_end']} "
              f"frames {e['start']}-{e['end']} peak={e['peak_motion']}")

    # ══════════════════════════════════════════════════════════════════════
    # DETECTION / TRACKING COUNTERS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("DETECTION & TRACKING COUNTERS")
    print("=" * 80)
    print(f"total raw YOLO detections ............ {len(det.raw_frames)} "
          f"({len(det.raw_frames)/max(1,result.frames_decoded)*100:.2f}% of frames)")
    print(f"confirmed tracking frames ............ {confirmed_frames}")
    print(f"predicted/coasting frames ............ {predicted_frames}")
    print(f"raw detections outside any event ..... {len(raw_outside)}"
          + (f" first few: {raw_outside[:10]}" if raw_outside else ""))
    print(f"tracking candidates/events ........... {len(events)}")
    print(f"sub-threshold candidates ............. {len(rejected)}")
    print(f"  -> rejected horizons registered .... {len(rejected_h)}")
    print(f"  -> no-confirmed horizons ........... {len(noconf_h)}")
    print(f"zero-detection activity windows ...... {len(untracked)}")
    print(f"boundary reasons ..................... {seg.get('boundary_reasons')}")

    # Merge suspects: events that bridged a long internal silence (D)
    print("\n" + "=" * 80)
    print("SUSPECTED MERGES (event bridged a gap > coast ceiling)")
    print("=" * 80)
    merge_suspects = []
    coast = seg["resolved_thresholds"]["max_coast_frames"]
    for e in events:
        g = (e.evidence or {}).get("max_internal_gap", 0)
        if g and g > coast:
            merge_suspects.append(e)
    if not merge_suspects:
        print("  (none)")
    for e in merge_suspects:
        g = (e.evidence or {}).get("max_internal_gap", 0)
        print(f"  event#{e.index} frames {e.start_frame}-{e.end_frame} "
              f"confirmed={e.confirmed_count} max_internal_gap={g} "
              f"internal_gaps={(e.evidence or {}).get('internal_gaps')}")

    # ══════════════════════════════════════════════════════════════════════
    # Persist
    # ══════════════════════════════════════════════════════════════════════
    out = {
        "video": {
            "path": str(video),
            "frames_decoded": result.frames_decoded,
            "fps": fps,
            "duration_sec": round(result.frames_decoded / fps, 3) if fps else None,
            "width": result.width,
            "height": result.height,
            "wall_seconds": round(wall, 2),
        },
        "detection": {
            "raw_detections": len(det.raw_frames),
            "confirmed_tracking_frames": confirmed_frames,
            "predicted_frames": predicted_frames,
            "raw_detections_outside_events": raw_outside,
            "events_total": len(events),
            "events_primary": len(primary),
            "events_sub_threshold": len(rejected),
            "zero_detection_windows": len(untracked),
            "horizons": horizons,
            "boundary_reasons": seg.get("boundary_reasons"),
        },
        "deliveries": {
            "accepted": len(result.deliveries),
            "validation": counts,
            "rows": accepted_rows,
        },
        "attribution": {k: len(v) for k, v in buckets.items()},
        "bursts": bursts,
        "buckets": buckets,
        "merge_suspects": [e.index for e in merge_suspects],
    }
    out_path = paths.root / "diagnostic.json"
    out_path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
