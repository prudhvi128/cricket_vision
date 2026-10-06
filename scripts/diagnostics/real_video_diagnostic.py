#!/usr/bin/env python
"""
real_video_diagnostic.py — DIAGNOSTIC ONLY. Changes no pipeline behaviour.

Runs the FULL video through the existing single-pass pipeline
(`TrackingService.analyse`) and records, per frame, what happened at every
stage, so that any lost batting event can be traced to the exact stage that
dropped it.

HOW IT INSTRUMENTS WITHOUT TOUCHING THE PIPELINE
------------------------------------------------
  * YOLO   : `DiagDetector` SUBCLASSES `BallDetector`. The authoritative answer
             still comes from `super().detect_with_confidence(frame)` — the
             unmodified production code path. In addition, one extra "shadow"
             predict is run at a low confidence floor purely to ACCOUNT for
             boxes the production filters would discard. The shadow result is
             never returned to the pipeline; it cannot change any outcome.
             It runs on the SAME already-decoded frame, so there is still
             exactly ONE decode and ONE tracker pass. No per-clip inference.
  * Kalman : `BallTracker.update` / `.reseed` are wrapped at runtime by thin
             delegating wrappers that record the returned `TrackResult` and the
             internal `missed_frames`. The original methods are called.
  * Segment: `EventSegmenter.observe` is wrapped the same way to record the
             `ContinuationKind` decided on every frame.
  * Vision : `SignalTimeline.record` is wrapped to capture the per-frame motion
             and pitch-corridor signals used for gap hunting.

Every wrapper delegates to the original. Nothing here is imported by the
application.

Usage (from the project root):

    python scripts\diagnostics\real_video_diagnostic.py --video real_cricket.mp4
"""

from __future__ import annotations

import argparse
import json
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

from app.core.config import (  # noqa: E402
    PipelineConfig,
    Paths,
    analysis_root,
)
from app.core.constants import DETECTOR_CONF_THRESHOLD  # noqa: E402
from app.schemas.tracking import Delivery  # noqa: E402
from app.segmentation.barriers import SignalTimeline  # noqa: E402
from app.segmentation.candidate_generation import EventSegmenter  # noqa: E402
from app.tracking import kalman_tracker as tracker_mod  # noqa: E402
from app.tracking.detector import (  # noqa: E402
    BallDetector,
    RESIZE_H,
    RESIZE_W,
)
from app.tracking.tracking_service import TrackingService  # noqa: E402

SHADOW_CONF = 0.01   # diagnostic floor only; never used by the pipeline

# ── Diagnostic recorders ──────────────────────────────────────────────────────
PER_FRAME: list[dict] = []
_state = {"frame": 0}


class DiagDetector(BallDetector):
    """Production detector, plus a shadow predict used only for accounting."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # `detect_with_confidence` is called exactly once per decoded frame by
        # the tracking loop, so a self-incrementing call counter IS the frame
        # index. (The detector is not told the frame number.)
        self._call = 0

    def detect_with_confidence(self, frame):
        self._call += 1
        idx = self._call - 1

        # ── AUTHORITATIVE: unmodified production code path ──────────────────
        cx, cy, conf = super().detect_with_confidence(frame)

        if 0 <= idx < len(PER_FRAME):
            rec = PER_FRAME[idx]
            rec["yolo_accepted"] = cx is not None
            rec["yolo_conf"] = conf
            rec["yolo_xy"] = [cx, cy] if cx is not None else None

        # ── SHADOW: accounting only. Never returned to the pipeline. ────────
        if self.model is not None and 0 <= idx < len(PER_FRAME):
            rec = PER_FRAME[idx]
            small = cv2.resize(frame, (RESIZE_W, RESIZE_H))
            r = self.model.predict(
                source=small, conf=SHADOW_CONF, verbose=False, device=self.device
            )[0]
            boxes = []
            if r.boxes is not None:
                for b in r.boxes.data.tolist():
                    x1, y1, x2, y2, score, cls = b
                    diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    boxes.append({
                        "conf": round(float(score), 4),
                        "cls": int(cls),
                        "diag": round(float(diag), 2),
                        "cx": int((x1 + x2) / 2),
                        "cy": int((y1 + y2) / 2),
                    })
            rec["shadow_boxes"] = boxes
            # Stage-2 accounting: how many boxes existed, and why each was lost.
            rec["n_boxes_any_conf"] = len(boxes)
            rec["n_boxes_at_conf"] = sum(
                1 for b in boxes if b["conf"] >= DETECTOR_CONF_THRESHOLD
            )
            rec["n_wrong_class"] = sum(
                1 for b in boxes
                if b["conf"] >= DETECTOR_CONF_THRESHOLD and b["cls"] != 0
            )
            rec["n_rejected_size"] = sum(
                1 for b in boxes
                if b["conf"] >= DETECTOR_CONF_THRESHOLD and b["cls"] == 0
                and not (4 <= b["diag"] <= 80)
            )
            rec["subthreshold_boxes"] = sum(
                1 for b in boxes if b["conf"] < DETECTOR_CONF_THRESHOLD
            )
        return cx, cy, conf


def _wrap_tracker():
    orig_update = tracker_mod.BallTracker.update
    orig_reseed = tracker_mod.BallTracker.reseed
    orig_reset = tracker_mod.BallTracker.reset_delivery

    def update(self, detection, frame_no):
        res = orig_update(self, detection, frame_no)
        rec = PER_FRAME[frame_no - 1]
        rec["kalman_source"] = res.source
        rec["kalman_gated_out"] = bool(res.gated_out)
        rec["kalman_missed"] = self.missed_frames
        rec["kalman_xy"] = list(res.position) if res.position else None
        rec["kf_initialized"] = bool(self.kf.initialized)
        return res

    def reseed(self, pt, frame_no):
        rec = PER_FRAME[frame_no - 1]
        rec["tracker_reseeded"] = True
        return orig_reseed(self, pt, frame_no)

    def reset_delivery(self):
        f = _state["frame"]
        if 1 <= f <= len(PER_FRAME):
            PER_FRAME[f - 1]["tracker_reset_delivery"] = True
        return orig_reset(self)

    tracker_mod.BallTracker.update = update
    tracker_mod.BallTracker.reseed = reseed
    tracker_mod.BallTracker.reset_delivery = reset_delivery


def _wrap_segmenter():
    orig_observe = EventSegmenter.observe
    orig_close = EventSegmenter.close_open_event

    def observe(self, ev):
        d = orig_observe(self, ev)
        rec = PER_FRAME[ev.frame - 1]
        rec["seg_kind"] = d.kind
        rec["seg_gap"] = d.boundary.gap_frames if d.boundary else None
        rec["seg_reason"] = d.boundary.reason if d.boundary else None
        if d.opened is not None:
            rec["seg_opened_idx"] = d.opened.index
        if d.closed is not None:
            rec["seg_closed_idx"] = d.closed.index
        return d

    def close_open_event(self, at_frame, reason="end_of_video"):
        seg = orig_close(self, at_frame, reason)
        if seg is not None:
            f = max(1, min(at_frame, len(PER_FRAME)))
            if 1 <= f <= len(PER_FRAME):
                PER_FRAME[f - 1]["seg_closed_at_eov"] = seg.index
        return seg

    EventSegmenter.observe = observe
    EventSegmenter.close_open_event = close_open_event


def _wrap_signals():
    orig_record = SignalTimeline.record

    def record(self, signal):
        rec = PER_FRAME[signal.frame - 1]
        rec["is_black"] = bool(signal.is_black)
        rec["is_scene_cut"] = bool(signal.is_scene_cut)
        rec["motion"] = round(float(signal.motion), 3)
        rec["pitch_motion"] = round(float(signal.pitch_motion), 3)
        rec["had_candidate"] = bool(signal.candidate)
        return orig_record(self, signal)

    SignalTimeline.record = record


def _delivery_light(d: Delivery) -> dict:
    return {
        "delivery_id": d.delivery_id,
        "detection_start_frame": d.detection_start_frame,
        "detection_end_frame": d.detection_end_frame,
        "ball_lost_frame": d.ball_lost_frame,
        "event_index": d.event.get("event_index") if d.event else None,
        "confirmed_count": d.event.get("confirmed_count") if d.event else None,
        "predicted_count": d.event.get("predicted_count") if d.event else None,
        "points": len(d.trajectory.points) if d.trajectory else 0,
        "gaps": d.trajectory.gaps if d.trajectory else [],
        "clip_start": d.clip.clip_start_frame if d.clip else None,
        "clip_end": d.clip.clip_end_frame if d.clip else None,
        "clip_path": d.clip_path,
        "validation": d.validation,
        "shot": getattr(d.shot, "type", None) if d.shot else None,
        "shot_error": getattr(d.shot, "error", None) if d.shot else None,
        "speed_kmh": d.bowling.speed_kmh if d.bowling else None,
        "quality_flags": list(d.quality.flags) if d.quality else [],
    }


def _counts(report) -> dict:
    """`ValidationReport.counts` is a property in some builds, a method in others."""
    if report is None:
        return {}
    c = getattr(report, "counts", None)
    if callable(c):
        return c()
    return c if isinstance(c, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--analysis-id", default="real_diagnostic")
    ap.add_argument("--out", default=str(ROOT / "data" / "real_diagnostic.json"))
    ap.add_argument("--no-clips", action="store_true",
                    help="keep_raw_clips off (saves disk; boundaries unaffected)")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.is_file():
        print(f"video not found: {video}")
        return 2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"video      : {video}")
    print(f"resolution : {w}x{h}  fps={fps}  reported_frames={reported}")

    PER_FRAME.clear()
    PER_FRAME.extend(
        {"frame": i + 1} for i in range(reported)
    )

    _wrap_tracker()
    _wrap_segmenter()
    _wrap_signals()

    detector = DiagDetector()
    if detector.model is None:
        print("detector failed to load")
        return 2

    import shutil
    shutil.rmtree(analysis_root(args.analysis_id), ignore_errors=True)
    paths = Paths(analysis_id=args.analysis_id).ensure()

    cfg = PipelineConfig(keep_raw_clips=not args.no_clips)

    last = [0.0]

    def progress(phase, done, total, detail):
        now = time.time()
        if now - last[0] >= 20:
            last[0] = now
            print(f"  [{now - t0:6.0f}s] {phase:18s} frame {done}/{total}  {detail}",
                  flush=True)

    print("starting single-pass run over the FULL video ...", flush=True)
    t0 = time.time()
    service = TrackingService(config=cfg)
    result = service.analyse(str(video), paths, progress=progress, detector=detector)
    elapsed = time.time() - t0
    print(f"pass complete in {elapsed:.1f}s "
          f"({result.frames_decoded / elapsed:.1f} frames/s)", flush=True)

    # ── Assemble the diagnostic document ─────────────────────────────────────
    for rec in PER_FRAME:
        rec["t_sec"] = round(rec["frame"] / fps, 3) if fps else None
        # Keep only class-0 boxes in the per-frame payload; wrong-class boxes
        # are already accounted for in the totals.
        boxes = rec.get("shadow_boxes") or []
        rec["shadow_boxes"] = [b for b in boxes if b["cls"] == 0]

    raw_boxes = [
        b["conf"] for rec in PER_FRAME for b in (rec.get("shadow_boxes") or [])
    ]
    # `result.segments` holds EventSegment objects; serialise them properly.
    segments = [
        s.as_dict() if hasattr(s, "as_dict") else s
        for s in (result.segments or [])
    ]

    doc = {
        "meta": {
            "video": str(video),
            "fps": fps,
            "width": w,
            "height": h,
            "reported_frames": reported,
            "decoded_frames": result.frames_decoded,
            "duration_sec": round(result.frames_decoded / fps, 3) if fps else None,
            "wall_seconds": round(elapsed, 2),
            "conf_threshold": DETECTOR_CONF_THRESHOLD,
            "shadow_conf": SHADOW_CONF,
            "min_delivery_frames": cfg.min_delivery_frames,
            "segmentation": result.segmentation_summary.get("resolved_thresholds"),
            "stage_seconds": result.stage_seconds,
        },
        "totals": {
            "frames": len(PER_FRAME),
            "yolo_accepted_frames": sum(1 for r in PER_FRAME if r.get("yolo_accepted")),
            "yolo_gated_out_frames": sum(
                1 for r in PER_FRAME if r.get("kalman_gated_out")),
            "kalman_predicted_frames": sum(
                1 for r in PER_FRAME if r.get("kalman_source") == "predicted"),
            "kalman_no_position_frames": sum(
                1 for r in PER_FRAME if r.get("kalman_source") is None),
            "frames_with_any_box": sum(
                1 for r in PER_FRAME if r.get("n_boxes_any_conf")),
            "frames_with_box_at_conf": sum(
                1 for r in PER_FRAME if r.get("n_boxes_at_conf")),
            "frames_with_only_subthreshold_boxes": sum(
                1 for r in PER_FRAME
                if r.get("n_boxes_any_conf", 0) > 0
                and r.get("n_boxes_at_conf", 0) == 0),
            "frames_rejected_by_size": sum(
                1 for r in PER_FRAME if r.get("n_rejected_size", 0) > 0),
            "total_boxes_any_conf": sum(
                r.get("n_boxes_any_conf", 0) for r in PER_FRAME),
            "total_boxes_at_conf": sum(
                r.get("n_boxes_at_conf", 0) for r in PER_FRAME),
            "total_subthreshold_boxes": sum(
                r.get("subthreshold_boxes", 0) for r in PER_FRAME),
            "total_rejected_by_size": sum(
                r.get("n_rejected_size", 0) for r in PER_FRAME),
            "total_wrong_class": sum(
                r.get("n_wrong_class", 0) for r in PER_FRAME),
        },
        "decision_kinds": dict(Counter(
            r.get("seg_kind") for r in PER_FRAME if r.get("seg_kind"))),
        "close_reasons": dict(Counter(
            r.get("seg_reason") for r in PER_FRAME if r.get("seg_reason"))),
        "segmentation_summary": result.segmentation_summary,
        "events": segments,
        "deliveries": [_delivery_light(d) for d in result.deliveries],
        "event_horizons": result.event_horizons,
        "warnings": result.warnings,
        "validation": result.validation.as_dict() if result.validation else None,
        "validation_counts": _counts(result.validation),
        "clip_stats": result.clip_stats,
        "box_confidences_any_conf": raw_boxes,
        "per_frame": PER_FRAME,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, default=str), encoding="utf-8")
    print(f"diagnostic written: {out}  ({out.stat().st_size/1e6:.1f} MB)")

    # Clean up the clips this diagnostic generated — they are throwaway output.
    shutil.rmtree(analysis_root(args.analysis_id), ignore_errors=True)

    print("\n--- headline ---")
    print(f"frames decoded        : {result.frames_decoded}")
    print(f"YOLO accepted frames  : {doc['totals']['yolo_accepted_frames']}")
    print(f"Kalman predicted      : {doc['totals']['kalman_predicted_frames']}")
    print(f"Kalman gate-rejected  : {doc['totals']['yolo_gated_out_frames']}")
    print(f"events found          : {result.segmentation_summary.get('events_total')}")
    print(f"events primary (>=8)  : {result.segmentation_summary.get('events_primary')}")
    print(f"events rejected       : {result.segmentation_summary.get('events_rejected')}")
    print(f"DELIVERIES            : {len(result.deliveries)}")
    print(f"untracked windows     : {sum(1 for h in result.event_horizons if h['kind']=='UNTRACKED_ACTIVITY_WINDOW')}")
    print(f"close reasons         : {doc['close_reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())