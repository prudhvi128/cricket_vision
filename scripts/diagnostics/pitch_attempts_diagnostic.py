"""
pitch_attempts_diagnostic.py — See why the pitch detector answered as it did.

The counters say 799 failures and 40 refusals; they do not say that the model
answers on some frames and not on the frame right next to it, that a refusal
means four corners came back clustered inside 1% of the frame, or that a
delivery's own trailing frames are where the pitch is actually readable. This
turns one run's attempt log into:

  <out>/pitch_attempts_<run>.csv    every attempt: frame, timestamp, outcome,
                                    rejection reason, confidence, keypoints,
                                    calibration state, queue wait, inference
  <out>/pitch_montage_<run>.jpg     the same attempts drawn on the frames that
                                    produced them, so the answer can be seen
  <out>/pitch_attempts_<run>.md     the counts behind the report

Usage:
    python scripts/diagnostics/pitch_attempts_diagnostic.py [--run RUN_ID]
                                                            [--out DIR]
                                                            [--video PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# scripts/<category>/<name>.py -> <root>
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import cv2  # noqa: E402

RUNS = ROOT / "data" / "runs"
DEFAULT_OUT = ROOT / "docs" / "diagnostics"

COLUMNS = [
    "frame", "timestamp", "outcome", "reason", "calibration_state",
    "confidence", "keypoint_count", "keypoints", "infer_ms", "wait_ms",
    "event_open", "in_delivery",
]


def newest_run() -> Path:
    """The most recently written analysis directory."""
    candidates = [p for p in RUNS.iterdir() if (p / "result.json").is_file()]
    return max(candidates, key=lambda p: (p / "result.json").stat().st_mtime)


def video_path(run_dir: Path) -> Path:
    """Where the run's source video lives, as recorded by the run itself."""
    tracking = json.loads((run_dir / "tracking.json").read_text(encoding="utf-8"))
    recorded = (tracking.get("video") or {}).get("path") or (
        (tracking.get("source_caps") or {}).get("path")
    )
    if recorded:
        p = Path(recorded)
        if p.is_file():
            return p
    for guess in (ROOT / "real_cricket.mp4", ROOT / "data" / "real_cricket.mp4"):
        if guess.is_file():
            return guess
    return Path()


def timestamp(frame: int, fps: float) -> str:
    if not fps:
        return ""
    seconds = frame / fps
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def attempt_rows(attempts: list[dict], fps: float) -> list[dict]:
    rows = []
    for a in attempts:
        kps = a.get("keypoints") or {}
        rows.append({
            "frame": a.get("frame"),
            "timestamp": timestamp(int(a.get("frame") or 0), fps),
            "outcome": a.get("outcome"),
            "reason": a.get("reason", ""),
            "calibration_state": a.get("calibration_state", ""),
            "confidence": a.get("confidence", ""),
            "keypoint_count": len(kps),
            "keypoints": ";".join(
                f"{name}:{xy[0]:.1f},{xy[1]:.1f}" for name, xy in kps.items()
            ),
            "infer_ms": a.get("infer_ms", ""),
            "wait_ms": a.get("wait_ms", ""),
            "event_open": a.get("event"),
            "in_delivery": a.get("hot"),
        })
    return rows


def select_for_montage(attempts: list[dict], per_kind: dict[str, int]) -> list[dict]:
    """A spread of every outcome, accepted first — the point is the contrast."""
    chosen: list[dict] = []
    for outcome, limit in per_kind.items():
        matches = [a for a in attempts if a.get("outcome") == outcome]
        if not matches:
            continue
        step = max(1, len(matches) // limit)
        chosen.extend(matches[::step][:limit])
    return chosen


def draw_tile(frame, attempt: dict, fps: float) -> "cv2.Mat":
    """One attempt as one labelled tile: what was asked, and what came back."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - 74), (w, h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)

    outcome = str(attempt.get("outcome"))
    colour = {
        "detected": (60, 220, 60),
        "refused": (0, 165, 255),
        "no_keypoints": (200, 200, 200),
        "timeout": (0, 0, 255),
    }.get(outcome, (255, 255, 255))

    for name, xy in (attempt.get("keypoints") or {}).items():
        x, y = int(xy[0]), int(xy[1])
        cv2.circle(canvas, (x, y), 6, colour, 2)
        cv2.putText(canvas, name, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    frame_no = int(attempt.get("frame") or 0)
    line1 = (f"frame {frame_no}  {timestamp(frame_no, fps)}  {outcome}"
             f"{'' if not attempt.get('reason') else '  ' + str(attempt['reason'])}")
    line2 = (f"conf={attempt.get('confidence')}  kps={len(attempt.get('keypoints') or {})}  "
             f"infer={attempt.get('infer_ms')}ms  queue={attempt.get('wait_ms')}ms  "
             f"calib={attempt.get('calibration_state')}")
    cv2.putText(canvas, line1, (10, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                colour, 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def montage(video: Path, attempts: list[dict], fps: float, out_path: Path,
            per_kind: dict[str, int]) -> int:
    chosen = select_for_montage(attempts, per_kind)
    if not chosen or not video.is_file():
        return 0

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    tiles = []
    for attempt in chosen:
        frame_no = int(attempt.get("frame") or 0)
        if not 0 < frame_no <= max(total, frame_no):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        tiles.append(draw_tile(frame, attempt, fps))
    cap.release()
    if not tiles:
        return 0

    tile_h = 270
    resized = []
    for tile in tiles:
        scale = tile_h / tile.shape[0]
        resized.append(cv2.resize(
            tile, (int(tile.shape[1] * scale), tile_h), interpolation=cv2.INTER_AREA
        ))
    cols = 3
    rows = (len(resized) + cols - 1) // cols
    tile_w = max(t.shape[1] for t in resized)
    sheet = None
    for r in range(rows):
        band = None
        for c in range(cols):
            i = r * cols + c
            if i >= len(resized):
                break
            pad = cv2.copyMakeBorder(resized[i], 0, 0, 0, tile_w - resized[i].shape[1],
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
            band = pad if band is None else cv2.hconcat([band, pad])
        if band is None:
            continue
        if band.shape[1] < tile_w * cols:
            band = cv2.copyMakeBorder(band, 0, 0, 0, tile_w * cols - band.shape[1],
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        sheet = band if sheet is None else cv2.vconcat([sheet, band])
    if sheet is None:
        return 0
    cv2.imwrite(str(out_path), sheet)
    return len(resized)


def write_markdown(path: Path, run_id: str, summary: dict, attempts: list[dict],
                   montage_path: Path, tiles: int) -> None:
    counts: dict[str, int] = {}
    for a in attempts:
        outcome = str(a.get("outcome"))
        counts[outcome] = counts.get(outcome, 0) + 1
    reasons: dict[str, int] = {}
    states: dict[str, int] = {}
    for a in attempts:
        if a.get("reason"):
            reasons[str(a["reason"])] = reasons.get(str(a["reason"]), 0) + 1
        state = str(a.get("calibration_state") or "-")
        states[state] = states.get(state, 0) + 1

    lines = [
        f"# Pitch attempt log — run `{run_id}`",
        "",
        "One row per attempt the worker actually sent to the model. The CSV next",
        "to this file has every column; the montage shows the frames themselves.",
        "",
        "## Outcomes",
        "",
        "| outcome | attempts | share |",
        "|---|---:|---:|",
    ]
    for outcome, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = 100.0 * n / max(1, len(attempts))
        lines.append(f"| {outcome} | {n} | {share:.1f}% |")
    lines += ["", "## Rejection reasons", "",
              "| reason | attempts |", "|---|---:|"]
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {n} |")
    if not reasons:
        lines.append("| (none) | 0 |")
    lines += ["", "## Calibration state at the attempt", "",
              "| state | attempts |", "|---|---:|"]
    for state, n in sorted(states.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {state} | {n} |")
    lines += ["", "## Summary recorded by the run", "", "```json",
              json.dumps(summary, indent=2, sort_keys=True), "```"]
    if tiles:
        lines += ["", f"## Montage", "", f"![pitch attempts]({montage_path.name})", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="analysis id (default: most recent run)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="output directory (default: docs/diagnostics)")
    parser.add_argument("--video", help="source video (default: as recorded)")
    parser.add_argument("--tiles", type=int, default=4,
                        help="montage tiles per outcome (default 4)")
    args = parser.parse_args()

    run_dir = RUNS / args.run if args.run else newest_run()
    if not (run_dir / "result.json").is_file():
        print(f"no result.json in {run_dir}")
        return 1
    run_id = run_dir.name
    doc = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    pitch = doc.get("pitch_detection") or {}
    diag = pitch.get("diagnostics") or {}
    attempts = diag.get("attempts") or []
    if not attempts:
        print(f"run {run_id} has no attempt log (it predates the instrumentation)")
        return 1

    fps = float((json.loads((run_dir / "tracking.json").read_text(encoding="utf-8"))
                 .get("video") or {}).get("fps") or 0.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"pitch_attempts_{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(attempt_rows(attempts, fps))

    video = Path(args.video) if args.video else video_path(run_dir)
    jpg_path = out / f"pitch_montage_{run_id}.jpg"
    tiles = montage(video, attempts, fps, jpg_path,
                    {"detected": args.tiles, "refused": args.tiles,
                     "no_keypoints": args.tiles, "timeout": 2})
    md_path = out / f"pitch_attempts_{run_id}.md"
    write_markdown(md_path, run_id, {k: pitch.get(k) for k in
                                     ("submitted", "detections", "failures",
                                      "refusals", "queue_drops",
                                      "detector_error")}, attempts,
                   jpg_path, tiles)

    print(f"run         {run_id}")
    print(f"attempts    {len(attempts)} logged "
          f"({diag.get('offered')} frames offered)")
    print(f"csv         {csv_path.relative_to(ROOT)}")
    print(f"report      {md_path.relative_to(ROOT)}")
    print(f"montage     {f'{jpg_path.relative_to(ROOT)} ({tiles} tiles)' if tiles else 'skipped (no frames)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
