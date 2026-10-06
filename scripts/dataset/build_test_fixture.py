"""
build_test_fixture.py — Assemble a multi-delivery test video from real footage.

WHY A FIXTURE HAS TO BE BUILT
-----------------------------
Neither repository contains a full-match video. What the Adarsh repository did
contain is 41 clips of real broadcast footage, each 640x360 @ 25 fps, 25 frames
(1.0 s), produced by its `AutoClipper` from a full match that is not present.
Those clips now live in-project at `data/datasets/source_clips/`, so this builder
depends on nothing outside this repository.

Concatenating them naively would not exercise the pipeline: 25-frame deliveries
sitting back-to-back are 25 frames apart, and the 60-frame delivery debounce
would reject almost all of them. So each clip is followed by a gap.

THE GAP IS SYNTHETIC, AND THAT IS THE ONE CAVEAT
-------------------------------------------------
Gap frames are black. Black guarantees the detector finds nothing, which is what
makes the gap a genuine "no delivery here" interval. It is not real footage, and
it means the fixture's camera framing is discontinuous between deliveries. Any
accuracy figure derived from this video describes this composite, not a real
match. Everything else about it — the cricket imagery, the ball, the model's
detections — is real.

Usage (from the project root):
    python scripts/dataset/build_test_fixture.py
    python scripts/dataset/build_test_fixture.py --limit 6 --out quick_fixture.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# scripts/dataset/build_test_fixture.py -> <root>/scripts/dataset -> <root>/scripts -> <root>
ROOT = Path(__file__).resolve().parents[2]
# The 41 real delivery clips were relocated in-project during cleanup, from a
# source repository that is no longer part of the tree.
SRC_DIR = ROOT / "data" / "datasets" / "source_clips"
OUT_DIR = ROOT / "data" / "samples"


def read_all_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    finally:
        cap.release()
    return frames


def build(
    out_path: Path,
    limit: int | None = None,
    gap_frames: int = 45,
    fps: float = 25.0,
) -> dict:
    if not SRC_DIR.is_dir():
        raise SystemExit(f"Source clips not found at {SRC_DIR}")

    clips = sorted(SRC_DIR.glob("delivery_*.mp4"))
    if not clips:
        raise SystemExit(f"No delivery clips in {SRC_DIR}")
    if limit:
        clips = clips[:limit]

    # Probe the first clip for geometry, and require every clip to match: a
    # silent resolution change mid-build would break the homogeneous frame size
    # that ClipWriter depends on.
    probe = read_all_frames(clips[0])
    if not probe:
        raise SystemExit(f"{clips[0]} decoded zero frames")
    h, w = probe[0].shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open writer for {out_path}")

    blank = np.zeros((h, w, 3), dtype=np.uint8)
    stats = {
        "clips": 0, "clip_frames": 0, "gap_frames": 0,
        "width": w, "height": h, "fps": fps,
    }

    try:
        for clip in clips:
            frames = read_all_frames(clip)
            if not frames:
                continue
            if frames[0].shape[:2] != (h, w):
                raise SystemExit(
                    f"{clip.name} is {frames[0].shape[1]}x{frames[0].shape[0]}, "
                    f"expected {w}x{h}"
                )
            for f in frames:
                writer.write(f)
            stats["clips"] += 1
            stats["clip_frames"] += len(frames)
            for _ in range(gap_frames):
                writer.write(blank)
            stats["gap_frames"] += gap_frames
    finally:
        writer.release()

    stats["total_frames"] = stats["clip_frames"] + stats["gap_frames"]
    stats["duration_sec"] = round(stats["total_frames"] / fps, 2)

    # Verify what was actually written rather than trusting the counters.
    cap = cv2.VideoCapture(str(out_path))
    if cap.isOpened():
        stats["verified_frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stats["verified_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        stats["verified_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Count frames by decoding, since CAP_PROP_FRAME_COUNT is metadata and
        # can disagree with reality.
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        stats["decoded_frame_count"] = n
        cap.release()

    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N clips")
    p.add_argument("--gap", type=int, default=45,
                   help="black frames inserted after each clip (default 45)")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    name = "full_fixture.mp4" if args.limit is None else f"quick_fixture_{args.limit}.mp4"
    out = Path(args.out) if args.out else OUT_DIR / name

    stats = build(out, limit=args.limit, gap_frames=args.gap, fps=args.fps)

    print(f"Fixture written: {out}")
    for k, v in stats.items():
        print(f"  {k:24s} {v}")
    print(
        f"\nNOTE: {stats['clips']} real delivery clips concatenated with "
        f"{stats['gap_frames']} synthetic black gap frames. The cricket "
        f"imagery is real; the gaps and the resulting discontinuity in camera "
        f"framing are not."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())