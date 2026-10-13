"""
draw_seg_overlays.py — Frame-seek visual check of accepted pitch quads.

Seeks individual frames from the source video (NOT a second decode pass) and
draws the accepted segmentation quads recorded during the A/B ON run, so the
corners can be eyeballed against the real footage.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core.config import PITCH_SEG_CONFIG  # noqa: E402
from app.core.constants import PITCH_SEG_DEFAULT_WEIGHTS  # noqa: E402
from app.pitch.detector import PitchDetectError  # noqa: E402
from app.pitch.segmentation_detector import LocalPitchSegDetector  # noqa: E402
from dataclasses import replace  # noqa: E402

VIDEO = ROOT / "real_cricket.mp4"
OUT = ROOT / "docs" / "diagnostics" / "seg_overlays"
# Midpoints of every 5th delivery from the A/B ON run: these frames were
# inside footage a delivery actually happened on, so the pitch was on camera.
FRAMES = [122, 1863, 4267, 6075, 7776, 9233, 10260, 11180]


def main() -> int:
    if not VIDEO.is_file():
        print(f"missing {VIDEO}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    detector = LocalPitchSegDetector.from_config(
        replace(PITCH_SEG_CONFIG, model=PITCH_SEG_DEFAULT_WEIGHTS)
    )
    capture = cv2.VideoCapture(str(VIDEO))
    for frame_number in FRAMES:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            print(f"frame {frame_number}: read failed")
            continue
        try:
            detection = detector.detect(frame, frame_number,
                                        (frame.shape[1], frame.shape[0]))
        except PitchDetectError as exc:
            # A frame where the model finds nothing is a legitimate outcome
            # (no_mask), not a crash: record it and keep going.
            cv2.putText(frame, f"f{frame_number} NO MASK", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            out = OUT / f"quad_{frame_number:05d}.jpg"
            cv2.imwrite(str(out), frame)
            print(f"frame {frame_number}: no mask ({exc}) -> {out.name}")
            continue
        points = np.array([[k.x, k.y] for k in detection.keypoints], np.int32)
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        for keypoint in detection.keypoints:
            cv2.circle(frame, (int(keypoint.x), int(keypoint.y)), 5,
                       (0, 0, 255), -1)
        label = f"f{frame_number} conf={detection.confidence:.2f}"
        cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        out = OUT / f"quad_{frame_number:05d}.jpg"
        cv2.imwrite(str(out), frame)
        print(f"frame {frame_number}: conf={detection.confidence:.2f} "
              f"corners={[(round(k.x), round(k.y)) for k in detection.keypoints]} "
              f"-> {out.name}")
    capture.release()
    stats = detector.stats()
    print(f"\ntally: samples={stats['samples']} accepted={stats['accepted']} "
          f"no_mask={stats['no_mask']} errors={stats['errors']}")
    print(f"overlays in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
