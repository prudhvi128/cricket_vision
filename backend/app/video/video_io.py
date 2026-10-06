"""
video_io.py — Opening, probing, and measuring video sources.

Everything in this module touches a `cv2.VideoCapture` or a video file on disk.
It is deliberately free of ML logic: the tracking service owns *what* is detected
in a frame, this module owns *how the container is opened and described*.

`get_fps` is ported verbatim from CricketTracker `tracker_modules/utils.py`.
`_is_readable_video` (renamed `is_readable_video`) moved here from the API
layer, because upload validation is a property of the file, not of the route.
"""

from __future__ import annotations

from pathlib import Path

import cv2


def get_fps(cap: cv2.VideoCapture, default: float = 30.0) -> float:
    """Extract FPS from VideoCapture; fall back to *default* if invalid."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 1000:
        print(f"⚠️  Invalid FPS from metadata ({fps}), defaulting to {default}")
        return default
    print(f"📹 Video FPS (from metadata): {fps:.3f}")
    return fps


def is_readable_video(path: Path) -> bool:
    """
    Confirm OpenCV can open the file and that it actually has frames.

    Catches the two failures that otherwise surface minutes later as an opaque
    tracking error: a file that is not a video at all, and a container whose
    header parses but which contains no frames. Returns immediately so an invalid
    upload is rejected at the door rather than after a model load.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None)
    except Exception:
        return False
    finally:
        cap.release()


__all__ = ["get_fps", "is_readable_video"]