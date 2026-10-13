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

import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple

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


def writer_candidates(
    codec_preference: tuple = (),
) -> Iterator[Tuple[str, int, Optional[int]]]:
    """
    Yield `(label, fourcc, apiPreference)` candidates for `cv2.VideoWriter`.

    The FIRST candidate is the Windows Media Foundation H.264 encoder
    (`cv2.CAP_MSMF` + FourCC `H264`). It produces browser-playable H.264
    without OpenCV's bundled FFmpeg wrapper attempting — and failing — to
    load Cisco's OpenH264 DLL, which is what logs
    "Failed to load OpenH264 library" / "Incorrect library version loaded"
    on machines that do not ship that DLL. On non-Windows platforms (or a
    build without MSMF) the MSMF candidate is skipped silently and the
    caller's `codec_preference` is used unchanged.

    After the MSMF candidate, every codec in `codec_preference` is yielded
    with the default backend (`apiPreference=None`), so the existing
    avc1-then-mp4v fallback chain is preserved exactly.
    """
    if sys.platform == "win32" and hasattr(cv2, "CAP_MSMF"):
        try:
            yield ("H264", cv2.VideoWriter_fourcc(*"H264"), cv2.CAP_MSMF)
        except Exception:
            pass
    for codec in codec_preference:
        try:
            yield (codec, cv2.VideoWriter_fourcc(*codec), None)
        except Exception:
            continue


def open_video_writer(
    path: Path,
    fps: float,
    size: tuple[int, int],
    codec_preference: tuple = (),
) -> tuple[Optional[cv2.VideoWriter], str, list[str]]:
    """
    Open a `cv2.VideoWriter` at *path*, trying `writer_candidates` in order.

    Returns `(writer, codec_label, errors)`. `writer` is None when no
    candidate opened; `codec_label` names the one that did. `isOpened()` is
    the whole check: a VideoWriter that fails to initialise does not raise —
    it returns an object whose `release()` discards the file — so without
    this check the failure is invisible until the pass has already completed.
    """
    errors: list[str] = []
    for label, fourcc, api in writer_candidates(codec_preference):
        try:
            if api is None:
                candidate = cv2.VideoWriter(str(path), fourcc, fps, size)
            else:
                candidate = cv2.VideoWriter(str(path), api, fourcc, fps, size)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            continue
        if candidate is not None and candidate.isOpened():
            return candidate, label, errors
        if candidate is not None:
            candidate.release()
        errors.append(f"{label}: isOpened() returned False")
    return None, "", errors


__all__ = [
    "get_fps",
    "is_readable_video",
    "open_video_writer",
    "writer_candidates",
]