"""
Shared test helpers: synthetic video and a stub detector.

The stub detector exists so the pipeline can be tested deterministically and
without a GPU. It is NOT a shortcut around the real detector — `test_real.py`
runs the actual YOLO model on real footage. This is for asserting the
surrounding machinery: frame mapping, provenance, segmentation, the single-pass
guarantee.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_video(
    path: Path,
    n_frames: int = 120,
    width: int = 320,
    height: int = 180,
    fps: float = 25.0,
    moving_dot_from: int | None = None,
    moving_dot_span: int = 20,
    background: int = 30,
) -> Path:
    """
    Write a synthetic video, optionally with a bright dot that a detector could
    find. Frames are dark so a JPEG round-trip stays small and lossless enough
    for exact-pixel assertions on the moving object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {path}")
    for i in range(n_frames):
        frame = np.full((height, width, 3), background, dtype=np.uint8)
        if moving_dot_from is not None and moving_dot_from <= i < moving_dot_from + moving_dot_span:
            t = i - moving_dot_from
            x = 20 + t * (width - 40) // max(1, moving_dot_span - 1)
            y = height - 30 - t * (height - 60) // max(1, moving_dot_span - 1)
            cv2.circle(frame, (int(x), int(y)), 6, (250, 250, 250), -1)
        writer.write(frame)
    writer.release()
    return path


class StubDetector:
    """
    Deterministic stand-in for BallDetector.

    Emits a position for frames within each configured window and None
    elsewhere, so delivery segmentation, the debounce, provenance and clip
    windowing can all be asserted exactly.
    """
    def __init__(self, windows: list[tuple[int, int]], model_path: str = "stub") -> None:
        self.model = object()          # tracking_service only checks this is not None
        self.model_path = model_path
        self.device = "stub"
        self.windows = windows
        self.calls = 0                 # how many times detect was called
        self.confidence = 0.9

    def detect(self, frame):
        cx, cy, _ = self.detect_with_confidence(frame)
        return None if cx is None else (cx, cy)

    def detect_with_confidence(self, frame):
        self.calls += 1
        h, w = frame.shape[:2]
        for start, end in self.windows:
            if start <= self.calls <= end:
                span = max(1, end - start)
                t = (self.calls - start) / span
                return (
                    int(w * (0.2 + 0.6 * t)),
                    int(h * (0.8 - 0.5 * t)),
                    self.confidence,
                )
        return None, None, None


def make_tagged_video(
    path: Path,
    n_frames: int,
    tags: list[tuple[int, int, int]],
    width: int = 320,
    height: int = 180,
    fps: float = 25.0,
    background: int = 30,
) -> Path:
    """
    Write a video whose frames carry a per-frame GREY VALUE identifying which
    delivery they belong to.

    This exists so clip isolation can be proven from the BYTES ON DISK rather
    than from the frame map. A frame map can claim a clip is correct; a decoded
    clip showing no frame from the neighbouring delivery is proof.

    `tags` is a list of (start_frame, end_frame, grey_value), 1-based and
    inclusive. Tag values should be well separated from `background` so JPEG and
    H.264 rounding cannot blur them together.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {path}")
    for i in range(1, n_frames + 1):
        value = background
        for start, end, tag in tags:
            if start <= i <= end:
                value = tag
        writer.write(np.full((height, width, 3), value, dtype=np.uint8))
    writer.release()
    return path


def decode_grey_means(path: Path) -> list[float]:
    """Mean grey level of every frame in a video, in order."""
    cap = cv2.VideoCapture(str(path))
    out: list[float] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.append(float(frame.mean()))
    finally:
        cap.release()
    return out


def classify_tag(grey: float, background: int, tags: list[tuple[int, int, int]],
                 tol: float = 18.0) -> str:
    """
    Map a decoded frame's mean grey back to which delivery it came from.

    Returns "background", "delivery_<grey>" or "unknown" when the value matches
    nothing — 'unknown' is a failure signal, never a pass.
    """
    best, best_dist = "unknown", tol
    if abs(grey - background) < best_dist:
        best, best_dist = "background", abs(grey - background)
    for _, _, tag in tags:
        d = abs(grey - tag)
        if d < best_dist:
            best, best_dist = f"delivery_{tag}", d
    return best


class CountingVideoCapture:
    """
    cv2.VideoCapture replacement that records every path opened.

    Used to prove the central architectural claim: the SOURCE video is opened
    exactly once, and no downstream stage reopens it.
    """

    def __init__(self) -> None:
        self.opened: list[str] = []

    def install(self):
        import cv2 as _cv2

        real = _cv2.VideoCapture
        recorder = self

        class _Wrapper:
            def __init__(self, *args, **kwargs):
                if args:
                    recorder.opened.append(str(args[0]))
                self._cap = real(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._cap, name)

            def __enter__(self):
                return self._cap.__enter__()

            def __exit__(self, *a):
                return self._cap.__exit__(*a)

        _cv2.VideoCapture = _Wrapper
        self._real = real
        return self

    def restore(self):
        import cv2 as _cv2

        if getattr(self, "_real", None) is not None:
            _cv2.VideoCapture = self._real

    def count_prefix(self, prefix: str) -> int:
        return sum(1 for p in self.opened if prefix in p)