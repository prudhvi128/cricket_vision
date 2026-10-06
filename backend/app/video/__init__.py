"""
video — Everything that touches frames as pixels rather than as numbers.

  video_io.py      opening, probing and measuring a video container.
  frame_buffer.py  `FrameRingBuffer` — the bounded retention that lets the shot
                   classifier read frames without a second decode.
  clips.py         `ClipWriterRegistry` — writes delivery clips and the
                   `ClipFrameMap` that maps clip positions back to source frames.
  renderer.py      low-level drawing primitives, byte-identical to upstream
                   CricketTracker `tracker_modules/renderer.py`.
  overlay.py       `OverlayRenderer` — composes those primitives into the
                   per-delivery overlay video and the pitch map.

Keeping the buffer, the clip writer and the overlay together is deliberate: they
share the frame-index conventions that `ClipFrameMap` defines, and the historical
defect in this system was those conventions drifting apart between modules.
"""

from .clips import ClipWriteResult, ClipWriterRegistry
from .frame_buffer import FrameRingBuffer
from .overlay import OverlayRenderer
from .video_io import get_fps, is_readable_video

__all__ = [
    "ClipWriteResult",
    "ClipWriterRegistry",
    "FrameRingBuffer",
    "OverlayRenderer",
    "get_fps",
    "is_readable_video",
]