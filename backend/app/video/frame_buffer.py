"""
ring_buffer.py — Bounded rolling frame store.

WHY THIS EXISTS
---------------
A delivery clip must begin BEFORE the delivery was detected, otherwise there is
no run-up and no bowler in frame. By the time we know a delivery has started,
those frames have already been decoded. Upstream CricketShot-Classification
solved this by re-opening the source video and seeking backwards
(`AutoClipper._extract_clip` -> `cap.set(CAP_PROP_POS_FRAMES)`).

That is not available to us. `cap.set(CAP_PROP_POS_FRAMES)` seeks to a
*keyframe* and decodes forward to the requested frame, so with H.264 B-frames or
variable frame rate the returned frame index is not the index you asked for.
Clip frame numbering must be exact for the overlay to land on the right ball
pixel, so seeking is unusable and the pre-roll has to be held in memory.

WHY JPEG, NOT RAW
-----------------
Raw 1080p is 5.93 MiB/frame. A 180-frame window would be 1.04 GiB. JPEG q92
costs roughly 45 MiB for the same window — a 23x saving — and q92 is
visually lossless for the purposes of a thin trajectory overlay.

Two caps are enforced independently (frame count AND bytes) so that a
high-resolution or high-bitrate feed cannot blow the memory budget on the frame
count alone, and a pathological sequence of incompressible frames cannot blow it
on the byte count alone.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import cv2
import numpy as np

from ..core.constants import (
    RING_BUFFER_CAPACITY_FRAMES,
    RING_BUFFER_JPEG_QUALITY,
    RING_BUFFER_MAX_BYTES,
)

log = logging.getLogger(__name__)


class FrameRingBuffer:
    """
    Frame-numbered rolling store of JPEG-compressed frames.

    Frames are keyed by their 1-based SOURCE frame number, which is the single
    authoritative timebase in the system. Nothing is ever guessed: a frame that
    was not decoded is simply absent, and a caller asking for it gets None
    rather than a substitute.
    """

    def __init__(
        self,
        capacity_frames: int = RING_BUFFER_CAPACITY_FRAMES,
        max_bytes: int = RING_BUFFER_MAX_BYTES,
        jpeg_quality: int = RING_BUFFER_JPEG_QUALITY,
    ) -> None:
        if capacity_frames < 1:
            raise ValueError("capacity_frames must be >= 1")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be 1..100")

        self.capacity_frames = capacity_frames
        self.max_bytes = max_bytes
        self.jpeg_quality = jpeg_quality

        self._frames: "OrderedDict[int, bytes]" = OrderedDict()
        self._bytes = 0

        # Instrumentation
        self.encoded_count = 0
        self.evicted_count = 0
        self.dropped_encode_count = 0
        self.dropped_oversize_count = 0
        self.peak_bytes = 0

    # ── Write ────────────────────────────────────────────────────────────────
    def push(self, frame_no: int, frame: np.ndarray) -> bool:
        """
        Append one decoded frame. Returns True if it was stored.

        Two frames can fail to be stored, both counted rather than raised:
          * the frame will not JPEG-encode, or
          * the encoded frame on its own exceeds `max_bytes`.

        Neither aborts the pass. Losing one frame degrades a clip's opening
        seconds; raising would discard an entire analysis. Callers detect the
        loss through `dropped_encode_count` / `dropped_oversize_count`, and the
        clip writer turns any resulting hole into an invalid frame map so the
        affected delivery is never overlaid.
        """
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            self.dropped_encode_count += 1
            log.warning("JPEG encode failed for frame %d; dropped", frame_no)
            return False

        payload = buf.tobytes()

        # Enforce the byte budget strictly. A single frame larger than the
        # whole budget is not stored: keeping it would mean the documented
        # ceiling is not a ceiling. The budget is a guard against pathological
        # input, so a frame that big means the configuration is too small for
        # this footage and dropping is the honest response.
        if len(payload) > self.max_bytes:
            self.dropped_oversize_count += 1
            log.warning(
                "Frame %d encodes to %d bytes, exceeding the %d byte ring "
                "budget on its own; dropped",
                frame_no, len(payload), self.max_bytes,
            )
            return False

        # Defensive: a frame_no already present means the decode loop restarted
        # or a caller double-pushed. Replace rather than corrupt the mapping.
        if frame_no in self._frames:
            self._bytes -= len(self._frames.pop(frame_no))

        self._frames[frame_no] = payload
        self._bytes += len(payload)
        self.encoded_count += 1
        if self._bytes > self.peak_bytes:
            self.peak_bytes = self._bytes

        self._evict()
        return True

    def _evict(self) -> None:
        """Drop oldest frames until both budgets are satisfied.

        The `len > 1` guard is a backstop only: `push` already refuses any frame
        larger than `max_bytes`, so the budget cannot be breached by a lone
        resident frame.
        """
        while len(self._frames) > self.capacity_frames or (
            self._bytes > self.max_bytes and len(self._frames) > 1
        ):
            _, payload = self._frames.popitem(last=False)
            self._bytes -= len(payload)
            self.evicted_count += 1

    # ── Read ─────────────────────────────────────────────────────────────────
    def __contains__(self, frame_no: int) -> bool:
        return frame_no in self._frames

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def byte_size(self) -> int:
        return self._bytes

    @property
    def first_frame(self) -> int | None:
        return next(iter(self._frames), None)

    @property
    def last_frame(self) -> int | None:
        return next(reversed(self._frames), None)

    def get(self, frame_no: int) -> np.ndarray | None:
        """Decode one frame, or None if it has already been evicted."""
        payload = self._frames.get(frame_no)
        if payload is None:
            return None
        return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)

    def peek(self, frame_no: int) -> bytes | None:
        """Raw JPEG bytes, for callers that want to avoid a decode round-trip."""
        return self._frames.get(frame_no)

    def iter_frames(self, start_frame: int, end_frame: int):
        """
        Yield (frame_no, ndarray) for every buffered frame in [start, end].

        Frames that are absent are skipped, not substituted. Yields in ascending
        frame order so a caller can stream straight to a writer.

        The caller is responsible for INVARIANT A (every frame in the requested
        range must be present) when the range is expected to be contiguous.
        """
        if end_frame < start_frame:
            return
        for frame_no in range(start_frame, end_frame + 1):
            payload = self._frames.get(frame_no)
            if payload is None:
                continue
            img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield frame_no, img

    def has_all(self, start_frame: int, end_frame: int) -> bool:
        """True when every frame in [start, end] is still resident (INVARIANT A)."""
        if end_frame < start_frame:
            return True
        return not self.missing_frames(start_frame, end_frame)

    def missing_frames(self, start_frame: int, end_frame: int) -> list[int]:
        """Explicit list of gaps in [start, end] — used to report, not to paper over."""
        return [
            n
            for n in range(start_frame, end_frame + 1)
            if n not in self._frames
        ]

    # ── Maintenance ──────────────────────────────────────────────────────────
    def trim_before(self, frame_no: int) -> int:
        """Drop everything strictly before frame_no. Returns the number dropped."""
        dropped = 0
        while self._frames:
            oldest = next(iter(self._frames))
            if oldest >= frame_no:
                break
            self._bytes -= len(self._frames.pop(oldest))
            dropped += 1
            self.evicted_count += 1
        return dropped

    def clear(self) -> None:
        self._frames.clear()
        self._bytes = 0

    def stats(self) -> dict:
        return {
            "capacity_frames": self.capacity_frames,
            "max_bytes": self.max_bytes,
            "jpeg_quality": self.jpeg_quality,
            "encoded_count": self.encoded_count,
            "resident_frames": len(self._frames),
            "evicted_count": self.evicted_count,
            "dropped_encode_count": self.dropped_encode_count,
            "dropped_oversize_count": self.dropped_oversize_count,
            "byte_size": self._bytes,
            "peak_bytes": self.peak_bytes,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
        }