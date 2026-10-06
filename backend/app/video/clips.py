"""
clip_writer.py — Streaming delivery-clip extraction from the tracking pass.

THE SINGLE-PASS RULE
---------------------
Clips are written by streaming frames that are ALREADY IN MEMORY (the ring
buffer) or that are arriving on the current iteration of the tracking loop.
This module never opens the source video and never seeks. That is the whole
point of the redesign: the source is decoded exactly once, and every clip
falls out of that one pass.

WHY STREAM AND NOT SEEK
-----------------------
`cap.set(CAP_PROP_POS_FRAMES, n)` seeks to the nearest preceding keyframe and
decodes forward. With H.264 B-frames or variable frame rate the frame you land
on is not the frame you asked for. Since the overlay is drawn using
`clip_frame = original_frame - frame_offset`, an off-by-N seek silently puts the
trajectory in the wrong place on screen. Sequential streaming keeps the
mapping exact and is measurably cheaper than seeking.

WHY THE CODEC FALLBACK CHAIN
----------------------------
Upstream CricketTracker hardcodes `cv2.VideoWriter_fourcc(*'avc1')` and never
checks whether the writer actually opened. On Windows the bundled OpenH264 is
frequently missing or the wrong version, in which case the failure surfaces only
when `release()` runs — after the entire tracking pass has been spent and
thrown away. Here every candidate is opened and `isOpened()`-checked before it
is accepted, and the attempt happens at clip-OPEN time rather than at close, so
the pipeline can degrade visibly instead of vanishing.

WHY `tick()` EXISTS
-------------------
Only one delivery can be open at a time (MIN_FRAMES_BETWEEN_DELIVERIES debounce),
so a single active writer suffices. `tick()` is called once per decoded frame
and appends that frame if it falls inside the open clip's window. It returns the
frame the writer still needs next, so the registry knows when to close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..core.constants import (
    SHOT_N_FRAMES,
    SHOT_SAMPLING_MAX_IN_MEMORY_BYTES,
    VIDEO_CODEC_PREFERENCE,
)
from ..schemas.tracking import ClipFrameMap, ClipMapError

log = logging.getLogger(__name__)


@dataclass
class ClipWriteResult:
    frame_map: ClipFrameMap
    path: Path
    codec_used: str
    missing_frames: list[int] = field(default_factory=list)
    # Frames sampled for shot classification, captured while the clip was being
    # written so the classifier does not have to read the clip back. Keyed by
    # 0-based position within the clip, so the order is exactly clip order.
    shot_frames: dict[int, np.ndarray] = field(default_factory=dict)
    shot_capture_truncated: bool = False


class ClipWriteError(Exception):
    """Raised when no usable codec is available to write a clip."""


class ClipWriter:
    """
    Writes one delivery clip frame-by-frame, in ascending source frame order.
    """

    def __init__(
        self,
        path: Path,
        fps: float,
        size: tuple[int, int],
        codec_preference: tuple = VIDEO_CODEC_PREFERENCE,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Guard against a container reporting fps=0 or NaN, which makes
        # VideoWriter emit a zero-timestamped file no browser will play.
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.size = (int(size[0]), int(size[1]))

        self.codec_used: str | None = None
        self.writer: cv2.VideoWriter | None = None
        self.codec_errors: list[str] = []

        self._open_with_fallback(codec_preference)

        self.frames_written = 0
        self.next_frame_needed: int | None = None
        self.window_start: int | None = None
        self.window_end: int | None = None

    # ── Construction ─────────────────────────────────────────────────────────
    def _open_with_fallback(self, codec_preference: tuple) -> None:
        """
        Try each FourCC in turn and take the first that actually opens.

        `isOpened()` is the whole point. A VideoWriter that fails to initialise
        does not raise — it returns an object whose release() discards the file.
        Without this check the failure is invisible until the tracking pass has
        already completed.
        """
        for codec in codec_preference:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                candidate = cv2.VideoWriter(
                    str(self.path), fourcc, self.fps, self.size
                )
            except Exception as exc:  # pragma: no cover - platform dependent
                self.codec_errors.append(f"{codec}: {exc}")
                continue

            if candidate is not None and candidate.isOpened():
                self.writer = candidate
                self.codec_used = codec
                log.info(
                    "Clip writer opened: codec=%s fps=%.3f size=%sx%s -> %s",
                    codec, self.fps, self.size[0], self.size[1], self.path.name,
                )
                return

            if candidate is not None:
                candidate.release()
            self.codec_errors.append(f"{codec}: isOpened() returned False")

        raise ClipWriteError(
            f"No usable video codec for {self.path.name}. Tried "
            f"{list(codec_preference)}: {'; '.join(self.codec_errors)}. "
            f"Install OpenH264 (https://github.com/cisco/openh264/releases) to "
            f"restore browser-playable H.264; mp4v will be used otherwise."
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def open_window(self, start_frame: int, end_frame: int) -> int:
        """
        Declare the inclusive source-frame window this clip covers.

        Returns the frame number the writer expects next, so the caller can
        detect completion without tracking counts.
        """
        self.window_start = int(start_frame)
        self.window_end = int(end_frame)
        self.next_frame_needed = int(start_frame)
        return self.next_frame_needed

    def truncate_window(self, new_end: int) -> None:
        """
        Cut the window short at *new_end* (inclusive).

        Used when the next delivery begins before the requested post-roll
        reaches. Nothing already written is affected — the writer simply stops
        accepting frames beyond the new end, so the clip cannot contain a second
        delivery. Never seek backwards or rewrite: the bytes already emitted stay
        exactly as they are, which is what keeps clip_frame numbering honest.
        """
        self.window_end = int(new_end)

    def write(self, frame_no: int, frame: np.ndarray) -> bool:
        """
        Append one frame. Returns False if it falls outside the open window.

        A frame whose dimensions differ from the clip's declared size is
        skipped and recorded rather than silently resized: rescaling would break
        the identity between stored trajectory pixel coordinates and clip pixel
        coordinates, which the overlay depends on.
        """
        if self.writer is None:
            raise ClipWriteError("ClipWriter used after close()")
        if self.next_frame_needed is None:
            return False
        if frame_no != self.next_frame_needed:
            return False
        if self.window_end is not None and frame_no > self.window_end:
            return False

        h, w = frame.shape[:2]
        if (w, h) != self.size:
            log.error(
                "Frame %d is %dx%d but clip %s is %dx%d; skipping to preserve "
                "trajectory coordinate mapping",
                frame_no, w, h, self.path.name, self.size[0], self.size[1],
            )
            self.next_frame_needed = frame_no + 1
            return False

        self.writer.write(frame)
        self.frames_written += 1
        self.next_frame_needed = frame_no + 1
        return True

    def skip(self, frame_no: int) -> bool:
        """
        Advance the window past a frame that cannot be written (evicted from the
        ring buffer, or a size mismatch).

        The window stays contiguous so that later frames keep their correct
        clip_frame numbers; the shortfall surfaces at finalize() as
        frames_written != frame_count, which marks the map invalid and prevents
        the overlay from being drawn on a mis-numbered clip.
        """
        if self.writer is None or self.next_frame_needed is None:
            return False
        if frame_no != self.next_frame_needed:
            return False
        self.next_frame_needed = frame_no + 1
        return True

    @property
    def complete(self) -> bool:
        """True once every frame in the declared window has been supplied."""
        if self.next_frame_needed is None or self.window_end is None:
            return False
        return self.next_frame_needed > self.window_end

    def close(self) -> int:
        """Flush and close. Returns frames actually written."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.next_frame_needed = None
        return self.frames_written

    def __enter__(self) -> "ClipWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ClipWriterRegistry:
    """
    Tracks the currently-open clip and streams frames into it during the pass.

    A clip opens when a delivery is FINALISED (the ball is lost), which is the
    earliest moment its start and end are both known. At that point the
    pre-roll frames are still resident in the ring buffer and are drained
    immediately; every subsequent frame arrives through `tick()`.

    ONE CLIP = ONE DELIVERY = ONE BATTING EVENT
    --------------------------------------------
    A clip's window is bounded by its neighbours so that it can never contain a
    second delivery:

      * the END is cut at `next_delivery_start - 1` by `close_before()`, which
        the tracking loop calls the moment a new delivery begins;
      * the START is pushed to `previous_clip_end + 1` by `open_for()`, so this
        clip cannot re-use frames the previous clip already spent.

    `_last_written_frame` is the highest source frame committed to ANY clip. It
    is the single source of truth for both bounds, which is what makes the
    clips disjoint and ordered rather than merely non-overlapping by intent.
    """

    def __init__(
        self,
        ring,
        codec_preference: tuple = VIDEO_CODEC_PREFERENCE,
        shot_frame_budget: int = SHOT_N_FRAMES,
        shot_max_in_memory_bytes: int = SHOT_SAMPLING_MAX_IN_MEMORY_BYTES,
    ) -> None:
        self.ring = ring
        self.codec_preference = codec_preference
        self.results: dict[int, ClipWriteResult] = {}
        self.errors: dict[int, str] = {}
        self._active: ClipWriter | None = None
        self._active_id: int | None = None
        self._active_map: ClipFrameMap | None = None
        self._missing: list[int] = []

        # Highest source frame written to any clip so far. The next clip cannot
        # start at or before it.
        self._last_written_frame: int = 0
        # Delivery that owns _last_written_frame, for log messages only.
        self._last_written_delivery_id: int | None = None

        # ── Exclusion Zones / Barriers ───────────────────────────────────────
        # List of (start_frame, end_frame, kind) for rejected candidates or
        # untracked activity windows that clips must not swallow.
        self._exclusion_zones: list[tuple[int, int, str]] = []

        # ── Shot-sampling capture ────────────────────────────────────────────
        # The frames the shot classifier needs are picked up as they stream past,
        # which is the whole point of the single pass: the classifier does not
        # have to seek backwards through a clip, and the source video is never
        # reopened. Which frames are needed is known at open time because the
        # window length is, so this costs one set lookup per frame.
        self.shot_frame_budget = shot_frame_budget
        self.shot_max_in_memory_bytes = shot_max_in_memory_bytes
        self._capture: dict[int, np.ndarray] = {}
        self._capture_needed: set[int] = set()
        self._capture_enabled = True

    # ── State ────────────────────────────────────────────────────────────────
    @property
    def active(self) -> bool:
        return self._active is not None

    @property
    def active_delivery_id(self) -> int | None:
        return self._active_id

    @property
    def last_written_frame(self) -> int:
        """Highest source frame committed to any clip. 0 when none written."""
        return self._last_written_frame

    def register_exclusion_zone(self, start_frame: int, end_frame: int, kind: str = "EXCLUSION_ZONE") -> None:
        """Record an impermeable barrier (e.g. rejected candidate or untracked activity)."""
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if end_frame < start_frame:
            end_frame = start_frame
        self._exclusion_zones.append((start_frame, end_frame, str(kind)))

    def exclusion_ranges(self) -> list[tuple[int, int]]:
        """All impermeable barrier spans as (start, end) pairs."""
        return [(a, b) for a, b, _ in self._exclusion_zones]

    def exclusion_zone_kinds(self) -> list[tuple[int, int, str]]:
        """All impermeable barrier spans with the kind that registered them."""
        return list(self._exclusion_zones)

    @property
    def active_next_frame(self) -> Optional[int]:
        """Source frame the open clip still needs, or None when no clip is open."""
        if self._active is None:
            return None
        return self._active.next_frame_needed

    @property
    def active_ball_lost_frame(self) -> Optional[int]:
        """Ball-loss frame of the open clip, or None when no clip is open."""
        if self._active_map is None:
            return None
        return self._active_map.ball_lost_frame

    def next_free_frame(self) -> int:
        """
        Earliest source frame a new clip may start on.

        A clip must not re-use a frame the previous clip already contains, so
        this is one past the previous clip's last frame.
        """
        return self._last_written_frame + 1

    # ── Open / stream / close ────────────────────────────────────────────────
    def open_for(
        self,
        delivery_id: int,
        start_frame: int,
        end_frame: int,
        fps: float,
        size: tuple[int, int],
        out_path: Path,
        post_roll_frames: int = 0,
        pre_roll_actual_sec: float | None = None,
        ball_lost_frame: int | None = None,
        anchor_frame: int | None = None,
        max_pre_roll_frames: int | None = None,
        max_post_roll_frames: int | None = None,
    ) -> ClipFrameMap:
        """
        Open a clip and drain its pre-roll window from the ring buffer.

        `start_frame` is the DESIRED start (detection - pre-roll). It is pushed
        forward to one past the previous clip's last frame, because a clip that
        began inside the previous delivery's post-roll would contain that
        delivery's batting event as well as its own. Pre-roll is sacrificed for
        contiguity; the earlier delivery keeps its post-roll because that is
        where its stroke is.

        Returns the ClipFrameMap so the caller can stamp clip_frame numbers onto
        the trajectory immediately, before any writing has happened.

        THIS IS ALSO THE LAST LINE OF DEFENCE FOR THE BARRIERS
        ------------------------------------------------------
        The caller already ran a bounded search over the horizon signals, but that
        search is advisory: horizons keep being discovered while the pass runs, and
        a zone registered a moment before this call may sit inside the window the
        search already chose. So the window is re-checked here, where enforcement
        is guaranteed:

          * the start is raised past any exclusion zone that CONTAINS it or that
            begins after it but before the clip's own anchor — the case the earlier
            loop missed, because it only compared zones against the requested start
            and silently ignored a zone beginning later inside the pre-roll;
          * the end is cut before any exclusion zone that begins after the ball was
            lost, so this clip can never run into footage filed against something
            else. Zones that begin before the ball was lost are NOT cut at, because
            doing so would remove the stroke;
          * `max_pre_roll_frames` / `max_post_roll_frames` are applied as hard
            bounds, so no caller can declare a window longer than the safety ceiling
            even if it forgets to cap it.
        """
        if self._active is not None:
            # Only one delivery can be open at a time (deliveries are strictly
            # sequential: a new one begins after the previous is finalised). If
            # this ever fires, the debounce is broken — close the old clip and
            # say so rather than corrupting both mappings.
            log.error(
                "Opening clip for delivery %d while delivery %s is still open; "
                "closing the previous clip early",
                delivery_id, self._active_id,
            )
            self._finish_active()

        requested_start = int(start_frame)
        requested_end = int(end_frame)

        # ── Hard roll caps ──────────────────────────────────────────────────
        # The resolved ceilings are the authority. A caller that hands over a
        # longer window gets it shortened rather than obeyed: the caps exist
        # because a longer window is where a neighbouring event hides.
        capped_start = requested_start
        if anchor_frame is not None and max_pre_roll_frames is not None:
            capped_start = max(
                capped_start, int(anchor_frame) - int(max_pre_roll_frames)
            )
        capped_end = requested_end
        if ball_lost_frame is not None and max_post_roll_frames is not None:
            capped_end = min(
                capped_end, int(ball_lost_frame) + int(max_post_roll_frames)
            )
        if capped_start != requested_start or capped_end != requested_end:
            log.info(
                "Delivery %d: window capped to %d-%d from the requested %d-%d by "
                "the resolved roll ceilings (pre<=%s post<=%s frames)",
                delivery_id, capped_start, capped_end, requested_start, requested_end,
                max_pre_roll_frames, max_post_roll_frames,
            )

        # ── Enforce disjointness with previous clip and exclusion zones ─────
        floor = self.next_free_frame()
        ceiling = None
        anchor = int(anchor_frame) if anchor_frame is not None else None
        for ez_start, ez_end, kind in self._exclusion_zones:
            # A zone entirely before the window cannot overlap it.
            if ez_end < capped_start:
                floor = max(floor, ez_end + 1)
                continue
            # A zone entirely after it can only shorten it.
            if ez_start > capped_end:
                continue
            if ez_start <= capped_start <= ez_end:
                # The requested start lands inside a zone filed against something
                # else. Step over it.
                floor = max(floor, ez_end + 1)
            elif anchor is not None and ez_start < anchor <= ez_end:
                # The zone begins after the requested start and reaches into this
                # delivery's own pre-roll. Previously ignored: the old loop only
                # compared zones against `start_frame`, so a barrier discovered
                # after the bounded search had chosen its window was let straight
                # through.
                floor = max(floor, ez_end + 1)
            elif (
                ez_start > capped_start
                and (ball_lost_frame is None or ez_start > int(ball_lost_frame))
            ):
                # The zone begins inside the post-roll, after the ball was lost, so
                # stopping before it costs no stroke and keeps another event's
                # footage out.
                stop = ez_start - 1
                ceiling = stop if ceiling is None else min(ceiling, stop)

        effective_start = max(capped_start, floor)
        effective_end = capped_end if ceiling is None else min(capped_end, ceiling)
        start_clamped = effective_start > capped_start
        end_clamped = effective_end < capped_end
        if start_clamped:
            log.info(
                "Delivery %d: pre-roll clamped from frame %d to %d; frames "
                "%d-%d already belong to delivery %s or an exclusion zone",
                delivery_id, capped_start, effective_start, capped_start,
                effective_start - 1, self._last_written_delivery_id,
            )
        if end_clamped:
            log.info(
                "Delivery %d: post-roll clipped at frame %d; an exclusion zone "
                "begins at %d",
                delivery_id, effective_end, effective_end + 1,
            )

        effective_end = max(effective_end, effective_start - 1)

        writer = ClipWriter(out_path, fps, size, self.codec_preference)

        frame_map = ClipFrameMap(
            clip_start_frame=effective_start,
            clip_end_frame=effective_end,
            frame_count=0,
            width=int(size[0]),
            height=int(size[1]),
            fps=fps,
            codec=writer.codec_used or "",
            post_roll_frames=post_roll_frames,
            pre_roll_actual_sec=pre_roll_actual_sec,
            ball_lost_frame=ball_lost_frame,
            start_clamped_by_previous=start_clamped,
        )
        writer.open_window(effective_start, effective_end)

        self._active = writer
        self._active_id = delivery_id
        self._active_map = frame_map
        self._missing = []
        self._last_written_delivery_id = delivery_id
        self._prepare_capture(frame_map, size)

        if effective_end < effective_start:
            # Nothing to write: the previous clip already reached past this
            # delivery's ball-loss frame, or a barrier consumed the whole window.
            # Emit an empty, invalid map rather than a clip containing the wrong
            # delivery, and name whichever bound applied.
            frame_map.frame_map_valid = False
            frame_map.frame_map_error = (
                f"empty window: start {effective_start} > end {effective_end}; "
                + (
                    "an exclusion zone removed the whole window"
                    if end_clamped
                    else "the previous clip's window already covered it"
                )
                + f" (last committed delivery "
                f"{self._last_written_delivery_id})"
            )
            self._finish_active()
            return frame_map

        # Drain only what the ring already holds. The window normally extends
        # past the current frame (post-roll has not been decoded yet), so
        # draining to `end_frame` would mark every future frame as missing.
        # Everything after the newest resident frame arrives via tick().
        newest_resident = self.ring.last_frame
        drain_to = min(effective_end, newest_resident) if newest_resident else effective_end
        self._drain_ring(drain_to)

        return frame_map

    def close_before(
        self,
        frame_no: int,
        next_delivery_id: int | None = None,
        reason: str | None = None,
        min_end: int | None = None,
    ) -> ClipWriteResult | None:
        """
        Close the open clip so that it contains NO frame at or after *frame_no*.

        Called the moment a new delivery begins, or a black gap / camera cut /
        untracked activity window is reached. Without this the previous
        delivery's post-roll would swallow the next ball, and one clip would
        contain two batting events — the defect this method exists to prevent.

        The truncation happens BEFORE `tick()` writes *frame_no*, so the frame
        that reveals the new delivery is never committed to the old clip. No
        frame already written is removed or renumbered; the window simply stops
        accepting frames, and the frame map is shortened to match what was
        actually written so the mapping stays exact.

        The check below is UNCONDITIONAL. `_drain_ring` writes ahead of the tracking
        loop from the ring buffer, so by the time a barrier is evaluated on frame
        F the writer may already have committed F — the drain for a delivery
        finalised at F reaches the newest resident frame, which is F itself.
        Truncating then shortens the map by one while the extra frame is already
        on disk, which is precisely how INVARIANT B was broken on real footage
        (delivery 29 of real_cricket.mp4: 79 frames written into a 78-frame map,
        which suppressed its overlay). The guard used to be opt-in via
        `only_if_unwritten`, and four of the six callers never opted in, so the
        protection the docstring described was almost never active.

        `min_end` is the floor the cut may not go below — normally the delivery's
        own ball-loss frame. It exists because "a barrier was reached" is not on
        its own a reason to cut: a barrier that sits BEFORE the ball was lost
        would remove the ball's own footage, and a barrier detected during the
        stroke would remove the stroke. A caller that cannot guarantee the
        barrier lies after the ball was lost passes the floor and this method
        declines to cut.
        """
        if self._active is None or self._active_map is None:
            return None

        needed = self._active.next_frame_needed
        if needed is None or needed > int(frame_no):
            # The writer has already committed *frame_no* itself (or past it).
            # Shortening the map now would leave those bytes on disk
            # unaccounted for and break INVARIANT B, so decline the cut and let
            # the barrier's exclusion zone protect the NEXT clip instead.
            return None

        new_end = int(frame_no) - 1
        if min_end is not None and new_end < int(min_end):
            # The barrier is behind the footage this clip is obliged to keep.
            log.info(
                "Delivery %s: refusing to cut at frame %d (%s) because that is "
                "before the ball-loss floor of %d; the clip keeps its own stroke",
                self._active_id, new_end, reason or "barrier", int(min_end),
            )
            return None
        if new_end >= self._active_map.clip_end_frame:
            # The requested window already ends before the new delivery; nothing
            # to cut. It will be closed by its own completion.
            return None

        log.info(
            "Delivery %s: cutting clip at frame %d (%s) because delivery %s "
            "begins there (requested end was %d)",
            self._active_id, new_end, reason or "barrier",
            next_delivery_id if next_delivery_id is not None else "another",
            self._active_map.clip_end_frame,
        )

        self._active.truncate_window(new_end)
        self._active_map.truncate_end(new_end, reason=reason)
        return self._finish_active()

    def _prepare_capture(self, frame_map: ClipFrameMap, size: tuple[int, int]) -> None:
        """
        Decide which clip-frame positions to retain for shot classification.

        Uses the same `np.linspace` rule as the classifier's file-based path, so
        the label does not depend on which path supplied the pixels. Capture is
        disabled up front when the memory cost is unreasonable (a 30-frame
        4K window is ~750 MB); the classifier then reads the clip instead.
        """
        self._capture = {}
        self._capture_needed = set()
        self._capture_enabled = True

        n = frame_map.frame_count
        if n <= 0:
            self._capture_enabled = False
            return

        stride = max(1, self.shot_frame_budget)
        frame_bytes = int(size[0]) * int(size[1]) * 3
        projected = stride * frame_bytes
        if projected > self.shot_max_in_memory_bytes:
            log.info(
                "Shot capture disabled: %d frames x %d bytes = %.0f MiB exceeds "
                "the %.0f MiB budget; the classifier will read the clip file",
                stride, frame_bytes, projected / 1048576,
                self.shot_max_in_memory_bytes / 1048576,
            )
            self._capture_enabled = False
            return

        if n <= stride:
            self._capture_needed = set(range(n))
        else:
            self._capture_needed = {
                int(i) for i in np.linspace(0, n - 1, stride, dtype=np.int64)
            }

    def _maybe_capture(self, frame_no: int, frame: np.ndarray) -> None:
        """Retain *frame* if it is one of the positions the classifier needs."""
        if not self._capture_enabled or self._active_map is None:
            return
        clip_idx = frame_no - self._active_map.frame_offset - 1  # 0-based
        if clip_idx in self._capture_needed and clip_idx not in self._capture:
            # The tracking loop does not mutate frames after decode (push encodes,
            # tick writes), so holding the reference is safe and avoids a
            # full-frame copy per sample.
            self._capture[clip_idx] = frame

    def _drain_ring(self, upto_frame: int) -> None:
        """
        Write every buffered frame <= upto_frame into the active clip.

        Walks the range in order rather than iterating what happens to be
        resident, because a hole must advance the window explicitly. Skipping it
        silently would renumber every later frame and misplace the overlay.
        """
        if self._active is None or self._active_map is None:
            return
        needed = self._active.next_frame_needed
        if needed is None or needed > upto_frame:
            return

        for frame_no in range(needed, upto_frame + 1):
            img = self.ring.get(frame_no)
            if img is None:
                self._missing.append(frame_no)
                self._active.skip(frame_no)
                continue
            self._active.write(frame_no, img)
            self._maybe_capture(frame_no, img)
            if frame_no > self._last_written_frame:
                self._last_written_frame = frame_no

        if self._active.complete:
            self._finish_active()

    def tick(self, frame_no: int, frame: np.ndarray) -> None:
        """
        Feed the current tracking-loop frame to the open clip.

        Cheap no-op when no clip is open, which is the overwhelmingly common
        case (a clip is open for a fraction of the video).
        """
        if self._active is None or self._active_map is None:
            return
        needed = self._active.next_frame_needed
        if needed is None:
            return
        if frame_no < needed:
            return

        if self._active.write(frame_no, frame):
            self._maybe_capture(frame_no, frame)
            if frame_no > self._last_written_frame:
                self._last_written_frame = frame_no
            if self._active.complete:
                self._finish_active()

    def _finish_active(self) -> ClipWriteResult | None:
        """Close the active clip and run INVARIANT A/B checks."""
        if self._active is None or self._active_map is None or self._active_id is None:
            return None

        writer = self._active
        frame_map = self._active_map
        delivery_id = self._active_id

        written = writer.close()
        frame_map._frames_written = written
        frame_map.finalize()

        # INVARIANT A — no frame in the declared window may be missing.
        missing = sorted(set(self._missing))
        if missing and frame_map.frame_map_valid:
            frame_map.frame_map_valid = False
            frame_map.frame_map_error = (
                f"{len(missing)} frame(s) missing from clip window, first at "
                f"original frame {missing[0]}"
            )

        result = ClipWriteResult(
            frame_map=frame_map,
            path=writer.path,
            codec_used=writer.codec_used or "",
            missing_frames=missing,
            shot_frames=dict(self._capture),
            shot_capture_truncated=(
                self._capture_enabled
                and len(self._capture) < len(self._capture_needed)
            ),
        )
        self.results[delivery_id] = result
        if missing:
            log.error(
                "Clip %s has frame-map gaps (first=%s); overlay will be skipped",
                writer.path.name, missing[0],
            )

        self._active = None
        self._active_id = None
        self._active_map = None
        self._missing = []
        self._capture = {}
        self._capture_needed = set()
        return result

    def flush_and_close(self) -> ClipWriteResult | None:
        """
        Close any clip still open at end of video.

        The window may extend past the last decoded frame (a delivery near the
        end of the video whose post-roll is truncated). The map is marked
        invalid in that case, because the declared window and the written frame
        count genuinely disagree — the video ended, we did not lose frames.
        """
        if self._active is None:
            return None
        log.info(
            "Closing clip for delivery %s at end of video (window end %s)",
            self._active_id, self._active_map.clip_end_frame if self._active_map else "?",
        )
        return self._finish_active()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def frame_map_for(self, delivery_id: int) -> ClipFrameMap | None:
        r = self.results.get(delivery_id)
        return r.frame_map if r else None

    def assert_mappable(self, delivery_id: int) -> ClipFrameMap:
        """
        Return the map for a delivery or raise — used by the overlay renderer.

        A trajectory drawn onto a clip with a broken frame mapping is worse than
        no trajectory at all: it looks correct and is not.
        """
        fm = self.frame_map_for(delivery_id)
        if fm is None:
            raise ClipMapError(f"No clip frame map for delivery {delivery_id}")
        if not fm.frame_map_valid:
            raise ClipMapError(
                f"Delivery {delivery_id} frame map invalid: {fm.frame_map_error}"
            )
        return fm