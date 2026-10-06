"""
event_horizons.py — EVENT HORIZONS for dynamic, barrier-bounded clip windows.

WHY THIS EXISTS
---------------
`EventSegmenter` decides WHERE ONE BATTING EVENT ENDS and the next begins, from
ball-tracking provenance alone. But the *clip window* around an accepted event
is a different question: how far may its pre-roll reach back, and its post-roll
reach forward, before it starts showing footage that belongs to something else?

The old answer was a static `detection_start - 4s` / `ball_lost + 2s`. That is
wrong wherever deliveries sit closer together than `PRE_ROLL + POST_ROLL`: the
window swallows a neighbouring event, a black transition, or a camera cut. This
module answers the window question from evidence the single pass already
produced — no second decode, no second detector run, no second filter run.

THE THREE TIERS OF BARRIER (from the fix design)
-----------------------------------------------
  TIER 1 — HARD BARRIERS
      black / blank frame runs   (nothing here is cricket action)
      scene cuts / camera cuts   (a clip may not span two camera segments)
      previously accepted clip end (clips must stay disjoint)
      an already-registered exclusion zone
  TIER 2 — EVENT HORIZONS
      activity windows with NO confirmed ball detection: a real delivery the
      detector missed. It is not promoted to a delivery, but a neighbouring
      clip's roll may never cross it.
      inactivity / reset valleys: the dead-ball pause between two events.
  TIER 3 — SUPPORTING
      full-frame motion energy is the fallback when no pitch ROI is usable, so
      the search still degrades gracefully on a moving or uncalibrated camera.

The search starts at the delivery's own reliable anchor and walks OUTWARD only
as far as the evidence allows, stopping at the nearest barrier. When no barrier
is found it stops at the roll CAP. A shorter, safe clip always wins over a
longer clip that might contain another event.

PURE DATA, NO VIDEO
-------------------
Everything here is measured during the pass and recorded frame by frame. The
values are cheap: a handful of scalars per decoded frame. The module never
opens a video and never seeks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

__all__ = [
    "ActivityWindow",
    "ActivityWindowTracker",
    "ClipWindow",
    "FrameSignal",
    "SignalTimeline",
]


# Why a roll search stopped. Persisted with the delivery so the boundary can be
# audited without re-running the pass.
STOP_BLACK_GAP = "black_gap"
STOP_SCENE_CUT = "scene_cut"
STOP_EXCLUSION_ZONE = "exclusion_zone"
STOP_INACTIVITY_VALLEY = "inactivity_valley"
STOP_MAX_PRE_ROLL = "max_pre_roll"
STOP_MAX_POST_ROLL = "max_post_roll"
STOP_START_OF_VIDEO = "start_of_video"
STOP_UNKNOWN = "unknown"


@dataclass(frozen=True)
class FrameSignal:
    """One frame's barrier-relevant measurements, recorded inside the pass."""

    frame: int
    is_black: bool = False
    is_scene_cut: bool = False
    motion: float = 0.0
    pitch_motion: float = 0.0
    confirmed: bool = False
    candidate: bool = False


@dataclass
class ClipWindow:
    """
    A resolved, barrier-bounded clip window.

    `start`/`end` are inclusive 1-based source frames. `start_reason` and
    `end_reason` say which barrier (or cap) determined each side, so a wrong
    boundary is diagnosable from the record alone.
    """

    start: int
    end: int
    start_reason: str
    end_reason: str
    pre_roll_frames: int = 0
    post_roll_frames: int = 0

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "start_reason": self.start_reason,
            "end_reason": self.end_reason,
            "pre_roll_frames": self.pre_roll_frames,
            "post_roll_frames": self.post_roll_frames,
        }


@dataclass
class ActivityWindow:
    """
    A contiguous run of motion above the activity threshold.

    `had_confirmed` is the whole point: a window with strong motion but no
    confirmed ball detection is a *candidate* that the tracker could not
    analyse, not proof that nothing happened. It becomes an exclusion zone so
    a later clip can never reach back through it.
    """

    start: int
    end: int
    had_confirmed: bool
    had_candidate: bool = False
    peak_motion: float = 0.0

    @property
    def duration(self) -> int:
        return self.end - self.start + 1

    @property
    def is_untracked(self) -> bool:
        return not self.had_confirmed

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "had_confirmed": self.had_confirmed,
            "had_candidate": self.had_candidate,
            "peak_motion": round(self.peak_motion, 3),
        }


class SignalTimeline:
    """
    Frame-indexed store of the lightweight per-frame signals.

    Storage is parallel lists indexed by `frame - 1`; appending one frame is
    O(1) and cheap enough not to disturb the single pass. The whole timeline for
    a 90 000-frame (1 hour @ 25 fps) video is a few megabytes.
    """

    def __init__(
        self,
        quiet_threshold: float = 1.5,
        active_threshold: float = 4.0,
        black_run_barrier: int = 3,
        fps: float = 25.0,
    ) -> None:
        self.quiet = float(quiet_threshold)
        self.active = float(active_threshold)
        self.black_run_barrier = int(max(1, black_run_barrier))
        self.fps = float(fps) if fps and fps > 0 else 25.0

        self._frame: list[int] = []
        self._black: list[bool] = []
        self._cut: list[bool] = []
        self._motion: list[float] = []
        self._pitch: list[float] = []
        self._confirmed: list[bool] = []
        self._candidate: list[bool] = []

    # ── Recording ────────────────────────────────────────────────────────────
    def record(self, signal: FrameSignal) -> None:
        self._frame.append(signal.frame)
        self._black.append(bool(signal.is_black))
        self._cut.append(bool(signal.is_scene_cut))
        self._motion.append(float(signal.motion))
        self._pitch.append(float(signal.pitch_motion))
        self._confirmed.append(bool(signal.confirmed))
        self._candidate.append(bool(signal.candidate))

    def __len__(self) -> int:
        return len(self._frame)

    @property
    def last_frame(self) -> int:
        return self._frame[-1] if self._frame else 0

    @property
    def first_frame(self) -> int:
        return self._frame[0] if self._frame else 0

    # ── Point queries ────────────────────────────────────────────────────────
    def _idx(self, frame: int) -> Optional[int]:
        if not self._frame:
            return None
        # Frames are recorded strictly in order and contiguously from frame 1,
        # so the index is direct. Fall back to a search if that ever breaks.
        i = frame - self._frame[0]
        if 0 <= i < len(self._frame) and self._frame[i] == frame:
            return i
        lo, hi = 0, len(self._frame) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._frame[mid] == frame:
                return mid
            if self._frame[mid] < frame:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def is_black(self, frame: int) -> bool:
        i = self._idx(frame)
        return bool(i is not None and self._black[i])

    def is_scene_cut(self, frame: int) -> bool:
        i = self._idx(frame)
        return bool(i is not None and self._cut[i])

    def motion(self, frame: int) -> float:
        i = self._idx(frame)
        return float(self._motion[i]) if i is not None else 0.0

    def pitch_motion(self, frame: int) -> float:
        i = self._idx(frame)
        return float(self._pitch[i]) if i is not None else 0.0

    def had_detection(self, frame: int) -> bool:
        i = self._idx(frame)
        if i is None:
            return False
        return bool(self._confirmed[i] or self._candidate[i])

    def smoothed_motion(self, frame: int, half: int = 2) -> float:
        """Short moving average, so a one-frame motion spike is not an anchor."""
        vals: list[float] = []
        for f in range(frame - half, frame + half + 1):
            i = self._idx(f)
            if i is not None:
                vals.append(self._motion[i])
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def black_run_at(self, frame: int, backward: bool) -> int:
        """Length of the contiguous black run starting at / ending at `frame`."""
        n = 0
        f = frame
        while self.is_black(f):
            n += 1
            f += -1 if backward else 1
        return n

    # ── Barrier tests ────────────────────────────────────────────────────────
    @staticmethod
    def in_ranges(frame: int, ranges: Iterable[tuple[int, int]]) -> bool:
        for a, b in ranges:
            if a <= frame <= b:
                return True
        return False

    def _hard_barrier(self, frame: int, excluded) -> Optional[str]:
        """Black / cut / exclusion at `frame`; the reason, or None."""
        if self.in_ranges(frame, excluded):
            return STOP_EXCLUSION_ZONE
        if self.is_black(frame):
            return STOP_BLACK_GAP
        if self.is_scene_cut(frame):
            return STOP_SCENE_CUT
        return None

    # ── The two searches ─────────────────────────────────────────────────────
    def safe_pre_roll_start(
        self,
        anchor: int,
        max_back: int,
        excluded: Iterable[tuple[int, int]] = (),
    ) -> tuple[int, str]:
        """
        Walk BACKWARD from `anchor` (the first reliable ball detection) and stop
        at the nearest barrier, or at the cap.

        The inactivity-valley rule only fires once motion has been seen to rise
        above `active` on the way back, so a quiet patch of ball flight is not
        mistaken for the reset. This is what separates two events in fixed-camera
        footage with no black frame and no cut.
        """
        excluded = tuple(excluded)
        limit = max(1, int(anchor) - int(max_back))
        f = int(anchor) - 1

        # PHASE A — cross this delivery's own action to reach the reset valley.
        # Anything louder than a quiet patch is the run-up / ball-flight of the
        # CURRENT delivery, so it is included; the first quiet patch is where the
        # current delivery's activity ends.
        while f >= limit:
            reason = self._hard_barrier(f, excluded)
            if reason == STOP_SCENE_CUT:
                # A cut between f-1 and f means the camera segment containing the
                # ball starts at f. Including f keeps us on the ball's camera and
                # stops us crossing into the previous one.
                return f, reason
            if reason is not None:
                # black / exclusion: the frame itself belongs to the barrier, so
                # the window starts just after it.
                return f + 1, reason
            if self.smoothed_motion(f) <= self.quiet:
                break
            f -= 1

        if f < limit:
            # No quiet frame anywhere in the allowed pre-roll: the delivery is
            # surrounded by continuous action. Take the full (capped) window.
            return (max(1, limit),
                    STOP_START_OF_VIDEO if limit == 1 else STOP_MAX_PRE_ROLL)

        # PHASE B — walk to the start of the reset valley. Stopping at the
        # valley's first frame means the window begins on the dead-ball pause
        # and never reaches back into the PREVIOUS event's stroke.
        while f - 1 >= limit:
            if self._hard_barrier(f - 1, excluded) is not None:
                break
            if self.smoothed_motion(f - 1) > self.quiet:
                break
            f -= 1
        return f, STOP_INACTIVITY_VALLEY

    def safe_post_roll_end(
        self,
        anchor: int,
        desired_end: int,
        excluded: Iterable[tuple[int, int]] = (),
        known_frame: Optional[int] = None,
        stroke_from: Optional[int] = None,
    ) -> tuple[int, str]:
        """
        Walk FORWARD from the last confirmed detection and stop at the nearest
        barrier found in the frames the pass has ALREADY recorded, or at the
        requested end.

        WHY THE ANCHOR IS THE LAST DETECTION, NOT ball_lost
        --------------------------------------------------
        An event may coast for up to the prediction ceiling after its last
        detection. That coasting span routinely crosses a black gap or a camera
        cut. Anchoring on `ball_lost` would step over the transition and drag it
        into the clip; anchoring on the last real measurement stops at it.

        BLACK / EXCLUSION APPLY THROUGHOUT; CUT / VALLEY ONLY AFTER THE STROKE
        ---------------------------------------------------------------------
        Between the last measurement and the stroke (`stroke_from`) only a black
        gap or an exclusion zone is meaningful — there is no stroke yet for a
        camera cut or a reset valley to be confused with. Once the stroke region
        is reached, camera cuts and inactivity valleys stop the window too.

        Frames past `known_frame` have not been decoded yet, so they cannot be
        judged here. Running out of known frames is NOT itself a barrier: the
        window keeps its full requested length and the streaming loop applies
        the same barriers as those frames arrive.
        """
        excluded = tuple(excluded)
        scan_to = int(desired_end)
        if known_frame is not None:
            scan_to = min(scan_to, int(known_frame))
        stroke_from = int(stroke_from) if stroke_from is not None else int(anchor)
        f = int(anchor) + 1
        saw_action = False
        while f <= scan_to:
            reason = self._hard_barrier(f, excluded)
            if reason == STOP_SCENE_CUT and f < stroke_from:
                reason = None
            if reason is not None:
                # End just BEFORE the barrier: it belongs to whatever comes next.
                return max(int(anchor), f - 1), reason
            if f >= stroke_from:
                m = self.smoothed_motion(f)
                if m >= self.active:
                    saw_action = True
                elif m <= self.quiet and saw_action:
                    # First quiet frame after the stroke: stop before it, keeping
                    # the follow-through and dropping the dead-ball reset.
                    return max(int(anchor), f - 1), STOP_INACTIVITY_VALLEY
            f += 1
        return int(desired_end), STOP_MAX_POST_ROLL

    # ── Untracked activity windows ───────────────────────────────────────────
    def activity_windows(
        self,
        start_frame: int,
        end_frame: int,
        min_window_frames: int = 4,
    ) -> list[ActivityWindow]:
        """
        Contiguous motion clusters in [start_frame, end_frame].

        A cluster with no confirmed detection at all is a real event the
        detector missed (or a non-cricket motion burst). Either way it is not
        safe to let a neighbouring clip reach through it.
        """
        windows: list[ActivityWindow] = []
        f = int(start_frame)
        end_frame = int(end_frame)
        active_start: Optional[int] = None
        had_confirmed = False
        had_candidate = False
        peak = 0.0
        while f <= end_frame:
            m = self.smoothed_motion(f)
            if m >= self.active:
                if active_start is None:
                    active_start = f
                    had_confirmed = False
                    had_candidate = False
                    peak = 0.0
                peak = max(peak, m)
                if self.had_detection(f):
                    i = self._idx(f)
                    if i is not None and self._confirmed[i]:
                        had_confirmed = True
                    else:
                        had_candidate = True
            else:
                if active_start is not None:
                    w_end = f - 1
                    if w_end - active_start + 1 >= min_window_frames:
                        windows.append(ActivityWindow(
                            start=active_start, end=w_end,
                            had_confirmed=had_confirmed,
                            had_candidate=had_candidate,
                            peak_motion=peak,
                        ))
                    active_start = None
            f += 1
        if active_start is not None:
            w_end = end_frame
            if w_end - active_start + 1 >= min_window_frames:
                windows.append(ActivityWindow(
                    start=active_start, end=w_end,
                    had_confirmed=had_confirmed,
                    had_candidate=had_candidate,
                    peak_motion=peak,
                ))
        return windows


class ActivityWindowTracker:
    """
    Streaming activity-window detector used inside the tracking loop.

    `update()` is fed per frame and returns a finished `ActivityWindow` when one
    closes (a sustained quiet run). This is what turns a zero-detection delivery
    into an exclusion zone *during* the pass, before a later delivery's pre-roll
    is computed.
    """

    def __init__(
        self,
        quiet_threshold: float,
        active_threshold: float,
        quiet_hold_frames: int = 4,
        min_window_frames: int = 4,
    ) -> None:
        self.quiet = float(quiet_threshold)
        self.active = float(active_threshold)
        self.quiet_hold = max(1, int(quiet_hold_frames))
        self.min_window = max(1, int(min_window_frames))

        self._open_start: Optional[int] = None
        self._peak = 0.0
        self._had_confirmed = False
        self._had_candidate = False
        self._quiet_run = 0

    @property
    def open_start(self) -> Optional[int]:
        return self._open_start

    @property
    def is_open(self) -> bool:
        return self._open_start is not None

    def update(self, frame: int, motion: float, had_confirmed: bool, had_candidate: bool) -> Optional[ActivityWindow]:
        """Feed one frame. Returns the closed window, if one just ended."""
        if self._open_start is None:
            if motion >= self.active:
                self._open_start = frame
                self._peak = motion
                self._had_confirmed = had_confirmed
                self._had_candidate = had_candidate
                self._quiet_run = 0
            return None

        if had_confirmed:
            self._had_confirmed = True
        if had_candidate:
            self._had_candidate = True
        self._peak = max(self._peak, motion)

        if motion <= self.quiet:
            self._quiet_run += 1
            if self._quiet_run >= self.quiet_hold:
                end = frame - self._quiet_run
                start = self._open_start
                window = ActivityWindow(
                    start=start, end=max(start, end),
                    had_confirmed=self._had_confirmed,
                    had_candidate=self._had_candidate,
                    peak_motion=self._peak,
                )
                self._open_start = None
                self._quiet_run = 0
                if window.duration >= self.min_window:
                    return window
        else:
            self._quiet_run = 0
        return None
