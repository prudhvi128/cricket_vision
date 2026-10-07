"""
service.py — Non-blocking pitch calibration for one analysis.

THE ONE RULE
------------
The tracking loop decodes each frame exactly once and runs the ball detector
exactly once per frame. This service may not add either, may not wait, and may
not fail. So:

    loop ──on_frame()──▶ queue (bounded, drop-on-full) ──▶ worker thread ──▶ model
                     ▲                                            │
                     └──────────── nothing blocks here ───────────┘
                                     state is read with snapshot()

`on_frame()` is called for EVERY frame. When the feature is off it is one
attribute check; when it is on it is still at most one non-blocking put, and
only every `interval_frames`-th frame ever reaches the model.

FOUR STATES, TOLD HONESTLY
--------------------------
    no_calibration       nothing validated yet              → nothing to draw
    calibrated           a fresh detection is in effect     → draw the pitch
    temporary_loss       recent detections failed, but the  → draw the LAST known
                         last good one is still fresh         pitch, marked stale
    calibration_expired  the last good one is too old       → nothing to draw

The same four states decide whether a client gets a ball position on the pitch:
`pitch_position` exists only while a calibration with a homography is in effect.
A stale boundary is never silently served as a current one, and an expired one
is never served at all.

WHAT EACH ATTEMPT COST, AND WHY IT WAS MADE
-------------------------------------------
Every frame the loop offers is answered with exactly one verdict, and every
verdict is counted: submitted, gated (with the reason), evicted from a full
queue, no keypoints, timeout, API error, refused (with the validator's reason),
or accepted. `summary()["diagnostics"]` carries the tally, the latency
distribution and the attempt records themselves, so "799 failures" is a list of
causes rather than one number.

Two spacing modes (see `PITCH_SAMPLING_MODE`):

    interval  every `interval_frames`-th frame, unconditionally — the original
              behaviour, and what a bare `PitchConfig()` still does.
    adaptive  the same base interval scaled by what the loop already knows —
              a delivery in progress, a quad that already covers this view, a
              corridor too dark to hold a pitch — and never more often than
              the queue can take. Nothing about calibration, validation or the
              geometry changes; only WHICH frames are offered changes.

    sequential backpressure-aware sequential processing: ONE inference at a
              time and never faster than the detector answers. A frame is
              offered only when nothing is in flight and the queue is empty,
              so no backlog of stale frames can build up, and the next offer
              waits out a wall-clock cooldown of `1 / max_attempts_per_sec`.
              A slow detector therefore stretches the interval by itself
              (the rate follows the latency instead of filling the queue);
              a fast one may raise the rate only up to the configured
              maximum. Calibration, validation, thresholds and geometry are
              untouched — only WHICH frames are offered, and when.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import replace
from typing import Any, Optional

from ..core.config import PITCH_CONFIG, PitchConfig
from ..core.constants import (
    PITCH_ATTEMPTS_PER_SEC_FLOOR,
    PITCH_CORRIDOR_MIN_LUMINANCE,
    PITCH_COVERED_INTERVAL_MULTIPLIER,
    PITCH_DIM_INTERVAL_MULTIPLIER,
    PITCH_DETECTOR_REARM_FRAMES,
    PITCH_DROUGHT_BACKOFF,
    PITCH_DROUGHT_INTERVAL_MULTIPLIER,
    PITCH_DROUGHT_INTERVAL_MULTIPLIER_IN_EVENT,
    PITCH_FORCE_INTERVAL_MULTIPLIER,
    PITCH_IDLE_INTERVAL_MULTIPLIER,
    PITCH_SAMPLING_MODE_ADAPTIVE,
    PITCH_SAMPLING_MODE_SEQUENTIAL,
)
from .calibration import PitchCalibration, PitchSmoother, build_calibration
from .detector import (
    Keypoint,
    PitchDetectError,
    PitchDetection,
    PitchDetectorUnavailable,
    PitchTimeout,
    RoboflowPitchDetector,
    scrub,
)

log = logging.getLogger(__name__)

# The four states a client may be told about.
STATE_NO_CALIBRATION = "no_calibration"
STATE_CALIBRATED = "calibrated"
STATE_TEMPORARY_LOSS = "temporary_loss"
STATE_EXPIRED = "calibration_expired"


class PitchService:
    """
    Interval-gated, thread-backed pitch calibration for one analysis.

    `detector` is injectable so tests can drive the whole state machine without
    a network. The default is a Roboflow client, constructed only when a key is
    configured — with no key this object is a no-op that still answers every
    snapshot with `no_calibration`, which is the truthful response.
    """

    def __init__(
        self,
        cfg: Optional[PitchConfig] = None,
        *,
        width: int,
        height: int,
        detector=None,
    ) -> None:
        self.cfg = cfg if cfg is not None else PITCH_CONFIG
        self.width = int(width)
        self.height = int(height)

        self._injected = detector is not None
        self._detector = detector
        self._detector_error: Optional[str] = None
        # Set on the first frame offered after the detector went quiet, and
        # cleared when it is given another chance (see `_maybe_rearm`).
        self._disabled_at_frame: Optional[int] = None
        self._rearms = 0

        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stopped = False

        # ── `sequential` mode's backpressure (see `_sequential_gate`)
        # `inflight` is written by the worker around one detector call and read
        # by the loop; `cooldown_until` is written by the loop after a
        # submission. Both are plain values under the GIL — `on_frame` must
        # still never wait on anything.
        self._inflight = False
        self._cooldown_until = 0.0

        # Calibration history, most recent last. Bounded: a delivery is
        # finalised within a few dozen frames of the live edge, so older entries
        # can never be asked for again.
        self._calibrations: list[PitchCalibration] = []
        self._history_cap = max(50, 4 * self.cfg.max_lost_frames)

        self._smoother = PitchSmoother(self.cfg.smoothing_alpha)

        self._last_submitted_frame = -10**9
        self._submitted = 0
        self._detections = 0
        self._failures = 0
        self._refusals = 0
        self._queue_drops = 0
        self._last_error: Optional[str] = None
        self._last_refusal: Optional[str] = None
        self._last_detection_frame: Optional[int] = None

        # ── Diagnostics: one verdict per frame offered, one record per attempt
        self._offered = 0
        self._covered_until = -10**9   # a quad at/below this frame describes the view
        # Empty answers in a row. The worker writes it, the loop reads it: a
        # plain int, so a torn read costs nothing and a missed increment only
        # delays the backoff by one attempt.
        self._drought = 0
        self._gated: dict[str, int] = {}
        self._categories: dict[str, int] = {}
        self._refusals_by_reason: dict[str, int] = {}
        self._attempts: list[dict] = []
        self._attempts_truncated = 0
        self._evicted_frames: list[int] = []
        self._queue_wait_ms: list[float] = []
        self._infer_ms: list[float] = []
        self._queue_depth_max = 0
        self._first_submit_at: Optional[float] = None
        self._last_submit_at: Optional[float] = None
        self._first_process_at: Optional[float] = None
        self._last_process_at: Optional[float] = None
        # Generous, but bounded: a nine-minute video produces ~900 attempts,
        # and this list is persisted inside `result.json`.
        self._attempt_cap = 4000
        self._evicted_cap = 5000

    # ── capability ────────────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        """
        Cheap enough to call once per frame: no locks, no I/O.

        A detector passed in by the caller is an explicit choice and does not
        need a credential; the Roboflow client is only built when the
        environment actually configured one.
        """
        if self._detector is None and not self.cfg.enabled:
            return False
        return self._detector_ok()

    def _detector_ok(self) -> bool:
        if self._detector_error is not None:
            return False
        if self._detector is None:
            self._detector = RoboflowPitchDetector(self.cfg)
        reason = getattr(self._detector, "disabled_reason", None)
        if reason:
            # A client that stopped answering (three timeouts in a row) is not
            # going to start answering; stop paying for submissions.
            self._detector_error = f"detector disabled: {reason}"
            return False
        return True

    # ── the loop's single call site ───────────────────────────────────────────
    def on_frame(
        self,
        frame_number: int,
        frame,
        *,
        corridor_lum: Optional[float] = None,
        event_open: Optional[bool] = None,
        near_event: Optional[bool] = None,
        scene_cut: bool = False,
        black: bool = False,
    ) -> None:
        """
        Offer one frame to the model. Never blocks, never raises, never waits.

        Everything that could be slow - the HTTP call, the JSON parse, the
        validation - happens on the worker thread after this returns.

        The keyword hints are the tracking loop's own per-frame signals. They
        are advisory: in `interval` mode they are ignored entirely, and in
        `adaptive` mode they only decide WHICH eligible frame is offered, never
        what a response means. `event_open` is a delivery in progress;
        `near_event` is the guard that runs past one, where the clip window
        still shows the pitch.
        """
        if self._detector_error is not None:
            self._maybe_rearm(frame_number)
        if not self.enabled:
            # A frame the loop offered but the detector refused — no
            # credential, or a client that stopped answering — still counts as
            # offered. Counting only the frames that survived this check would
            # shrink `offered` the moment the detector goes quiet, which is
            # exactly when the report most needs to say how much of the video
            # went unsampled.
            self._offered += 1
            with self._lock:
                self._gated["disabled"] = self._gated.get("disabled", 0) + 1
            return
        # Only the loop thread ever writes this counter, so it needs no lock —
        # `on_frame` must stay free of anything it could wait on.
        self._offered += 1
        gate = self._gate(
            frame_number,
            corridor_lum=corridor_lum,
            event_open=event_open,
            near_event=near_event,
            scene_cut=scene_cut,
            black=black,
        )
        if gate is not None:
            with self._lock:
                self._gated[gate] = self._gated.get(gate, 0) + 1
            return

        now = time.perf_counter()
        item = (frame_number, frame, now, bool(event_open),
                bool(event_open) or bool(near_event))
        try:
            self._queue.put_nowait(item)
            self._submitted += 1
        except queue.Full:
            # Latest wins: discard the frame already waiting (the loop has
            # moved on) and try to publish the newer one.
            try:
                evicted = self._queue.get_nowait()
                self._note_eviction(evicted)
                self._queue.put_nowait(item)
                self._submitted += 1
            except (queue.Empty, queue.Full):
                with self._lock:
                    self._queue_drops += 1
                return
        self._last_submitted_frame = frame_number
        if self.cfg.sampling_mode == PITCH_SAMPLING_MODE_SEQUENTIAL:
            # Measured from the submission, not from the answer: the cooldown
            # is the ceiling, and a round trip longer than it simply wins.
            self._cooldown_until = now + self._cooldown_seconds
        with self._lock:
            if self._first_submit_at is None:
                self._first_submit_at = now
            self._last_submit_at = now
            depth = self._queue.qsize()
            if depth > self._queue_depth_max:
                self._queue_depth_max = depth
        self._ensure_worker()

    def _maybe_rearm(self, frame_number: int) -> None:
        """
        Give a client that timed out one more chance, long after it went quiet.

        Three consecutive timeouts mean the service is having a bad minute, not
        that the credential died — but without this, the disable lasts for the
        rest of the video, and a five-minute outage in the first half silently
        empties the second half of every pitch attempt. Once per re-arm window
        the detector is tried again; if it times out three more times it goes
        back to sleep for another window. Errors that are not a timeout-driven
        disable (no credential, no SDK) are permanent and are left alone.
        """
        if not self._detector_error.startswith("detector disabled:"):
            return
        with self._lock:
            if self._disabled_at_frame is None:
                self._disabled_at_frame = frame_number
                return
            if frame_number - self._disabled_at_frame < PITCH_DETECTOR_REARM_FRAMES:
                return
            # The window restarts when the next disable is noticed, so a client
            # that answers for a while and then fails again waits a full window
            # instead of being re-armed on the very frame it gave up.
            self._disabled_at_frame = None
            self._detector_error = None
            self._rearms += 1
        det = self._detector
        if det is not None:
            rearm = getattr(det, "rearm", None)
            if callable(rearm):
                rearm()
            elif hasattr(det, "disabled_reason"):
                # Test doubles and any future detector that exposes only the
                # flag: clearing it is the whole of what a re-arm means here.
                det.disabled_reason = None

    def _gate(
        self,
        frame_number: int,
        *,
        corridor_lum: Optional[float],
        event_open: Optional[bool],
        near_event: Optional[bool] = None,
        scene_cut: bool,
        black: bool,
    ) -> Optional[str]:
        """
        None → submit this frame; otherwise the reason it was not submitted.

        `interval` mode is the original rule and nothing more: one frame every
        `interval_frames`, whatever the hints say, so a checkout that never
        configured a mode behaves exactly as it always has.

        `sequential` mode never reaches the rules below — it answers first, and
        only in terms of the detector's own backpressure (see
        `_sequential_gate`).

        `adaptive` mode scales that base interval:

            in-event  an event is open, or the guard that runs past one still
                      covers this frame — the clip window shows the pitch here,
                      so the model is asked at the base rate;
            covered   a quad measured within `min(interval * 6, max_lost_frames)`
                      frames already describes this view — asking again now buys
                      nothing and costs a worker slot;
            idle      neither of the above — pitch footage is a minority of the
                      video, so the model is asked an eighth as often;
            dim       the pitch corridor is darker than any frame the model has
                      answered in (measured, not guessed) — half as often, never
                      never, because a threshold is a prior and not a proof;
            drought   the model has answered "nothing there" this many times in
                      a row: the SCENE is what is not answering, so ask a tenth
                      as often outside a delivery, a quarter as often inside
                      one, and let a cut restart the conversation;
            force     at least every `interval * 40` frames regardless, so a
                      camera that keeps moving is never starved of attempts;
            busy      the queue already holds work: only a frame from INSIDE a
                      delivery may displace it, and the black frame is refused
                      outright — no exposure, no pitch.

        Nothing here relaxes a validation rule, extends a calibration or
        changes what a response means.
        """
        base = max(1, self.cfg.interval_frames)
        mode = self.cfg.sampling_mode
        if mode == PITCH_SAMPLING_MODE_SEQUENTIAL:
            return self._sequential_gate(black=black)
        if mode != PITCH_SAMPLING_MODE_ADAPTIVE:
            return None if frame_number - self._last_submitted_frame >= base else "spacing"

        if black:
            return "black"
        if scene_cut:
            # New content: the previous answers say nothing about this shot,
            # so the drought that was suppressing attempts no longer applies.
            self._drought = 0
            if self._covered_until > frame_number:
                # The cut replaced the view the quad describes: stop counting it
                # as covered so the new view is measured rather than assumed.
                self._covered_until = frame_number - 1

        # A caller that supplies no event hints gets no event penalty — silence
        # is not evidence that the frame is between deliveries.
        if event_open is None and near_event is None:
            in_delivery = True
        else:
            in_delivery = bool(event_open) or bool(near_event)

        elapsed = frame_number - self._last_submitted_frame
        spacing = base
        if frame_number <= self._covered_until:
            spacing = max(spacing, self._covered_frames)
        elif not in_delivery:
            spacing = max(spacing, base * PITCH_IDLE_INTERVAL_MULTIPLIER)
        if corridor_lum is not None and corridor_lum < PITCH_CORRIDOR_MIN_LUMINANCE:
            spacing *= PITCH_DIM_INTERVAL_MULTIPLIER
        if self._drought >= PITCH_DROUGHT_BACKOFF:
            factor = (PITCH_DROUGHT_INTERVAL_MULTIPLIER_IN_EVENT if in_delivery
                      else PITCH_DROUGHT_INTERVAL_MULTIPLIER)
            spacing = max(spacing, base * factor)
        if elapsed >= base * PITCH_FORCE_INTERVAL_MULTIPLIER:
            spacing = 0
        if elapsed < spacing:
            return "spacing"
        if self._queue.qsize() > 0 and not in_delivery:
            return "busy"
        return None

    def _sequential_gate(self, *, black: bool) -> Optional[str]:
        """
        Backpressure-aware sequential spacing: ask only when the last answer is
        in, and at most `max_attempts_per_sec` times a second.

        Four checks, cheapest first, and nothing else — no delivery hints, no
        corridor brightness, no drought, no coverage window. Which frame to ask
        about is decided by the detector's own pace alone:

            black     no exposure, no pitch: never worth a request;
            cooldown  the wall clock is inside `1 / max_attempts_per_sec`
                      since the last submission — the configured ceiling;
            busy      a call is in flight OR something is still queued. One
                      inference at a time, and a frame is offered only when
                      neither holds, so the queue can never hold a backlog of
                      stale frames and a slow answer simply stretches the
                      interval (the rate follows the latency rather than
                      the queue filling up);
            —         otherwise this is the next eligible frame after the
                      last inference finished: submit it.

        A fast detector therefore raises the rate only up to the ceiling; a
        slow one lowers it without limit and without a single dropped frame.
        """
        if black:
            return "black"
        if time.perf_counter() < self._cooldown_until:
            return "cooldown"
        if self._inflight or not self._queue.empty():
            return "busy"
        return None

    @property
    def _cooldown_seconds(self) -> float:
        """Seconds between two submissions: the configured rate ceiling."""
        rate = float(self.cfg.max_attempts_per_sec)
        if rate <= 0:
            rate = PITCH_ATTEMPTS_PER_SEC_FLOOR
        return 1.0 / rate

    @property
    def _covered_frames(self) -> int:
        """
        How long a measured quad counts as "this view is already covered".

        Bounded by `max_lost_frames` on purpose: skipping submissions past the
        point where the calibration itself expires would trade a live quad for
        an expired one.
        """
        base = max(1, self.cfg.interval_frames)
        return max(1, min(base * PITCH_COVERED_INTERVAL_MULTIPLIER,
                          max(1, self.cfg.max_lost_frames)))

    def _note_eviction(self, item: Any) -> None:
        """A frame lost its queue slot to a newer one: count it, and keep it."""
        with self._lock:
            self._queue_drops += 1
            if isinstance(item, tuple) and item and len(self._evicted_frames) < self._evicted_cap:
                self._evicted_frames.append(int(item[0]))

    # ── reads ─────────────────────────────────────────────────────────────────
    def snapshot(self, frame_number: int) -> dict:
        """
        The per-delivery pitch block: what a client needs to decide whether to
        draw the pitch and where.

        Always the SAME keys, whatever happened — a client should be able to read
        `pitch.detected` without checking that `pitch` exists first.
        """
        cal, state = self._state_at(frame_number)
        return self._block(cal, state)

    def calibration_for_delivery(
        self, start_frame: int, end_frame: int
    ) -> Optional[PitchCalibration]:
        """
        The quad that describes ONE delivery's footage, or None.

        A delivery clip is one continuous shot, so a quad measured anywhere
        inside it describes that footage — the live edge's freshness at the
        delivery's LAST frame is the wrong question, and it is why a delivery
        spanning a camera-lit period could otherwise lose geometry it genuinely
        has. Preference order:

          1. the most recent calibration measured inside `[start, end]`;
          2. failing that, the most recent one the state machine would still
             call current at the delivery's end (`age <= fresh_frames`), which
             is how a quad detected just before a short delivery is treated.

        Never an expired quad: one measured outside the window and older than
        `max_lost_frames` describes a different view of the pitch.
        """
        with self._lock:
            in_window = [
                c for c in self._calibrations
                if start_frame <= c.frame_number <= end_frame
            ]
            if in_window:
                return in_window[-1]
            fresh = [
                c for c in self._calibrations
                if c.frame_number <= end_frame
                and end_frame - c.frame_number <= self._fresh_frames
            ]
            return fresh[-1] if fresh else None

    def snapshot_for_delivery(self, start_frame: int, end_frame: int) -> dict:
        """
        `snapshot()` scoped to a delivery window rather than to one frame.

        With a calibration measured during the delivery the four states still
        describe it honestly: `calibrated` while that quad is fresh at the
        delivery's end, `temporary_loss` (labelled as a previous calibration)
        once it is stale. Without one, this is exactly `snapshot(end_frame)` —
        `temporary_loss`, `calibration_expired` or `no_calibration`.
        """
        cal = self.calibration_for_delivery(start_frame, end_frame)
        if cal is None:
            return self.snapshot(end_frame)
        if end_frame - cal.frame_number <= self._fresh_frames:
            return self._block(cal, STATE_CALIBRATED)
        return self._block(cal, STATE_TEMPORARY_LOSS)

    def _block(self, cal: Optional[PitchCalibration], state: str) -> dict:
        """The block itself, shared by every snapshot entry point."""
        using_previous = state == STATE_TEMPORARY_LOSS
        calibrated = state in (STATE_CALIBRATED, STATE_TEMPORARY_LOSS)
        detected = state == STATE_CALIBRATED

        if cal is None:
            return _empty_block(state)
        return {
            "state": state,
            "detected": detected,
            "calibrated": calibrated,
            "using_previous_calibration": using_previous,
            "confidence": cal.confidence,
            "keypoints": {k: [round(v[0], 2), round(v[1], 2)] for k, v in cal.keypoints.items()},
            "corners": {k: [round(v[0], 2), round(v[1], 2)] for k, v in cal.corners.items()},
            "center": (
                [round(cal.center[0], 2), round(cal.center[1], 2)] if cal.center else None
            ),
            "homography_available": cal.homography is not None,
            "homography": (
                [[round(float(v), 8) for v in row] for row in cal.homography]
                if cal.homography is not None else None
            ),
            "image_size": [int(cal.image_size[0]), int(cal.image_size[1])],
            "frame_number": cal.frame_number,
            "model_id": self.cfg.model_id,
        }

    def summary(self) -> dict:
        """Analysis-level record. Contains configuration, never the key."""
        with self._lock:
            summary = {
                # Whether this analysis runs the feature at all: a credential
                # configured in the environment, or a detector the caller put
                # here deliberately (a test double, or a local model later).
                "enabled": bool(self.cfg.enabled) or self._injected,
                "config": self.cfg.metadata(),
                "submitted": self._submitted,
                "detections": self._detections,
                "failures": self._failures,
                "refusals": self._refusals,
                "queue_drops": self._queue_drops,
                "last_detection_frame": self._last_detection_frame,
                "last_error": self._last_error,
                "last_refusal": self._last_refusal,
                "detector_error": self._detector_error,
                "calibration_updates": len(self._calibrations),
            }
            summary["diagnostics"] = self._diagnostics_locked()
        return summary

    def _diagnostics_locked(self) -> dict:
        """
        The per-attempt record. Must be called with `self._lock` held.

        Three questions, answered from measurements rather than inference:
        which frames were offered and why the rest were not, what each attempt
        cost (queue wait, round trip) and what came of it, and how fast the
        producer and the single consumer actually ran.
        """
        submitted = sum(self._categories.values())
        return {
            "sampling_mode": self.cfg.sampling_mode,
            "base_interval_frames": max(1, self.cfg.interval_frames),
            "offered": self._offered,
            "categories": _sorted_counts(self._categories),
            "gated": _sorted_counts(self._gated),
            "refusals": _sorted_counts(self._refusals_by_reason),
            # Empty answers in a row when the pass ended: the gate's reason for
            # holding back where it held back.
            "drought": self._drought,
            # Times a timeout-disabled client was given another chance.
            "detector_rearms": self._rearms,
            "latency_ms": _stats(self._infer_ms),
            "queue_wait_ms": _stats(self._queue_wait_ms),
            "queue": {
                "capacity": self._queue.maxsize,
                "max_depth_observed": self._queue_depth_max,
                "evictions": self._queue_drops,
            },
            # `sequential` mode's backpressure, reported in the same units the
            # rest of this block uses: the ceiling, the cooldown it implies, and
            # whether a call is running right now (the snapshot's moment).
            "backpressure": {
                "in_flight": self._inflight,
                "cooldown_ms": round(self._cooldown_seconds * 1000, 2),
                "max_attempts_per_sec": self.cfg.max_attempts_per_sec,
            },
            "throughput": {
                "submitted_per_s": _rate(self._submitted, self._first_submit_at, self._last_submit_at),
                "processed_per_s": _rate(submitted, self._first_process_at, self._last_process_at),
                "active_seconds": _span(self._first_submit_at, self._last_process_at),
            },
            "attempts": list(self._attempts),
            "attempts_truncated": self._attempts_truncated,
            "evicted_frames": list(self._evicted_frames),
            "covered_until_frame": self._covered_until,
        }

    def stop(self, timeout: float = 5.0) -> None:
        """
        Drain what is queued and join the worker.

        Bounded on purpose: if a call is genuinely hung (the SDK offers no
        request timeout of its own) the worker is abandoned rather than the
        analysis, because a daemon thread costs nothing next to a pipeline that
        refuses to finish.
        """
        self._stopped = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    # ── worker ────────────────────────────────────────────────────────────────
    def _ensure_worker(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._worker, name="pitch-detector", daemon=True
            )
            self._thread.start()

    def _worker(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stopped:
                    return
                continue
            if item is None:
                return
            frame_number, frame, queued_at, event_open, hot = item
            # One call at a time, and the gate is told about it before anything
            # else can happen: while this is set no further frame is offered, so
            # the queue cannot grow a backlog behind a slow answer.
            self._inflight = True
            try:
                self._process(frame_number, frame, queued_at, event_open, hot)
            except Exception as exc:  # noqa: BLE001 — a worker must never die
                self._record_failure(f"unexpected: {scrub(str(exc), self.cfg.api_key)}")
            finally:
                self._inflight = False
            if self._stopped and self._queue.empty():
                return

    def _process(
        self,
        frame_number: int,
        frame,
        queued_at: Optional[float] = None,
        event_open: Optional[bool] = None,
        hot: bool = False,
    ) -> None:
        detector = self._detector
        if detector is None:
            return
        now = time.perf_counter()
        if self._first_process_at is None:
            self._first_process_at = now
        self._last_process_at = now
        rec: dict[str, Any] = {
            "frame": frame_number,
            # Time the frame waited for the worker — the queue's own latency,
            # the number that says whether the queue was ever the bottleneck.
            "wait_ms": None if queued_at is None else round((now - queued_at) * 1000, 2),
            "event": event_open,
            # Whether the gate considered this frame part of a delivery (open
            # event, or inside the guard that runs past one).
            "hot": bool(hot),
            # What the state machine said about calibration at this frame, so a
            # diagnostic report can show a rejection happening to a delivery
            # that already had (or already lacked) a quad.
            "calibration_state": self._state_at(frame_number)[1],
        }
        try:
            try:
                detection = detector.detect(frame, frame_number, (self.width, self.height))
            finally:
                # Recorded for every outcome: an empty answer costs the same
                # round trip as a full one, and the report needs to say so.
                rec["infer_ms"] = round((time.perf_counter() - now) * 1000, 2)
        except PitchDetectorUnavailable as exc:
            # No credential, or the SDK is not installed: submissions are futile.
            with self._lock:
                self._detector_error = str(exc)[:300]
                self._failures += 1
                self._last_error = str(exc)[:300]
                self._finish_locked(rec, "detector_unavailable")
            return
        except PitchTimeout as exc:
            self._record_failure(str(exc), category="timeout", rec=rec)
            return
        except PitchDetectError as exc:
            self._record_failure(str(exc), rec=rec)
            return
        except Exception as exc:  # noqa: BLE001
            self._record_failure(
                f"unexpected: {scrub(str(exc), self.cfg.api_key)}", rec=rec
            )
            return

        detection = self._smooth(detection)
        # Keep what the model actually said even when validation rejects it:
        # a report that can only show accepted detections cannot explain a
        # refusal, and the refusal is the thing needing explanation.
        rec["confidence"] = (
            None if detection.confidence is None
            else round(float(detection.confidence), 4)
        )
        rec["keypoints"] = {
            k.name: [round(k.x, 2), round(k.y, 2)] for k in detection.keypoints
        }
        calibration, reason = build_calibration(
            detection, cfg=self.cfg, image_size=(self.width, self.height)
        )
        if calibration is None:
            self._record_refusal(reason, rec=rec)
            return

        rec["confidence"] = calibration.confidence
        rec["keypoints"] = {
            k: [round(v[0], 2), round(v[1], 2)] for k, v in calibration.keypoints.items()
        }
        with self._lock:
            self._calibrations.append(calibration)
            if len(self._calibrations) > self._history_cap:
                del self._calibrations[: len(self._calibrations) - self._history_cap]
            self._detections += 1
            self._last_detection_frame = frame_number
            self._drought = 0
            self._finish_locked(rec, "detected")
        # A measured quad covers this view for as long as re-measuring it would
        # add nothing (bounded by the calibration's own expiry, never beyond).
        self._covered_until = frame_number + self._covered_frames

    def _smooth(self, detection: PitchDetection) -> PitchDetection:
        """
        Temporal smoothing, applied BEFORE validation so a jittery boundary never
        reaches the client. Per-keypoint confidences travel unchanged: averaging
        a confidence is a claim about the model, not about the geometry.
        """
        named = {k.name: (k.x, k.y) for k in detection.keypoints}
        smoothed, confidence = self._smoother.apply(named, detection.confidence)
        conf_by_name = {k.name: k.confidence for k in detection.keypoints}
        keypoints = [
            Keypoint(name, xy[0], xy[1], conf_by_name.get(name))
            for name, xy in smoothed.items()
        ]
        return replace(detection, keypoints=keypoints, confidence=confidence)

    # ── state machine ─────────────────────────────────────────────────────────
    def _state_at(self, frame_number: int) -> tuple[Optional[PitchCalibration], str]:
        with self._lock:
            usable = [
                c for c in self._calibrations if c.frame_number <= frame_number
            ]
            cal = usable[-1] if usable else None
        if cal is None:
            return None, STATE_NO_CALIBRATION

        age = frame_number - cal.frame_number
        if age <= self._fresh_frames:
            return cal, STATE_CALIBRATED
        if age <= self.cfg.max_lost_frames:
            return cal, STATE_TEMPORARY_LOSS
        return None, STATE_EXPIRED

    @property
    def _fresh_frames(self) -> int:
        """
        How far behind the live edge a detection still counts as current.

        Two scheduled submissions may legitimately be in flight (one queued,
        one being answered) before a calibration is called stale — a model that
        answers slower than the interval must not be reported as a loss while
        it is, in fact, still answering.
        """
        return 2 * max(1, self.cfg.interval_frames) + 5

    # ── bookkeeping ───────────────────────────────────────────────────────────
    def _record_failure(
        self, message: str, *, category: Optional[str] = None, rec: Optional[dict] = None
    ) -> None:
        with self._lock:
            self._failures += 1
            self._last_error = scrub(message, self.cfg.api_key)[:300]
            outcome = category or _classify_failure(message)
            if outcome == "no_keypoints":
                # A content answer, and an empty one: the scene is what failed
                # to answer. Transport failures say nothing about the scene and
                # leave the drought alone.
                self._drought += 1
            self._finish_locked(rec, outcome)

    def _record_refusal(self, reason: str, rec: Optional[dict] = None) -> None:
        with self._lock:
            self._refusals += 1
            self._last_refusal = reason[:120]
            self._refusals_by_reason[reason[:120]] = (
                self._refusals_by_reason.get(reason[:120], 0) + 1
            )
            # Keypoints came back — the model can see a pitch here, however the
            # validator judged this particular frame.
            self._drought = 0
            if rec is not None:
                rec["reason"] = reason[:120]
            self._finish_locked(rec, "refused")

    def _finish_locked(self, rec: Optional[dict], outcome: str) -> None:
        """File one attempt's verdict. `self._lock` must already be held."""
        self._categories[outcome] = self._categories.get(outcome, 0) + 1
        if rec is None:
            return
        rec["outcome"] = outcome
        if rec.get("wait_ms") is not None:
            self._queue_wait_ms.append(rec["wait_ms"])
        if rec.get("infer_ms") is not None:
            self._infer_ms.append(rec["infer_ms"])
        if len(self._attempts) < self._attempt_cap:
            self._attempts.append(rec)
        else:
            self._attempts_truncated += 1


# ── diagnostics helpers ──────────────────────────────────────────────────────
def _stats(values: list) -> dict:
    """Count, mean and percentiles of a latency sample, in milliseconds."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    def quantile(p: float) -> float:
        return round(ordered[int(round(p * (len(ordered) - 1)))], 2)
    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 2),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "max": round(ordered[-1], 2),
    }


def _rate(count: int, first: Optional[float], last: Optional[float]) -> Optional[float]:
    """Events per second over the span the first and last one were stamped."""
    if count < 2 or first is None or last is None or last <= first:
        return None
    return round((count - 1) / (last - first), 3)


def _span(first: Optional[float], last: Optional[float]) -> Optional[float]:
    if first is None or last is None or last < first:
        return None
    return round(last - first, 3)


def _sorted_counts(counts: dict) -> dict:
    """Largest count first, then alphabetical — a stable report ordering."""
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _classify_failure(message: str) -> str:
    """
    Which of the expected failures this one was.

    Ordered by specificity: the dominant outcome (an answer with no keypoints)
    first, rate limiting before the generic API bucket — a 429 must never be
    buried inside "api_error", because it is the one verdict that would mean
    the diagnosis is wrong — then transport, then anything else.
    """
    text = (message or "").lower()
    if "no keypoints" in text:
        return "no_keypoints"
    if "no response after" in text or "consecutive" in text:
        return "timeout"
    if "429" in text or "too many requests" in text:
        return "rate_limited"
    if "status_code=" in text:
        return "api_error"
    if any(hint in text for hint in (
        "connection", "max retries", "read timed out", "newconnection",
        "temporary failure in name resolution", "proxy",
    )):
        return "network"
    if "model returned nothing" in text:
        return "empty_response"
    return "error"


def _empty_block(state: str) -> dict:
    """The block with every key present and nothing invented."""
    return {
        "state": state,
        "detected": False,
        "calibrated": False,
        "using_previous_calibration": False,
        "confidence": None,
        "keypoints": {},
        "corners": {},
        "center": None,
        "homography_available": False,
        "homography": None,
        "image_size": None,
        "frame_number": None,
        "model_id": None,
    }


__all__ = [
    "STATE_CALIBRATED",
    "STATE_EXPIRED",
    "STATE_NO_CALIBRATION",
    "STATE_TEMPORARY_LOSS",
    "PitchService",
]
