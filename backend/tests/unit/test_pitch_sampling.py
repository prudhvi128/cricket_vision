"""
test_pitch_sampling.py - which frames get offered, and what each attempt cost.

Two things are pinned down here, both of them new:

1. THE GATE. `interval` mode must behave exactly as it always has (one frame
   per `interval_frames`, whatever the loop's hints say — that is what a bare
   `PitchConfig()` still does, and what the existing suite already asserts).
   `adaptive` mode scales that base interval by delivery activity, corridor
   brightness, calibration coverage and queue back-pressure, and every one of
   those rules is a SPATIAL decision about which frame to send — never a
   change to what a response means.

2. THE DIAGNOSTIC RECORD. Every frame the loop offers leaves a trace: submitted,
   or gated with a reason; every submitted frame ends in exactly one category
   (detected / refused / no_keypoints / timeout / api_error / ...), with its
   queue wait and its round trip measured. That is what turns "799 failures"
   into a list of causes.
"""

from __future__ import annotations

import dataclasses
import json
import time
import unittest

from app.core.config import PitchConfig
from app.core.constants import (
    PITCH_ATTEMPTS_PER_SEC_FLOOR,
    PITCH_COVERED_INTERVAL_MULTIPLIER,
    PITCH_DETECTOR_REARM_FRAMES,
    PITCH_DIM_INTERVAL_MULTIPLIER,
    PITCH_DROUGHT_BACKOFF,
    PITCH_DROUGHT_INTERVAL_MULTIPLIER,
    PITCH_DROUGHT_INTERVAL_MULTIPLIER_IN_EVENT,
    PITCH_FORCE_INTERVAL_MULTIPLIER,
    PITCH_IDLE_INTERVAL_MULTIPLIER,
    PITCH_MAX_ATTEMPTS_PER_SEC,
    PITCH_SAMPLING_MODE_ADAPTIVE,
    PITCH_SAMPLING_MODE_DEFAULT,
    PITCH_SAMPLING_MODE_INTERVAL,
    PITCH_SAMPLING_MODE_SEQUENTIAL,
)
from app.pitch.detector import Keypoint, PitchDetectError, PitchDetection, PitchTimeout
from app.pitch.service import PitchService

W, H = 320, 180
SIZE = (W, H)
KEY = "SAMPLING_KEY_MUST_NEVER_APPEAR"
BRIGHT = 200.0   # corridor luminance: a pitch, plainly visible
DARK = 40.0      # corridor luminance: crowd, graphic, night


def quad_detection(frame_number: int, image_size=SIZE, confidence=0.9) -> PitchDetection:
    width, height = image_size
    coords = [
        (0.1 * width, 0.1 * height),
        (0.9 * width, 0.1 * height),
        (0.9 * width, 0.9 * height),
        (0.1 * width, 0.9 * height),
    ]
    return PitchDetection(
        keypoints=[Keypoint(f"p{i}", x, y, confidence) for i, (x, y) in enumerate(coords, 1)],
        frame_number=frame_number,
        image_size=image_size,
        confidence=confidence,
    )


class FakeDetector:
    def __init__(self, result=None, error=None, delay: float = 0.0) -> None:
        self.result = result
        self.error = error
        self.delay = delay
        self.calls = 0
        self.disabled_reason = None

    def detect(self, frame, frame_number: int, image_size) -> PitchDetection:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if callable(self.result):
            return self.result(frame_number, image_size)
        if self.result is None:
            return quad_detection(frame_number, image_size)
        return self.result


def service(mode=PITCH_SAMPLING_MODE_ADAPTIVE, interval=5, detector=None, **cfg_kw) -> PitchService:
    cfg = dataclasses.replace(
        PitchConfig(api_key=""),
        interval_frames=interval,
        sampling_mode=mode,
        **cfg_kw,
    )
    return PitchService(cfg, width=W, height=H, detector=detector or FakeDetector())


# ── The gate ─────────────────────────────────────────────────────────────────
class TestIntervalModeIsUntouched(unittest.TestCase):
    """The default must be byte-for-byte the old rule."""

    def test_hints_do_not_change_a_single_decision(self):
        svc = service(mode=PITCH_SAMPLING_MODE_INTERVAL, interval=10)
        svc._last_submitted_frame = 0
        # Bright, dark, black, in-delivery, out-of-delivery: all the same.
        for kwargs in (
            {"corridor_lum": BRIGHT, "event_open": True},
            {"corridor_lum": DARK, "event_open": False},
            {"corridor_lum": DARK, "event_open": False, "black": True, "scene_cut": True},
            {},
        ):
            self.assertIsNone(svc._gate(10, **{
                "corridor_lum": kwargs.get("corridor_lum"),
                "event_open": kwargs.get("event_open"),
                "scene_cut": kwargs.get("scene_cut", False),
                "black": kwargs.get("black", False),
            }))
            self.assertEqual(
                svc._gate(9, corridor_lum=kwargs.get("corridor_lum"),
                          event_open=kwargs.get("event_open"),
                          scene_cut=kwargs.get("scene_cut", False),
                          black=kwargs.get("black", False)),
                "spacing",
            )

    def test_unknown_mode_resolves_to_the_default(self):
        self.assertEqual(PitchConfig(api_key="", sampling_mode="nonsense").sampling_mode,
                         PITCH_SAMPLING_MODE_DEFAULT)
        self.assertEqual(PitchConfig(api_key="", sampling_mode="  ADAPTIVE ".lower()).sampling_mode,
                         "adaptive")
        self.assertEqual(PitchConfig(api_key="").sampling_mode, PITCH_SAMPLING_MODE_INTERVAL)

    def test_the_rate_ceiling_is_configurable_and_never_zero(self):
        cfg = PitchConfig(api_key="", sampling_mode=PITCH_SAMPLING_MODE_SEQUENTIAL,
                          max_attempts_per_sec=0.0)
        self.assertEqual(cfg.max_attempts_per_sec, PITCH_ATTEMPTS_PER_SEC_FLOOR,
                         "a rate that would divide by zero is corrected, not used")
        self.assertEqual(PitchConfig(api_key="").max_attempts_per_sec,
                         PITCH_MAX_ATTEMPTS_PER_SEC)
        self.assertIn("max_attempts_per_sec", PitchConfig(api_key="").metadata())


class TestAdaptiveGate(unittest.TestCase):
    def setUp(self):
        self.svc = service(interval=5)

    def gate(self, frame, **kwargs):
        kwargs.setdefault("corridor_lum", BRIGHT)
        kwargs.setdefault("event_open", True)
        return self.svc._gate(
            frame,
            corridor_lum=kwargs.get("corridor_lum"),
            event_open=kwargs.get("event_open"),
            near_event=kwargs.get("near_event"),
            scene_cut=kwargs.get("scene_cut", False),
            black=kwargs.get("black", False),
        )

    def test_inside_a_delivery_the_base_interval_stands(self):
        self.svc._last_submitted_frame = 0
        self.assertEqual(self.gate(4), "spacing")
        self.assertIsNone(self.gate(5))

    def test_the_guard_that_runs_past_a_delivery_counts_as_the_delivery(self):
        # The clip window outlives the segmenter's event: frames after it close
        # are where the model actually answers, so they are not "idle".
        self.svc._last_submitted_frame = 0
        self.assertEqual(self.gate(4, event_open=False, near_event=True), "spacing")
        self.assertIsNone(self.gate(5, event_open=False, near_event=True))

    def test_between_deliveries_the_model_is_asked_an_eighth_as_often(self):
        self.svc._last_submitted_frame = 0
        self.assertEqual(self.gate(5, event_open=False), "spacing")
        self.assertIsNone(self.gate(5 * PITCH_IDLE_INTERVAL_MULTIPLIER, event_open=False))

    def test_a_caller_that_gives_no_hints_is_not_treated_as_idle(self):
        self.svc._last_submitted_frame = 0
        self.assertIsNone(self.gate(5, event_open=None, near_event=None),
                          "silence is not evidence that the frame is between deliveries")

    def test_a_measured_quad_is_not_re_measured_immediately(self):
        self.svc._process(100, None)          # accepted: covers the view
        self.svc._last_submitted_frame = 100
        covered = 5 * PITCH_COVERED_INTERVAL_MULTIPLIER
        self.assertEqual(self.gate(101), "spacing")
        self.assertIsNone(self.gate(100 + covered + 1))

    def test_coverage_never_outlives_the_calibration_itself(self):
        svc = service(interval=5, max_lost_frames=12)
        self.assertLessEqual(svc._covered_frames, 12)
        self.assertEqual(svc._covered_frames, 12)  # min(5*6, 12)

    def test_a_dark_corridor_halves_the_rate_but_never_stops_it(self):
        self.svc._last_submitted_frame = 0
        self.assertEqual(self.gate(5, corridor_lum=DARK), "spacing")
        self.assertIsNone(self.gate(5 * PITCH_DIM_INTERVAL_MULTIPLIER, corridor_lum=DARK))

    def test_a_black_frame_is_never_submitted_not_even_when_overdue(self):
        self.svc._last_submitted_frame = -10**9
        self.assertEqual(self.gate(1, black=True), "black")

    def test_a_forced_attempt_bounds_the_longest_silence(self):
        # Between deliveries, and in a scene the model has stopped answering:
        # spacing would be 5 * 40 = 200 frames, and the force floor is the same
        # 200, so no frame can wait longer than that.
        self.svc._last_submitted_frame = 0
        self.svc._drought = PITCH_DROUGHT_BACKOFF
        force = 5 * PITCH_FORCE_INTERVAL_MULTIPLIER
        self.assertEqual(self.gate(force - 1, event_open=False), "spacing")
        self.assertIsNone(self.gate(force, event_open=False))

    def test_a_drought_backs_the_sampler_off_and_a_delivery_brings_it_back(self):
        self.svc._last_submitted_frame = 0
        self.svc._drought = PITCH_DROUGHT_BACKOFF
        # Between deliveries: a rare keepalive, not the idle rate.
        self.assertEqual(self.gate(5 * PITCH_IDLE_INTERVAL_MULTIPLIER, event_open=False),
                         "spacing")
        self.assertIsNone(
            self.gate(5 * PITCH_DROUGHT_INTERVAL_MULTIPLIER, event_open=False))
        # Inside a delivery the question is still worth asking, just less often.
        self.assertEqual(self.gate(5 * PITCH_DROUGHT_INTERVAL_MULTIPLIER_IN_EVENT - 1),
                         "spacing")
        self.assertIsNone(
            self.gate(5 * PITCH_DROUGHT_INTERVAL_MULTIPLIER_IN_EVENT))

    def test_a_scene_cut_starts_the_conversation_again(self):
        self.svc._drought = PITCH_DROUGHT_BACKOFF
        self.svc._last_submitted_frame = 0
        # Before the cut the backoff is in force: one attempt in 200 frames.
        self.assertEqual(self.gate(5 * PITCH_DROUGHT_INTERVAL_MULTIPLIER - 1,
                                   event_open=False), "spacing")
        # The cut opens content the old answers say nothing about: back to the
        # ordinary idle rate, not the drought rate.
        self.assertEqual(self.gate(5 * PITCH_IDLE_INTERVAL_MULTIPLIER - 1,
                                   event_open=False, scene_cut=True), "spacing")
        self.assertIsNone(self.gate(5 * PITCH_IDLE_INTERVAL_MULTIPLIER,
                                    event_open=False, scene_cut=True))
        self.assertEqual(self.svc._drought, 0,
                         "new content says nothing about the old answers")

    def test_only_an_empty_answer_builds_a_drought(self):
        svc = service(detector=FakeDetector(
            error=PitchDetectError("response held no keypoints")))
        svc._record_failure("response held no keypoints")
        self.assertEqual(svc.summary()["diagnostics"]["drought"], 1)
        svc._record_failure("HTTPCallErrorError(status_code=429, ...)")
        self.assertEqual(svc.summary()["diagnostics"]["drought"], 1,
                         "a transport failure is not evidence about the scene")
        svc._record_refusal("keypoint_hull_too_small")
        self.assertEqual(svc.summary()["diagnostics"]["drought"], 0,
                         "the model can see a pitch here")

    def test_a_scene_cut_ends_coverage_so_the_new_view_is_measured(self):
        self.svc._last_submitted_frame = 46
        self.svc._covered_until = 1000
        # Still covered: the fresh quad is being waited out, not re-measured.
        self.assertEqual(self.gate(51), "spacing")
        # The cut replaced that view, so the same frame becomes eligible again.
        self.assertIsNone(self.gate(51, scene_cut=True))
        self.assertEqual(self.svc._covered_until, 50)

    def test_a_busy_queue_only_gives_way_to_a_delivery(self):
        self.svc._last_submitted_frame = -100
        self.svc._queue.put_nowait((1, None, time.perf_counter(), False, False))
        try:
            self.assertEqual(self.gate(5, event_open=False), "busy")
            # The guard counts: a frame from the delivery's own tail may
            # displace a frame that was only waiting.
            self.assertIsNone(self.gate(5, event_open=False, near_event=True))
            self.assertIsNone(self.gate(5, event_open=True))
        finally:
            self.svc._queue.get_nowait()

    def test_an_event_frame_displaces_a_stale_idle_frame(self):
        svc = service(interval=5)
        svc._ensure_worker = lambda: None          # hold the queue still
        svc.on_frame(1, None, event_open=False, corridor_lum=BRIGHT)
        self.assertEqual(svc.summary()["submitted"], 1)
        svc.on_frame(6, None, event_open=False, corridor_lum=BRIGHT)
        self.assertEqual(svc.summary()["submitted"], 1, "an idle frame must wait its turn")
        svc.on_frame(11, None, event_open=True, corridor_lum=BRIGHT)
        self.assertEqual(svc.summary()["submitted"], 2, "a delivery frame may queue behind it")
        svc.on_frame(16, None, event_open=True, corridor_lum=BRIGHT)
        summary = svc.summary()
        self.assertEqual(summary["submitted"], 3)
        self.assertEqual(summary["queue_drops"], 1, "the stale idle frame lost its slot")
        self.assertEqual(summary["diagnostics"]["evicted_frames"], [1])


# ── `sequential`: backpressure-aware, one call at a time ──────────────────────
class TestSequentialGate(unittest.TestCase):
    """The gate answers in terms of the detector's pace, and nothing else."""

    def setUp(self):
        self.svc = service(mode=PITCH_SAMPLING_MODE_SEQUENTIAL, interval=5)

    def tearDown(self):
        self.svc.stop()

    def gate(self, **kwargs):
        return self.svc._gate(
            100,
            corridor_lum=kwargs.get("corridor_lum", BRIGHT),
            event_open=kwargs.get("event_open", True),
            near_event=kwargs.get("near_event"),
            scene_cut=kwargs.get("scene_cut", False),
            black=kwargs.get("black", False),
        )

    def test_an_idle_detector_with_no_cooldown_takes_the_frame(self):
        self.assertIsNone(self.gate())

    def test_a_black_frame_is_never_sent(self):
        self.assertEqual(self.gate(black=True), "black")

    def test_the_cooldown_holds_the_next_offer_back(self):
        self.svc._cooldown_until = time.perf_counter() + 5.0
        self.assertEqual(self.gate(), "cooldown")
        self.svc._cooldown_until = time.perf_counter() - 1.0
        self.assertIsNone(self.gate())

    def test_one_inference_at_a_time_and_no_backlog_behind_it(self):
        self.svc._inflight = True
        self.assertEqual(self.gate(), "busy", "a call is running: ask nothing")
        self.svc._inflight = False
        self.svc._queue.put_nowait((1, None, time.perf_counter(), False, False))
        try:
            self.assertEqual(self.gate(), "busy", "something is queued: ask nothing")
        finally:
            self.svc._queue.get_nowait()
        self.assertIsNone(self.gate(), "the answer is in: this is the next frame")

    def test_the_cooldown_is_the_configured_ceiling(self):
        self.assertEqual(self.svc._cooldown_seconds, 0.5)          # 2.0 attempts/s
        slower = service(mode=PITCH_SAMPLING_MODE_SEQUENTIAL,
                         max_attempts_per_sec=1.5)
        self.assertAlmostEqual(slower._cooldown_seconds, 1 / 1.5, places=6)

    def test_an_offered_frame_records_the_backpressure_reason(self):
        self.svc._inflight = True
        self.svc.on_frame(1, None, corridor_lum=BRIGHT, event_open=True)
        summary = self.svc.summary()
        self.assertEqual(summary["submitted"], 0)
        self.assertEqual(summary["diagnostics"]["offered"], 1)
        self.assertEqual(summary["diagnostics"]["gated"], {"busy": 1})

    def test_a_submission_starts_the_cooldown(self):
        self.svc.on_frame(1, None, corridor_lum=BRIGHT, event_open=True)
        self.assertEqual(self.svc.summary()["submitted"], 1)
        self.assertGreater(self.svc._cooldown_until, time.perf_counter(),
                           "the next offer waits out 1/max_attempts_per_sec")

    def test_the_ceiling_reaches_the_diagnostics(self):
        backpressure = self.svc.summary()["diagnostics"]["backpressure"]
        self.assertEqual(backpressure["max_attempts_per_sec"], PITCH_MAX_ATTEMPTS_PER_SEC)
        self.assertEqual(backpressure["cooldown_ms"], 500.0)
        self.assertFalse(backpressure["in_flight"])

    def test_the_other_modes_cannot_see_the_cooldown_or_the_backpressure(self):
        # `interval` and `adaptive` keep their own rules byte for byte: a
        # pending cooldown or a running call does not gate them.
        for mode in (PITCH_SAMPLING_MODE_INTERVAL, PITCH_SAMPLING_MODE_ADAPTIVE):
            svc = service(mode=mode, interval=10)
            svc._last_submitted_frame = 0
            svc._cooldown_until = time.perf_counter() + 10.0
            svc._inflight = True
            self.assertIsNone(
                svc._gate(10, corridor_lum=None, event_open=None,
                          scene_cut=False, black=False),
                f"{mode} must behave exactly as before",
            )


class TestSequentialNeverFillsTheQueue(unittest.TestCase):
    """Measured behaviour: the rate follows the detector, drops stay at zero."""

    def offer_for(self, svc: PitchService, seconds: float, step: float = 0.004) -> float:
        """Offer frames as fast as the loop decodes them, for a wall-clock span."""
        started = time.perf_counter()
        frame = 0
        while time.perf_counter() - started < seconds:
            svc.on_frame(frame, None, corridor_lum=BRIGHT, event_open=True)
            frame += 1
            time.sleep(step)
        return time.perf_counter() - started

    def test_a_slow_detector_slows_the_sampler_instead_of_dropping_frames(self):
        svc = service(mode=PITCH_SAMPLING_MODE_SEQUENTIAL,
                      detector=FakeDetector(delay=0.6))   # slower than the 0.5 s ceiling
        try:
            elapsed = self.offer_for(svc, 1.3)
        finally:
            svc.stop(timeout=3.0)
        summary = svc.summary()
        diag = summary["diagnostics"]
        self.assertEqual(summary["queue_drops"], 0, "nothing was queued behind a slow answer")
        self.assertLessEqual(diag["queue"]["max_depth_observed"], 1)
        self.assertEqual(summary["submitted"],
                         sum(diag["categories"].values()),
                         "every submission was answered, none left behind")
        # The interval stretched to the detector's own pace: one attempt per
        # ~0.6 s, so 1.3 s cannot hold more than a handful.
        self.assertGreaterEqual(summary["submitted"], 2)
        self.assertLessEqual(summary["submitted"], int(elapsed / 0.6) + 2)
        self.assertGreater(diag["gated"].get("cooldown", 0)
                           + diag["gated"].get("busy", 0), 0)

    def test_a_fast_detector_rises_only_to_the_configured_ceiling(self):
        svc = service(mode=PITCH_SAMPLING_MODE_SEQUENTIAL,
                      detector=FakeDetector(), max_attempts_per_sec=4.0)  # 0.25 s apart
        try:
            elapsed = self.offer_for(svc, 0.9)
        finally:
            svc.stop(timeout=3.0)
        summary = svc.summary()
        self.assertEqual(summary["queue_drops"], 0)
        self.assertGreaterEqual(summary["submitted"], 3)
        self.assertLessEqual(summary["submitted"], int(elapsed / 0.25) + 2)

    def test_the_pass_still_builds_a_moving_camera_calibration(self):
        svc = service(mode=PITCH_SAMPLING_MODE_SEQUENTIAL,
                      detector=FakeDetector(), max_attempts_per_sec=50.0)
        try:
            for frame in range(0, 60, 3):
                svc.on_frame(frame, None, corridor_lum=BRIGHT, event_open=True)
                time.sleep(0.03)   # the loop decodes: frames arrive over time
        finally:
            svc.stop()
        summary = svc.summary()
        self.assertGreaterEqual(summary["detections"], 1)
        self.assertGreaterEqual(summary["calibration_updates"], 1)
        snapshot = svc.snapshot(59)
        self.assertEqual(snapshot["state"], "calibrated")
        self.assertTrue(snapshot["homography_available"],
                        "the geometry the calibration produces is untouched")
        self.assertTrue(snapshot["homography"])


# ── The diagnostic record ────────────────────────────────────────────────────
class TestDiagnostics(unittest.TestCase):
    def test_every_attempt_ends_in_exactly_one_category(self):
        svc = service(detector=FakeDetector())
        svc._process(1, None, time.perf_counter() - 0.01, True)          # detected
        for frame, error in [
            (2, PitchDetectError("response held no keypoints")),
            (3, PitchTimeout("no response after 15.0s (1 consecutive)")),
            (4, PitchDetectError("HTTPCallErrorError(status_code=500, ...)")),
            (5, PitchDetectError("Max retries exceeded with url: /infer")),
        ]:
            svc._record_failure(str(error), rec={"frame": frame, "wait_ms": 1.0})
        svc._record_refusal("keypoint_hull_too_small", rec={"frame": 6, "wait_ms": 1.0})
        svc.stop()

        summary = svc.summary()
        diag = summary["diagnostics"]
        self.assertEqual(diag["categories"].get("detected"), 1)
        self.assertEqual(diag["categories"].get("no_keypoints"), 1)
        self.assertEqual(diag["categories"].get("timeout"), 1)
        self.assertEqual(diag["categories"].get("api_error"), 1)
        self.assertEqual(diag["categories"].get("network"), 1)
        self.assertEqual(diag["categories"].get("refused"), 1)
        self.assertEqual(diag["refusals"], {"keypoint_hull_too_small": 1})
        self.assertEqual(
            sum(diag["categories"].values()),
            summary["detections"] + summary["failures"] + summary["refusals"],
        )
        self.assertEqual(len(diag["attempts"]), 6)

    def test_a_rate_limit_is_its_own_category(self):
        self.assertEqual(
            _classify("HTTPCallErrorError(status_code=429, ...)"), "rate_limited"
        )
        self.assertEqual(_classify("response held no keypoints"), "no_keypoints")
        self.assertEqual(_classify("HTTPCallErrorError(status_code=500)"), "api_error")
        self.assertEqual(_classify("Max retries exceeded with url: /infer"), "network")
        self.assertEqual(_classify("something entirely new"), "error")

    def test_queue_wait_and_round_trip_are_both_measured(self):
        svc = service(detector=FakeDetector(delay=0.02))
        svc._process(7, None, time.perf_counter() - 0.05, True)
        diag = svc.summary()["diagnostics"]
        self.assertEqual(diag["latency_ms"]["n"], 1)
        self.assertGreaterEqual(diag["latency_ms"]["mean"], 15)
        self.assertEqual(diag["queue_wait_ms"]["n"], 1)
        self.assertGreaterEqual(diag["queue_wait_ms"]["mean"], 40)
        self.assertEqual(diag["attempts"][0]["frame"], 7)
        self.assertEqual(diag["attempts"][0]["outcome"], "detected")
        self.assertIn("keypoints", diag["attempts"][0])

    def test_offered_frames_reconcile_with_submissions_and_gates(self):
        svc = service(interval=5)
        for frame in range(0, 50):
            svc.on_frame(frame, None, corridor_lum=BRIGHT, event_open=True)
        svc.stop()
        diag = svc.summary()["diagnostics"]
        gated = sum(diag["gated"].values())
        self.assertEqual(diag["offered"], 50)
        self.assertEqual(diag["offered"], diag.get("offered"))
        self.assertEqual(svc.summary()["submitted"] + gated, 50)
        self.assertEqual(diag["sampling_mode"], PITCH_SAMPLING_MODE_ADAPTIVE)
        self.assertEqual(diag["base_interval_frames"], 5)

    def test_the_credential_never_reaches_the_diagnostics(self):
        svc = PitchService(
            dataclasses.replace(PitchConfig(api_key=KEY), interval_frames=5),
            width=W, height=H, detector=FakeDetector(error=PitchDetectError(f"api_key={KEY}")),
        )
        svc._process(1, None, None, None)
        payload = json.dumps(svc.summary()["diagnostics"])
        self.assertNotIn(KEY, payload)

    def test_the_attempt_record_says_whether_the_frame_was_a_delivery_frame(self):
        svc = service()
        svc._process(1, None, None, False, True)    # guard frame: delivery tail
        svc._process(2, None, None, False, False)   # genuinely between deliveries
        attempts = svc.summary()["diagnostics"]["attempts"]
        self.assertEqual([a["hot"] for a in attempts], [True, False])
        self.assertEqual([a["event"] for a in attempts], [False, False])

    def test_a_disabled_service_submits_nothing_but_still_counts_offers(self):
        svc = service(detector=FakeDetector())
        svc._detector = None
        svc.cfg = dataclasses.replace(svc.cfg, api_key="")
        for frame in range(10):
            svc.on_frame(frame, None)
        summary = svc.summary()
        self.assertEqual(summary["submitted"], 0)
        self.assertEqual(summary["detections"], 0)
        self.assertEqual(summary["diagnostics"]["offered"], 10,
                         "frames the detector refused are still frames offered")
        self.assertEqual(summary["diagnostics"]["gated"], {"disabled": 10})


class TestDetectorRearm(unittest.TestCase):
    """A bad minute must not silence the rest of the video."""

    def service_with_dead_client(self):
        det = FakeDetector()
        svc = service(detector=det)
        det.disabled_reason = "timeout"
        return svc, det

    def test_a_timeout_disable_is_not_forever(self):
        svc, det = self.service_with_dead_client()
        svc.on_frame(1, None)      # the disable is noticed here
        svc.on_frame(2, None)      # the window starts here
        for frame in range(3, 2 + PITCH_DETECTOR_REARM_FRAMES):
            svc.on_frame(frame, None)
        summary = svc.summary()
        self.assertEqual(summary["diagnostics"]["detector_rearms"], 0)
        self.assertEqual(summary["submitted"], 0)
        # The window elapsed: the client is tried again.
        svc.on_frame(2 + PITCH_DETECTOR_REARM_FRAMES, None)
        summary = svc.summary()
        self.assertEqual(summary["diagnostics"]["detector_rearms"], 1)
        self.assertTrue(svc.enabled)
        self.assertIsNone(summary["detector_error"])
        self.assertEqual(summary["submitted"], 1)

    def test_three_more_timeouts_put_it_back_to_sleep(self):
        svc, det = self.service_with_dead_client()
        svc.on_frame(1, None)
        svc.on_frame(2, None)
        svc.on_frame(2 + PITCH_DETECTOR_REARM_FRAMES, None)     # rearmed
        det.disabled_reason = "timeout"                         # and failed again
        svc.on_frame(3 + PITCH_DETECTOR_REARM_FRAMES, None)     # noticed here
        start = 4 + PITCH_DETECTOR_REARM_FRAMES
        svc.on_frame(start, None)                               # window starts
        for frame in range(start + 1, start + PITCH_DETECTOR_REARM_FRAMES):
            svc.on_frame(frame, None)
        self.assertEqual(svc.summary()["diagnostics"]["detector_rearms"], 1,
                         "a fresh failure waits a full window")
        svc.on_frame(start + PITCH_DETECTOR_REARM_FRAMES, None)
        self.assertEqual(svc.summary()["diagnostics"]["detector_rearms"], 2)

    def test_a_permanent_error_is_never_retried(self):
        svc = PitchService(
            dataclasses.replace(PitchConfig(api_key=""), interval_frames=5,
                                sampling_mode=PITCH_SAMPLING_MODE_ADAPTIVE),
            width=W, height=H, detector=FakeDetector(),
        )
        svc._detector_error = "no credential configured"
        svc._detector = None
        for frame in range(1, PITCH_DETECTOR_REARM_FRAMES * 2, 50):
            svc.on_frame(frame, None, corridor_lum=BRIGHT, event_open=True)
        summary = svc.summary()
        self.assertEqual(summary["diagnostics"]["detector_rearms"], 0)
        self.assertEqual(summary["submitted"], 0)


def _classify(message: str) -> str:
    from app.pitch.service import _classify_failure
    return _classify_failure(message)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
