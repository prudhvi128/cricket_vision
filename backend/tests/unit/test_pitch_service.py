"""
test_pitch_service.py - the background service the tracking loop talks to.

Unit tests over `app.pitch.service.PitchService` with an injected detector: no
network, no video, no API key. What they pin down:

  * with no credential and no injected detector the service is inert - no
    submissions, no worker thread, and every snapshot still answers with the
    honest `no_calibration` block (all thirteen keys, nothing invented);
  * an injected detector works WITHOUT a credential, because it is an explicit
    choice by the caller;
  * the interval gate submits one frame per `interval_frames` and nothing else;
  * the four states are a function of age alone: calibrated while fresh,
    temporary_loss while the last good calibration is still within
    `max_lost_frames`, expired past it - and never a calibration from the
    future;
  * failures, refusals and their reasons are counted for the analysis record,
    and a credential in an error message never survives into it;
  * `on_frame` never blocks: a detector slower than the loop drops frames
    instead of waiting, and `stop()` drains what is queued.
"""

from __future__ import annotations

import dataclasses
import json
import time
import unittest

from app.core.config import PitchConfig
from app.pitch.calibration import build_calibration
from app.pitch.detector import (
    Keypoint,
    PitchDetectError,
    PitchDetection,
    PitchDetectorUnavailable,
)
from app.pitch.service import (
    STATE_CALIBRATED,
    STATE_EXPIRED,
    STATE_NO_CALIBRATION,
    STATE_TEMPORARY_LOSS,
    PitchService,
)

W, H = 320, 180
SIZE = (W, H)
KEY = "SERVICE_KEY_SHOULD_NEVER_APPEAR"
EMPTY_BLOCK_KEYS = {
    "state",
    "detected",
    "calibrated",
    "using_previous_calibration",
    "confidence",
    "keypoints",
    "corners",
    "center",
    "homography_available",
    "homography",
    "image_size",
    "frame_number",
    "model_id",
}


def quad_detection(frame_number: int, image_size=SIZE, confidence=0.9) -> PitchDetection:
    """A detection the validator accepts: a rectangle at 10%..90% of the frame."""
    width, height = image_size
    coords = [
        (0.1 * width, 0.1 * height),
        (0.9 * width, 0.1 * height),
        (0.9 * width, 0.9 * height),
        (0.1 * width, 0.9 * height),
    ]
    return PitchDetection(
        keypoints=[
            Keypoint(f"p{i}", x, y, confidence) for i, (x, y) in enumerate(coords, 1)
        ],
        frame_number=frame_number,
        image_size=image_size,
        confidence=confidence,
    )


class FakeDetector:
    """A detector a test can steer: return a value, raise, or take its time."""

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


def service(cfg=None, detector=None, *, width=W, height=H) -> PitchService:
    return PitchService(
        cfg if cfg is not None else PitchConfig(api_key=""),
        width=width,
        height=height,
        detector=detector,
    )


# ── Capability ───────────────────────────────────────────────────────────────
class TestCapability(unittest.TestCase):
    def test_no_credential_means_no_work_at_all(self):
        svc = service()
        self.assertFalse(svc.enabled)
        for frame in range(5):
            svc.on_frame(frame, None)
        self.assertEqual(svc.summary()["submitted"], 0)
        self.assertIsNone(svc._thread, "a disabled service must not start a worker")

    def test_the_disabled_service_still_answers_every_snapshot(self):
        snapshot = service().snapshot(0)
        self.assertEqual(set(snapshot), EMPTY_BLOCK_KEYS)
        self.assertEqual(snapshot["state"], STATE_NO_CALIBRATION)
        self.assertFalse(snapshot["detected"])
        self.assertFalse(snapshot["calibrated"])
        self.assertFalse(snapshot["homography_available"])
        self.assertIsNone(snapshot["confidence"])
        self.assertEqual(snapshot["keypoints"], {})
        self.assertEqual(snapshot["corners"], {})

    def test_an_injected_detector_needs_no_credential(self):
        svc = service(detector=FakeDetector())
        self.assertTrue(svc.enabled)
        svc.on_frame(0, None)
        svc.stop()
        self.assertEqual(svc.summary()["submitted"], 1)
        self.assertTrue(svc.summary()["enabled"])

    def test_summary_reports_the_injection_without_a_key(self):
        summary = service(detector=FakeDetector()).summary()
        self.assertTrue(summary["enabled"])
        self.assertFalse(summary["config"]["api_key_configured"])
        self.assertNotIn("api_key", summary["config"])

    def test_the_credential_never_reaches_the_summary(self):
        svc = service(cfg=PitchConfig(api_key=KEY))
        payload = json.dumps(svc.summary())
        self.assertNotIn(KEY, payload)
        self.assertIn("api_key_configured", payload)

    def test_a_disabled_detector_stops_submissions(self):
        fake = FakeDetector()
        fake.disabled_reason = "timeout"
        svc = service(detector=fake)
        self.assertFalse(svc.enabled)
        svc.on_frame(0, None)
        self.assertEqual(svc.summary()["submitted"], 0)
        self.assertIn("timeout", svc.summary()["detector_error"])


# ── The interval gate ────────────────────────────────────────────────────────
class TestIntervalGating(unittest.TestCase):
    def setUp(self):
        self.fake = FakeDetector()
        self.svc = service(
            dataclasses.replace(PitchConfig(api_key=""), interval_frames=10),
            detector=self.fake,
        )

    def tearDown(self):
        self.svc.stop()

    def test_only_every_tenth_frame_is_submitted(self):
        for frame in range(11):  # 0..10 inclusive
            self.svc.on_frame(frame, None)
        self.assertEqual(self.svc.summary()["submitted"], 2)

    def test_the_gate_is_by_distance_not_by_count(self):
        self.svc.on_frame(0, None)
        self.svc.on_frame(4, None)  # too soon
        self.svc.on_frame(10, None)  # exactly one interval later
        self.assertEqual(self.svc.summary()["submitted"], 2)

    def test_a_one_frame_interval_submits_every_frame(self):
        svc = service(
            dataclasses.replace(PitchConfig(api_key=""), interval_frames=1),
            detector=FakeDetector(),
        )
        try:
            for frame in range(5):
                svc.on_frame(frame, None)
            self.assertEqual(svc.summary()["submitted"], 5)
        finally:
            svc.stop()


# ── The state machine ────────────────────────────────────────────────────────
class TestStateMachine(unittest.TestCase):
    """Driven through `_process` directly: no worker, no timing, no races."""

    def setUp(self):
        self.svc = service(detector=FakeDetector())

    def process(self, frame_number: int) -> None:
        self.svc._process(frame_number, None)

    def test_before_any_detection_there_is_nothing_to_draw(self):
        snapshot = self.svc.snapshot(0)
        self.assertEqual(set(snapshot), EMPTY_BLOCK_KEYS)
        self.assertEqual(snapshot["state"], STATE_NO_CALIBRATION)

    def test_a_fresh_detection_is_calibrated(self):
        self.process(100)
        snapshot = self.svc.snapshot(100)
        self.assertEqual(snapshot["state"], STATE_CALIBRATED)
        self.assertTrue(snapshot["detected"])
        self.assertTrue(snapshot["calibrated"])
        self.assertFalse(snapshot["using_previous_calibration"])
        self.assertTrue(snapshot["homography_available"])
        self.assertIsNotNone(snapshot["homography"])
        self.assertEqual(snapshot["frame_number"], 100)
        self.assertEqual(snapshot["image_size"], [W, H])
        self.assertEqual(snapshot["model_id"], PitchConfig().model_id)
        self.assertEqual(len(snapshot["corners"]), 4)
        self.assertEqual(len(snapshot["keypoints"]), 4)

    def test_a_calibration_is_never_served_before_its_own_frame(self):
        self.process(100)
        self.assertEqual(self.svc.snapshot(99)["state"], STATE_NO_CALIBRATION)

    def test_the_fresh_window_tolerates_a_slow_model(self):
        """Two scheduled submissions may be in flight at once."""
        self.process(100)
        fresh = self.svc._fresh_frames  # 2 * interval + 5
        self.assertEqual(fresh, 25)
        self.assertEqual(self.svc.snapshot(100 + fresh)["state"], STATE_CALIBRATED)

    def test_beyond_the_fresh_window_the_last_calibration_is_marked_stale(self):
        self.process(100)
        snapshot = self.svc.snapshot(100 + self.svc._fresh_frames + 1)
        self.assertEqual(snapshot["state"], STATE_TEMPORARY_LOSS)
        self.assertFalse(snapshot["detected"])
        self.assertTrue(snapshot["calibrated"])
        self.assertTrue(snapshot["using_previous_calibration"])
        # The boundary is still available to draw - as stale, not as current.
        self.assertEqual(len(snapshot["corners"]), 4)
        self.assertTrue(snapshot["homography_available"])

    def test_past_max_lost_the_calibration_is_expired(self):
        self.process(100)
        snapshot = self.svc.snapshot(100 + self.svc.cfg.max_lost_frames + 1)
        self.assertEqual(snapshot["state"], STATE_EXPIRED)
        self.assertEqual(set(snapshot), EMPTY_BLOCK_KEYS)
        self.assertFalse(snapshot["calibrated"])
        self.assertFalse(snapshot["homography_available"])
        self.assertIsNone(snapshot["image_size"])

    def test_a_new_detection_restores_the_calibrated_state(self):
        self.process(100)
        self.process(200)
        self.assertEqual(self.svc.snapshot(200)["state"], STATE_CALIBRATED)
        self.assertEqual(self.svc.snapshot(200)["frame_number"], 200)

    def test_the_bookkeeping_counts_what_happened(self):
        self.process(100)
        summary = self.svc.summary()
        self.assertEqual(summary["detections"], 1)
        self.assertEqual(summary["calibration_updates"], 1)
        self.assertEqual(summary["last_detection_frame"], 100)
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(summary["refusals"], 0)


# ── The delivery window ──────────────────────────────────────────────────────
class TestDeliveryWindow(unittest.TestCase):
    """`snapshot_for_delivery` / `calibration_for_delivery`.

    A delivery is one continuous shot, so a quad measured anywhere inside it
    describes that footage. Reading the live edge at the delivery's LAST frame
    is the wrong question: it used to mark a delivery `calibration_expired` for
    a quad that was perfectly good, and a delivery with geometry lost every
    metre measurement it was entitled to.
    """

    def setUp(self):
        self.svc = service(detector=FakeDetector())
        self.process = self.svc._process

    def tearDown(self):
        self.svc.stop()

    def test_a_quad_measured_inside_the_delivery_is_used_even_when_it_is_stale_at_the_end(self):
        self.process(100, None)
        # The delivery runs long after the fresh window has passed...
        self.assertEqual(self.svc.snapshot(500)["state"], STATE_EXPIRED)
        # ...but the footage it describes was measured during it.
        cal = self.svc.calibration_for_delivery(90, 500)
        self.assertIsNotNone(cal)
        self.assertEqual(cal.frame_number, 100)
        snapshot = self.svc.snapshot_for_delivery(90, 500)
        # Honest about freshness (stale, labelled as previous) rather than
        # throwing away a measurement that belongs to this delivery.
        self.assertEqual(snapshot["state"], STATE_TEMPORARY_LOSS)
        self.assertTrue(snapshot["calibrated"])
        self.assertTrue(snapshot["using_previous_calibration"])
        self.assertTrue(snapshot["homography_available"])

    def test_a_quad_detected_just_before_a_short_delivery_counts(self):
        self.process(100, None)
        cal = self.svc.calibration_for_delivery(105, 120)
        self.assertIsNotNone(cal)
        self.assertEqual(self.svc.snapshot_for_delivery(105, 120)["state"], STATE_CALIBRATED)

    def test_never_an_expired_quad(self):
        """Outside the window and past max_lost_frames describes another view."""
        self.process(100, None)
        self.assertIsNone(self.svc.calibration_for_delivery(1000, 1100))
        self.assertEqual(self.svc.snapshot_for_delivery(1000, 1100)["state"], STATE_EXPIRED)

    def test_the_most_recent_quad_inside_the_window_wins(self):
        self.process(100, None)
        self.process(300, None)
        cal = self.svc.calibration_for_delivery(150, 400)
        self.assertEqual(cal.frame_number, 300)

    def test_a_delivery_with_no_detection_at_all_answers_the_same_as_a_snapshot(self):
        snapshot = self.svc.snapshot_for_delivery(10, 50)
        self.assertEqual(set(snapshot), EMPTY_BLOCK_KEYS)
        self.assertEqual(snapshot["state"], STATE_NO_CALIBRATION)


# ── Failure and refusal bookkeeping ──────────────────────────────────────────
class TestBookkeeping(unittest.TestCase):
    def test_a_detection_error_is_counted_not_raised(self):
        svc = service(detector=FakeDetector(error=PitchDetectError("model said no")))
        svc._process(1, None)
        summary = svc.summary()
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["detections"], 0)
        self.assertEqual(summary["last_error"], "model said no")
        self.assertEqual(svc.snapshot(1)["state"], STATE_NO_CALIBRATION)

    def test_a_missing_credential_stops_further_submissions(self):
        svc = service(
            detector=FakeDetector(
                error=PitchDetectorUnavailable("no ROBOFLOW_API_KEY configured")
            )
        )
        self.assertTrue(svc.enabled)
        svc._process(1, None)
        summary = svc.summary()
        self.assertEqual(summary["failures"], 1)
        self.assertIn("ROBOFLOW_API_KEY", summary["detector_error"])
        self.assertFalse(svc.enabled)
        svc.on_frame(0, None)
        self.assertEqual(svc.summary()["submitted"], 0)

    def test_an_invalid_detection_is_a_refusal_with_a_reason(self):
        thin = PitchDetection(
            keypoints=[Keypoint("p1", 10.0, 10.0), Keypoint("p2", 100.0, 10.0)],
            frame_number=1,
            image_size=SIZE,
            confidence=0.9,
        )
        svc = service(detector=FakeDetector(result=thin))
        svc._process(1, None)
        summary = svc.summary()
        self.assertEqual(summary["refusals"], 1)
        self.assertEqual(summary["detections"], 0)
        self.assertEqual(summary["last_refusal"], "only_2_usable_keypoints")

    def test_an_unexpected_error_is_scrubbed(self):
        svc = service(cfg=PitchConfig(api_key=KEY))
        svc._detector = FakeDetector(error=RuntimeError(f"api_key={KEY}&model=x"))
        svc._process(1, None)
        summary = svc.summary()
        self.assertEqual(summary["failures"], 1)
        self.assertNotIn(KEY, summary["last_error"])
        self.assertIn("api_key=***", summary["last_error"])

    def test_a_refusal_never_leaves_a_calibration_behind(self):
        thin = PitchDetection(
            keypoints=[Keypoint("p1", 10.0, 10.0)],
            frame_number=5,
            image_size=SIZE,
        )
        svc = service(detector=FakeDetector(result=thin))
        svc._process(5, None)
        self.assertEqual(svc.snapshot(5)["state"], STATE_NO_CALIBRATION)


# ── The loop's promise: never block ──────────────────────────────────────────
class TestNonBlocking(unittest.TestCase):
    def test_on_frame_returns_while_the_model_is_still_busy(self):
        slow = FakeDetector(delay=0.3)
        svc = service(
            dataclasses.replace(PitchConfig(api_key=""), interval_frames=1),
            detector=slow,
        )
        started = time.perf_counter()
        try:
            for frame in range(8):
                svc.on_frame(frame, None)
            elapsed = time.perf_counter() - started
            self.assertLess(
                elapsed,
                0.25,
                "on_frame waited for a detector that was already busy",
            )
            # Latest wins: the loop outran the model, so frames were dropped
            # rather than queued without bound.
            self.assertGreaterEqual(svc.summary()["queue_drops"], 1)
        finally:
            svc.stop(timeout=3.0)

    def test_stop_drains_the_queued_work(self):
        svc = service(
            dataclasses.replace(PitchConfig(api_key=""), interval_frames=1),
            detector=FakeDetector(),
        )
        svc.on_frame(0, None)
        svc.on_frame(1, None)
        svc.stop(timeout=3.0)
        self.assertFalse(svc._thread.is_alive())
        self.assertGreaterEqual(svc.summary()["detections"], 1)

    def test_stop_without_a_worker_is_harmless(self):
        svc = service()
        svc.stop()
        self.assertIsNone(svc._thread)

    def test_on_frame_never_raises(self):
        svc = service(
            dataclasses.replace(PitchConfig(api_key=""), interval_frames=1),
            detector=FakeDetector(error=RuntimeError("boom")),
        )
        try:
            for frame in range(3):
                svc.on_frame(frame, None)  # must not raise
            svc.stop(timeout=3.0)
            self.assertGreaterEqual(svc.summary()["failures"], 1)
        finally:
            svc.stop(timeout=3.0)


# ── The service agrees with the validator ────────────────────────────────────
class TestAgreementWithTheValidator(unittest.TestCase):
    def test_a_snapshot_is_the_calibration_the_validator_produced(self):
        svc = service(detector=FakeDetector())
        svc._process(42, None)
        snapshot = svc.snapshot(42)
        cal, reason = build_calibration(
            quad_detection(42), cfg=PitchConfig(api_key=""), image_size=SIZE
        )
        self.assertEqual(reason, "")
        self.assertEqual(snapshot["corners"], {
            k: [round(v[0], 2), round(v[1], 2)] for k, v in cal.corners.items()
        })
        self.assertEqual(snapshot["confidence"], cal.confidence)
        self.assertEqual(snapshot["center"], [
            round(cal.center[0], 2), round(cal.center[1], 2)
        ])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
