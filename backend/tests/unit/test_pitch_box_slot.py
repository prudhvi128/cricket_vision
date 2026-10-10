"""
test_pitch_box_slot.py - the queue the tracking loop offers frames into.

`PitchBoxSlot` is the only part of the optional local pitch-box feature that
runs inside the tracking loop, so its contract matters more than its output:

  * `offer()` must NEVER block - one modulo, one put, done;
  * only the LATEST offered frame is ever kept (latest-wins), and an offered
    frame the worker did not reach is counted as `evicted`, never hidden;
  * `latest_before(frame)` must refuse to answer about a frame AFTER the one
    asked for, because a delivery finalised late must not be handed an
    estimate measured outside its own window;
  * `stop()` must terminate even if the worker is mid-inference;
  * every frame offered is accounted for in `summary()`.

The detector is a two-line stub: these tests are about the plumbing, not about
YOLO or about geometry (that is `test_pitch_box_corners.py`).
"""

from __future__ import annotations

import time
import unittest

from app.pitch.local_box_detector import PitchBox, PitchBoxSlot


class StubDetector:
    """Answers with a fixed box; counts calls the way the real one does."""

    def __init__(self, box=PitchBox(100, 50, 300, 400, 0.9), fail=False) -> None:
        self.box = box
        self.fail = fail
        self.calls = 0

    def detect(self, frame):  # noqa: ANN001 - frame type is irrelevant here
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return self.box

    def stats(self) -> dict:
        return {
            "model_path": "stub",
            "conf": 0.25,
            "imgsz": 640,
            "frames_seen": self.calls,
            "boxes": self.calls,
            "no_box": 0,
            "errors": 0,
            "last_error": None,
        }


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.01):
    """Poll `predicate` until it holds or `timeout` expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def latest_at_or_after(slot: PitchBoxSlot, frame: int):
    """The current estimate if it was measured at or past `frame`, else None."""
    estimate = slot.latest_before(10**9)
    if estimate is not None and estimate["frame_index"] >= frame:
        return estimate
    return None


class TestPitchBoxSlot(unittest.TestCase):

    def _slot(self, detector=None, **kw) -> PitchBoxSlot:
        self.addCleanup(lambda: slot.stop(timeout=2.0))
        slot = PitchBoxSlot(detector or StubDetector(), **kw)
        return slot

    def test_offered_frames_reach_the_detector_and_land(self) -> None:
        slot = self._slot(StubDetector(), every_n=1)
        for n in range(1, 6):
            slot.offer(n, None)
        estimate = wait_for(lambda: slot.latest_before(10**9))
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate["corners_source"], "box_estimate")
        self.assertEqual(estimate["corners_valid"], True)
        self.assertEqual(estimate["corners_reason"], "ok")
        # The box, converted by the module under test's own default inset.
        self.assertEqual(estimate["box"], [100.0, 50.0, 300.0, 400.0])
        self.assertEqual(estimate["box_conf"], 0.9)
        self.assertEqual(set(estimate["corners"]), {"P1", "P2", "P3", "P4"})

    def test_every_n_gates_which_frames_are_offered(self) -> None:
        detector = StubDetector()
        slot = self._slot(detector, every_n=5)
        for n in range(1, 21):
            slot.offer(n, None)
        self.assertEqual(slot.offered, 4)          # 5, 10, 15, 20
        slot.stop()                                # drain before counting
        # None of the other 16 frames were ever offered, and every one that
        # was is accounted for exactly once.
        self.assertEqual(slot.detector.stats()["frames_seen"] + slot.evicted, 4)

    def test_latest_before_refuses_a_frame_from_the_future(self) -> None:
        slot = self._slot(StubDetector(), every_n=1)
        # Offered one at a time and waited for, so all ten frames are in the
        # history - otherwise the boundary below would be testing eviction.
        for n in range(1, 11):
            slot.offer(n, None)
            self.assertIsNotNone(wait_for(lambda n=n: latest_at_or_after(slot, n)))
        latest = slot.latest_before(10**9)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["frame_index"], 10)
        # A window that ended earlier still gets an answer - just never an
        # answer measured after it. This is what stops a delivery emitted late
        # from borrowing an estimate from its own future.
        earlier = slot.latest_before(9)
        self.assertIsNotNone(earlier)
        self.assertLessEqual(earlier["frame_index"], 9)
        # Nothing at all was measured that early.
        self.assertIsNone(slot.latest_before(0))

    def test_an_estimate_stays_valid_until_superseded(self) -> None:
        slot = self._slot(StubDetector(), every_n=1)
        slot.offer(1, None)
        first = wait_for(lambda: latest_at_or_after(slot, 1))
        self.assertIsNotNone(first)
        self.assertEqual(first["frame_index"], 1)
        slot.offer(2, None)
        second = wait_for(lambda: latest_at_or_after(slot, 2))
        self.assertIsNotNone(second)
        self.assertEqual(second["frame_index"], 2)
        # The answer is a copy all the way down: rewriting what we were handed
        # does not corrupt the slot's own state.
        first["corners"]["P1"][0] = -1.0
        first["box"][0] = -1.0
        third = slot.latest_before(10**9)
        self.assertNotEqual(third["corners"]["P1"][0], -1.0)
        self.assertNotEqual(third["box"][0], -1.0)

    def test_offer_never_blocks_and_counts_what_it_drops(self) -> None:
        # A worker that is busy the whole time: every offer past the first is
        # dropped rather than queued, and the loop never waits for it.
        class Slow(StubDetector):
            def detect(self, frame):  # noqa: ANN001
                time.sleep(0.05)
                return super().detect(frame)

        slot = self._slot(Slow(), every_n=1)
        started = time.monotonic()
        for n in range(1, 41):
            slot.offer(n, None)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.5, "offer() must not wait for inference")
        self.assertEqual(slot.offered, 40)
        self.assertGreater(slot.evicted, 0)
        slot.stop()
        # Every offered frame is accounted for exactly once: either the worker
        # ran it or the slot evicted it to stay current.
        self.assertEqual(
            slot.detector.stats()["frames_seen"] + slot.evicted,
            slot.offered,
        )

    def test_summary_reports_every_offer(self) -> None:
        slot = self._slot(StubDetector(), every_n=1, inset=0.1)
        # Offered one at a time and waited for, so a quiescent producer must
        # never have to evict anything.
        for n in range(1, 6):
            slot.offer(n, None)
            self.assertIsNotNone(wait_for(lambda n=n: latest_at_or_after(slot, n)))
        slot.stop()
        summary = slot.summary()
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["every_n"], 1)
        self.assertEqual(summary["inset"], 0.1)
        self.assertEqual(summary["offered"], 5)
        self.assertEqual(summary["evicted"], 0)
        self.assertEqual(summary["frames_seen"], 5)

    def test_a_detector_that_raises_does_not_kill_the_worker(self) -> None:
        slot = self._slot(StubDetector(fail=True), every_n=1)
        slot.offer(1, None)
        slot.offer(2, None)
        # No estimate appears, but the slot keeps working and reports honestly.
        time.sleep(0.3)
        self.assertIsNone(slot.latest_before(10**9))
        self.assertEqual(slot.offered, 2)
        slot.offer(3, None)
        self.assertEqual(slot.offered, 3)

    def test_stop_terminates_a_busy_worker(self) -> None:
        class Slow(StubDetector):
            def detect(self, frame):  # noqa: ANN001
                time.sleep(0.2)
                return super().detect(frame)

        slot = PitchBoxSlot(Slow(), every_n=1)
        slot.offer(1, None)
        slot.offer(2, None)
        started = time.monotonic()
        slot.stop(timeout=5.0)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertFalse(slot._thread.is_alive())

    def test_offering_after_stop_is_ignored(self) -> None:
        slot = PitchBoxSlot(StubDetector(), every_n=1)
        slot.stop(timeout=5.0)
        slot.offer(1, None)
        self.assertEqual(slot.offered, 0)


if __name__ == "__main__":
    unittest.main()
