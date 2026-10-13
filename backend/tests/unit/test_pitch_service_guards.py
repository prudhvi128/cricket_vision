"""
test_pitch_service_guards.py — the estimate guards inside PitchService.

A pitch quad is an ESTIMATE, and estimates decay. Three situations must not
hand a delivery a quad the evidence no longer supports:

  * SCENE CHANGE: a quad measured before a camera cut describes the PREVIOUS
    shot. `_state_at` and `calibration_for_delivery` must refuse it — the
    next delivery gets `no_calibration` until the model measures the new
    view, not a boundary from a different camera angle. The EMA smoother is
    reset here too (`_note_scene_change`), so the first post-cut detection
    seeds fresh rather than averaging against pre-cut coordinates.
  * CONFIDENCE REGRESSION: a fresh 0.91 quad must not be replaced by a 0.55
    quad just because the latter arrived later. Only while the current one
    is fresh: a stale quad is not worth defending.
  * CENTROID DISPLACEMENT: a detection whose keypoint centroid jumped past
    `smoothing_max_centroid_shift` of the frame diagonal describes a moved
    camera the cut detector did not flag. It REPLACES the smoothed state
    instead of blending into it — averaging two camera positions produces a
    ghost that is neither, and always smaller.

All three are driven through `_process` directly: no worker, no timing, no
races.
"""

from __future__ import annotations

import unittest

from app.core.config import PitchConfig
from app.core.constants import (
    PITCH_MOTION_INVALIDATE_MULTIPLIER,
    PITCH_SEG_CONFIDENCE_TOLERANCE,
    SCENE_CUT_DIFF_THRESHOLD,
)
from app.pitch.calibration import build_calibration
from app.pitch.detector import Keypoint, PitchDetection
from app.pitch.service import (
    STATE_CALIBRATED,
    STATE_NO_CALIBRATION,
    PitchService,
)

W, H = 320, 180
SIZE = (W, H)


def quad_detection(frame_number: int, confidence: float = 0.9,
                   image_size=SIZE, box=(0.1, 0.1, 0.9, 0.9)) -> PitchDetection:
    """A detection the validator accepts: a rectangle inside the given box."""
    width, height = image_size
    x0, y0, x1, y1 = box
    coords = [
        (x0 * width, y0 * height),
        (x1 * width, y0 * height),
        (x1 * width, y1 * height),
        (x0 * width, y1 * height),
    ]
    return PitchDetection(
        keypoints=[
            Keypoint(f"p{i}", x, y, confidence) for i, (x, y) in enumerate(coords, 1)
        ],
        frame_number=frame_number,
        image_size=image_size,
        confidence=confidence,
    )


def shifted_quad_detection(frame_number: int, dx: float, dy: float = 0.0,
                           confidence: float = 0.9, image_size=SIZE,
                           box=(0.1, 0.1, 0.9, 0.9)) -> PitchDetection:
    """The same rectangle translated by (dx, dy) — a camera pan, not a new quad."""
    base = quad_detection(frame_number, confidence, image_size, box)
    return PitchDetection(
        keypoints=[
            Keypoint(k.name, k.x + dx, k.y + dy, k.confidence)
            for k in base.keypoints
        ],
        frame_number=base.frame_number,
        image_size=image_size,
        confidence=confidence,
    )


class FakeDetector:
    """Answers with a quad at a caller-chosen confidence."""

    disabled_reason = None

    def __init__(self, confidence: float = 0.9) -> None:
        self.confidence = confidence
        self.calls = 0

    def detect(self, frame, frame_number: int, image_size) -> PitchDetection:
        self.calls += 1
        return quad_detection(frame_number, self.confidence, image_size)


# The panning tests use a smaller quad (20%..60% of the frame) so a 40-px pan
# per call stays inside the frame — an off-frame keypoint would be dropped by
# the bounds check and the detection refused for an unrelated reason.
PAN_BOX = (0.2, 0.2, 0.6, 0.6)


class PanningDetector:
    """Answers with the rectangle translated further each call — a camera pan."""

    disabled_reason = None

    def __init__(self, step_px: float, confidence: float = 0.9) -> None:
        self.step_px = step_px
        self.confidence = confidence
        self.calls = 0

    def detect(self, frame, frame_number: int, image_size) -> PitchDetection:
        self.calls += 1
        return shifted_quad_detection(
            frame_number, self.calls * self.step_px, confidence=self.confidence,
            image_size=image_size, box=PAN_BOX,
        )


def service(detector, **cfg_overrides) -> PitchService:
    # smoothing_alpha=1.0 by default: the confidence guards are about WHICH
    # estimate is kept, not about the EMA blending two of them — those tests
    # must see raw confidences. The smoother tests override alpha explicitly.
    defaults = dict(api_key="", smoothing_alpha=1.0)
    defaults.update(cfg_overrides)
    return PitchService(
        PitchConfig(**defaults),
        width=W, height=H, detector=detector,
    )


class TestSceneCutInvalidation(unittest.TestCase):
    def test_a_calibration_survives_until_a_cut_is_noted(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        self.assertEqual(svc.snapshot(12)["state"], STATE_CALIBRATED)
        cal = svc.calibration_for_delivery(8, 20)
        self.assertIsNotNone(cal)

    def test_a_cut_invalidates_quads_measured_before_it(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        # The view changes at frame 50. `_note_scene_change` directly: the
        # ordering-sensitive tests below must not race the worker thread that
        # `on_frame` also submits the frame to (the routing test further down
        # covers the on_frame -> note path).
        svc._note_scene_change(50)
        # The quad from frame 10 described the previous shot: refused.
        self.assertEqual(svc.snapshot(52)["state"], STATE_NO_CALIBRATION)
        self.assertIsNone(svc.calibration_for_delivery(40, 60))

    def test_a_quad_measured_on_the_cut_frame_describes_the_new_view(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        svc._note_scene_change(50)
        svc._process(50, None)
        self.assertEqual(svc.snapshot(50)["state"], STATE_CALIBRATED)
        cal = svc.calibration_for_delivery(40, 60)
        self.assertIsNotNone(cal)
        self.assertEqual(cal.frame_number, 50)

    def test_strong_camera_motion_alone_is_treated_as_a_cut(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        threshold = PITCH_MOTION_INVALIDATE_MULTIPLIER * SCENE_CUT_DIFF_THRESHOLD
        # Proves the ROUTING only: motion at/above the threshold becomes a
        # scene change. (on_frame also submits the frame to the worker, so the
        # invalidation consequence is asserted via _note_scene_change above,
        # not against a state the worker could overwrite.)
        svc.on_frame(20, None, motion=threshold + 1.0)
        with svc._lock:
            self.assertIn(20, svc._scene_cuts)

    def test_ordinary_motion_does_not_invalidate(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        threshold = PITCH_MOTION_INVALIDATE_MULTIPLIER * SCENE_CUT_DIFF_THRESHOLD
        svc.on_frame(20, None, motion=threshold - 1.0)
        with svc._lock:
            self.assertEqual(svc._scene_cuts, [])
        # Still the same view: the frame-10 quad is fresh here (the service's
        # default interval is 10, so fresh_frames = 25).
        self.assertEqual(svc.snapshot(20)["state"], STATE_CALIBRATED)

    def test_a_delivery_spanning_a_cut_refuses_the_pre_cut_quad(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        svc._note_scene_change(30)
        svc._process(80, None)   # a quad for the NEW view exists too
        cal = svc.calibration_for_delivery(5, 85)
        self.assertIsNotNone(cal)
        self.assertEqual(cal.frame_number, 80)

    def test_a_delivery_entirely_before_the_cut_keeps_its_quad(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        svc._note_scene_change(90)
        cal = svc.calibration_for_delivery(5, 20)
        self.assertIsNotNone(cal)
        self.assertEqual(cal.frame_number, 10)

    def test_the_same_cut_frame_is_noted_once(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc.on_frame(50, None, scene_cut=True)
        svc.on_frame(50, None, motion=100.0)
        with svc._lock:
            self.assertEqual(svc._scene_cuts, [50])
            self.assertEqual(svc._scene_changes, 1)

    def test_the_scene_change_count_reaches_the_diagnostics(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc.on_frame(50, None, scene_cut=True)
        svc.on_frame(90, None, scene_cut=True)
        diagnostics = svc.summary()["diagnostics"]
        self.assertEqual(diagnostics["scene_changes"], 2)
        self.assertEqual(diagnostics["last_scene_cut_frame"], 90)


class TestConfidenceRegressionGuard(unittest.TestCase):
    def test_a_fresh_high_confidence_estimate_is_kept(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        # A materially weaker answer arrives while the 0.9 quad is fresh.
        det.confidence = 0.5
        svc._process(15, None)
        # The refusal is recorded, and the 0.9 quad is still the one served.
        self.assertEqual(svc.summary()["last_refusal"],
                         "lower_confidence_than_current_estimate")
        snapshot = svc.snapshot(15)
        self.assertEqual(snapshot["state"], STATE_CALIBRATED)
        self.assertAlmostEqual(snapshot["confidence"], 0.9, places=4)
        self.assertEqual(svc.summary()["detections"], 1)
        self.assertEqual(svc.summary()["refusals"], 1)

    def test_a_weak_answer_inside_the_tolerance_is_accepted(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        det.confidence = 0.9 - PITCH_SEG_CONFIDENCE_TOLERANCE
        svc._process(15, None)
        self.assertEqual(svc.summary()["detections"], 2)
        self.assertEqual(svc.summary()["refusals"], 0)

    def test_a_stronger_answer_replaces_the_current_one(self):
        det = FakeDetector(0.7)
        svc = service(det)
        svc._process(10, None)
        det.confidence = 0.95
        svc._process(15, None)
        self.assertEqual(svc.summary()["detections"], 2)
        self.assertAlmostEqual(svc.snapshot(15)["confidence"], 0.95, places=4)

    def test_a_weak_answer_after_the_current_one_went_stale_is_accepted(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        # Age the 0.9 quad past `fresh_frames` but within `max_lost_frames`:
        # it is temporary_loss, not calibrated, and not worth defending.
        stale_frame = 10 + svc._fresh_frames + 1
        det.confidence = 0.5
        svc._process(stale_frame, None)
        self.assertEqual(svc.summary()["detections"], 2)
        self.assertEqual(svc.summary()["refusals"], 0)

    def test_the_refusal_names_the_reason_in_the_diagnostics(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        # 0.7 clears the keypoint confidence floor (0.5) so the detection is
        # VALID — and is still refused by the guard, which is the point.
        det.confidence = 0.7
        svc._process(15, None)
        diagnostics = svc.summary()["diagnostics"]
        self.assertEqual(
            diagnostics["refusals"], {"lower_confidence_than_current_estimate": 1}
        )
        self.assertEqual(diagnostics["categories"]["refused"], 1)


class TestSourceTravelsIntoTheCalibration(unittest.TestCase):
    def test_a_segmentation_detection_is_labelled_in_its_calibration(self):
        det = FakeDetector(0.9)
        svc = service(det)
        svc._process(10, None)
        cal = svc.calibration_for_delivery(1, 20)
        self.assertIsNotNone(cal)
        self.assertEqual(cal.source, "pitch_keypoints")

    def test_a_segmentation_source_survives_build_calibration(self):
        detection = quad_detection(7, 0.9)
        detection = PitchDetection(
            keypoints=detection.keypoints,
            frame_number=detection.frame_number,
            image_size=detection.image_size,
            confidence=detection.confidence,
            source="pitch_segmentation",
        )
        cal, reason = build_calibration(detection, cfg=PitchConfig(api_key=""),
                                        image_size=SIZE)
        self.assertEqual(reason, "")
        self.assertEqual(cal.source, "pitch_segmentation")


# ── The EMA smoother's own guards ────────────────────────────────────────────
class TestSmootherResetOnSceneChange(unittest.TestCase):
    """`_note_scene_change` clears the EMA state, so the first post-cut
    detection seeds fresh instead of averaging against the previous view."""

    def test_a_scene_change_resets_the_smoother(self):
        det = FakeDetector(0.9)
        svc = service(det, smoothing_alpha=0.4)
        svc._process(10, None)
        # The smoother now holds the frame-10 quad.
        self.assertIsNotNone(svc._smoother.last_centroid)
        svc._note_scene_change(50)
        self.assertIsNone(
            svc._smoother.last_centroid,
            "a scene change must clear the EMA state, not blend across the cut",
        )

    def test_the_first_detection_after_a_cut_is_not_blended_with_the_old_view(self):
        det = FakeDetector(0.9)
        svc = service(det, smoothing_alpha=0.4)
        svc._process(10, None)
        svc._note_scene_change(50)
        # A detection for the NEW view arrives 40% as far from the old
        # centroid as the old quad was wide. If the smoother still held the
        # pre-cut state, `apply` would blend and land strictly between the two.
        svc._process(51, None)
        cal = svc.calibration_for_delivery(51, 51)
        self.assertIsNotNone(cal)
        # With alpha=0.4 the frame-51 corners are the detection's own, unblended.
        expected = quad_detection(51).keypoints
        for name in ("p1", "p2", "p3", "p4"):
            got = cal.keypoints[name]
            want = next(k for k in expected if k.name == name)
            self.assertAlmostEqual(got[0], want.x, places=6, msg=name)
            self.assertAlmostEqual(got[1], want.y, places=6, msg=name)

    def test_a_stable_view_keeps_its_smoothing(self):
        """Nothing moved, nothing was reset: the EMA still averages jitter."""
        det = FakeDetector(0.9)
        svc = service(det, smoothing_alpha=0.4)
        svc._process(10, None)
        # A 4-px jitter — far below the displacement gate — applied to the
        # next detection. Driven through `_smooth` directly so the test does
        # not depend on the worker queue.
        jittered = shifted_quad_detection(20, dx=4.0)
        smoothed = svc._smooth(jittered)
        raw = {k.name: (k.x, k.y) for k in jittered.keypoints}
        for k in smoothed.keypoints:
            self.assertLess(k.x, raw[k.name][0], "jitter must be pulled back")
            self.assertGreater(k.x, raw[k.name][0] - 4.0, "but not frozen")
        self.assertEqual(svc.summary()["diagnostics"]["smoother_resets"], 0)

    def test_the_reset_count_reaches_the_diagnostics(self):
        det = FakeDetector(0.9)
        svc = service(det, smoothing_alpha=0.4)
        svc._process(10, None)
        svc._note_scene_change(50)
        svc._note_scene_change(90)
        self.assertEqual(svc.summary()["diagnostics"]["smoother_resets"], 2)


class TestCentroidDisplacementGate(unittest.TestCase):
    """A detection whose centroid jumped past the gate REPLACES the smoothed
    state; the gate is configurable and can be disabled."""

    def test_a_large_jump_replaces_rather_than_blends(self):
        det = PanningDetector(step_px=40.0)  # 40 px/call on a 320x180 frame
        svc = service(det, smoothing_alpha=0.4)
        svc._process(10, None)   # quad at PAN_BOX, call 1 (+40 px)
        svc._process(20, None)   # shifted another +40 px — past the gate
        cal = svc.calibration_for_delivery(20, 20)
        self.assertIsNotNone(cal)
        # The stored quad must be the RAW frame-20 detection, not a blend
        # between the frame-10 quad and it (which would sit ~24 px short).
        want = shifted_quad_detection(20, dx=80.0, box=PAN_BOX)
        for name in ("p1", "p2", "p3", "p4"):
            got = cal.keypoints[name]
            wk = next(k for k in want.keypoints if k.name == name)
            self.assertAlmostEqual(got[0], wk.x, places=6, msg=name)
            self.assertAlmostEqual(got[1], wk.y, places=6, msg=name)

    def test_a_small_jitter_still_blends(self):
        det = PanningDetector(step_px=2.0)  # 2 px/call — below the gate
        svc = service(det, smoothing_alpha=0.4)
        svc._process(10, None)
        svc._process(20, None)
        cal = svc.calibration_for_delivery(20, 20)
        self.assertIsNotNone(cal)
        raw = shifted_quad_detection(20, dx=4.0, box=PAN_BOX)
        wk = next(k for k in raw.keypoints if k.name == "p1")
        seed = quad_detection(10, box=PAN_BOX).keypoints[0]
        got = cal.keypoints["p1"]
        # Blended: strictly between the frame-10 position and the raw frame-20.
        self.assertLess(got[0], wk.x)
        self.assertGreater(got[0], seed.x)
        self.assertEqual(svc.summary()["diagnostics"]["smoother_resets"], 0)

    def test_the_gate_threshold_is_configurable(self):
        det = PanningDetector(step_px=40.0)
        # 0.5 of the diagonal (~184 px on 320x180): the 40-px jump is now
        # ordinary jitter as far as this service is concerned.
        svc = service(det, smoothing_alpha=0.4, smoothing_max_centroid_shift=0.5)
        svc._process(10, None)
        svc._process(20, None)
        cal = svc.calibration_for_delivery(20, 20)
        self.assertIsNotNone(cal)
        raw = shifted_quad_detection(20, dx=80.0, box=PAN_BOX)
        wk = next(k for k in raw.keypoints if k.name == "p1")
        self.assertLess(
            cal.keypoints["p1"][0], wk.x,
            "a wider gate must let the jump blend like ordinary jitter",
        )

    def test_a_disabled_gate_never_replaces(self):
        det = PanningDetector(step_px=40.0)
        svc = service(det, smoothing_alpha=0.4, smoothing_max_centroid_shift=0.0)
        svc._process(10, None)
        svc._process(20, None)
        self.assertEqual(svc.summary()["diagnostics"]["smoother_resets"], 0)

    def test_the_first_detection_is_never_gated(self):
        det = PanningDetector(step_px=40.0)
        svc = service(det, smoothing_alpha=0.4)
        svc._process(10, None)
        # Nothing held yet: the first detection seeds, whatever its position.
        self.assertEqual(svc.summary()["detections"], 1)
        self.assertEqual(svc.summary()["diagnostics"]["smoother_resets"], 0)

    def test_the_gate_counts_resets_in_the_diagnostics(self):
        det = PanningDetector(step_px=40.0)
        svc = service(det, smoothing_alpha=0.4)
        svc._process(10, None)
        svc._process(20, None)
        svc._process(30, None)
        # Two pans after the seed: two resets.
        self.assertEqual(svc.summary()["diagnostics"]["smoother_resets"], 2)


if __name__ == "__main__":
    unittest.main()
