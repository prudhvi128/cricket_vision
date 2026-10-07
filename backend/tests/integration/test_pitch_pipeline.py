"""
test_pitch_pipeline.py - the optional pitch detector inside a REAL analysis.

The unit tests prove each piece works on its own. This proves they fit
together in the pass that actually writes `result.json`:

  * the ball tracking is untouched - the same video with and without the
    pitch detector produces identical deliveries (frames, trajectory, sources,
    confidences), because `on_frame` adds no decode and no wait;
  * the analysis-level record says whether the feature ran, how often it
    answered, and never contains the credential;
  * every delivery carries a pitch block with the SAME keys whether or not a
    calibration existed, and - when the detector answered - a real quad plus
    the homography;
  * the frontend projection exposes `pitch_position` on every trajectory point
    and keeps the matrix itself server-side.

The pipeline is driven end to end (stub ball detector, synthetic video, fake
pitch detector), so this is the served record rather than a hand-built
fixture.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from app.core.config import Paths, PipelineConfig, PitchConfig
from app.pitch.detector import Keypoint, PitchDetection
from app.pitch.service import PitchService
from app.schemas.frontend import serialize_delivery
from app.tracking.tracking_service import TrackingService
from tests.helpers import StubDetector, make_video

W, H, FPS = 320, 180, 25.0
WINDOW = (30, 60)
CREDENTIAL = "TEST_KEY_THAT_MUST_NEVER_PERSIST"

# The block a client receives - never the matrix, the model id, the source
# dimensions or the frame it came from. `orientation` IS served: it is what
# tells a pitch map which end of the quad the batter is at, without which a
# bounce could be drawn at the wrong end of the pitch.
FRONTEND_PITCH_KEYS = {
    "state",
    "detected",
    "calibrated",
    "confidence",
    "using_previous_calibration",
    "keypoints",
    "corners",
    "center",
    "homography_available",
    "orientation",
}
INTERNAL_PITCH_KEYS = {"homography", "model_id", "image_size", "frame_number"}

# Every key the internal pitch block has, whatever happened. `map_homography`
# (quad -> metres, then normalised with the batting end at y=1) is the matrix
# the analytics used and is always present - null when no quad could be
# oriented - so every delivery's block has the same shape.
INTERNAL_BLOCK_KEYS = FRONTEND_PITCH_KEYS | INTERNAL_PITCH_KEYS | {"map_homography"}


class FakePitchDetector:
    """Always answers with a rectangle covering 10%..90% of the frame."""

    disabled_reason = None

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame, frame_number: int, image_size) -> PitchDetection:
        self.calls += 1
        width, height = image_size
        coords = [
            (0.1 * width, 0.1 * height),
            (0.9 * width, 0.1 * height),
            (0.9 * width, 0.9 * height),
            (0.1 * width, 0.9 * height),
        ]
        return PitchDetection(
            keypoints=[
                Keypoint(f"p{i}", x, y, 0.93) for i, (x, y) in enumerate(coords, 1)
            ],
            frame_number=frame_number,
            image_size=image_size,
            confidence=0.93,
        )


class TestPitchPipeline(unittest.TestCase):
    """Both runs are computed once: the comparison is the point of the test."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        cls.video = make_video(
            cls.dir / "source.mp4", n_frames=200, width=W, height=H, fps=FPS
        )
        cls.pitch_service = PitchService(
            PitchConfig(api_key=CREDENTIAL),
            width=W,
            height=H,
            detector=FakePitchDetector(),
        )
        cls.on = cls._run("test_pitch_pipeline_on", cls.pitch_service)
        cls.off = cls._run("test_pitch_pipeline_off", None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def _run(cls, analysis_id: str, pitch_service=None):
        paths = Paths(analysis_id=analysis_id).ensure()
        service = TrackingService(PipelineConfig(), pitch_service=pitch_service)
        return service.analyse(
            str(cls.video), paths, detector=StubDetector([WINDOW])
        )

    # ── ball tracking is untouched ───────────────────────────────────────────
    def test_the_pitch_detector_changes_nothing_about_the_deliveries(self):
        self.assertGreaterEqual(len(self.on.deliveries), 1)
        self.assertEqual(len(self.on.deliveries), len(self.off.deliveries))
        for active, idle in zip(self.on.deliveries, self.off.deliveries):
            self.assertEqual(
                active.detection_start_frame, idle.detection_start_frame
            )
            self.assertEqual(active.detection_end_frame, idle.detection_end_frame)
            self.assertEqual(active.release_frame, idle.release_frame)
            self.assertEqual(
                len(active.trajectory.points), len(idle.trajectory.points)
            )
            self.assertEqual(
                [(p.original_frame, p.x, p.y, p.source, p.confidence)
                 for p in active.trajectory.points],
                [(p.original_frame, p.x, p.y, p.source, p.confidence)
                 for p in idle.trajectory.points],
            )

    def test_the_single_decode_guarantee_still_holds_with_the_feature_on(self):
        self.assertEqual(self.on.frames_decoded, self.off.frames_decoded)

    def test_the_pitch_detector_really_ran(self):
        summary = self.pitch_service.summary()
        self.assertGreaterEqual(summary["submitted"], 1)
        self.assertGreaterEqual(summary["detections"], 1)
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(summary["refusals"], 0)

    # ── the analysis-level record ────────────────────────────────────────────
    def test_the_record_says_the_feature_ran(self):
        summary = self.on.pitch_summary
        self.assertIsInstance(summary, dict)
        self.assertTrue(summary["enabled"])
        self.assertGreaterEqual(summary["detections"], 1)
        self.assertTrue(summary["config"]["api_key_configured"])

    def test_the_disabled_run_says_so_honestly(self):
        summary = self.off.pitch_summary
        self.assertFalse(summary["enabled"])
        self.assertEqual(summary["submitted"], 0)
        self.assertEqual(summary["detections"], 0)

    def test_no_credential_reaches_any_record(self):
        self.assertNotIn(CREDENTIAL, json.dumps(self.on.pitch_summary))
        self.assertNotIn(
            CREDENTIAL, json.dumps(self.on.tracking_document(PipelineConfig()))
        )

    def test_the_tracking_document_carries_the_pitch_record(self):
        document = self.on.tracking_document(PipelineConfig())
        self.assertIn("pitch_detection", document)
        self.assertTrue(document["pitch_detection"]["enabled"])
        json.dumps(document["pitch_detection"])  # serialisable

    # ── the per-delivery block ───────────────────────────────────────────────
    def test_every_delivery_has_the_same_keys(self):
        for delivery in self.on.deliveries:
            self.assertEqual(set(delivery.pitch), INTERNAL_BLOCK_KEYS)

    def test_a_delivery_is_served_from_the_calibration_in_its_own_frame(self):
        states = set()
        for delivery in self.on.deliveries:
            pitch = delivery.pitch
            states.add(pitch["state"])
            self.assertTrue(pitch["calibrated"], pitch["state"])
            self.assertTrue(pitch["detected"] or pitch["using_previous_calibration"])
            self.assertEqual(len(pitch["corners"]), 4)
            self.assertEqual(len(pitch["keypoints"]), 4)
            self.assertTrue(pitch["homography_available"])
            self.assertIsNotNone(pitch["homography"])
            self.assertEqual(pitch["image_size"], [W, H])
            self.assertIsNotNone(pitch["confidence"])
        self.assertIn("calibrated", states)

    def test_without_the_feature_the_block_is_empty_but_present(self):
        for delivery in self.off.deliveries:
            pitch = delivery.pitch
            self.assertEqual(set(pitch), INTERNAL_BLOCK_KEYS)
            self.assertEqual(pitch["state"], "no_calibration")
            self.assertFalse(pitch["calibrated"])
            self.assertFalse(pitch["homography_available"])
            self.assertEqual(pitch["corners"], {})
            self.assertIsNone(pitch["confidence"])

    # ── what the browser actually gets ───────────────────────────────────────
    def test_the_matrix_stays_server_side(self):
        for delivery in self.on.deliveries:
            dump = serialize_delivery(
                delivery.as_dict(), analysis_id="pitch"
            ).pitch.model_dump()
            self.assertEqual(set(dump), FRONTEND_PITCH_KEYS)
            self.assertTrue(dump["homography_available"])
            for key in INTERNAL_PITCH_KEYS:
                self.assertNotIn(key, dump)

    def test_every_trajectory_point_gets_a_pitch_position(self):
        checked = 0
        for delivery in self.on.deliveries:
            out = serialize_delivery(delivery.as_dict(), analysis_id="pitch")
            for point in out.trajectory.points:
                position = point.pitch_position
                self.assertIsNotNone(position)
                self.assertTrue(math.isfinite(position.x))
                self.assertTrue(math.isfinite(position.y))
                # The quad is the fake detector's 10%..90% rectangle, so
                # `inside_pitch` must agree with the source pixels - a
                # predicted point that ran past the boundary says so.
                inside = (
                    0.1 * W - 1e-6 <= point.x <= 0.9 * W + 1e-6
                    and 0.1 * H - 1e-6 <= point.y <= 0.9 * H + 1e-6
                )
                self.assertEqual(
                    position.inside_pitch, inside, (point.x, point.y, position)
                )
                checked += 1
            self.assertTrue(
                any(
                    point.pitch_position.inside_pitch
                    for point in out.trajectory.points
                )
            )
        self.assertGreater(checked, 0)

    def test_without_a_homography_no_position_is_invented(self):
        for delivery in self.off.deliveries:
            out = serialize_delivery(delivery.as_dict(), analysis_id="pitch")
            self.assertFalse(out.pitch.homography_available)
            for point in out.trajectory.points:
                self.assertIsNone(point.pitch_position)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
