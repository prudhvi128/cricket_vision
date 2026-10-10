"""
test_pitch_box_feature.py - the optional LOCAL pitch-BOX detector inside a
REAL analysis.

`box_corners.py` is proven pure, `PitchBoxSlot` is proven to obey its queue
contract; this proves the two of them FIT into the pass that writes
`result.json` without disturbing anything the pass already did:

  * OFF (the default) adds nothing: no `pitch_box_detection` record, no
    `box_estimate` on any delivery, and the tracking document's top-level
    keys are exactly what this build shipped with before the feature;
  * ON, every delivery carries an estimate built by `box_corners` from the
    detector's box, and the estimate is labelled `box_estimate` so it cannot
    be mistaken for a measurement;
  * the estimate never claims a frame from the future - it is always measured
    at or before the delivery's own window;
  * it cannot calibrate anything: no `pitch.corners`, no homography,
    `geometry_calibrated` stays False, and speed / length / line / swing stay
    null because they must;
  * the browser is unaffected: the serializer exposes the same pitch keys it
    always did, with no `box_estimate` among them;
  * the ball tracking is untouched: same frames, same trajectory, same
    pitch block, with and without the feature;
  * the real weights, when present, load and produce a usable box.

The pipeline is driven end to end with a stub ball detector and a synthetic
video, so this is the served record rather than a hand-built fixture.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.core.config import MODELS_DIR, PITCH_BOX_CONFIG, Paths, PipelineConfig, PitchConfig
from app.pitch.box_corners import box_to_corners, corners_valid
from app.pitch.detector import Keypoint, PitchDetection
from app.pitch.local_box_detector import LocalPitchBoxDetector, PitchBox, PitchBoxSlot
from app.pitch.service import PitchService
from app.schemas.frontend import serialize_delivery
from app.tracking.tracking_service import TrackingService
from tests.helpers import StubDetector, make_video

W, H, FPS = 320, 180, 25.0
WINDOW = (30, 60)

# The box the stub detector always returns: strictly inside a 320x180 frame.
STUB_BOX = (80.0, 40.0, 240.0, 150.0)
STUB_CONF = 0.91

# The keys `tracking_document` had before this feature existed. If one of
# these ever disappears, something other than an added optional key changed.
BASELINE_TRACKING_KEYS = {
    "analysis_id",
    "calibration",
    "clip_writer",
    "config",
    "deliveries",
    "performance",
    "pipeline_version",
    "pitch_detection",
    "ring_buffer",
    "schema_version",
    "segmentation",
    "validation",
    "video",
    "warnings",
}

ESTIMATE_KEYS = {
    "corners",
    "corners_source",
    "corners_valid",
    "corners_reason",
    "box",
    "box_conf",
    "inset",
    "frame_index",
}


class StubBoxDetector:
    """Always answers with `STUB_BOX`, counting every frame it is given."""

    def __init__(self) -> None:
        self.frames_seen = 0

    def detect(self, frame_bgr) -> PitchBox:  # noqa: ANN001 - shape is irrelevant
        self.frames_seen += 1
        return PitchBox(*STUB_BOX, conf=STUB_CONF)

    def stats(self) -> dict:
        return {
            "model_path": "stub",
            "conf": 0.25,
            "imgsz": 640,
            "frames_seen": self.frames_seen,
            "boxes": self.frames_seen,
            "no_box": 0,
            "errors": 0,
            "last_error": None,
        }


class FakePitchDetector:
    """Always answers with a rectangle covering 10%..90% of the frame."""

    disabled_reason = None

    def detect(self, frame, frame_number: int, image_size) -> PitchDetection:  # noqa: ANN001
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


def _run(analysis_id: str, video: Path, *, pitch_service=None, box_detector=None):
    paths = Paths(analysis_id=analysis_id).ensure()
    service = TrackingService(
        PipelineConfig(),
        pitch_service=pitch_service,
        pitch_box_detector=box_detector,
    )
    result = service.analyse(str(video), paths, detector=StubDetector([WINDOW]))
    return paths, result


class TestPitchBoxFeature(unittest.TestCase):
    """OFF and ON of the same video, computed once for every test here."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        cls.video = make_video(
            cls.dir / "source.mp4", n_frames=200, width=W, height=H, fps=FPS
        )
        cls.off_paths, cls.off = _run("test_pitch_box_off", cls.video)
        cls.box_detector = StubBoxDetector()
        cls.on_paths, cls.on = _run(
            "test_pitch_box_on", cls.video, box_detector=cls.box_detector
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ── OFF is inert ──────────────────────────────────────────────────────────
    def test_off_adds_no_record_to_the_result(self):
        self.assertIsNone(self.off.pitch_box_summary)

    def test_off_adds_no_key_to_the_tracking_document(self):
        document = self.off.tracking_document(PipelineConfig())
        self.assertNotIn("pitch_box_detection", document)
        self.assertEqual(set(document), BASELINE_TRACKING_KEYS)
        json.dumps(document)

    def test_off_adds_nothing_to_tracking_json_on_disk(self):
        document = json.loads(self.off_paths.tracking_json.read_text())
        self.assertNotIn("pitch_box_detection", document)
        self.assertEqual(set(document), BASELINE_TRACKING_KEYS)

    def test_off_leaves_every_pitch_block_alone(self):
        self.assertGreaterEqual(len(self.off.deliveries), 1)
        for delivery in self.off.deliveries:
            self.assertNotIn("box_estimate", delivery.pitch)

    def _result_document(self, tracking):
        """The API-facing `result.json` for a tracking pass, as the pipeline
        assembles it. `result_document` reads only what the pass returned, so
        the whole shot + overlay stage is not needed to check its keys."""
        from app.pipeline.analysis_pipeline import PipelineResult

        pipeline = PipelineResult(analysis_id=tracking.analysis_id, tracking=tracking)
        pipeline.deliveries = list(tracking.deliveries)
        return pipeline.result_document(Paths(analysis_id=tracking.analysis_id))

    def test_off_adds_no_record_to_the_result_document(self):
        document = self._result_document(self.off)
        self.assertNotIn("pitch_box_detection", document)
        for delivery in document["deliveries"]:
            self.assertNotIn("box_estimate", delivery["pitch"])
        json.dumps(document)

    # ── ON records its own work ───────────────────────────────────────────────
    def test_the_detector_really_ran(self):
        summary = self.on.pitch_box_summary
        self.assertIsInstance(summary, dict)
        self.assertTrue(summary["enabled"])
        self.assertGreater(summary["frames_seen"], 0)
        self.assertGreater(summary["boxes"], 0)
        self.assertEqual(summary["errors"], 0)
        # Everything offered is accounted for: the worker ran it or the slot
        # evicted it to stay current.
        self.assertEqual(
            summary["frames_seen"] + summary["evicted"], summary["offered"]
        )
        json.dumps(summary)

    def test_the_record_reaches_the_tracking_document(self):
        document = self.on.tracking_document(PipelineConfig())
        self.assertIn("pitch_box_detection", document)
        self.assertEqual(set(document), BASELINE_TRACKING_KEYS | {"pitch_box_detection"})
        # The document was not reshuffled: the pre-feature keys are all still
        # there with the values they had without the feature (the two runs
        # share every pipeline constant).
        off_document = self.off.tracking_document(PipelineConfig())
        for key in ("schema_version", "pipeline_version"):
            self.assertEqual(document[key], off_document[key])

    def test_the_record_reaches_tracking_json_on_disk(self):
        document = json.loads(self.on_paths.tracking_json.read_text())
        self.assertIn("pitch_box_detection", document)
        self.assertTrue(document["pitch_box_detection"]["enabled"])

    def test_the_record_reaches_the_result_document_too(self):
        document = self._result_document(self.on)
        self.assertIn("pitch_box_detection", document)
        self.assertTrue(document["pitch_box_detection"]["enabled"])
        for delivery in document["deliveries"]:
            self.assertIn("box_estimate", delivery["pitch"])
        json.dumps(document)

    def test_the_stub_really_saw_frames(self):
        self.assertGreater(self.box_detector.frames_seen, 0)

    # ── the per-delivery estimate ─────────────────────────────────────────────
    def test_every_delivery_carries_the_estimate(self):
        self.assertGreaterEqual(len(self.on.deliveries), 1)
        for delivery in self.on.deliveries:
            self.assertIn("box_estimate", delivery.pitch)

    def test_an_estimate_has_the_documented_shape(self):
        for delivery in self.on.deliveries:
            estimate = delivery.pitch["box_estimate"]
            self.assertIsNotNone(estimate, delivery.delivery_id)
            self.assertEqual(set(estimate), ESTIMATE_KEYS)
            self.assertEqual(set(estimate["corners"]), {"P1", "P2", "P3", "P4"})
            self.assertEqual(estimate["corners_source"], "box_estimate")
            self.assertTrue(estimate["corners_valid"])
            self.assertEqual(estimate["corners_reason"], "ok")
            self.assertEqual(estimate["box"], list(STUB_BOX))
            self.assertEqual(estimate["box_conf"], STUB_CONF)
            self.assertEqual(estimate["inset"], PITCH_BOX_CONFIG.inset)
            json.dumps(estimate)   # serialisable into result.json

    def test_the_corners_are_the_boxes_corners_in_the_documented_order(self):
        expected = box_to_corners(*STUB_BOX, top_inset=PITCH_BOX_CONFIG.inset)
        valid, reason = corners_valid(expected)
        self.assertTrue(valid, reason)
        for delivery in self.on.deliveries:
            corners = delivery.pitch["box_estimate"]["corners"]
            for name, point in zip(("P1", "P2", "P3", "P4"), expected):
                self.assertAlmostEqual(corners[name][0], round(point[0], 2), places=2)
                self.assertAlmostEqual(corners[name][1], round(point[1], 2), places=2)

    def test_no_estimate_is_taken_from_a_frame_in_the_future(self):
        for delivery in self.on.deliveries:
            estimate = delivery.pitch["box_estimate"]
            self.assertIsNotNone(estimate)
            # Mirror the rule the pipeline uses: a delivery's window ends at
            # its ball-lost frame (or the segment's end when there is none),
            # and an estimate measured after that point is not this delivery's.
            limit = delivery.ball_lost_frame or delivery.event["end_frame"]
            self.assertLessEqual(estimate["frame_index"], limit)

    # ── it cannot calibrate or measure anything ───────────────────────────────
    def test_it_cannot_make_a_delivery_calibrated(self):
        for delivery in self.on.deliveries:
            pitch = delivery.pitch
            self.assertFalse(pitch["calibrated"])
            self.assertFalse(pitch["homography_available"])
            self.assertEqual(pitch["corners"], {})
            self.assertIsNone(pitch["homography"])
            self.assertFalse(delivery.bowling.geometry_calibrated)

    def test_speed_and_line_stay_null_because_they_must(self):
        for delivery in self.on.deliveries:
            bowling = delivery.bowling
            self.assertIsNone(bowling.speed_kmh)
            self.assertIsNone(bowling.length)
            self.assertIsNone(bowling.line)
            self.assertIsNone(bowling.swing)
            self.assertIsNone(bowling.bounce_angle)

    def test_geometry_stays_uncalibrated_in_the_result(self):
        for delivery in self.on.deliveries:
            self.assertFalse(delivery.bowling.geometry_calibrated)
        self.assertFalse(self.on.geometry_calibrated)

    # ── the browser is unaffected ─────────────────────────────────────────────
    def test_the_serializer_exposes_no_estimate(self):
        for delivery in self.on.deliveries:
            body = serialize_delivery(
                delivery.as_dict(), analysis_id="pitch_box"
            ).model_dump()
            self.assertNotIn("box_estimate", body["pitch"])
            json.dumps(body)

    def test_the_off_and_on_served_pitch_keys_match(self):
        off_body = serialize_delivery(
            self.off.deliveries[0].as_dict(), analysis_id="a"
        ).model_dump()
        on_body = serialize_delivery(
            self.on.deliveries[0].as_dict(), analysis_id="a"
        ).model_dump()
        self.assertEqual(set(off_body["pitch"]), set(on_body["pitch"]))

    # ── the ball tracking is untouched ────────────────────────────────────────
    def test_the_two_runs_agree_on_every_delivery(self):
        self.assertEqual(len(self.on.deliveries), len(self.off.deliveries))
        for active, idle in zip(self.on.deliveries, self.off.deliveries):
            self.assertEqual(active.detection_start_frame, idle.detection_start_frame)
            self.assertEqual(active.detection_end_frame, idle.detection_end_frame)
            self.assertEqual(active.release_frame, idle.release_frame)
            self.assertEqual(
                [(p.original_frame, p.x, p.y, p.source, p.confidence)
                 for p in active.trajectory.points],
                [(p.original_frame, p.x, p.y, p.source, p.confidence)
                 for p in idle.trajectory.points],
            )

    def test_the_two_runs_agree_on_the_pitch_block_ignoring_the_estimate(self):
        for active, idle in zip(self.on.deliveries, self.off.deliveries):
            on_block = dict(active.pitch)
            off_block = dict(idle.pitch)
            on_block.pop("box_estimate", None)
            self.assertEqual(on_block, off_block)

    def test_the_single_decode_guarantee_holds_with_the_feature_on(self):
        self.assertEqual(self.on.frames_decoded, self.off.frames_decoded)


class TestEstimateNextToRealCalibration(unittest.TestCase):
    """With a pitch detector AND the box detector, the estimate sits beside the
    measured quad rather than replacing it."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        cls.video = make_video(
            cls.dir / "source.mp4", n_frames=200, width=W, height=H, fps=FPS
        )
        cls.pitch_service = PitchService(
            PitchConfig(api_key="KEY_THAT_MUST_NEVER_PERSIST"),
            width=W,
            height=H,
            detector=FakePitchDetector(),
        )
        cls.paths, cls.result = _run(
            "test_pitch_box_calibrated",
            cls.video,
            pitch_service=cls.pitch_service,
            box_detector=StubBoxDetector(),
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_both_are_present(self):
        for delivery in self.result.deliveries:
            self.assertIsNotNone(delivery.pitch["box_estimate"])
            self.assertTrue(delivery.pitch["calibrated"], delivery.pitch["state"])

    def test_the_measured_quad_is_still_the_measured_quad(self):
        # 10%..90% of the frame, from the fake KEYPOINT detector - not the box.
        # The measured quad names its corners top_left..bottom_left, the box
        # estimate names its own P1..P4: two different objects, two different
        # vocabularies, deliberately.
        for delivery in self.result.deliveries:
            corners = delivery.pitch["corners"]
            self.assertEqual(len(corners), 4)
            self.assertAlmostEqual(corners["top_left"][0], 0.1 * W, places=1)
            self.assertAlmostEqual(corners["top_right"][0], 0.9 * W, places=1)
            self.assertAlmostEqual(corners["bottom_left"][1], 0.9 * H, places=1)
            self.assertTrue(delivery.pitch["homography_available"])
            self.assertIsNotNone(delivery.pitch["homography"])

    def test_the_estimate_is_a_different_number_and_says_so(self):
        for delivery in self.result.deliveries:
            estimate = delivery.pitch["box_estimate"]
            self.assertEqual(estimate["corners_source"], "box_estimate")
            # The box spans x 80..240 while the quad spans x 32..288: two
            # different objects, recorded as two different objects.
            self.assertAlmostEqual(estimate["corners"]["P1"][0], 80.0, places=1)
            self.assertNotAlmostEqual(
                estimate["corners"]["P1"][0],
                delivery.pitch["corners"]["bottom_left"][0],
                places=1,
            )

    def test_no_credential_reaches_the_record(self):
        document = self.result.tracking_document(PipelineConfig())
        self.assertNotIn("KEY_THAT_MUST_NEVER_PERSIST", json.dumps(document))


@unittest.skipUnless(
    (MODELS_DIR / "pitch_detect.pt").is_file(),
    "pitch_detect.pt not present in backend/models",
)
class TestRealWeights(unittest.TestCase):
    """The shipped weights, loaded by the code path the pipeline uses."""

    def setUp(self):
        # Per test, so each one sees its own counters. The weights themselves
        # are cached process-wide by `load_pitch_box_model`, so only the first
        # of these pays the load.
        self.detector = LocalPitchBoxDetector(
            str(MODELS_DIR / "pitch_detect.pt"),
            conf=PITCH_BOX_CONFIG.conf,
            imgsz=PITCH_BOX_CONFIG.imgsz,
        )

    def test_a_synthetic_frame_is_handled_without_raising(self):
        import numpy as np

        box = self.detector.detect(np.zeros((H, W, 3), dtype=np.uint8))
        stats = self.detector.stats()
        # Nothing to find in a black frame - but the model loaded, ran, and
        # every frame it was given is accounted for.
        self.assertIsNone(box)
        self.assertEqual(stats["frames_seen"], 1)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(stats["last_error"], None)
        self.assertEqual(
            stats["frames_seen"], stats["boxes"] + stats["no_box"] + stats["errors"]
        )

    def test_a_real_broadcast_frame_finds_the_pitch(self):
        video = Path(__file__).resolve().parents[3] / "real_cricket.mp4"
        if not video.is_file():
            self.skipTest("real_cricket.mp4 not present")
        import cv2

        capture = cv2.VideoCapture(str(video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 4000)
        ok, frame = capture.read()
        capture.release()
        self.assertTrue(ok)
        height, width = frame.shape[:2]

        box = self.detector.detect(frame)
        self.assertIsNotNone(box, "the shipped weights find the pitch on broadcast footage")
        self.assertGreater(box.conf, 0.5)
        # ORIGINAL frame pixels, not the model's letterboxed 640 input.
        self.assertLessEqual(box.x2, width)
        self.assertLessEqual(box.y2, height)
        corners = box_to_corners(
            box.x1, box.y1, box.x2, box.y2, top_inset=PITCH_BOX_CONFIG.inset
        )
        valid, reason = corners_valid(corners)
        self.assertTrue(valid, reason)

    def test_the_slot_runs_the_real_detector_end_to_end(self):
        import time

        import numpy as np

        detector = LocalPitchBoxDetector(
            str(MODELS_DIR / "pitch_detect.pt"), conf=0.25, imgsz=640
        )
        slot = PitchBoxSlot(detector, every_n=1, inset=PITCH_BOX_CONFIG.inset)
        self.addCleanup(slot.stop, 5.0)
        for n in range(1, 6):
            slot.offer(n, np.zeros((H, W, 3), dtype=np.uint8))
            time.sleep(0.01)
        slot.stop()
        summary = slot.summary()
        self.assertGreaterEqual(summary["frames_seen"], 1)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["enabled"], True)


if __name__ == "__main__":
    unittest.main()
