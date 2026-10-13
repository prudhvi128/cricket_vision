"""
test_pitch_seg_feature.py - the optional LOCAL pitch-SEGMENTATION detector
inside a REAL analysis.

`mask_geometry.py` is proven pure, the detector is proven to obey its
contract; this proves the two FIT into the pass that writes `result.json`
without contaminating the metric analytics:

  * OFF (the default under test, conftest sets CRICKET_PITCH_SEG_MODEL=off)
    adds nothing: no `pitch_seg_detection` record, and the tracking document's
    top-level keys are exactly what this build shipped with before the feature;
  * ON (a stub seg detector injected by the caller, the same way the box
    feature is tested), the seg quad runs the NORMAL calibration state machine
    - the pitch block gets corners and the frame homography, labelled with
    `source="pitch_segmentation"` so the metric gate can see where it came from;
  * the gate admits seg as a metric source but only when its GEOMETRY passes
    the per-delivery checks (enough detected points, aspect agrees with the
    ball's travel, homography finite, corners map inside the pitch). This
    stub's flight has too few detected points after the bounce cut, so the
    delivery is honestly refused: `map_homography` and `orientation` stay
    null, `geometry_calibrated` stays False, and speed / length / line /
    swing stay null because they must - no number is ever fabricated from a
    bad quad;
  * the browser is unaffected: the serializer exposes the same pitch keys it
    always did;
  * the ball tracking is untouched: same frames, same trajectory, with and
    without the feature;
  * a configured-but-missing weights file degrades to the OFF behaviour with
    an error in the log, rather than crashing or faking a detection.

The pipeline is driven end to end with a stub ball detector and a synthetic
video, so this is the served record rather than a hand-built fixture.
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.core.config import PITCH_SEG_CONFIG, Paths, PipelineConfig
from app.core.constants import PITCH_SEG_DEFAULT_WEIGHTS
from app.pitch.detector import Keypoint, PitchDetection
from app.pitch.segmentation_detector import (
    PITCH_DETECTION_SOURCE_SEGMENTATION,
    build_pitch_seg_summary,
)
from app.schemas.frontend import serialize_delivery
from app.tracking.tracking_service import TrackingService
from tests.helpers import StubDetector, make_video

W, H, FPS = 320, 180, 25.0
WINDOW = (30, 60)

# The quad the stub seg detector always answers with: a plausible pitch mask
# boundary, strictly inside a 320x180 frame, 10%..90% like the box feature's
# stub so the same calibration validator accepts it.
STUB_QUAD = [
    (0.1 * W, 0.1 * H),
    (0.9 * W, 0.1 * H),
    (0.9 * W, 0.9 * H),
    (0.1 * W, 0.9 * H),
]
STUB_CONF = 0.91

# The keys `tracking_document` had before this feature existed.
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

# Every key the internal pitch block has, whatever happened - the same set
# test_pitch_pipeline.py pins: the segmentation feature adds NO key to the
# delivery block; its quad travels through the calibration that was already
# there. `source` names the detector that measured the quad;
# `geometry_reason` is the metric gate's recorded refusal (null when it
# accepted) — both shipped together as provenance.
INTERNAL_BLOCK_KEYS = {
    "state", "detected", "calibrated", "confidence",
    "using_previous_calibration", "keypoints", "corners", "center",
    "homography_available", "orientation", "homography", "model_id",
    "image_size", "frame_number", "map_homography",
    "source", "geometry_reason",
}


class StubSegDetector:
    """Always answers with `STUB_QUAD`, labelled as a segmentation source."""

    disabled_reason = None

    def __init__(self) -> None:
        self.frames_seen = 0

    def detect(self, frame, frame_number: int, image_size) -> PitchDetection:  # noqa: ANN001
        self.frames_seen += 1
        width, height = image_size
        return PitchDetection(
            keypoints=[
                Keypoint(f"p{i}", x, y, STUB_CONF)
                for i, (x, y) in enumerate(STUB_QUAD, 1)
            ],
            frame_number=frame_number,
            image_size=image_size,
            confidence=STUB_CONF,
            source=PITCH_DETECTION_SOURCE_SEGMENTATION,
        )

    def stats(self) -> dict:
        return {
            "enabled": True,
            "model_path": "stub",
            "device": "stub",
            "conf": 0.5,
            "imgsz": 640,
            "samples": self.frames_seen,
            "accepted": self.frames_seen,
            "no_mask": 0,
            "errors": 0,
            "rejected": {},
            "last_error": None,
            "latency_ms": {"n": 0, "mean": None, "p50": None, "p95": None},
            "last_prediction": None,
        }


def _run(analysis_id: str, video: Path, *, seg_detector=None):
    paths = Paths(analysis_id=analysis_id).ensure()
    service = TrackingService(
        PipelineConfig(),
        pitch_seg_detector=seg_detector,
    )
    return paths, service.analyse(str(video), paths, detector=StubDetector([WINDOW]))


class TestPitchSegFeature(unittest.TestCase):
    """OFF and ON of the same video, computed once for every test here."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        cls.video = make_video(
            cls.dir / "source.mp4", n_frames=200, width=W, height=H, fps=FPS
        )
        cls.off_paths, cls.off = _run("test_pitch_seg_off", cls.video)
        cls.seg_detector = StubSegDetector()
        cls.on_paths, cls.on = _run(
            "test_pitch_seg_on", cls.video, seg_detector=cls.seg_detector
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _result_document(self, tracking):
        """The API-facing `result.json` for a tracking pass, as the pipeline
        assembles it. `result_document` reads only what the tracking pass
        returned, so the shot + overlay stage is not needed to check keys."""
        from app.pipeline.analysis_pipeline import PipelineResult

        pipeline = PipelineResult(analysis_id=tracking.analysis_id, tracking=tracking)
        pipeline.deliveries = list(tracking.deliveries)
        return pipeline.result_document(Paths(analysis_id=tracking.analysis_id))

    # ── OFF is inert ──────────────────────────────────────────────────────────
    def test_off_adds_no_record_to_the_result(self):
        self.assertIsNone(self.off.pitch_seg_summary)

    def test_off_adds_no_key_to_the_tracking_document(self):
        document = self.off.tracking_document(PipelineConfig())
        self.assertNotIn("pitch_seg_detection", document)
        self.assertEqual(set(document), BASELINE_TRACKING_KEYS)
        json.dumps(document)

    def test_off_adds_nothing_to_tracking_json_on_disk(self):
        document = json.loads(self.off_paths.tracking_json.read_text())
        self.assertNotIn("pitch_seg_detection", document)
        self.assertEqual(set(document), BASELINE_TRACKING_KEYS)

    def test_off_adds_no_record_to_the_result_document(self):
        document = self._result_document(self.off)
        self.assertNotIn("pitch_seg_detection", document)
        json.dumps(document)

    # ── ON records its own work ───────────────────────────────────────────────
    def test_the_detector_really_ran(self):
        self.assertGreater(self.seg_detector.frames_seen, 0)
        summary = self.on.pitch_seg_summary
        self.assertIsInstance(summary, dict)
        self.assertTrue(summary["enabled"])
        self.assertGreater(summary["samples"], 0)
        self.assertEqual(summary["errors"], 0)
        json.dumps(summary)

    def test_the_record_reaches_the_tracking_document(self):
        document = self.on.tracking_document(PipelineConfig())
        self.assertIn("pitch_seg_detection", document)
        self.assertEqual(
            set(document), BASELINE_TRACKING_KEYS | {"pitch_seg_detection"}
        )
        off_document = self.off.tracking_document(PipelineConfig())
        for key in ("schema_version", "pipeline_version"):
            self.assertEqual(document[key], off_document[key])

    def test_the_record_reaches_tracking_json_on_disk(self):
        document = json.loads(self.on_paths.tracking_json.read_text())
        self.assertIn("pitch_seg_detection", document)
        self.assertTrue(document["pitch_seg_detection"]["enabled"])

    def test_the_record_reaches_the_result_document_too(self):
        document = self._result_document(self.on)
        self.assertIn("pitch_seg_detection", document)
        self.assertTrue(document["pitch_seg_detection"]["enabled"])
        json.dumps(document)

    def test_the_summary_says_metric_analytics_are_checked_per_delivery(self):
        summary = self.on.pitch_seg_summary
        # The seg source is now eligible for metric geometry; the per-delivery
        # geometric validation decides, and its verdict lives in the run's
        # pitch_geometry_stats.
        self.assertEqual(summary["calibration"]["geometry_calibrated"], "per_delivery")
        self.assertIsNone(summary["calibration"]["reason"])
        self.assertEqual(
            summary["calibration"]["metric_analytics"],
            "subject_to_geometric_validation",
        )
        # The service filing travels with the detector tally.
        self.assertIn("service", summary)
        self.assertGreater(summary["service"]["submitted"], 0)

    # ── the seg quad becomes the calibration, and nothing more ────────────────
    def test_every_delivery_has_the_unchanged_pitch_block_keys(self):
        self.assertGreaterEqual(len(self.on.deliveries), 1)
        for delivery in self.on.deliveries:
            self.assertEqual(set(delivery.pitch), INTERNAL_BLOCK_KEYS)

    def test_the_seg_quad_is_the_pitch_blocks_corners(self):
        for delivery in self.on.deliveries:
            pitch = delivery.pitch
            self.assertTrue(pitch["calibrated"], pitch["state"])
            self.assertEqual(len(pitch["corners"]), 4)
            self.assertTrue(pitch["homography_available"])
            self.assertIsNotNone(pitch["homography"])
            # The stub's rectangle, in the corners vocabulary the block uses.
            self.assertAlmostEqual(
                pitch["corners"]["bottom_left"][0], STUB_QUAD[0][0], places=1
            )
            self.assertAlmostEqual(
                pitch["corners"]["top_right"][1], STUB_QUAD[1][1], places=1
            )

    def test_the_metric_gate_refuses_the_seg_quad(self):
        for delivery in self.on.deliveries:
            pitch = delivery.pitch
            # The frame homography (quad -> image) exists; the METRIC one
            # (quad -> metres) does not, and the orientation that would let a
            # pitch map be drawn to scale is not invented either.
            self.assertIsNone(pitch["map_homography"])
            self.assertIsNone(pitch["orientation"])

    def test_it_cannot_make_a_delivery_geometry_calibrated(self):
        for delivery in self.on.deliveries:
            self.assertFalse(delivery.bowling.geometry_calibrated)

    def test_speed_and_line_stay_null_because_they_must(self):
        for delivery in self.on.deliveries:
            bowling = delivery.bowling
            self.assertIsNone(bowling.speed_kmh)
            self.assertIsNone(bowling.length)
            self.assertIsNone(bowling.line)
            self.assertIsNone(bowling.swing)
            self.assertIsNone(bowling.bounce_angle)

    def test_the_rejection_is_counted_in_the_geometry_stats(self):
        # With the seg detector as the only pitch source, this stub's flight
        # has too few detected points after the bounce cut, so the GEOMETRIC
        # check refuses it. The source-based refusal is gone (seg is now
        # eligible), but nothing was quietly rounded into a measurement: the
        # delivery still has no geometry.
        stats = self.on.pitch_geometry_stats
        self.assertEqual(stats["deliveries_with_geometry"], 0)
        # The old categorical source refusal must NOT be the reason any more.
        self.assertIsNone(
            stats["rejections"].get("segmentation_estimate_is_not_metric_geometry")
        )
        # A geometric reason IS recorded for every delivery.
        self.assertEqual(
            stats["rejections"].get("too_few_detected_points"),
            stats["deliveries"],
        )

    # ── the browser is unaffected ─────────────────────────────────────────────
    def test_the_serializer_exposes_no_seg_specific_key(self):
        for delivery in self.on.deliveries:
            body = serialize_delivery(
                delivery.as_dict(), analysis_id="pitch_seg"
            ).model_dump()
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

    def test_the_single_decode_guarantee_holds_with_the_feature_on(self):
        self.assertEqual(self.on.frames_decoded, self.off.frames_decoded)


class TestMissingWeightsDegradeToOff(unittest.TestCase):
    """Configured but the file is gone: the analysis runs exactly as it would
    without the feature, and says so in the log."""

    def test_the_analysis_completes_without_the_feature(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        video = make_video(
            Path(tmp.name) / "source.mp4", n_frames=120, width=W, height=H, fps=FPS
        )
        missing = replace(PITCH_SEG_CONFIG, model="pitch_seg_missing_weights.pt")
        with patch(
            "app.tracking.tracking_service.PITCH_SEG_CONFIG", missing
        ), self.assertLogs(
            "app.tracking.tracking_service", level=logging.ERROR
        ) as captured:
            paths, result = _run("test_pitch_seg_missing", video)
        self.assertTrue(
            any("weights not found" in line for line in captured.output),
            captured.output,
        )
        self.assertIsNone(result.pitch_seg_summary)
        document = result.tracking_document(PipelineConfig())
        self.assertNotIn("pitch_seg_detection", document)
        self.assertEqual(set(document), BASELINE_TRACKING_KEYS)
        self.assertGreaterEqual(len(result.deliveries), 1)


@unittest.skipUnless(
    (Path(__file__).resolve().parents[2] / "models" / PITCH_SEG_DEFAULT_WEIGHTS).is_file(),
    f"{PITCH_SEG_DEFAULT_WEIGHTS} not present in backend/models",
)
class TestRealWeightsThroughThePass(unittest.TestCase):
    """The shipped checkpoint, driven through the whole tracking pass on a
    synthetic video: the model loads, runs, and every frame it is given is
    accounted for - even when a flat grey frame gives it nothing to find."""

    def test_the_real_model_runs_end_to_end(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        video = make_video(
            Path(tmp.name) / "source.mp4", n_frames=60, width=W, height=H, fps=FPS
        )
        from app.pitch.segmentation_detector import LocalPitchSegDetector

        detector = LocalPitchSegDetector.from_config(
            replace(PITCH_SEG_CONFIG, model=PITCH_SEG_DEFAULT_WEIGHTS)
        )
        _, result = _run("test_pitch_seg_real", video, seg_detector=detector)
        summary = result.pitch_seg_summary
        self.assertIsNotNone(summary)
        self.assertTrue(summary["enabled"])
        self.assertEqual(
            summary["samples"],
            summary["accepted"] + summary["no_mask"] + summary["errors"],
        )
        self.assertEqual(summary["errors"], 0)
        self.assertGreater(summary["samples"], 0)
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()
