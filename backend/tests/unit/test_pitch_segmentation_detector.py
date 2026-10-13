"""
test_pitch_segmentation_detector.py — the local mask detector, with a stubbed
ultralytics model.

The detector implements the PitchService interface: one frame in, one
`PitchDetection` (four mask-derived keypoints) or a filed error out. These
tests prove the contract without a GPU or the 20 MB checkpoint:

  * a valid pitch mask becomes four named keypoints and a `pitch_segmentation`
    source, so the metric-geometry gate can see where it came from;
  * every other answer — no masks, a crowd, a triangle, a NaN polygon, an
    inference exception — is REFUSED WITH A REASON and counted;
  * `samples == accepted + no_mask + errors`, always;
  * a missing checkpoint raises PitchDetectorUnavailable (the service then
    stops submitting) rather than crashing the pass;
  * when CUDA is unavailable the device resolves to CPU instead of failing.
"""

from __future__ import annotations

import unittest

import numpy as np

from app.core.constants import (
    PITCH_DETECTION_SOURCE_KEYPOINTS,
    PITCH_DETECTION_SOURCE_SEGMENTATION,
)
from app.pitch.detector import PitchDetectError, PitchDetectorUnavailable
from app.pitch.segmentation_detector import (
    LocalPitchSegDetector,
    build_pitch_seg_summary,
    describe_pitch_seg,
    pitch_seg_enabled,
    resolve_seg_device,
    seg_pitch_config,
)

W, H = 1280, 720

PITCH_POLYGON = [
    (540.0, 120.0), (740.0, 120.0), (900.0, 640.0), (380.0, 640.0),
]
# Same trapezoid with a boundary dense enough to look like a real mask.
PITCH_POLYGON_DENSE = [
    (540.0, 120.0), (590.0, 120.0), (640.0, 120.0), (690.0, 120.0), (740.0, 120.0),
    (780.0, 250.0), (820.0, 400.0), (860.0, 520.0), (900.0, 640.0),
    (700.0, 640.0), (500.0, 640.0), (380.0, 640.0),
    (420.0, 520.0), (470.0, 400.0), (505.0, 250.0),
]
CROWD_POLYGON = [
    (0.0, 0.0), (1280.0, 0.0), (1280.0, 400.0), (0.0, 420.0),
]


# ── ultralytics stand-ins ───────────────────────────────────────────────────
class _Boxes:
    def __init__(self, confs, clss):
        self.conf = np.asarray(confs, dtype=np.float32)
        self.cls = np.asarray(clss, dtype=np.float32)
        self.xyxy = np.zeros((len(confs), 4), dtype=np.float32)


class _Masks:
    def __init__(self, polygons):
        self.xy = [np.asarray(p, dtype=np.float32) for p in polygons]


class _Result:
    def __init__(self, polygons, confs, clss=None):
        self.masks = _Masks(polygons) if polygons else None
        self.boxes = _Boxes(confs, clss if clss is not None else [0] * len(confs))


class _Model:
    """Stands in for `ultralytics.YOLO`: `predict` returns canned results."""

    def __init__(self, results):
        self._results = results
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        if isinstance(self._results, Exception):
            raise self._results
        return self._results


def _detector(results=None, **kwargs) -> LocalPitchSegDetector:
    det = LocalPitchSegDetector(
        "stub.pt",
        conf=kwargs.pop("conf", 0.5),
        imgsz=kwargs.pop("imgsz", 640),
        **kwargs,
    )
    # Tests never load real weights: the model is injected, so the cache and
    # CUDA are irrelevant here.
    det._model = _Model(results) if results is not None else None
    return det


FRAME = np.zeros((H, W, 3), dtype=np.uint8)


class TestValidDetection(unittest.TestCase):
    def test_a_pitch_mask_becomes_four_named_keypoints(self):
        det = _detector([_Result([PITCH_POLYGON_DENSE], [0.9])])
        detection = det.detect(FRAME, 100, (W, H))
        self.assertEqual(len(detection.keypoints), 4)
        self.assertEqual(
            {k.name for k in detection.keypoints},
            {"top_left", "top_right", "bottom_right", "bottom_left"},
        )
        self.assertAlmostEqual(detection.confidence, 0.9, places=4)
        self.assertEqual(detection.frame_number, 100)
        self.assertEqual(detection.image_size, (W, H))
        # The corners are the mask's own extremes, in source pixels.
        named = detection.named
        self.assertAlmostEqual(named["top_left"][0], 540.0, delta=15.0)
        self.assertAlmostEqual(named["bottom_right"][1], 640.0, delta=15.0)

    def test_the_source_is_segmentation_so_the_metric_gate_can_see_it(self):
        det = _detector([_Result([PITCH_POLYGON_DENSE], [0.9])])
        detection = det.detect(FRAME, 1, (W, H))
        self.assertEqual(detection.source, PITCH_DETECTION_SOURCE_SEGMENTATION)
        # …and the keypoint default is what the abandoned path still gets.
        from app.pitch.detector import PitchDetection
        self.assertEqual(
            PitchDetection(keypoints=[], frame_number=1, image_size=(W, H)).source,
            PITCH_DETECTION_SOURCE_KEYPOINTS,
        )

    def test_every_frame_is_accounted_for(self):
        det = _detector([_Result([PITCH_POLYGON_DENSE], [0.9])])
        det.detect(FRAME, 1, (W, H))
        det.detect(FRAME, 2, (W, H))
        stats = det.stats()
        self.assertEqual(stats["samples"], 2)
        self.assertEqual(stats["accepted"], 2)
        self.assertEqual(stats["no_mask"], 0)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(
            stats["samples"], stats["accepted"] + stats["no_mask"] + stats["errors"]
        )

    def test_the_accepted_prediction_is_remembered_for_diagnostics(self):
        det = _detector([_Result([PITCH_POLYGON_DENSE], [0.91])])
        det.detect(FRAME, 42, (W, H))
        prediction = det.last_prediction
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.frame_index, 42)
        self.assertAlmostEqual(prediction.conf, 0.91, places=4)
        self.assertGreater(prediction.area_fraction, 0.01)
        history = det.history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["frame_index"], 42)
        self.assertIn("polygon", history[0])

    def test_stats_are_serialisable(self):
        import json
        det = _detector([_Result([PITCH_POLYGON_DENSE], [0.9])])
        det.detect(FRAME, 1, (W, H))
        json.dumps(det.stats())


class TestRefusals(unittest.TestCase):
    def test_no_masks_is_a_no_keypoints_answer(self):
        det = _detector([_Result(None, [])])
        with self.assertRaises(PitchDetectError) as ctx:
            det.detect(FRAME, 5, (W, H))
        self.assertIn("no keypoints", str(ctx.exception))
        stats = det.stats()
        self.assertEqual(stats["no_mask"], 1)
        self.assertEqual(stats["accepted"], 0)

    def test_an_empty_results_list_is_a_no_keypoints_answer(self):
        det = _detector([])
        with self.assertRaises(PitchDetectError):
            det.detect(FRAME, 5, (W, H))
        self.assertEqual(det.stats()["no_mask"], 1)

    def test_a_crowd_mask_is_rejected_with_its_reason(self):
        det = _detector([_Result([CROWD_POLYGON], [0.9])])
        with self.assertRaises(PitchDetectError) as ctx:
            det.detect(FRAME, 5, (W, H))
        self.assertIn("mask_area_too_large", str(ctx.exception))
        self.assertEqual(det.stats()["rejected"], {"mask_area_too_large": 1})

    def test_a_low_confidence_instance_is_rejected(self):
        det = _detector([_Result([PITCH_POLYGON_DENSE], [0.3])], conf=0.5)
        with self.assertRaises(PitchDetectError):
            det.detect(FRAME, 5, (W, H))
        self.assertEqual(det.stats()["rejected"], {"below_confidence": 1})

    def test_a_wrong_class_instance_is_rejected(self):
        det = _detector([_Result([PITCH_POLYGON_DENSE], [0.9], clss=[3])])
        with self.assertRaises(PitchDetectError):
            det.detect(FRAME, 5, (W, H))
        self.assertEqual(det.stats()["rejected"], {"wrong_class": 1})

    def test_a_nan_polygon_is_rejected_not_repaired(self):
        dirty = PITCH_POLYGON_DENSE + [(float("nan"), float("nan"))] * 3
        det = _detector([_Result([dirty], [0.9])])
        # The NaN vertices are dropped; the clean remainder still forms a
        # valid mask, so this must SUCCEED — a malformed vertex is not a
        # reason to refuse an otherwise honest mask.
        detection = det.detect(FRAME, 5, (W, H))
        self.assertEqual(len(detection.keypoints), 4)

    def test_an_inference_exception_is_filed_as_an_error(self):
        det = _detector(RuntimeError("CUDA out of memory"))
        with self.assertRaises(PitchDetectError) as ctx:
            det.detect(FRAME, 5, (W, H))
        self.assertIn("inference failed", str(ctx.exception))
        stats = det.stats()
        self.assertEqual(stats["errors"], 1)
        self.assertIn("CUDA out of memory", stats["last_error"])

    def test_the_tally_always_adds_up(self):
        det = _detector([_Result([CROWD_POLYGON], [0.9])])
        for frame in range(1, 4):
            with self.assertRaises(PitchDetectError):
                det.detect(FRAME, frame, (W, H))
        stats = det.stats()
        self.assertEqual(
            stats["samples"], stats["accepted"] + stats["no_mask"] + stats["errors"]
        )


class TestSelection(unittest.TestCase):
    def test_the_best_valid_instance_wins(self):
        det = _detector([_Result(
            [PITCH_POLYGON_DENSE, CROWD_POLYGON], [0.9, 0.95]
        )])
        # The crowd is refused by validation despite its higher confidence:
        # validity comes first, confidence only ranks the VALID.
        detection = det.detect(FRAME, 1, (W, H))
        self.assertAlmostEqual(detection.confidence, 0.9, places=4)
        self.assertEqual(det.stats()["rejected"], {"mask_area_too_large": 1})

    def test_two_valid_candidates_within_the_tie_band_prefer_consistency(self):
        shifted = [(x + 400.0, y) for x, y in PITCH_POLYGON_DENSE]
        det = _detector(None)
        # First frame: accept the left-hand mask.
        det._model = _Model([_Result([PITCH_POLYGON_DENSE], [0.9])])
        det.detect(FRAME, 1, (W, H))
        # Second frame: both masks present, equal confidence. The one next
        # door to the last accepted estimate wins the tie.
        det._model = _Model([_Result(
            [PITCH_POLYGON_DENSE, shifted], [0.9, 0.9]
        )])
        detection = det.detect(FRAME, 2, (W, H))
        self.assertAlmostEqual(detection.named["top_left"][0], 540.0, delta=15.0)

    def test_a_confidently_better_candidate_beats_consistency(self):
        shifted = [(x + 400.0, y) for x, y in PITCH_POLYGON_DENSE]
        det = _detector(None)
        det._model = _Model([_Result([PITCH_POLYGON_DENSE], [0.9])])
        det.detect(FRAME, 1, (W, H))
        # 0.99 vs 0.90: outside the tie band, so confidence wins outright.
        det._model = _Model([_Result(
            [PITCH_POLYGON_DENSE, shifted], [0.9, 0.99]
        )])
        detection = det.detect(FRAME, 2, (W, H))
        self.assertAlmostEqual(detection.named["top_left"][0], 940.0, delta=15.0)


class TestUnavailableAndDevice(unittest.TestCase):
    def test_a_missing_checkpoint_raises_detector_unavailable(self):
        det = LocalPitchSegDetector("definitely_missing_weights.pt")
        with self.assertRaises(PitchDetectorUnavailable):
            det.detect(FRAME, 1, (W, H))
        # Cached: the second attempt fails the same way, without a second load.
        with self.assertRaises(PitchDetectorUnavailable):
            det.detect(FRAME, 2, (W, H))

    def test_from_config_refuses_an_off_configuration(self):
        from dataclasses import replace
        from app.core.config import PITCH_SEG_CONFIG
        off = replace(PITCH_SEG_CONFIG, model="off")
        with self.assertRaises(PitchDetectorUnavailable):
            LocalPitchSegDetector.from_config(off)

    def test_cpu_is_always_honoured(self):
        self.assertEqual(resolve_seg_device("cpu"), "cpu")

    def test_auto_resolves_to_cuda_or_cpu_without_crashing(self):
        device = resolve_seg_device("")
        self.assertIn(device, {"cuda", "cpu"})


class TestServiceWiring(unittest.TestCase):
    def test_seg_pitch_config_is_inert_and_quad_shaped(self):
        from app.core.config import PITCH_SEG_CONFIG
        cfg = seg_pitch_config(PITCH_SEG_CONFIG)
        self.assertFalse(cfg.remote_enabled)
        self.assertEqual(cfg.api_key, "")
        self.assertEqual(cfg.min_keypoints, 4)
        self.assertTrue(cfg.model_id.startswith("pitch_seg:"))
        self.assertEqual(cfg.interval_frames, PITCH_SEG_CONFIG.every_n)

    def test_the_summary_states_that_metric_analytics_are_checked_per_delivery(self):
        det = _detector([_Result([PITCH_POLYGON_DENSE], [0.9])])
        det.detect(FRAME, 1, (W, H))
        summary = build_pitch_seg_summary(det, {
            "submitted": 3, "detections": 1, "failures": 2, "refusals": 0,
            "queue_drops": 0,
            "diagnostics": {"categories": {"detected": 1},
                            "refusals": {}, "latency_ms": {"n": 1},
                            "scene_changes": 0, "last_scene_cut_frame": None},
            "detector_error": None,
        })
        self.assertTrue(summary["enabled"])
        # The source is eligible for metric geometry; the per-delivery
        # geometric validation decides, and its verdict lives in the run's
        # pitch_geometry_stats, not here.
        self.assertEqual(summary["calibration"]["geometry_calibrated"], "per_delivery")
        self.assertIsNone(summary["calibration"]["reason"])
        self.assertEqual(
            summary["calibration"]["metric_analytics"],
            "subject_to_geometric_validation",
        )
        self.assertEqual(summary["service"]["submitted"], 3)
        self.assertEqual(summary["last_prediction"]["frame_index"], 1)
        import json
        json.dumps(summary)

    def test_describe_and_enabled_follow_the_off_tokens(self):
        from dataclasses import replace
        from app.core.config import PITCH_SEG_CONFIG
        # conftest sets CRICKET_PITCH_SEG_MODEL=off for every test, so the
        # module-level config is the OFF case; an ON case is built explicitly.
        off = replace(PITCH_SEG_CONFIG, model="off")
        self.assertFalse(pitch_seg_enabled(off))
        self.assertIn("OFF", describe_pitch_seg(off))
        on = replace(PITCH_SEG_CONFIG, model="pitch_seg_yolo11s.pt")
        self.assertTrue(pitch_seg_enabled(on))
        self.assertIn("ON", describe_pitch_seg(on))


class TestRealWeights(unittest.TestCase):
    """The shipped checkpoint, loaded by the code path the pipeline uses.

    The module-level PITCH_SEG_CONFIG is switched off under test by conftest,
    so these tests rebuild an ON config pointing at the real file — the
    weights exist in the repo, and only their presence may skip anything.
    """

    def setUp(self):
        from dataclasses import replace
        from app.core.config import MODELS_DIR, PITCH_SEG_CONFIG
        from app.core.constants import PITCH_SEG_DEFAULT_WEIGHTS
        # The off-switch makes the module-level model_path None, so the file
        # itself is checked; only its absence may skip anything.
        path = MODELS_DIR / PITCH_SEG_DEFAULT_WEIGHTS
        if not path.is_file():
            self.skipTest("pitch_seg_yolo11s.pt not present in backend/models")
        self.detector = LocalPitchSegDetector.from_config(
            replace(PITCH_SEG_CONFIG, model=PITCH_SEG_DEFAULT_WEIGHTS)
        )

    def test_a_black_frame_is_a_refusal_not_a_crash(self):
        det = self.detector
        with self.assertRaises(PitchDetectError):
            det.detect(np.zeros((H, W, 3), dtype=np.uint8), 1, (W, H))
        stats = det.stats()
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(
            stats["samples"], stats["accepted"] + stats["no_mask"] + stats["errors"]
        )

    def test_a_real_broadcast_frame_finds_the_pitch(self):
        from pathlib import Path
        video = Path(__file__).resolve().parents[3] / "real_cricket.mp4"
        if not video.is_file():
            self.skipTest("real_cricket.mp4 not present")
        import cv2
        capture = cv2.VideoCapture(str(video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 1850)
        ok, frame = capture.read()
        capture.release()
        self.assertTrue(ok)
        height, width = frame.shape[:2]
        detection = self.detector.detect(frame, 1850, (width, height))
        self.assertEqual(len(detection.keypoints), 4)
        self.assertGreater(detection.confidence, 0.5)
        self.assertEqual(detection.source, PITCH_DETECTION_SOURCE_SEGMENTATION)
        # ORIGINAL frame pixels: every corner inside the frame.
        for keypoint in detection.keypoints:
            self.assertGreaterEqual(keypoint.x, -0.05 * width)
            self.assertLessEqual(keypoint.x, 1.05 * width)
            self.assertGreaterEqual(keypoint.y, -0.05 * height)
            self.assertLessEqual(keypoint.y, 1.05 * height)


if __name__ == "__main__":
    unittest.main()
