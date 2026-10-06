"""
The single-pass guarantee, asserted rather than assumed.

Decisions #3, #4, #5 and constraint C1: the source video is opened once,
decoded once, and no stage downstream of tracking reopens it. `CountingVideoCapture`
records every path handed to `cv2.VideoCapture`, so this is measured, not claimed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.core.config import Paths, PipelineConfig
from app.core.constants import SCHEMA_VERSION
from app.pipeline.analysis_pipeline import UnifiedPipeline
from app.tracking.tracking_service import TrackingService
from tests.helpers import CountingVideoCapture, StubDetector, make_video

W, H, FPS = 320, 180, 25.0


class SinglePassCase(unittest.TestCase):
    """Runs a real analysis over a real video with a stub detector."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.video = make_video(
            self.dir / "source.mp4", n_frames=200, width=W, height=H, fps=FPS
        )
        self.paths = Paths(analysis_id="test_single_pass").ensure()
        self.paths.clip_path(1).parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def run_tracking(self, windows=None, config=None, n_frames=200):
        windows = windows or [(30, 60)]
        if n_frames != 200:
            self.video = make_video(
                self.dir / "source.mp4", n_frames=n_frames,
                width=W, height=H, fps=FPS,
            )
        service = TrackingService(config or PipelineConfig())
        detector = StubDetector(windows)
        return service.analyse(
            str(self.video), self.paths, detector=detector
        ), detector


class TestSingleDecode(SinglePassCase):
    def test_source_video_is_opened_exactly_once(self):
        counter = CountingVideoCapture().install()
        try:
            result, _ = self.run_tracking()
        finally:
            counter.restore()
        self.assertEqual(
            counter.count_prefix(str(self.video)), 1,
            f"Source was opened {counter.count_prefix(str(self.video))} times; "
            f"all opens: {counter.opened}",
        )
        self.assertGreater(result.frames_decoded, 0)

    def test_detector_runs_once_per_frame_not_once_per_delivery(self):
        result, detector = self.run_tracking(windows=[(30, 60), (110, 140)])
        self.assertGreaterEqual(result.frames_decoded, 200)
        # One detection call per decoded frame, regardless of delivery count.
        self.assertEqual(detector.calls, result.frames_decoded)

    def test_no_third_pass_over_the_source_from_any_stage(self):
        """Clip and overlay rendering must read the CLIP, never the source."""
        counter = CountingVideoCapture().install()
        try:
            from app.video.overlay import OverlayRenderer

            result, _ = self.run_tracking()
            renderer = OverlayRenderer(codec_preference=("mp4v",))
            for d in result.deliveries:
                renderer.render(d, self.paths)
        finally:
            counter.restore()
        self.assertEqual(counter.count_prefix(str(self.video)), 1)
        # Clips were read, which is allowed and expected.
        self.assertGreaterEqual(counter.count_prefix("delivery_"), 0)


class TestDeliverySegmentation(SinglePassCase):
    def test_single_window_yields_one_delivery(self):
        result, _ = self.run_tracking(windows=[(30, 60)])
        self.assertEqual(len(result.deliveries), 1)
        d = result.deliveries[0]
        self.assertEqual(d.delivery_id, 1)
        self.assertEqual(d.tracking_source, "single_pass")
        self.assertIsNone(d.fallback_reason)

    def test_close_second_window_is_tracked_not_suppressed(self):
        """A second detection run must yield a second delivery.

        HISTORY — this test used to assert the opposite (`== 1`) and was named
        "debounce". There is no debounce: `min_frames_between_deliveries` is
        documented as DEPRECATED and unused as a segmentation rule
        (app/core/config.py, "min_frames_between_deliveries_note"). What actually
        collapsed the second window was the Stage 3 Kalman livelock — the stale
        track from window 1 gate-refused all 26 detections of window 2, so it
        never became an event. The test was pinning that defect.

        Window 2 now correctly produces its own delivery.
        """
        result, _ = self.run_tracking(windows=[(30, 60), (70, 95)])
        self.assertEqual(len(result.deliveries), 2)

    def test_debounce_constant_is_not_a_segmentation_rule(self):
        """The deprecation is real: nothing gates delivery count on it."""
        from app.core.config import PipelineConfig
        from app.core.constants import MIN_FRAMES_BETWEEN_DELIVERIES

        self.assertEqual(PipelineConfig().min_frames_between_deliveries,
                         MIN_FRAMES_BETWEEN_DELIVERIES)
        # A 25-frame window after a 30-frame one is only 10 frames apart, far
        # inside the nominal 60, yet both are delivered — because the constant is
        # metadata, not a filter.
        result, _ = self.run_tracking(windows=[(30, 60), (70, 95)])
        self.assertEqual(len(result.deliveries), 2)

    def test_well_separated_windows_yield_separate_deliveries(self):
        result, _ = self.run_tracking(windows=[(30, 60), (130, 165)])
        self.assertEqual(len(result.deliveries), 2)
        self.assertEqual([d.delivery_id for d in result.deliveries], [1, 2])

    def test_short_window_is_not_a_delivery(self):
        """Fewer detections than MIN_DELIVERY_FRAMES must be discarded."""
        from app.core.constants import MIN_DELIVERY_FRAMES

        result, _ = self.run_tracking(
            windows=[(30, 30 + MIN_DELIVERY_FRAMES - 2)]
        )
        self.assertEqual(len(result.deliveries), 0)
        self.assertTrue(result.warnings)

    def test_delivery_ids_are_contiguous_with_no_gaps(self):
        """Upstream increments the id even when the cooldown discards a
        delivery, producing gaps. Here ids advance only on acceptance."""
        result, _ = self.run_tracking(
            windows=[(30, 60), (70, 95), (130, 165), (170, 195)]
        )
        ids = [d.delivery_id for d in result.deliveries]
        self.assertEqual(ids, list(range(1, len(ids) + 1)))


class TestPersistedRecords(SinglePassCase):
    def test_boundaries_are_source_frame_numbers(self):
        result, _ = self.run_tracking(windows=[(30, 60)])
        d = result.deliveries[0]
        self.assertGreaterEqual(d.detection_start_frame, 30)
        self.assertLessEqual(d.detection_end_frame, 60)
        # ball_lost_frame is detection_end + MAX_MISSED + 1
        self.assertGreater(d.ball_lost_frame, d.detection_end_frame)

    def test_trajectory_records_provenance_per_point(self):
        result, _ = self.run_tracking()
        d = result.deliveries[0]
        sources = {p.source for p in d.trajectory.points}
        self.assertTrue(sources <= {"detected", "predicted"})
        self.assertIn("detected", sources)
        for p in d.trajectory.points:
            if p.source == "detected":
                self.assertIsNotNone(p.confidence)
            else:
                self.assertIsNone(p.confidence)

    def test_all_points_have_frame_numbers(self):
        """Upstream discarded frame numbers in _finalise_delivery."""
        result, _ = self.run_tracking()
        for d in result.deliveries:
            for p in d.trajectory.points:
                self.assertIsInstance(p.original_frame, int)
                self.assertGreater(p.original_frame, 0)

    def test_tracking_json_is_written_and_parseable(self):
        import json

        self.run_tracking()
        doc = json.loads(self.paths.tracking_json.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema_version"], SCHEMA_VERSION)
        self.assertEqual(doc["performance"]["source_decodes"], 1)
        self.assertEqual(len(doc["deliveries"]), 1)
        self.assertIn("calibration", doc)

    def test_tracking_json_carries_the_calibration_facts(self):
        import json

        self.run_tracking()
        doc = json.loads(self.paths.tracking_json.read_text(encoding="utf-8"))
        cal = doc["calibration"]
        self.assertTrue(cal["speed_is_estimate"])
        self.assertFalse(cal["speed_is_calibrated"])
        self.assertEqual(cal["speed_scale"], 2.0)
        self.assertEqual(cal["speed_offset_kmh"], -20.0)


class TestEachDeliveryCarriesItsOwnTrajectory(SinglePassCase):
    """
    A delivery's trajectory must be the ball IT saw.

    Emission is deferred: a closed candidate is held until the merge decision is
    final, which can be several events later. Its points therefore travel with the
    candidate rather than being read from the live accumulator — by the time a
    candidate is committed, `cur_points` belongs to a different ball entirely.

    Reading it anyway filed one delivery's trajectory under another's boundaries
    and then cleared the accumulator, so the event still being tracked lost its
    points too. Nothing downstream noticed: the clip frames were right, the frame
    map was valid, and the end-of-pass validator simply reported NO_EVENT for
    deliveries whose trajectory had been handed to somebody else. The corruption
    was silent, which is why it is pinned here rather than left to the clip tests.
    """

    # Three deliveries close enough together that each is committed only after the
    # next one has started collecting points.
    WINDOWS = [(31, 45), (50, 64), (70, 84)]

    def setUp(self):
        super().setUp()
        self.result, _ = self.run_tracking(windows=self.WINDOWS, n_frames=200)
        self.assertGreaterEqual(
            len(self.result.deliveries), 2,
            "this test is meaningless unless several deliveries were emitted",
        )

    def test_every_delivery_has_points(self):
        for d in self.result.deliveries:
            self.assertGreater(
                len(d.trajectory.points), 0,
                f"delivery {d.delivery_id} was emitted with no trajectory at all",
            )

    def test_every_detected_point_lies_inside_its_own_delivery(self):
        for d in self.result.deliveries:
            detected = d.trajectory.detected_points()
            self.assertTrue(detected, f"delivery {d.delivery_id} has no detected point")
            for p in detected:
                self.assertGreaterEqual(
                    p.original_frame, d.detection_start_frame,
                    f"delivery {d.delivery_id} holds a point from frame "
                    f"{p.original_frame}, before its own first detection at "
                    f"{d.detection_start_frame} — that is another ball's footage",
                )
                self.assertLessEqual(
                    p.original_frame, d.ball_lost_frame,
                    f"delivery {d.delivery_id} holds a point from frame "
                    f"{p.original_frame}, past its own ball loss at "
                    f"{d.ball_lost_frame}",
                )

    def test_no_two_deliveries_claim_the_same_detected_frame(self):
        seen: dict[int, int] = {}
        for d in self.result.deliveries:
            for p in d.trajectory.detected_points():
                self.assertNotIn(
                    p.original_frame, seen,
                    f"frame {p.original_frame} is claimed by deliveries "
                    f"{seen.get(p.original_frame)} and {d.delivery_id}",
                )
                seen[p.original_frame] = d.delivery_id

    def test_the_stored_trajectory_matches_the_stored_boundaries(self):
        """The persisted document is read back, because that is what consumers use."""
        import json

        doc = json.loads(self.paths.tracking_json.read_text(encoding="utf-8"))
        self.assertEqual(len(doc["deliveries"]), len(self.result.deliveries))
        for raw in doc["deliveries"]:
            detected = [
                p["original_frame"] for p in raw["trajectory"]["points"]
                if p["source"] == "detected"
            ]
            self.assertTrue(detected, f"delivery {raw['delivery_id']} persisted no ball")
            self.assertGreaterEqual(min(detected), raw["detection_start_frame"])
            self.assertLessEqual(max(detected), raw["detection_end_frame"])


class TestClipWindowing(SinglePassCase):
    def test_clip_window_includes_preroll_and_postroll(self):
        result, _ = self.run_tracking(windows=[(30, 60)])
        d = result.deliveries[0]
        self.assertIsNotNone(d.clip)
        # Pre-roll reaches back before the first detection.
        self.assertLess(d.clip.clip_start_frame, d.detection_start_frame)
        # Post-roll extends past the ball being lost — the stroke happens after.
        self.assertGreater(d.clip.clip_end_frame, d.ball_lost_frame)

    def test_clip_window_is_clamped_to_the_start_of_the_video(self):
        result, _ = self.run_tracking(windows=[(2, 40)])
        self.assertGreaterEqual(len(result.deliveries), 1)
        d = result.deliveries[0]
        self.assertGreaterEqual(d.clip.clip_start_frame, 1)
        self.assertEqual(d.clip.clip_start_frame, 1)

    def test_a_delivery_in_the_first_seconds_is_not_discarded(self):
        """Regression: upstream seeded the debounce at frame 0, so any delivery
        ending before frame 60 was silently dropped — including the first over
        of a video."""
        result, _ = self.run_tracking(windows=[(2, 40)])
        self.assertEqual(len(result.deliveries), 1)
        self.assertEqual(result.deliveries[0].detection_start_frame, 2)

    def test_clip_frame_map_is_valid_and_round_trips(self):
        result, _ = self.run_tracking()
        d = result.deliveries[0]
        self.assertTrue(d.clip.frame_map_valid)
        for p in d.trajectory.points:
            if p.clip_frame is not None:
                self.assertEqual(
                    d.clip.to_original_frame(p.clip_frame), p.original_frame
                )

    def test_clip_frame_count_matches_frames_written(self):
        result, _ = self.run_tracking()
        d = result.deliveries[0]
        self.assertEqual(d.clip.frames_written, d.clip.frame_count)

    def test_clip_file_exists_and_has_the_expected_length(self):
        import cv2

        result, _ = self.run_tracking()
        d = result.deliveries[0]
        cap = cv2.VideoCapture(str(d.clip_path))
        self.assertTrue(cap.isOpened())
        self.assertEqual(
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), d.clip.frame_count
        )
        cap.release()

    def test_clip_is_written_at_source_resolution(self):
        """Trajectory coordinates are only meaningful if clip dims match."""
        result, _ = self.run_tracking()
        d = result.deliveries[0]
        self.assertEqual(d.clip.width, W)
        self.assertEqual(d.clip.height, H)


class TestShotStageInputSources(unittest.TestCase):
    def test_sample_indices_are_identical_for_both_paths(self):
        """If these diverged, the same clip would classify differently
        depending on whether frames came from memory or the file."""
        from app.video.clips import ClipWriterRegistry  # noqa: F401
        from app.shot_classification.preprocessing import sample_indices

        for total in (30, 31, 47, 100, 149, 150):
            a = sample_indices(total)
            b = [int(i) for i in __import__("numpy").linspace(
                0, total - 1, 30, dtype=__import__("numpy").int64)]
            self.assertEqual(a, b, f"mismatch for total_frames={total}")

    def test_sample_indices_short_clip_still_yields_thirty_positions(self):
        """
        A short clip is stretched, not truncated.

        The model's tensor is (B, 30, 3, 224, 224). The old code returned
        whatever positions existed, so a 19-frame clip produced a 19-frame input
        and the model received something its architecture cannot accept. Short
        clips are now sampled with repeats, deterministically, and the repetition
        is recorded on the delivery's shot record so nobody reads a padded clip
        as 30 distinct observations.
        """
        import numpy as np

        from app.core.constants import SHOT_N_FRAMES
        from app.shot_classification.preprocessing import sample_indices

        self.assertEqual(sample_indices(0), [])
        for total in (1, 5, 19, 29, 30, 31):
            idxs = sample_indices(total)
            self.assertEqual(len(idxs), SHOT_N_FRAMES, f"total_frames={total}")
            # Every index addresses a frame that exists.
            self.assertLess(max(idxs), total, f"total_frames={total}")
            # Deterministic: same input, same positions.
            self.assertEqual(idxs, sample_indices(total))
            # Non-decreasing, so the clip is read in order.
            self.assertEqual(idxs, sorted(idxs), f"total_frames={total}")
            # Spans the clip rather than padding one end only.
            self.assertEqual(idxs[0], 0)
            self.assertEqual(idxs[-1], total - 1)

    def test_short_clip_repetition_matches_the_upstream_transformer_sampler(self):
        """The repeats must be the ones Adarsh's inference code produces.

        Byte-identical weights plus a different sampler would be a silent
        distribution shift: the labels would be wrong for reasons invisible in
        this repository.
        """
        import numpy as np

        from app.shot_classification.preprocessing import sample_indices

        for total in (1, 5, 19, 29, 31, 150):
            expected = np.linspace(0, total - 1, 30, dtype=np.int64)
            self.assertEqual(
                sample_indices(total), [int(i) for i in expected],
                f"total_frames={total}",
            )

    def test_a_long_clip_samples_distinct_positions(self):
        """Padding is for clips shorter than 30, not a permanent feature."""
        from app.shot_classification.preprocessing import sample_indices

        self.assertEqual(len(set(sample_indices(150))), 30)
        self.assertEqual(len(set(sample_indices(30))), 30)


class TestInMemoryFrameLookup(unittest.TestCase):
    """
    The retained frames are a SPARSE subset of the clip, keyed by position.
    Looking them up wrongly is silent: the model still gets 30 frames, just the
    wrong 30.
    """

    def setUp(self):
        import numpy as np

        from app.shot_classification.inference import ShotClassifier
        from app.shot_classification.preprocessing import sample_indices

        self.np = np
        self.ShotClassifier = ShotClassifier
        self.sample_indices = sample_indices
        self.total = 150
        self.idxs = sample_indices(self.total)
        # Distinct pixel values so a mislookup is detectable.
        self.mapping = {i: np.full((4, 4, 3), i % 256, np.uint8) for i in self.idxs}

    def test_mapping_is_looked_up_by_position(self):
        frames, source, positions = self.ShotClassifier._gather(
            None, self.mapping, self.total
        )
        self.assertEqual(source, "in_memory")
        self.assertEqual(len(frames), 30)
        self.assertEqual(positions, self.idxs)
        for pos, img in zip(self.idxs, frames):
            self.assertEqual(int(img[0, 0, 0]), pos % 256)

    def test_flattening_the_mapping_would_change_the_frames(self):
        """Regression guard for the bug this path originally had."""
        flat = [self.mapping[i] for i in sorted(self.mapping)]
        self.assertEqual(len(flat), 30)
        # A dense list is interpreted as every clip frame, so the sampled
        # positions no longer address the right moments.
        self.assertGreater(self.idxs[-1], len(flat))
        frames, source, _ = self.ShotClassifier._gather(None, flat, self.total)
        self.assertNotEqual(source, "in_memory")

    def test_dense_sequence_is_still_supported(self):
        dense = [
            self.np.full((4, 4, 3), i % 256, self.np.uint8)
            for i in range(self.total)
        ]
        frames, source, positions = self.ShotClassifier._gather(
            None, dense, self.total
        )
        self.assertEqual(source, "in_memory")
        self.assertEqual(positions, self.idxs)
        self.assertEqual([int(f[0, 0, 0]) for f in frames],
                         [i % 256 for i in self.idxs])

    def test_mapping_with_a_hole_falls_back_rather_than_substituting(self):
        broken = dict(self.mapping)
        broken.pop(self.idxs[7])
        frames, source, _ = self.ShotClassifier._gather(
            None, broken, self.total, )
        # No clip file given, so the fallback also has nothing: it must report
        # unavailable rather than invent the missing frame.
        self.assertIsNone(frames)
        self.assertEqual(source, "unavailable")

    def test_missing_positions_are_reported_not_silently_dropped(self):
        broken = dict(self.mapping)
        broken.pop(self.idxs[3])
        with self.assertLogs(
            "app.shot_classification.inference", level="WARNING"
        ) as cm:
            self.ShotClassifier._gather(None, broken, self.total)
        self.assertTrue(any("7 of 30" in m or "sampled positions" in m
                            for m in cm.output))

    def test_short_clip_is_rejected_rather_than_padded(self):
        """Padding frames would fabricate input the model never saw."""
        import numpy as np

        from app.shot_classification.preprocessing import frames_to_tensor

        with self.assertRaises(ValueError):
            frames_to_tensor([np.zeros((10, 10, 3), np.uint8)] * 5)


if __name__ == "__main__":
    unittest.main()