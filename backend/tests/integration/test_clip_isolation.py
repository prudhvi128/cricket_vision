"""
test_clip_isolation.py — ONE DELIVERY = ONE CLIP = ONE BATTING EVENT.

The defect these tests exist for
-------------------------------
A delivery clip's window is naturally wider than the ball's own trajectory:

    start = detection_start - PRE_ROLL     (4 s of run-up)
    end   = ball_lost      + POST_ROLL    (2 s, because the STROKE happens
                                            after the ball is lost)

Two failures follow from that, and both were observed in the Phase 3 run:

  1. START OVERLAP. Delivery B's pre-roll reaches back into delivery A's
     post-roll, so B's clip opens on A's batting event.
  2. END OVERLAP. A's post-roll runs past the moment B's ball appears, so A's
     clip contains B's delivery and B's stroke.

Either one puts two batting events in one clip, which is unusable for shot
classification and simply wrong as a product artefact.

The required invariant is:

    ONE DELIVERY = ONE DELIVERY CLIP = ONE BATTING EVENT

    desired_start = detection_start - PRE_ROLL
    desired_end   = ball_lost       + POST_ROLL
    actual_start  = max(desired_start, previous_clip_end + 1)
    actual_end    = min(desired_end,   next_delivery_start - 1)

Contiguity outranks roll length. When deliveries are closer together than
PRE_ROLL + POST_ROLL the two bounds conflict, and the later delivery loses
pre-roll rather than the earlier one losing post-roll — post-roll is where the
stroke is.

HOW THESE TESTS PROVE IT
------------------------
Not by reading the frame map, which is a claim. Each delivery's frames are
written with a distinct GREY VALUE, the clip is decoded from disk, and every
decoded frame is attributed back to the delivery it came from. A clip that
leaked a neighbour's frames, or that padded with frames the application
invented, fails on pixels — not on bookkeeping.

Deliberately NOT tested here: any per-clip re-tracking. The single
full-video YOLO11 + Kalman pass is unchanged by this fix; only the window
arithmetic moved.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.core.config import Paths, PipelineConfig
from app.tracking.tracking_service import TrackingService

from tests.helpers import (
    StubDetector,
    classify_tag,
    decode_grey_means,
    make_tagged_video,
)

W, H, FPS = 320, 180, 25.0
BG = 30          # background / gap frames
TAG_A = 210      # delivery A's ball is on screen
TAG_B = 90       # delivery B's ball is on screen

# Frame windows for the stub detector. `calls` equals the 1-based frame number
# because the tracking loop calls the detector exactly once per decoded frame.
A_WINDOW = (31, 45)
B_WINDOW = (130, 145)      # close enough that A's post-roll would cross it
B_WINDOW_FAR = (200, 215)  # far enough that only B's PRE-roll overlaps A

# The resolved roll ceilings (3.0 s pre, 1.8 s post) are a HARD bound on the
# window, applied on top of whatever the configured roll asks for. Several tests
# below deliberately ask for a 4 s roll so the window REACHES the neighbouring
# delivery and has to be truncated at it — that is the mechanism they exist to
# test, and it can only be reached with the ceiling raised above the requested
# roll. `wide_window` raises it; the tests that want the production default
# leave it alone, and `TestRollCeilingsBoundTheWindow` covers the ceiling itself.
def wide_window(cfg: PipelineConfig, seconds: float = 6.0) -> PipelineConfig:
    return replace(
        cfg,
        segmentation=replace(
            cfg.segmentation,
            max_pre_roll_seconds=seconds,
            max_post_roll_seconds=seconds,
        ),
    )


class ClipIsolationCase(unittest.TestCase):
    """Runs a real single-pass analysis over a real tagged video."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.tags = [(*A_WINDOW, TAG_A), (*B_WINDOW, TAG_B)]
        self.n_frames = 320

    def tearDown(self):
        self._tmp.cleanup()

    def analyse(self, b_window, config=None, n_frames=None):
        self.tags = [(*A_WINDOW, TAG_A), (*b_window, TAG_B)]
        self.n_frames = n_frames or self.n_frames
        video = make_tagged_video(
            self.dir / "source.mp4",
            n_frames=self.n_frames,
            tags=self.tags,
            width=W, height=H, fps=FPS, background=BG,
        )
        self.video = video
        paths = Paths(analysis_id="test_clip_isolation").ensure()
        detector = StubDetector([A_WINDOW, b_window])
        service = TrackingService(config or PipelineConfig())
        result = service.analyse(str(video), paths, detector=detector)
        return result

    def tags_in_clip(self, delivery) -> list[str]:
        """Attribute every decoded frame of a written clip to a delivery."""
        greys = decode_grey_means(Path(delivery.clip_path))
        return [classify_tag(g, BG, self.tags) for g in greys]


class TestPostRollStopsAtNextDelivery(ClipIsolationCase):
    """
    The case named in the requirement:

        A's requested post-roll crosses into B
            =>  A's clip ends at B.detection_start - 1
            =>  A's clip contains no frame belonging to B
    """

    def setUp(self):
        super().setUp()
        # 4 s post-roll = 100 frames. A is lost at ~54, so A would run to 154,
        # straight through B's ball at 130. The ceiling is raised above 4 s so
        # the request survives and the window really does reach B.
        self.cfg = wide_window(replace(PipelineConfig(), post_roll_seconds=4.0))
        self.result = self.analyse(B_WINDOW, config=self.cfg)
        self.assertEqual(
            len(self.result.deliveries), 2,
            f"expected two deliveries, got {len(self.result.deliveries)}",
        )
        self.a, self.b = self.result.deliveries

    def test_a_clip_ends_exactly_one_frame_before_b_is_detected(self):
        self.assertIsNotNone(self.a.clip)
        self.assertIsNotNone(self.b.clip)
        self.assertEqual(
            self.a.clip.clip_end_frame,
            self.b.detection_start_frame - 1,
            f"A's clip ends at {self.a.clip.clip_end_frame} but B's ball is "
            f"detected at {self.b.detection_start_frame}",
        )

    def test_a_clip_contains_no_frame_belonging_to_b(self):
        """The proof the requirement asks for, read off the written file."""
        observed = self.tags_in_clip(self.a)
        self.assertNotIn(
            f"delivery_{TAG_B}", observed,
            f"A's clip contains {observed.count(f'delivery_{TAG_B}')} frame(s) "
            f"from delivery B",
        )
        self.assertNotIn(
            "unknown", observed,
            "A's clip contains frames that match no source delivery — the "
            "application appears to have generated frames of its own",
        )

    def test_a_clip_truncation_is_recorded_not_hidden(self):
        self.assertEqual(
            self.a.clip.truncated_before_frame, self.b.detection_start_frame
        )
        # The post-roll actually granted is reported separately from the one
        # requested, so a consumer can see it was shortened.
        self.assertLess(
            self.a.clip.post_roll_actual_frames, self.a.clip.post_roll_frames
        )

    def test_b_clip_still_contains_b_itself(self):
        """Truncation must not be so aggressive that B loses its own ball."""
        observed = self.tags_in_clip(self.b)
        self.assertIn(
            f"delivery_{TAG_B}", observed,
            "B's clip lost its own ball — the window was cut on the wrong side",
        )

    def test_a_clip_ends_no_later_than_it_would_have_without_the_fix(self):
        """Guards the fix from being silently reverted to unbounded post-roll."""
        unbounded = self.a.clip.clip_end_frame
        self.assertLess(
            unbounded, self.b.clip.clip_start_frame,
            "A's and B's clips overlap",
        )


class TestPreRollStopsAtPreviousDelivery(ClipIsolationCase):
    """
    The other direction: B's pre-roll must not open inside A's post-roll.

    This is the overlap the Phase 3 run actually exhibited — delivery 19's clip
    was 2211-2379 and delivery 20's was 2351-2531, i.e. 29 shared frames.
    """

    def setUp(self):
        super().setUp()
        # A 4 s pre-roll is what makes B's window reach back into A's footage, which is
        # the overlap under test. The ceiling is raised so the request survives.
        self.cfg = wide_window(replace(PipelineConfig(), pre_roll_seconds=4.0,
                                       post_roll_seconds=2.0))
        self.result = self.analyse(B_WINDOW_FAR, config=self.cfg,
                                   n_frames=400)
        self.assertEqual(len(self.result.deliveries), 2)
        self.a, self.b = self.result.deliveries

    def test_b_clip_starts_after_a_clip_ends(self):
        self.assertGreaterEqual(
            self.b.clip.clip_start_frame, self.a.clip.clip_end_frame + 1,
            f"B starts at {self.b.clip.clip_start_frame} but A's clip runs to "
            f"{self.a.clip.clip_end_frame}",
        )

    def test_b_clip_contains_no_frame_belonging_to_a(self):
        observed = self.tags_in_clip(self.b)
        self.assertNotIn(
            f"delivery_{TAG_A}", observed,
            f"B's clip contains {observed.count(f'delivery_{TAG_A}')} frame(s) "
            f"from delivery A's ball",
        )

    def test_b_start_clamp_is_recorded(self):
        self.assertTrue(self.b.clip.start_clamped_by_previous)

    def test_b_clip_does_not_start_before_its_own_detection(self):
        """A clamped start may never precede the delivery it belongs to."""
        self.assertLessEqual(
            self.b.clip.clip_start_frame, self.b.detection_start_frame
        )


class TestClipsAreDisjointAndOrdered(ClipIsolationCase):
    """The general invariant, over a run with several deliveries."""

    def _three_deliveries(self):
        windows = [(31, 45), (130, 145), (240, 255)]
        tags = [(w[0], w[1], 210 - i * 60) for i, w in enumerate(windows)]
        video = make_tagged_video(
            self.dir / "source.mp4", n_frames=520, tags=tags,
            width=W, height=H, fps=FPS, background=BG,
        )
        paths = Paths(analysis_id="test_clip_isolation").ensure()
        result = TrackingService(PipelineConfig()).analyse(
            str(video), paths, detector=StubDetector(windows)
        )
        return result, tags

    def test_every_pair_of_clips_is_disjoint(self):
        result, _ = self._three_deliveries()
        clips = [d.clip for d in result.deliveries if d.clip]
        self.assertGreaterEqual(len(clips), 3)
        for i in range(len(clips) - 1):
            self.assertLess(
                clips[i].clip_end_frame, clips[i + 1].clip_start_frame,
                f"clip {i} ({clips[i].clip_start_frame}-{clips[i].clip_end_frame}) "
                f"overlaps clip {i + 1} "
                f"({clips[i + 1].clip_start_frame}-{clips[i + 1].clip_end_frame})",
            )

    def test_no_clip_contains_another_deliverys_ball(self):
        """Every clip must be free of every OTHER delivery's ball frames."""
        result, tags = self._three_deliveries()
        self.assertGreaterEqual(len(result.deliveries), 3)
        for d in result.deliveries:
            own_start = d.detection_start_frame
            for other in result.deliveries:
                if other.delivery_id == d.delivery_id:
                    continue
                overlap = not (
                    d.clip.clip_end_frame < other.detection_start_frame
                    or d.clip.clip_start_frame > other.detection_end_frame
                )
                self.assertFalse(
                    overlap,
                    f"delivery {d.delivery_id}'s clip "
                    f"{d.clip.clip_start_frame}-{d.clip.clip_end_frame} overlaps "
                    f"delivery {other.delivery_id}'s ball "
                    f"{other.detection_start_frame}-{other.detection_end_frame}",
                )

    def test_each_clip_carries_only_its_own_delivery_or_background(self):
        result, tags = self._three_deliveries()
        for d in result.deliveries:
            own = next(t for t in tags
                       if d.detection_start_frame >= t[0]
                       and d.detection_end_frame <= t[1])
            greys = decode_grey_means(Path(d.clip_path))
            observed = {classify_tag(g, BG, tags) for g in greys}
            allowed = {"background", f"delivery_{own[2]}"}
            self.assertTrue(
                observed <= allowed,
                f"delivery {d.delivery_id}'s clip contains {observed - allowed}",
            )


class TestRollCeilingsBoundTheWindow(ClipIsolationCase):
    """
    The isolation above relied on TRUNCATION: a wide window was cut back at the
    neighbour. The resolved ceilings remove the need for that by refusing the
    wide window in the first place, which is the stronger guarantee — it holds
    even when nothing announces the neighbour in time to truncate.

    These tests pin the ceiling as a bound that a caller cannot exceed, including
    when the caller asks for more than policy allows.
    """

    def _ceiling_config(self, pre, post, cap_pre, cap_post):
        return replace(
            PipelineConfig(), pre_roll_seconds=pre, post_roll_seconds=post,
            segmentation=replace(
                PipelineConfig().segmentation,
                max_pre_roll_seconds=cap_pre, max_post_roll_seconds=cap_post,
            ),
        )

    def test_window_never_exceeds_the_post_roll_ceiling(self):
        cfg = self._ceiling_config(4.0, 6.0, 3.0, 1.0)
        result = self.analyse(B_WINDOW, config=cfg, n_frames=320)
        cap = int(round(cfg.segmentation.max_post_roll_seconds * FPS))
        for d in result.deliveries:
            self.assertLessEqual(
                d.clip.clip_end_frame - d.ball_lost_frame, cap,
                f"delivery {d.delivery_id} runs {d.clip.clip_end_frame - d.ball_lost_frame} "
                f"frames past ball-loss, above the {cap}-frame ceiling",
            )

    def test_window_never_exceeds_the_pre_roll_ceiling(self):
        cfg = self._ceiling_config(6.0, 2.0, 1.0, 3.0)
        result = self.analyse(B_WINDOW_FAR, config=cfg, n_frames=400)
        cap = int(round(cfg.segmentation.max_pre_roll_seconds * FPS))
        for d in result.deliveries:
            self.assertLessEqual(
                d.detection_start_frame - d.clip.clip_start_frame, cap,
                f"delivery {d.delivery_id} reaches back "
                f"{d.detection_start_frame - d.clip.clip_start_frame} frames "
                f"before its first detection, above the {cap}-frame ceiling",
            )

    def test_a_ceiling_that_alone_prevents_overlap_beats_truncation(self):
        """
        With the production ceiling, A's window ends before B's ball even
        appears — so nothing needs to be cut, and the isolation holds with a
        frame to spare.
        """
        result = self.analyse(B_WINDOW, config=PipelineConfig(), n_frames=320)
        self.assertEqual(len(result.deliveries), 2)
        a, b = result.deliveries
        self.assertLess(
            a.clip.clip_end_frame, b.detection_start_frame,
            "A's clip reaches B's ball, so the ceiling did not bound the window",
        )
        self.assertIsNone(
            a.clip.truncated_before_frame,
            "the ceiling already stopped the window, so no cut was needed",
        )

    def test_configured_roll_and_applied_ceiling_are_both_reported(self):
        """
        A shortened window must be visible as such. If the ceiling silently
        truncated the roll, the recorded reasoning would claim the wider window
        the caller actually asked for.
        """
        cfg = self._ceiling_config(4.0, 6.0, 3.0, 1.0)
        result = self.analyse(B_WINDOW, config=cfg, n_frames=320)
        window = result.deliveries[0].event["evidence"]["clip_window"]
        self.assertEqual(
            window["configured_post_roll_frames"],
            int(round(cfg.post_roll_seconds * FPS)),
        )
        self.assertEqual(
            window["max_post_roll_frames"],
            int(round(cfg.segmentation.max_post_roll_seconds * FPS)),
        )
        self.assertLess(
            window["max_post_roll_frames"],
            window["configured_post_roll_frames"],
            "this test is meaningless unless the ceiling binds",
        )


class TestNoApplicationInsertedFrames(ClipIsolationCase):
    """
    'No black filler generated by our application' — checked on pixels.

    A clip whose frames do not correspond one-to-one with a contiguous run of
    source frames contains either dropped or invented frames. Both destroy the
    original_frame -> clip_frame mapping the overlay depends on.
    """

    def test_clip_length_equals_its_declared_window(self):
        result = self.analyse(B_WINDOW, config=wide_window(replace(
            PipelineConfig(), post_roll_seconds=4.0)))
        for d in result.deliveries:
            self.assertIsNotNone(d.clip)
            self.assertEqual(
                d.clip.frames_written,
                d.clip.clip_end_frame - d.clip.clip_start_frame + 1,
            )
            self.assertEqual(
                len(decode_grey_means(Path(d.clip_path))),
                d.clip.clip_end_frame - d.clip.clip_start_frame + 1,
                f"delivery {d.delivery_id}: file length disagrees with window",
            )

    def test_frame_map_survives_truncation(self):
        """clip_frame must round-trip exactly even after the window was cut."""
        result = self.analyse(B_WINDOW, config=wide_window(replace(
            PipelineConfig(), post_roll_seconds=4.0)))
        for d in result.deliveries:
            fm = d.clip
            self.assertTrue(fm.frame_map_valid, fm.frame_map_error)
            self.assertEqual(fm.frames_written, fm.frame_count)
            inside = [
                p for p in d.trajectory.points
                if fm.contains_original_frame(p.original_frame)
            ]
            for p in inside:
                self.assertEqual(
                    p.clip_frame, fm.to_clip_frame(p.original_frame)
                )
                self.assertEqual(
                    fm.to_original_frame(p.clip_frame), p.original_frame
                )

    def test_points_outside_a_truncated_window_carry_no_clip_frame(self):
        """Truncation must clear stale clip_frame values, not leave them."""
        result = self.analyse(B_WINDOW, config=wide_window(replace(
            PipelineConfig(), post_roll_seconds=4.0)))
        a = result.deliveries[0]
        self.assertIsNotNone(a.clip.truncated_before_frame)
        for p in a.trajectory.points:
            if a.clip.contains_original_frame(p.original_frame):
                self.assertIsNotNone(p.clip_frame)
            else:
                self.assertIsNone(
                    p.clip_frame,
                    f"point at frame {p.original_frame} is outside the clip "
                    f"window {a.clip.clip_start_frame}-{a.clip.clip_end_frame} "
                    f"but still carries clip_frame {p.clip_frame}",
                )

    def test_original_frame_numbers_remain_the_source_of_truth(self):
        result = self.analyse(B_WINDOW, config=wide_window(replace(
            PipelineConfig(), post_roll_seconds=4.0)))
        for d in result.deliveries:
            for p in d.trajectory.points:
                self.assertIsInstance(p.original_frame, int)
                self.assertGreaterEqual(p.original_frame, 1)
                self.assertLessEqual(
                    p.original_frame, result.frames_decoded
                )


class TestNoRetrackingWasIntroduced(ClipIsolationCase):
    """The fix must be window arithmetic only — no second detector pass."""

    def test_detector_runs_once_per_frame_and_source_opens_once(self):
        from tests.helpers import CountingVideoCapture

        tags = [(*A_WINDOW, TAG_A), (*B_WINDOW, TAG_B)]
        video = make_tagged_video(
            self.dir / "source.mp4", n_frames=320, tags=tags,
            width=W, height=H, fps=FPS, background=BG,
        )
        paths = Paths(analysis_id="test_clip_isolation").ensure()
        detector = StubDetector([A_WINDOW, B_WINDOW])
        counter = CountingVideoCapture().install()
        try:
            result = TrackingService(PipelineConfig()).analyse(
                str(video), paths, detector=detector
            )
        finally:
            counter.restore()
        self.assertEqual(counter.count_prefix(str(video)), 1)
        self.assertEqual(detector.calls, result.frames_decoded)
        self.assertEqual(
            {d.tracking_source for d in result.deliveries}, {"single_pass"}
        )


if __name__ == "__main__":
    unittest.main()
