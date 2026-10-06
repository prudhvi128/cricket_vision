"""
test_event_horizons.py — the invariant over the hard generalisation cases.

ONE DELIVERY CLIP = EXACTLY ONE BATTING EVENT.

`test_clip_isolation.py` proves the two clips do not overlap when both deliveries
are tracked. This suite covers the cases that made the old windowing fail even
when it *looked* disjoint:

  * a delivery the detector missed entirely (zero detections) between two
    tracked deliveries,
  * a sub-threshold delivery (1..7 detections) between two tracked deliveries,
  * a black-gap transition,
  * a scene cut,
  * a temporary ball occlusion inside one event,
  * several resolutions / frame rates,
  * the first and last delivery in a video.

The tests work at two levels:

  1. Unit tests on `SignalTimeline` / `ActivityWindowTracker` — the barrier
     search itself, isolated and deterministic (no video, no model).
  2. Integration tests through the real single-pass `TrackingService` with a
     stub detector, asserting the written clip frame ranges and the registered
     event horizons.

Nothing here asserts a *number* of deliveries: the count is an output.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Paths, PipelineConfig
from app.segmentation.barriers import (
    ActivityWindowTracker,
    FrameSignal,
    SignalTimeline,
)
from app.tracking.tracking_service import TrackingService

from tests.helpers import StubDetector, make_tagged_video

W, H, FPS = 320, 180, 25.0
BG = 30


# ═══════════════════════════════════════════════════════════════════════════
# 1. Unit tests: the barrier search itself
# ═══════════════════════════════════════════════════════════════════════════
class TestSignalTimelinePreRoll(unittest.TestCase):
    def _timeline(self, frames):
        tl = SignalTimeline(quiet_threshold=1.5, active_threshold=4.0, fps=25.0)
        for f in frames:
            tl.record(f)
        return tl

    def test_black_gap_stops_pre_roll(self):
        # Frames 1..40 black, 41..100 active content; anchor at 100.
        frames = []
        for f in range(1, 41):
            frames.append(FrameSignal(frame=f, is_black=True, motion=0.0))
        for f in range(41, 101):
            frames.append(FrameSignal(frame=f, motion=10.0, confirmed=False))
        tl = self._timeline(frames)
        start, reason = tl.safe_pre_roll_start(100, max_back=75)
        self.assertEqual(reason, "black_gap")
        self.assertEqual(start, 41)  # first frame after the black run

    def test_scene_cut_stops_pre_roll_at_the_cut(self):
        frames = []
        for f in range(1, 61):
            frames.append(FrameSignal(frame=f, motion=10.0))
        # A camera cut between 60 and 61.
        frames.append(FrameSignal(frame=61, is_scene_cut=True, motion=20.0))
        for f in range(62, 101):
            frames.append(FrameSignal(frame=f, motion=10.0))
        tl = self._timeline(frames)
        start, reason = tl.safe_pre_roll_start(100, max_back=75)
        self.assertEqual(reason, "scene_cut")
        self.assertEqual(start, 61)

    def test_exclusion_zone_stops_pre_roll(self):
        frames = [FrameSignal(frame=f, motion=10.0) for f in range(1, 101)]
        tl = self._timeline(frames)
        start, reason = tl.safe_pre_roll_start(
            100, max_back=75, excluded=[(70, 85)]
        )
        self.assertEqual(reason, "exclusion_zone")
        self.assertEqual(start, 86)

    def test_no_barrier_falls_back_to_the_cap(self):
        frames = [FrameSignal(frame=f, motion=10.0) for f in range(1, 201)]
        tl = self._timeline(frames)
        start, reason = tl.safe_pre_roll_start(200, max_back=75)
        self.assertEqual(reason, "max_pre_roll")
        self.assertEqual(start, 125)

    def test_inactivity_valley_stops_after_action(self):
        # Action around 60-80, quiet thereafter; anchor at 100.
        frames = []
        for f in range(1, 61):
            frames.append(FrameSignal(frame=f, motion=0.0))
        for f in range(61, 81):
            frames.append(FrameSignal(frame=f, motion=10.0))  # previous action
        for f in range(81, 101):
            frames.append(FrameSignal(frame=f, motion=0.0))   # reset valley
        tl = self._timeline(frames)
        start, reason = tl.safe_pre_roll_start(100, max_back=75)
        self.assertEqual(reason, "inactivity_valley")
        # Starts in the valley, and never reaches back into the previous action.
        self.assertGreater(start, 80)


class TestSignalTimelinePostRoll(unittest.TestCase):
    def test_black_gap_stops_post_roll(self):
        frames = [FrameSignal(frame=f, motion=10.0) for f in range(1, 121)]
        for f in range(111, 121):
            frames[f - 1] = FrameSignal(frame=f, is_black=True, motion=0.0)
        tl = SignalTimeline(quiet_threshold=1.5, active_threshold=4.0, fps=25.0)
        for fr in frames:
            tl.record(fr)
        end, reason = tl.safe_post_roll_end(100, desired_end=130)
        self.assertEqual(reason, "black_gap")
        self.assertEqual(end, 110)

    def test_running_out_of_known_frames_keeps_the_full_window(self):
        tl = SignalTimeline(fps=25.0)
        for f in range(1, 101):
            tl.record(FrameSignal(frame=f, motion=10.0))
        end, reason = tl.safe_post_roll_end(
            90, desired_end=140, known_frame=100
        )
        self.assertEqual(end, 140)  # not cut to the finalise frame


class TestActivityWindowTracker(unittest.TestCase):
    def test_untracked_window_opens_and_closes(self):
        trk = ActivityWindowTracker(quiet_threshold=1.5, active_threshold=4.0,
                                    quiet_hold_frames=4, min_window_frames=4)
        closed = None
        for f in range(1, 60):
            motion = 25.0 if 20 <= f <= 40 else 0.0
            w = trk.update(f, motion, had_confirmed=False, had_candidate=False)
            if w is not None:
                closed = w
        self.assertIsNotNone(closed)
        self.assertTrue(closed.is_untracked)
        self.assertEqual(closed.start, 20)
        self.assertEqual(closed.end, 40)

    def test_window_with_confirmed_detection_is_not_untracked(self):
        trk = ActivityWindowTracker(quiet_threshold=1.5, active_threshold=4.0,
                                    quiet_hold_frames=4, min_window_frames=4)
        closed = None
        for f in range(1, 60):
            motion = 25.0 if 20 <= f <= 40 else 0.0
            confirmed = f in (30, 31)
            w = trk.update(f, motion, had_confirmed=confirmed, had_candidate=False)
            if w is not None:
                closed = w
        self.assertIsNotNone(closed)
        self.assertFalse(closed.is_untracked)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Integration through the real single pass
# ═══════════════════════════════════════════════════════════════════════════
def write_activity_video(path, n_frames, activity_windows, width=W, height=H,
                         fps=FPS, background=BG, active_value=55):
    """
    Video with strong-but-not-cut motion during `activity_windows` and flat
    background elsewhere. The motion is 25 grey levels, above the activity
    threshold (4.0) and below the scene-cut threshold (35), so it is detected as
    activity without being mistaken for a camera cut.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {path}")

    def in_window(i):
        return any(a <= i <= b for a, b in activity_windows)

    for i in range(1, n_frames + 1):
        if in_window(i):
            # Alternate inside the window so every frame has real inter-frame
            # motion: above the activity threshold (4.0) and below the scene-cut
            # threshold (35), so it is activity without looking like a cut.
            value = active_value if i % 2 == 0 else background
        else:
            value = background
        writer.write(np.full((height, width, 3), value, dtype=np.uint8))
    writer.release()
    return path


class EventHorizonCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def analyse(self, video, windows, config=None, analysis_id="test_event_horizons"):
        paths = Paths(analysis_id=analysis_id).ensure()
        service = TrackingService(config or PipelineConfig())
        return service.analyse(
            str(video), paths, detector=StubDetector(windows)
        )

    # ── Zero-detection untracked event between two tracked deliveries ──────
    def test_untracked_zero_detection_event_is_never_swallowed(self):
        a_window = (31, 45)
        c_window = (200, 215)
        activity = [(120, 170)]
        video = write_activity_video(self.dir / "source.mp4", 260, activity)
        result = self.analyse(video, [a_window, c_window])
        self.assertEqual(len(result.deliveries), 2)

        kinds = [h["kind"] for h in result.event_horizons]
        self.assertIn("UNTRACKED_ACTIVITY_WINDOW", kinds)
        untracked = [
            h for h in result.event_horizons
            if h["kind"] == "UNTRACKED_ACTIVITY_WINDOW"
        ][0]
        self.assertEqual(untracked["start_frame"], 120)
        # The window closes a frame or two after the last active frame (the
        # quiet run has to be observed), so assert the neighbourhood, not an
        # exact edge.
        self.assertGreaterEqual(untracked["end_frame"], 170)
        self.assertLessEqual(untracked["end_frame"], 174)

        c = result.deliveries[1]
        self.assertGreater(
            c.clip.clip_start_frame, untracked["end_frame"],
            "C's clip reached back across the untracked activity window",
        )
        self.assertLessEqual(c.clip.clip_start_frame, c.detection_start_frame)

    # ── Sub-threshold candidate between two tracked deliveries ─────────────
    def test_subthreshold_candidate_is_a_temporal_barrier(self):
        a_window = (31, 45)
        b_window = (120, 126)     # 7 detections: below the minimum of 8
        c_window = (200, 215)
        video = make_tagged_video(
            self.dir / "source.mp4", n_frames=260,
            tags=[(31, 45, 210), (120, 126, 150), (200, 215, 90)],
            background=BG,
        )
        result = self.analyse(video, [a_window, b_window, c_window])
        kinds = [h["kind"] for h in result.event_horizons]
        self.assertIn("REJECTED_EVENT_CANDIDATE", kinds)
        # Only A and C become deliveries; B is a horizon, not a delivery.
        self.assertEqual(len(result.deliveries), 2)
        c = result.deliveries[1]
        self.assertGreater(
            c.clip.clip_start_frame, 126,
            "C's clip reached back across a rejected sub-threshold delivery",
        )

    # ── Black gap is a hard barrier ────────────────────────────────────────
    def test_black_gap_is_never_inside_a_clip(self):
        # background 0 => everything that is not a delivery is a black gap.
        video = make_tagged_video(
            self.dir / "source.mp4", n_frames=260, background=0,
            tags=[(31, 45, 210), (200, 215, 90)],
        )
        result = self.analyse(video, [(31, 45), (200, 215)])
        self.assertEqual(len(result.deliveries), 2)
        for d in result.deliveries:
            greys = []
            cap = cv2.VideoCapture(d.clip_path)
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    greys.append(float(frame.mean()))
            finally:
                cap.release()
            self.assertTrue(greys)
            self.assertTrue(
                all(g > 3.0 for g in greys),
                f"delivery {d.delivery_id}'s clip contains a black frame",
            )

    # ── Temporary occlusion inside ONE event does not split it ─────────────
    def test_temporary_occlusion_stays_one_event(self):
        # Continuous trajectory, but the detector misses frames 60-63.
        class GappyDetector:
            def __init__(self):
                self.model = object()
                self.model_path = "gappy"
                self.calls = 0

            def detect_with_confidence(self, frame):
                self.calls += 1
                f = self.calls
                if 30 <= f <= 95 and not (60 <= f <= 63):
                    t = (f - 30) / 65.0
                    return (int(W * (0.2 + 0.6 * t)), int(H * (0.8 - 0.5 * t)), 0.9)
                return None, None, None

        video = make_tagged_video(
            self.dir / "source.mp4", n_frames=200, background=BG,
            tags=[(30, 95, 210)],
        )
        paths = Paths(analysis_id="test_event_horizons").ensure()
        result = TrackingService(PipelineConfig()).analyse(
            str(video), paths, detector=GappyDetector()
        )
        self.assertEqual(
            len(result.deliveries), 1,
            "a 4-frame occlusion split one delivery into two",
        )
        d = result.deliveries[0]
        self.assertLessEqual(d.detection_start_frame, 32)
        self.assertGreaterEqual(d.detection_end_frame, 93)

    # ── Different frame rate and resolution ────────────────────────────────
    def test_different_fps_and_resolution(self):
        video = make_tagged_video(
            self.dir / "source.mp4", n_frames=300, width=640, height=360,
            fps=50.0, background=BG,
            tags=[(60, 90, 210), (200, 230, 90)],
        )
        result = self.analyse(
            video, [(60, 90), (200, 230)],
            config=PipelineConfig(),
        )
        self.assertEqual(len(result.deliveries), 2)
        clips = [d.clip for d in result.deliveries]
        self.assertLess(clips[0].clip_end_frame, clips[1].clip_start_frame)
        for d in result.deliveries:
            self.assertEqual(d.clip.width, 640)
            self.assertEqual(d.clip.height, 360)

    # ── First and last delivery in a video ─────────────────────────────────
    def test_first_and_last_delivery(self):
        video = make_tagged_video(
            self.dir / "source.mp4", n_frames=240, background=BG,
            tags=[(2, 20, 210), (210, 238, 90)],
        )
        result = self.analyse(video, [(2, 20), (210, 238)])
        self.assertEqual(len(result.deliveries), 2)
        self.assertEqual(result.deliveries[0].clip.clip_start_frame, 1)
        self.assertGreaterEqual(
            result.deliveries[-1].clip.clip_end_frame,
            result.deliveries[-1].detection_end_frame,
        )
        # Clips must stay disjoint and ordered.
        self.assertLess(
            result.deliveries[0].clip.clip_end_frame,
            result.deliveries[-1].clip.clip_start_frame,
        )

    # ── The validator never reports a second event inside a clip ───────────
    def test_no_clip_contains_two_tracked_events(self):
        a, b, c = (31, 45), (130, 145), (240, 255)
        video = make_tagged_video(
            self.dir / "source.mp4", n_frames=320, background=BG,
            tags=[(31, 45, 210), (130, 145, 150), (240, 255, 90)],
        )
        result = self.analyse(video, [a, b, c])
        self.assertIsNotNone(result.validation)
        self.assertEqual(
            len(result.validation.by_classification("MULTIPLE_EVENTS")), 0
        )


if __name__ == "__main__":
    unittest.main()
