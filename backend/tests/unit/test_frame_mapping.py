"""
Frame mapping: the contract that makes overlay coordinates trustworthy.

If `original_frame -> delivery_id -> clip_frame -> (x, y)` is wrong, the overlay
lands on the wrong ball pixel while looking entirely plausible. These tests
assert the mapping rather than trusting it.
"""

from __future__ import annotations

import unittest

from app.schemas.tracking import ClipFrameMap, Trajectory, TrajectoryPoint


def make_map(start: int, end: int, **kw) -> ClipFrameMap:
    return ClipFrameMap(
        clip_start_frame=start,
        clip_end_frame=end,
        frame_count=0,
        width=320,
        height=180,
        fps=25.0,
        codec="mp4v",
        **kw,
    )


class TestClipFrameMapArithmetic(unittest.TestCase):
    def test_offset_and_frame_count(self):
        fm = make_map(100, 199)
        self.assertEqual(fm.frame_offset, 99)
        self.assertEqual(fm.frame_count, 100)

    def test_first_clip_frame_maps_to_one(self):
        fm = make_map(100, 199)
        self.assertEqual(fm.to_clip_frame(100), 1)

    def test_last_clip_frame_is_the_window_length(self):
        fm = make_map(100, 199)
        self.assertEqual(fm.to_clip_frame(199), 100)

    def test_roundtrip_is_exact_over_the_whole_window(self):
        fm = make_map(37, 400)
        for original in range(fm.clip_start_frame, fm.clip_end_frame + 1):
            self.assertEqual(fm.to_original_frame(fm.to_clip_frame(original)), original)

    def test_contains_original_frame(self):
        fm = make_map(50, 60)
        self.assertTrue(fm.contains_original_frame(50))
        self.assertTrue(fm.contains_original_frame(60))
        self.assertFalse(fm.contains_original_frame(49))
        self.assertFalse(fm.contains_original_frame(61))

    def test_clip_starting_at_frame_one(self):
        fm = make_map(1, 10)
        self.assertEqual(fm.frame_offset, 0)
        self.assertEqual(fm.to_clip_frame(1), 1)
        self.assertEqual(fm.to_original_frame(10), 10)


class TestInvariantB(unittest.TestCase):
    """INVARIANT B: frames_written must equal frame_count."""

    def test_complete_write_keeps_map_valid(self):
        fm = make_map(10, 29)
        fm._frames_written = 20
        fm.finalize()
        self.assertTrue(fm.frame_map_valid)
        self.assertEqual(fm.frames_written, 20)
        self.assertIsNone(fm.frame_map_error)

    def test_short_write_marks_map_invalid(self):
        fm = make_map(10, 29)
        fm._frames_written = 18
        fm.finalize()
        self.assertFalse(fm.frame_map_valid)
        self.assertIn("18", fm.frame_map_error)
        self.assertIn("20", fm.frame_map_error)

    def test_overlong_write_marks_map_invalid(self):
        fm = make_map(10, 29)
        fm._frames_written = 22
        fm.finalize()
        self.assertFalse(fm.frame_map_valid)


class TestAssignClipFrames(unittest.TestCase):
    def test_only_points_inside_the_window_get_clip_frames(self):
        fm = make_map(50, 70)
        traj = Trajectory(points=[
            TrajectoryPoint(original_frame=45, x=1, y=1, source="detected"),
            TrajectoryPoint(original_frame=50, x=2, y=2, source="detected"),
            TrajectoryPoint(original_frame=60, x=3, y=3, source="predicted"),
            TrajectoryPoint(original_frame=70, x=4, y=4, source="detected"),
            TrajectoryPoint(original_frame=75, x=5, y=5, source="detected"),
        ])
        fm.assign_clip_frames(traj)
        # Points outside the window keep real tracking data but have no place
        # in this clip, so clip_frame stays None rather than being faked.
        self.assertIsNone(traj.points[0].clip_frame)
        self.assertEqual(traj.points[1].clip_frame, 1)
        self.assertEqual(traj.points[2].clip_frame, 11)
        self.assertEqual(traj.points[3].clip_frame, 21)
        self.assertIsNone(traj.points[4].clip_frame)

    def test_every_stamped_point_round_trips(self):
        fm = make_map(101, 199)
        traj = Trajectory(points=[
            TrajectoryPoint(original_frame=n, x=n, y=n, source="detected")
            for n in range(101, 200)
        ])
        fm.assign_clip_frames(traj)
        for p in traj.points:
            self.assertIsNotNone(p.clip_frame)
            self.assertEqual(fm.to_original_frame(p.clip_frame), p.original_frame)
            self.assertGreaterEqual(p.clip_frame, 1)
            self.assertLessEqual(p.clip_frame, fm.frame_count)


class TestTrajectoryIntegrity(unittest.TestCase):
    def test_gaps_are_reported_not_filled(self):
        traj = Trajectory(points=[
            TrajectoryPoint(original_frame=10, x=0, y=0, source="detected"),
            TrajectoryPoint(original_frame=11, x=1, y=1, source="predicted"),
            TrajectoryPoint(original_frame=15, x=5, y=5, source="detected"),
            TrajectoryPoint(original_frame=16, x=6, y=6, source="detected"),
        ])
        # Frames 12-14 have no tracked point. Nothing is invented to cover them.
        self.assertEqual(traj.gaps, [[12, 14]])
        self.assertEqual(traj.point_count, 4)
        self.assertEqual(traj.continuity, "gapped")

    def test_continuous_trajectory_has_no_gaps(self):
        traj = Trajectory(points=[
            TrajectoryPoint(original_frame=n, x=n, y=n, source="detected")
            for n in range(100, 110)
        ])
        self.assertEqual(traj.gaps, [])
        self.assertEqual(traj.continuity, "continuous")

    def test_trailing_gap_is_reported_to_the_last_frame(self):
        traj = Trajectory(points=[
            TrajectoryPoint(original_frame=10, x=0, y=0, source="detected"),
            TrajectoryPoint(original_frame=11, x=1, y=1, source="detected"),
            TrajectoryPoint(original_frame=18, x=2, y=2, source="detected"),
        ])
        self.assertEqual(traj.gaps, [[12, 17]])

    def test_multiple_gaps_are_all_reported(self):
        traj = Trajectory(points=[
            TrajectoryPoint(original_frame=n, x=n, y=n, source="detected")
            for n in (1, 2, 6, 7, 20, 21)
        ])
        self.assertEqual(traj.gaps, [[3, 5], [8, 19]])

    def test_detected_points_excludes_predictions(self):
        traj = Trajectory(points=[
            TrajectoryPoint(original_frame=1, x=0, y=0, source="detected"),
            TrajectoryPoint(original_frame=2, x=1, y=1, source="predicted"),
            TrajectoryPoint(original_frame=3, x=2, y=2, source="detected"),
        ])
        self.assertEqual(traj.detected_count, 2)
        self.assertEqual(traj.predicted_count, 1)
        self.assertEqual([p.original_frame for p in traj.detected_points()], [1, 3])


if __name__ == "__main__":
    unittest.main()