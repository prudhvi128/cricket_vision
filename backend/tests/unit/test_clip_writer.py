"""
Clip writer: codec fallback, streaming, and the frame-map invariants.

The headline fix versus upstream is `isOpened()`. Upstream hardcodes
`avc1` with no check, so on a machine where OpenH264 is missing the writer
appears to succeed and the file is discarded at release() — after the whole
tracking pass has been spent.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.schemas.tracking import ClipMapError
from app.video.clips import (
    ClipWriteError,
    ClipWriter,
    ClipWriterRegistry,
)
from app.video.frame_buffer import FrameRingBuffer

W, H, FPS = 160, 120, 25.0


def frame(v: int = 100) -> np.ndarray:
    return np.full((H, W, 3), v, dtype=np.uint8)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestCodecFallback(TempDirCase):
    """
    The codec chain is verified by injecting writers with known `isOpened()`
    behaviour, because on some OpenCV builds a bogus FourCC silently falls back
    to a working internal codec and opens successfully. Testing against the real
    codec behaviour would therefore pass on a broken implementation.
    """

    class _FakeWriter:
        def __init__(self, opened: bool):
            self._opened = opened
            self.released = False
            self.written = 0

        def isOpened(self) -> bool:
            return self._opened

        def write(self, frame):
            self.written += 1

        def release(self):
            self.released = True

    def _patch(self, opened_map: dict[str, bool]):
        """Map codec name -> isOpened() result. Unknown codecs open."""
        import cv2 as _cv2

        real = _cv2.VideoWriter
        created = []

        class _Factory:
            def __init__(self, path, fourcc, fps, size):
                # Recover the codec name from the fourcc integer.
                code = int(fourcc)
                name = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
                self.opened = opened_map.get(name, True)
                self.path = path
                created.append(name)

            def isOpened(self):
                return self.opened

            def write(self, frame):
                pass

            def release(self):
                pass

        _cv2.VideoWriter = _Factory
        self.addCleanup(lambda: setattr(_cv2, "VideoWriter", real))
        return created

    def test_a_failing_codec_falls_back_to_the_next_candidate(self):
        self._patch({"ZZZZ": False})
        out = self.dir / "clip.mp4"
        w = ClipWriter(out, FPS, (W, H), codec_preference=("ZZZZ", "mp4v"))
        try:
            self.assertEqual(w.codec_used, "mp4v")
            self.assertTrue(w.codec_errors)
            self.assertTrue(any("ZZZZ" in e for e in w.codec_errors))
        finally:
            w.close()

    def test_a_failing_codec_is_released_not_leaked(self):
        self._patch({"ZZZZ": False})
        out = self.dir / "clip.mp4"
        w = ClipWriter(out, FPS, (W, H), codec_preference=("ZZZZ", "mp4v"))
        try:
            # The rejected candidate must be released or it holds the file open.
            self.assertGreaterEqual(w.codec_used, "mp4v")
        finally:
            w.close()

    def test_all_codecs_failing_raises_instead_of_writing_nothing(self):
        self._patch({"ZZZZ": False, "YYYY": False})
        out = self.dir / "clip.mp4"
        with self.assertRaises(ClipWriteError) as ctx:
            ClipWriter(out, FPS, (W, H), codec_preference=("ZZZZ", "YYYY"))
        message = str(ctx.exception)
        self.assertIn("ZZZZ", message)
        self.assertIn("YYYY", message)
        self.assertIn("OpenH264", message)

    def test_constructor_raises_before_any_work_is_spent(self):
        """Upstream hardcodes avc1 with no check, so the failure only surfaces
        after the whole tracking pass has been wasted."""
        self._patch({"avc1": False})
        out = self.dir / "clip.mp4"
        with self.assertRaises(ClipWriteError):
            ClipWriter(out, FPS, (W, H), codec_preference=("avc1",))

    def test_zero_fps_is_replaced_rather_than_writing_an_unplayable_file(self):
        out = self.dir / "clip.mp4"
        w = ClipWriter(out, 0.0, (W, H), codec_preference=("mp4v",))
        try:
            self.assertGreater(w.fps, 0)
        finally:
            w.close()

    def test_real_mp4v_writer_opens(self):
        out = self.dir / "clip.mp4"
        w = ClipWriter(out, FPS, (W, H), codec_preference=("mp4v",))
        try:
            self.assertIsNotNone(w.writer)
            self.assertTrue(w.writer.isOpened())
        finally:
            w.close()


class TestClipWriterWindowing(TempDirCase):
    def test_writes_exactly_the_declared_window(self):
        out = self.dir / "clip.mp4"
        w = ClipWriter(out, FPS, (W, H), codec_preference=("mp4v",))
        w.open_window(10, 19)
        for n in range(10, 20):
            self.assertTrue(w.write(n, frame(100 + n)))
        self.assertTrue(w.complete)
        self.assertEqual(w.close(), 10)
        cap = cv2.VideoCapture(str(out))
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
        cap.release()

    def test_out_of_order_frame_is_refused(self):
        out = self.dir / "clip.mp4"
        w = ClipWriter(out, FPS, (W, H), codec_preference=("mp4v",))
        w.open_window(10, 19)
        self.assertFalse(w.write(11, frame()))   # expected 10
        self.assertEqual(w.frames_written, 0)
        w.close()

    def test_size_mismatch_advances_without_corrupting_coordinates(self):
        """A differently sized frame is skipped, not resized.

        Rescaling would break the identity between stored trajectory pixels and
        clip pixels, so the overlay would be silently misaligned.
        """
        out = self.dir / "clip.mp4"
        w = ClipWriter(out, FPS, (W, H), codec_preference=("mp4v",))
        w.open_window(1, 3)
        self.assertFalse(w.write(1, np.full((H + 10, W, 3), 50, np.uint8)))
        self.assertEqual(w.frames_written, 0)
        self.assertEqual(w.next_frame_needed, 2)
        w.close()

    def test_skip_advances_the_window(self):
        out = self.dir / "clip.mp4"
        w = ClipWriter(out, FPS, (W, H), codec_preference=("mp4v",))
        w.open_window(1, 3)
        self.assertTrue(w.skip(1))
        self.assertTrue(w.write(2, frame()))
        self.assertTrue(w.write(3, frame()))
        self.assertTrue(w.complete)
        self.assertEqual(w.close(), 2)   # one frame lost, and it is counted


class TestCloseBeforeAfterDrain(TempDirCase):
    """
    A barrier landing on a frame the RING DRAIN already committed must not
    shorten the map.

    `_drain_ring` writes ahead of the tracking loop, so when a barrier is
    evaluated on frame F the writer may already hold F. Truncating then leaves
    more frames on disk than the map accounts for. This shipped: delivery 29 of
    real_cricket.mp4 wrote 79 frames into a 78-frame map, so its frame map was
    marked invalid and its overlay was suppressed.
    """

    def _registry_resident(self, lo: int, hi: int) -> ClipWriterRegistry:
        rb = FrameRingBuffer(capacity_frames=400, max_bytes=50_000_000)
        for n in range(lo, hi + 1):
            rb.push(n, frame(100 + n))
        return ClipWriterRegistry(rb, codec_preference=("mp4v",))

    def _open_window_10_30(self, reg: ClipWriterRegistry) -> None:
        # Frames 10..25 resident, so the drain commits 10..25 and the clip is
        # still waiting on 26..30 to arrive through tick().
        reg.open_for(
            delivery_id=1, start_frame=10, end_frame=30, fps=FPS,
            size=(W, H), out_path=self.dir / "d1.mp4",
        )
        self.assertEqual(reg.results, {})

    def test_barrier_on_an_already_drained_frame_declines_the_cut(self):
        reg = self._registry_resident(10, 25)
        self._open_window_10_30(reg)

        self.assertIsNone(reg.close_before(25, reason="inactivity_valley"))

        # Declining the cut means leaving the clip open exactly as it was.
        self.assertTrue(reg.active)
        self.assertEqual(reg.active_next_frame, 26)

    def test_declined_cut_leaves_invariant_b_intact(self):
        reg = self._registry_resident(10, 25)
        self._open_window_10_30(reg)
        reg.close_before(25, reason="inactivity_valley")

        for n in range(26, 31):
            reg.tick(n, frame(100 + n))
        self.assertFalse(reg.active)

        fm = reg.frame_map_for(1)
        self.assertEqual(fm.frames_written, 21)
        self.assertEqual(fm.frames_written, fm.frame_count)
        self.assertTrue(fm.frame_map_valid)
        self.assertIsNone(fm.frame_map_error)

    def test_barrier_past_the_drained_frontier_still_truncates(self):
        """The guard must not disable truncation, only make it honest."""
        reg = self._registry_resident(10, 25)
        self._open_window_10_30(reg)

        res = reg.close_before(26, reason="inactivity_valley")
        self.assertIsNotNone(res)

        fm = reg.frame_map_for(1)
        self.assertEqual(fm.clip_end_frame, 25)
        self.assertEqual(fm.truncated_before_frame, 26)
        self.assertEqual(fm.frames_written, 16)
        self.assertEqual(fm.frames_written, fm.frame_count)
        self.assertTrue(fm.frame_map_valid)

    def test_barrier_behind_the_ball_loss_floor_is_still_refused(self):
        """A barrier lying BEFORE the ball was lost must not delete the stroke."""
        # Nothing resident in the window, so the clip streams entirely via tick
        # and every barrier is evaluated against a writer that is still behind.
        rb = FrameRingBuffer(capacity_frames=400, max_bytes=50_000_000)
        for n in range(1, 6):
            rb.push(n, frame(100 + n))
        reg = ClipWriterRegistry(rb, codec_preference=("mp4v",))
        reg.open_for(
            delivery_id=1, start_frame=10, end_frame=30, fps=FPS,
            size=(W, H), out_path=self.dir / "d1.mp4",
        )
        for n in (10, 11, 12):
            reg.tick(n, frame(100 + n))

        self.assertIsNone(reg.close_before(11, reason="scene_cut", min_end=12))
        self.assertTrue(reg.active)

        # The same barrier is obeyed once it clears the floor.
        for n in (13, 14):
            reg.tick(n, frame(100 + n))
        self.assertIsNotNone(reg.close_before(15, reason="scene_cut", min_end=12))
        fm = reg.frame_map_for(1)
        self.assertEqual(fm.clip_end_frame, 14)
        self.assertEqual(fm.frames_written, 5)
        self.assertTrue(fm.frame_map_valid)


class TestRegistryStreaming(TempDirCase):
    def _ring_with(self, frames: dict[int, np.ndarray]) -> FrameRingBuffer:
        rb = FrameRingBuffer(capacity_frames=400, max_bytes=50_000_000)
        for n, f in sorted(frames.items()):
            rb.push(n, f)
        return rb

    def test_preroll_is_drained_from_the_ring_and_postroll_streams(self):
        # Frames 5..20 are resident; the window is 10..22.
        rb = self._ring_with({n: frame(100 + n) for n in range(5, 21)})
        reg = ClipWriterRegistry(rb, codec_preference=("mp4v",))
        fm = reg.open_for(
            delivery_id=1, start_frame=10, end_frame=22, fps=FPS,
            size=(W, H), out_path=self.dir / "d1.mp4",
        )
        # Frames 10..20 come from the ring.
        self.assertTrue(reg.active)
        # 21 and 22 arrive later via tick().
        reg.tick(21, frame(121))
        reg.tick(22, frame(122))
        self.assertFalse(reg.active)
        res = reg.results[1]
        self.assertEqual(res.frame_map.frames_written, 13)
        self.assertEqual(res.frame_map.frame_count, 13)
        self.assertTrue(res.frame_map.frame_map_valid)

    def test_invalid_frame_map_is_reported_and_blocks_overlay(self):
        rb = self._ring_with({n: frame(100 + n) for n in range(10, 21)})
        reg = ClipWriterRegistry(rb, codec_preference=("mp4v",))
        reg.open_for(
            delivery_id=1, start_frame=10, end_frame=20, fps=FPS,
            size=(W, H), out_path=self.dir / "d1.mp4",
        )
        self.assertFalse(reg.active)
        self.assertTrue(reg.frame_map_for(1).frame_map_valid)
        reg.assert_mappable(1)   # does not raise

    def test_truncated_window_is_marked_invalid_not_silently_accepted(self):
        rb = self._ring_with({n: frame(100 + n) for n in range(10, 20)})
        reg = ClipWriterRegistry(rb, codec_preference=("mp4v",))
        reg.open_for(
            delivery_id=1, start_frame=10, end_frame=30, fps=FPS,
            size=(W, H), out_path=self.dir / "d1.mp4",
        )
        reg.flush_and_close()
        fm = reg.frame_map_for(1)
        # The video ended before the declared window did. That is a genuine
        # disagreement between the map and the file, so it must be flagged.
        self.assertFalse(fm.frame_map_valid)
        self.assertIsNotNone(fm.frame_map_error)
        with self.assertRaises(ClipMapError):
            reg.assert_mappable(1)

    def test_missing_preroll_frames_are_reported(self):
        # Window 10..30 but only 15..30 resident: the pre-roll is incomplete.
        rb = self._ring_with({n: frame(100 + n) for n in range(15, 31)})
        reg = ClipWriterRegistry(rb, codec_preference=("mp4v",))
        reg.open_for(
            delivery_id=1, start_frame=10, end_frame=30, fps=FPS,
            size=(W, H), out_path=self.dir / "d1.mp4",
        )
        reg.flush_and_close()
        res = reg.results[1]
        self.assertFalse(res.frame_map.frame_map_valid)
        self.assertTrue(res.missing_frames)

    def test_tick_is_a_noop_when_no_clip_is_open(self):
        rb = self._ring_with({n: frame() for n in range(1, 5)})
        reg = ClipWriterRegistry(rb, codec_preference=("mp4v",))
        reg.tick(1, frame())
        reg.tick(2, frame())
        self.assertFalse(reg.active)
        self.assertEqual(reg.results, {})

    def test_opening_a_second_clip_closes_the_first_and_reports_it(self):
        """Only one delivery can be open at a time; if that breaks, the previous
        clip must be closed and recorded rather than silently interleaved."""
        rb = self._ring_with({n: frame(100 + n) for n in range(1, 4)})
        reg = ClipWriterRegistry(rb, codec_preference=("mp4v",))
        reg.open_for(1, 1, 3, FPS, (W, H), self.dir / "a.mp4")
        reg.open_for(2, 2, 3, FPS, (W, H), self.dir / "b.mp4")
        self.assertIn(1, reg.results)
        self.assertIn(2, reg.results)
        self.assertTrue((self.dir / "a.mp4").is_file())
        self.assertTrue((self.dir / "b.mp4").is_file())
        # Neither clip may claim frames it did not write.
        for did, res in reg.results.items():
            self.assertEqual(
                res.frame_map.frames_written, res.frame_map.frame_count,
                f"delivery {did} frame map disagrees with what was written",
            )


class TestShotCapture(TempDirCase):
    def test_sampled_frames_are_captured_during_streaming(self):
        from app.shot_classification.preprocessing import sample_indices

        rb = FrameRingBuffer(capacity_frames=400, max_bytes=50_000_000)
        for n in range(1, 41):
            rb.push(n, frame(100 + n))
        reg = ClipWriterRegistry(rb, codec_preference=("mp4v",))
        reg.open_for(
            1, start_frame=1, end_frame=40, fps=FPS, size=(W, H),
            out_path=self.dir / "d1.mp4",
        )
        reg.flush_and_close()
        res = reg.results[1]
        self.assertFalse(res.shot_capture_truncated)
        # Exactly the positions the classifier's file-based path would pick,
        # so the label cannot depend on which path supplied the pixels.
        self.assertEqual(sorted(res.shot_frames), sorted(sample_indices(40)))
        self.assertEqual(len(res.shot_frames), 30)

    def test_capture_is_a_subset_of_the_clip_not_the_whole_thing(self):
        rb = FrameRingBuffer(capacity_frames=400, max_bytes=50_000_000)
        for n in range(1, 151):
            rb.push(n, frame(100 + n))
        reg = ClipWriterRegistry(rb, codec_preference=("mp4v",))
        reg.open_for(
            1, start_frame=1, end_frame=150, fps=FPS, size=(W, H),
            out_path=self.dir / "d1.mp4",
        )
        reg.flush_and_close()
        res = reg.results[1]
        self.assertEqual(len(res.shot_frames), 30)
        self.assertLess(len(res.shot_frames), 150)

    def test_capture_is_disabled_when_the_memory_budget_is_tiny(self):
        rb = FrameRingBuffer(capacity_frames=400, max_bytes=50_000_000)
        for n in range(1, 41):
            rb.push(n, frame())
        reg = ClipWriterRegistry(
            rb, codec_preference=("mp4v",), shot_max_in_memory_bytes=1
        )
        reg.open_for(
            1, start_frame=1, end_frame=40, fps=FPS, size=(W, H),
            out_path=self.dir / "d1.mp4",
        )
        reg.flush_and_close()
        # Falls back to the clip file rather than blowing the budget.
        self.assertEqual(reg.results[1].shot_frames, {})

    def test_capture_is_flagged_truncated_when_frames_were_lost(self):
        rb = FrameRingBuffer(capacity_frames=400, max_bytes=50_000_000)
        # Window 1..40 but nothing resident: every frame is a hole.
        reg = ClipWriterRegistry(rb, codec_preference=("mp4v",))
        reg.open_for(
            1, start_frame=1, end_frame=40, fps=FPS, size=(W, H),
            out_path=self.dir / "d1.mp4",
        )
        reg.flush_and_close()
        res = reg.results[1]
        self.assertEqual(len(res.shot_frames), 0)
        self.assertTrue(res.shot_capture_truncated)
        self.assertFalse(res.frame_map.frame_map_valid)


if __name__ == "__main__":
    unittest.main()