"""Ring buffer: eviction, retrieval, gap reporting."""

from __future__ import annotations

import unittest

import numpy as np

from app.video.frame_buffer import FrameRingBuffer


def frame(value: int = 128, h: int = 32, w: int = 48) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestRingBufferBasics(unittest.TestCase):
    def test_push_and_get_roundtrip(self):
        rb = FrameRingBuffer(capacity_frames=10, max_bytes=10_000_000)
        f = frame(200)
        self.assertTrue(rb.push(1, f))
        out = rb.get(1)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, f.shape)
        # JPEG is lossy, so allow a small tolerance rather than exact equality.
        self.assertLess(abs(int(out[0, 0, 0]) - 200), 6)

    def test_get_missing_returns_none_not_a_substitute(self):
        rb = FrameRingBuffer(capacity_frames=5, max_bytes=10_000_000)
        self.assertIsNone(rb.get(99))

    def test_iter_frames_is_ascending_and_skips_holes(self):
        rb = FrameRingBuffer(capacity_frames=10, max_bytes=10_000_000)
        for n in (1, 2, 3, 5, 6):
            rb.push(n, frame(n * 10))
        got = [n for n, _ in rb.iter_frames(1, 6)]
        self.assertEqual(got, [1, 2, 3, 5, 6])

    def test_missing_frames_reports_holes_explicitly(self):
        rb = FrameRingBuffer(capacity_frames=10, max_bytes=10_000_000)
        for n in (1, 2, 5):
            rb.push(n, frame())
        self.assertEqual(rb.missing_frames(1, 5), [3, 4])
        self.assertFalse(rb.has_all(1, 5))
        self.assertTrue(rb.has_all(1, 2))


class TestRingBufferEviction(unittest.TestCase):
    def test_frame_cap_evicts_oldest(self):
        rb = FrameRingBuffer(capacity_frames=3, max_bytes=100_000_000)
        for n in range(1, 6):
            rb.push(n, frame())
        self.assertEqual(len(rb), 3)
        self.assertIsNone(rb.get(1))
        self.assertIsNotNone(rb.get(5))
        self.assertEqual(rb.first_frame, 3)

    def test_byte_cap_independent_of_frame_cap(self):
        # Capacity allows far more frames than the byte budget will hold.
        rb = FrameRingBuffer(capacity_frames=10_000, max_bytes=20_000)
        for n in range(1, 200):
            rb.push(n, frame(120, h=64, w=64))
        self.assertLessEqual(rb.byte_size, 20_000)
        self.assertLess(len(rb), 200)
        self.assertGreater(rb.evicted_count, 0)

    def test_trim_before_drops_older_frames(self):
        rb = FrameRingBuffer(capacity_frames=100, max_bytes=10_000_000)
        for n in range(1, 11):
            rb.push(n, frame())
        dropped = rb.trim_before(6)
        self.assertEqual(dropped, 5)
        self.assertEqual(rb.first_frame, 6)

    def test_repeated_push_of_same_frame_replaces_without_corrupting_bytes(self):
        rb = FrameRingBuffer(capacity_frames=10, max_bytes=10_000_000)
        rb.push(7, frame(50))
        before = rb.byte_size
        rb.push(7, frame(50))
        self.assertEqual(len(rb), 1)
        self.assertEqual(rb.byte_size, before)

    def test_stats_are_self_consistent(self):
        rb = FrameRingBuffer(capacity_frames=4, max_bytes=10_000_000)
        for n in range(1, 9):
            rb.push(n, frame())
        s = rb.stats()
        self.assertEqual(s["encoded_count"], 8)
        self.assertEqual(s["evicted_count"], 4)
        self.assertEqual(s["resident_frames"], 4)
        self.assertEqual(s["first_frame"], 5)
        self.assertEqual(s["last_frame"], 8)
        self.assertGreater(s["peak_bytes"], 0)


class TestRingBufferGuards(unittest.TestCase):
    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            FrameRingBuffer(capacity_frames=0)
        with self.assertRaises(ValueError):
            FrameRingBuffer(max_bytes=0)
        with self.assertRaises(ValueError):
            FrameRingBuffer(jpeg_quality=0)
        with self.assertRaises(ValueError):
            FrameRingBuffer(jpeg_quality=101)

    def test_jpeg_is_far_smaller_than_raw_for_camera_like_content(self):
        """The reason this class exists: 1080p raw would be ~1 GiB for 180 frames."""
        raw_bytes = 1920 * 1080 * 3
        # Real broadcast frames are smooth; model that with a gradient plus mild
        # noise rather than uniform random, which is JPEG's worst case.
        grad = np.tile(
            np.linspace(0, 255, 1920, dtype=np.uint8)[None, :, None], (1080, 1, 3)
        )
        noisy = np.clip(
            grad.astype(np.int16)
            + np.random.default_rng(0).integers(-6, 7, grad.shape),
            0, 255,
        ).astype(np.uint8)

        rb = FrameRingBuffer(capacity_frames=10, max_bytes=100_000_000, jpeg_quality=92)
        rb.push(1, noisy)
        jpeg_bytes = rb.byte_size
        self.assertLess(jpeg_bytes, raw_bytes / 5)

    def test_byte_cap_survives_incompressible_content(self):
        """Pure noise is JPEG's worst case; the byte cap, not the codec, is the
        real guarantee that memory stays bounded."""
        rb = FrameRingBuffer(capacity_frames=1000, max_bytes=3_000_000)
        rng = np.random.default_rng(1)
        for n in range(1, 40):
            rb.push(n, rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8))
        self.assertLessEqual(rb.byte_size, 3_000_000)

    def test_frame_larger_than_the_whole_budget_is_refused(self):
        """Otherwise the documented ceiling would not actually be a ceiling."""
        rb = FrameRingBuffer(capacity_frames=100, max_bytes=1_000)
        rng = np.random.default_rng(2)
        oversize = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
        self.assertFalse(rb.push(1, oversize))
        self.assertEqual(len(rb), 0)
        self.assertEqual(rb.byte_size, 0)
        self.assertEqual(rb.dropped_oversize_count, 1)
        # A small frame still goes in afterwards: the buffer is not poisoned.
        self.assertTrue(rb.push(2, frame(50, h=16, w=16)))


if __name__ == "__main__":
    unittest.main()