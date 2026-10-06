"""
test_result_frontend_response.py — What `GET /api/analysis/{id}/result` serves.

The contract this pins down:

  * the body is an ARRAY of deliveries, so a client does
    `deliveries[currentIndex]` for Previous/Next and gets that delivery's video,
    trajectory, bounce, bowling analytics, shot and quality state in one object;
  * every entry has the identical schema — a client never has to check which
    keys exist;
  * no internal/debug object and no filesystem path is exposed;
  * `source` still distinguishes detected from predicted points;
  * nulls stay null when a value could not legitimately be computed;
  * `clip_url` actually serves video bytes;
  * the internal document, with all of its detail, is still available on
    `/result/internal` — nothing was deleted to make the array clean.

The pipeline is exercised end to end (stub detector, synthetic video), so this
is the served response, not a hand-built fixture.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from app.api.routes import router
from app.core.config import Paths
from app.pipeline.analysis_pipeline import UnifiedPipeline
from app.schemas.frontend import DeliveryOut
from tests.helpers import StubDetector, make_video

W, H, FPS = 320, 180, 25.0

TOP_LEVEL_KEYS = {
    "delivery_id",
    "delivery_ref",
    "video",
    "trajectory",
    "bounce",
    "bowling",
    "shot",
    "quality",
}

# Anything the internal document carries that must never reach a browser.
FORBIDDEN_KEYS = {
    "event", "validation", "provenance", "boundary", "evidence", "clip_window",
    "purity", "merge_checks", "weak_confirmation", "reacquired_frames",
    "confirmed_frame_ranges", "predicted_frame_ranges", "clip_path",
    "clip_frame", "original_frame", "speed_samples", "overlay", "summary",
    "quarantine", "performance", "config", "warnings", "errors",
    "class_probabilities", "detection_start_frame", "detection_end_frame",
    "ball_lost_frame", "release_frame", "frame_offset", "gaps",
    "event_ranges", "tracking_source", "index",
}

WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/]")


def walk(node, path="$"):
    """Yield (path, key, value) for every object key in a JSON document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, value
            yield from walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from walk(value, f"{path}[{i}]")


class TestFrontendResultArray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        video = make_video(
            Path(cls._tmp.name) / "source.mp4", n_frames=400, width=W, height=H, fps=FPS
        )
        cls.paths = Paths(analysis_id="frontend_schema_test").ensure()
        pipeline = UnifiedPipeline(detector=StubDetector([(30, 70), (200, 240)]))
        pipeline._build_classifier = lambda emit, result: None  # type: ignore
        cls.internal = pipeline.run(
            source_video=str(video), paths=cls.paths, progress=None
        )
        app = FastAPI()
        app.include_router(router, prefix="/api")
        # main.py mounts DATA_DIR at /data, which makes the stored
        # `/data/runs/<id>/clips/...` URLs resolve. Tests write to the temp runs
        # directory instead, so `/data/runs` is mounted there to reproduce the
        # same URL → file mapping a browser sees in production.
        app.mount(
            "/data/runs",
            StaticFiles(directory=str(cls.paths.root.parent), check_dir=False),
            name="data",
        )
        cls.client = TestClient(app)
        cls.body = cls.client.get(
            f"/api/analysis/{cls.paths.analysis_id}/result"
        ).json()

    @classmethod
    def tearDownClass(cls):
        import shutil

        from app.core.config import analysis_root

        shutil.rmtree(analysis_root(cls.paths.analysis_id), ignore_errors=True)
        cls._tmp.cleanup()

    # ── The array requirement ────────────────────────────────────────────────
    def test_the_response_is_an_array(self):
        self.assertIsInstance(self.body, list)

    def test_the_array_holds_every_served_delivery_in_order(self):
        self.assertEqual(
            [d["delivery_id"] for d in self.body],
            [d.delivery_id for d in self.internal.deliveries],
        )

    def test_previous_next_needs_nothing_beyond_the_array_index(self):
        """Index ± 1 yields the neighbouring delivery, complete."""
        for index, delivery in enumerate(self.body):
            self.assertEqual(set(delivery), TOP_LEVEL_KEYS, index)
            # The fields a previous/next swap depends on are all here.
            self.assertIsInstance(delivery["video"]["clip_url"], str)
            self.assertIsInstance(delivery["trajectory"]["points"], list)
            self.assertIn("bounce", delivery)
            self.assertIn("bowling", delivery)
            self.assertIn("shot", delivery)
            self.assertIn("quality", delivery)

    def test_every_delivery_has_the_identical_schema(self):
        first = self.body[0]
        for delivery in self.body[1:]:
            self.assertEqual(set(delivery), set(first))
            for block in ("video", "trajectory", "bounce", "bowling", "shot",
                          "quality"):
                self.assertEqual(
                    set(delivery[block]), set(first[block]), delivery["delivery_id"]
                )
            self.assertEqual(
                set(delivery["trajectory"]["points"][0]),
                set(first["trajectory"]["points"][0]),
            )

    def test_the_schema_matches_the_documented_model(self):
        for delivery in self.body:
            DeliveryOut.model_validate(delivery)  # raises on any stray key

    # ── Nothing internal leaks ───────────────────────────────────────────────
    def test_no_internal_key_is_present_anywhere(self):
        for path, key, _ in walk(self.body):
            self.assertNotIn(key, FORBIDDEN_KEYS, f"{path}.{key}")

    def test_no_filesystem_path_is_exposed(self):
        import json

        text = json.dumps(self.body)
        self.assertIsNone(WINDOWS_PATH.search(text), "a Windows path leaked")
        self.assertNotIn("\\\\", text)
        self.assertNotIn("clip_path", text)

    def test_the_internal_document_is_untouched_by_the_serializer(self):
        """
        The point of the architecture: cleaning the response must not clean the
        pipeline. Everything the serializer drops is still served by
        `/result/internal` for diagnostics.
        """
        internal = self.client.get(
            f"/api/analysis/{self.paths.analysis_id}/result/internal"
        ).json()
        self.assertEqual(internal["analysis_id"], self.paths.analysis_id)
        self.assertIn("summary", internal)
        self.assertIn("quarantine", internal)
        self.assertIn("performance", internal)
        self.assertIn("calibration", internal)
        for delivery in internal["deliveries"]:
            self.assertIn("event", delivery)
            self.assertIn("validation", delivery)
            self.assertIn("provenance", delivery)
            self.assertIn("clip_path", delivery)
            self.assertTrue(delivery["event"] or delivery["validation"])

        # …and the two documents describe the same deliveries.
        self.assertEqual(
            [d["delivery_id"] for d in internal["deliveries"]],
            [d["delivery_id"] for d in self.body],
        )

    # ── Trajectory ───────────────────────────────────────────────────────────
    def test_points_keep_the_detected_predicted_distinction(self):
        sources = set()
        for delivery in self.body:
            traj = delivery["trajectory"]
            self.assertEqual(
                traj["detected_points"] + traj["predicted_points"],
                len(traj["points"]),
                delivery["delivery_id"],
            )
            for point in traj["points"]:
                self.assertIn(point["source"], ("detected", "predicted"))
                self.assertEqual(set(point),
                                 {"frame", "x", "y", "source", "confidence"})
                sources.add(point["source"])
                if point["source"] == "detected":
                    self.assertIsInstance(point["confidence"], float)
            self.assertIn(traj["continuity"], ("continuous", "gapped"))
        self.assertIn("detected", sources, "no real measurement survived")

    def test_point_frames_are_source_frames_inside_the_clip(self):
        for delivery in self.body:
            video = delivery["video"]
            if video["start_frame"] is None:
                continue
            for point in delivery["trajectory"]["points"]:
                self.assertGreaterEqual(point["frame"], video["start_frame"])
                self.assertLessEqual(point["frame"], video["end_frame"])

    # ── Bounce / bowling / shot ──────────────────────────────────────────────
    def test_the_bounce_block_has_only_the_documented_fields(self):
        for delivery in self.body:
            self.assertEqual(
                set(delivery["bounce"]),
                {"detected", "frame", "x", "y", "ground_x_m", "ground_y_m"},
            )

    def test_uncalibrated_analytics_are_null_not_invented(self):
        """This run has no pitch corners: nothing derived from the ground plane."""
        for delivery in self.body:
            bowling = delivery["bowling"]
            self.assertIsNone(bowling["speed_kmh"])
            self.assertIsNone(bowling["line"])
            self.assertIsNone(bowling["length"])
            self.assertIsNone(bowling["swing"])
            self.assertIsNone(bowling["bounce_angle"])
            self.assertFalse(bowling["calibrated"])
            self.assertIn("release_angle", bowling)  # pixel-space: still computed

    def test_shot_is_null_when_the_classifier_did_not_run(self):
        for delivery in self.body:
            self.assertIsNone(delivery["shot"]["classification"])
            self.assertIsNone(delivery["shot"]["confidence"])

    # ── Quality ──────────────────────────────────────────────────────────────
    def test_quality_is_four_concise_fields(self):
        for delivery in self.body:
            quality = delivery["quality"]
            self.assertEqual(
                set(quality),
                {"trajectory_valid", "tracking_confidence",
                 "requires_human_review", "flags"},
            )
            self.assertIsInstance(quality["trajectory_valid"], bool)
            self.assertIsInstance(quality["requires_human_review"], bool)
            self.assertIsInstance(quality["flags"], list)
            for flag in quality["flags"]:
                self.assertIsInstance(flag, str)
                self.assertTrue(flag)
            if quality["tracking_confidence"] is not None:
                self.assertGreaterEqual(quality["tracking_confidence"], 0.0)
                self.assertLessEqual(quality["tracking_confidence"], 1.0)

    def test_validation_is_never_served_verbatim(self):
        import json

        text = json.dumps(self.body)
        for verbose in (
            "event_ranges", "start_margin_frames", "end_margin_frames",
            "own_event_index", "event_count_estimate", "clip_start",
        ):
            self.assertNotIn(verbose, text, verbose)

    # ── Media ────────────────────────────────────────────────────────────────
    def test_clip_url_is_a_url_for_every_delivery(self):
        for delivery in self.body:
            url = delivery["video"]["clip_url"]
            self.assertIsInstance(url, str, delivery["delivery_id"])
            self.assertTrue(url.startswith("/"), url)
            self.assertFalse(WINDOWS_PATH.search(url), url)

    def test_clip_url_serves_the_video_bytes(self):
        for delivery in self.body:
            response = self.client.get(delivery["video"]["clip_url"])
            self.assertEqual(response.status_code, 200, delivery["video"]["clip_url"])
            self.assertTrue(
                response.headers["content-type"].startswith("video/"),
                response.headers["content-type"],
            )

    def test_video_block_describes_a_playable_clip(self):
        for delivery in self.body:
            video = delivery["video"]
            self.assertIsInstance(video["start_frame"], int)
            self.assertIsInstance(video["end_frame"], int)
            self.assertGreater(video["end_frame"], video["start_frame"])
            self.assertEqual(video["fps"], FPS)
            self.assertEqual(video["width"], W)
            self.assertEqual(video["height"], H)
            frames = video["end_frame"] - video["start_frame"] + 1
            self.assertAlmostEqual(
                video["duration_seconds"], round(frames / FPS, 3), places=3
            )

    # ── Routes ───────────────────────────────────────────────────────────────
    def test_the_legacy_plural_route_returns_the_same_array(self):
        plural = self.client.get(
            f"/api/analyses/{self.paths.analysis_id}/result"
        ).json()
        self.assertIsInstance(plural, list)
        self.assertEqual(plural, self.body)

    def test_an_unknown_analysis_is_still_404(self):
        self.assertEqual(
            self.client.get("/api/analysis/nope/result").status_code, 404
        )
        self.assertEqual(
            self.client.get("/api/analysis/nope/result/internal").status_code, 404
        )

    def test_a_still_running_analysis_is_409_not_a_partial_array(self):
        from app.api.deps import get_jobs

        job = get_jobs().create(source_name="later.mp4")
        job.status = "running"
        response = self.client.get(f"/api/analysis/{job.analysis_id}/result")
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
