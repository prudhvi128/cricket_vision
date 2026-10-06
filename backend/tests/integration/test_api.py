"""API surface: routes, reference data, SSE shape, result schema."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.routes import router
from app.core.config import Paths
from app.reference import reference_document
from tests.helpers import StubDetector, make_video
from app.pipeline.analysis_pipeline import UnifiedPipeline

W, H, FPS = 320, 180, 25.0


class TestReferenceEndpoint(unittest.TestCase):
    """The reference route exists because the two merged frontends disagreed
    on the enum strings."""

    def setUp(self):
        from fastapi import FastAPI

        self.app = FastAPI()
        self.app.include_router(router, prefix="/api")
        self.client = TestClient(self.app)

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_reference_publishes_the_authoritative_labels(self):
        doc = self.client.get("/api/reference").json()
        self.assertIn("Cover", doc["shot_classes"])
        self.assertEqual(len(doc["shot_classes"]), 10)
        # "Leg Side"/"Off Side" — the backend's strings, not the frontend's
        # "Leg"/"Off".
        self.assertIn("Leg Side", doc["line_labels"])
        self.assertIn("Off Side", doc["line_labels"])
        self.assertNotIn("Leg", doc["line_labels"])

    def test_reference_exposes_calibration_facts(self):
        doc = self.client.get("/api/reference").json()
        cal = doc["calibration"]
        self.assertTrue(cal["speed_is_estimate"])
        self.assertFalse(cal["speed_is_calibrated"])
        self.assertIn("speed_scale", cal)
        self.assertIn("speed_offset_kmh", cal)

    def test_reference_explains_provenance_vocabularies(self):
        doc = self.client.get("/api/reference").json()
        self.assertEqual(set(doc["trajectory_sources"]), {"detected", "predicted"})
        self.assertEqual(
            set(doc["tracking_sources"]), {"single_pass", "clip_fallback"}
        )


class TestCancelRouteExists(unittest.TestCase):
    """Upstream's frontend called /api/cancel, which did not exist."""

    def setUp(self):
        from fastapi import FastAPI

        self.app = FastAPI()
        self.app.include_router(router, prefix="/api")
        self.client = TestClient(self.app)

    def test_cancel_returns_a_honest_answer_not_a_404(self):
        r = self.client.post("/api/analyses/does_not_exist/cancel")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["cancel_requested"])
        self.assertIn("detail", body)

    def test_status_for_unknown_analysis_is_explicit(self):
        body = self.client.get("/api/analyses/nope/status").json()
        self.assertEqual(body["status"], "unknown")

    def test_result_for_unknown_analysis_is_404(self):
        self.assertEqual(
            self.client.get("/api/analyses/nope/result").status_code, 404
        )

    def test_clip_name_traversal_is_rejected(self):
        for bad in ("../../secret", "delivery_001.mov", "etc_passwd"):
            r = self.client.get(f"/api/analyses/x/clips/{bad}")
            self.assertIn(r.status_code, (400, 404), f"accepted {bad}")


class TestProgressStreamShape(unittest.TestCase):
    def test_progress_is_a_real_event_stream(self):
        """The merged frontend used EventSource against a JSON route."""
        from fastapi import FastAPI

        from app.api.deps import JobRegistry, get_jobs

        app = FastAPI()
        app.include_router(router, prefix="/api")
        jobs: JobRegistry = get_jobs()
        job = jobs.create(source_name="t.mp4")
        job.status = "completed"
        job.phase = "complete"

        client = TestClient(app)
        with client.stream("GET", f"/api/analyses/{job.analysis_id}/progress") as r:
            self.assertEqual(r.status_code, 200)
            self.assertTrue(
                r.headers["content-type"].startswith("text/event-stream"),
                r.headers["content-type"],
            )
            body = "".join(r.iter_text())
        self.assertIn("event: progress", body)
        self.assertIn("event: done", body)
        self.assertIn("data: ", body)
        # Each data line must be valid JSON on its own.
        for line in body.splitlines():
            if line.startswith("data: "):
                json.loads(line[6:])


class TestResultSchema(unittest.TestCase):
    """Run one real end-to-end analysis, then assert the persisted document.

    The frontend-facing `/result` array has its own suite in
    `test_result_frontend_response.py`; these tests cover the internal document
    that `/result/internal` serves, which is where summary, calibration,
    performance and the full per-delivery records still live.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = Path(cls._tmp.name)
        video = make_video(d / "source.mp4", n_frames=200, width=W, height=H, fps=FPS)
        cls.paths = Paths(analysis_id="schema_test").ensure()
        # Inject the stub detector: this test is about the served schema, and
        # the real model finds no ball in a synthetic grey video.
        pipeline = UnifiedPipeline(detector=StubDetector([(30, 60)]))
        # Shot model is skipped: this test is about the served shape.
        pipeline._build_classifier = lambda emit, result: None  # type: ignore
        cls.result = pipeline.run(
            source_video=str(video), paths=cls.paths, progress=None
        )
        cls.result.shot_model_available = False
        UnifiedPipeline._persist(cls.paths, cls.result)

    @classmethod
    def tearDownClass(cls):
        import shutil

        from app.core.config import analysis_root

        shutil.rmtree(analysis_root(cls.paths.analysis_id), ignore_errors=True)
        cls._tmp.cleanup()

    def setUp(self):
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.mount("/data", __import__(
            "fastapi.staticfiles", fromlist=["StaticFiles"]
        ).StaticFiles(directory=str(self.paths.root.parent), check_dir=False),
            name="data")
        self.client = TestClient(app)

    def test_result_has_a_summary(self):
        doc = self.client.get(
            f"/api/analyses/{self.paths.analysis_id}/result/internal"
        ).json()
        self.assertIn("summary", doc)
        self.assertIn("deliveries", doc["summary"])
        self.assertIn("shots", doc["summary"])

    def test_result_carries_the_calibration_block(self):
        """A client rendering speed_kmh receives the facts that make it an
        estimate in the same payload."""
        doc = self.client.get(
            f"/api/analyses/{self.paths.analysis_id}/result/internal"
        ).json()
        self.assertTrue(doc["calibration"]["speed_is_estimate"])
        self.assertFalse(doc["calibration"]["speed_is_calibrated"])

    def test_reports_one_source_decode(self):
        doc = self.client.get(
            f"/api/analyses/{self.paths.analysis_id}/result/internal"
        ).json()
        self.assertEqual(doc["performance"]["source_decodes"], 1)

    def test_tracking_endpoint_excludes_shot_labels(self):
        doc = self.client.get(
            f"/api/analyses/{self.paths.analysis_id}/tracking"
        ).json()
        for d in doc["deliveries"]:
            self.assertIsNone(d["shot"])

    def test_trajectory_endpoint_returns_the_stored_points(self):
        doc = self.client.get(
            f"/api/analyses/{self.paths.analysis_id}/trajectory/1"
        ).json()
        self.assertIn("trajectory", doc)
        pts = doc["trajectory"]["points"]
        self.assertGreater(len(pts), 0)
        for p in pts[:5]:
            self.assertIn(p["source"], ("detected", "predicted"))
            self.assertIsInstance(p["original_frame"], int)
            self.assertIsInstance(p["x"], int)
            self.assertIsInstance(p["y"], int)

    def test_pitch_map_fields_are_nullable_not_fake(self):
        """A null bounce must be null, never a fabricated coordinate."""
        doc = self.client.get(
            f"/api/analyses/{self.paths.analysis_id}/trajectory/1"
        ).json()
        b = doc["bounce"]
        if b["x_px"] is None:
            self.assertIsNone(b["y_px"])
        if b["x_norm"] is not None:
            self.assertGreaterEqual(b["x_norm"], 0.0)
            self.assertLessEqual(b["x_norm"], 1.0)

    def test_speed_fields_distinguish_estimate_from_measurement(self):
        doc = self.client.get(
            f"/api/analyses/{self.paths.analysis_id}/result/internal"
        ).json()
        for d in doc["deliveries"]:
            b = d["bowling"]
            self.assertIn("speed_kmh", b)
            self.assertIn("speed_calibrated", b)
            self.assertIn("speed_imputed", b)
            self.assertFalse(b["speed_calibrated"])
            if b["speed_kmh"] is None:
                self.assertFalse(b["speed_imputed"])


if __name__ == "__main__":
    unittest.main()