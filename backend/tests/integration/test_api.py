"""
API surface: the six routes, their errors, and the document they serve.

`test_api_contract.py` pins the route inventory and the progress semantics;
this module pins what each route answers for a real analysis, and that the
diagnostics the API no longer serves are still written to disk.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.core.config import Paths
from app.pipeline.analysis_pipeline import UnifiedPipeline
from app.storage.persistence import read_json
from tests.helpers import StubDetector, make_video

W, H, FPS = 320, 180, 25.0


def client_for() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


class TestHealth(unittest.TestCase):
    def setUp(self):
        self.client = client_for()

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_health_says_which_versions_it_is_serving(self):
        body = self.client.get("/api/health").json()
        self.assertIn("schema_version", body)
        self.assertIn("pipeline_version", body)
        self.assertIn("analyses_tracked_in_memory", body)


class TestRemovedSurfaceStaysRemoved(unittest.TestCase):
    """
    Nothing in the old surface may come back through a different spelling.

    A route that is not in the contract must 404 rather than answer with
    something a client could accidentally depend on.
    """

    def setUp(self):
        self.client = client_for()

    def test_status_of_an_unknown_analysis_is_a_clean_404(self):
        r = self.client.get("/api/analysis/does_not_exist/status")
        self.assertEqual(r.status_code, 404)
        self.assertIn("detail", r.json())

    def test_result_of_an_unknown_analysis_is_a_clean_404(self):
        r = self.client.get("/api/analysis/does_not_exist/result")
        self.assertEqual(r.status_code, 404)
        self.assertIn("detail", r.json())

    def test_the_legacy_plural_spelling_is_not_routed(self):
        for path in (
            "/api/analyses/does_not_exist/status",
            "/api/analyses/does_not_exist/result",
        ):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_progress_cancellation_and_reference_are_gone(self):
        for path in (
            "/api/analysis/does_not_exist/progress",
            "/api/analysis/does_not_exist/progress/json",
            "/api/reference",
            "/api/analyses",
        ):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_cancellation_cannot_be_requested(self):
        r = self.client.post("/api/analysis/does_not_exist/cancel")
        self.assertEqual(r.status_code, 404)

    def test_the_data_directory_is_not_mounted(self):
        """Clip bytes leave through the clips route, never a static tree."""
        r = self.client.get("/data/runs/does_not_exist/clips/delivery_001.mp4")
        self.assertEqual(r.status_code, 404)

    def test_a_clip_name_traversal_is_rejected(self):
        for bad in ("delivery_001.mov", "etc_passwd"):
            r = self.client.get(f"/api/analysis/x/clips/{bad}")
            self.assertEqual(r.status_code, 400, f"accepted {bad}")


class TestPersistedDocument(unittest.TestCase):
    """Run one real end-to-end analysis, then assert the persisted document.

    The served envelope has its own suite in
    `test_result_frontend_response.py`; these tests cover `result.json`, which
    is where summary, calibration, performance and the full per-delivery
    records live — on disk, reachable by no route.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = Path(cls._tmp.name)
        video = make_video(d / "source.mp4", n_frames=200, width=W, height=H, fps=FPS)
        cls.paths = Paths(analysis_id="schema_test").ensure()
        # Inject the stub detector: this test is about the persisted schema, and
        # the real model finds no ball in a synthetic grey video.
        pipeline = UnifiedPipeline(detector=StubDetector([(30, 60)]))
        # Shot model is skipped: this test is about the stored shape.
        pipeline._build_classifier = lambda emit, result: None  # type: ignore
        cls.result = pipeline.run(
            source_video=str(video), paths=cls.paths, progress=None
        )
        cls.result.shot_model_available = False
        UnifiedPipeline._persist(cls.paths, cls.result)
        cls.document = read_json(cls.paths.result_json)
        cls.client = client_for()

    @classmethod
    def tearDownClass(cls):
        import shutil

        from app.core.config import analysis_root

        shutil.rmtree(analysis_root(cls.paths.analysis_id), ignore_errors=True)
        cls._tmp.cleanup()

    def test_the_document_is_reachable_only_from_disk(self):
        """Neither the served envelope nor any route exposes it wholesale."""
        self.assertIsNotNone(self.document)
        self.assertEqual(
            self.client.get(
                f"/api/analysis/{self.paths.analysis_id}/result/internal"
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                f"/api/analysis/{self.paths.analysis_id}/tracking"
            ).status_code,
            404,
        )

    def test_result_has_a_summary(self):
        doc = self.document
        self.assertIn("summary", doc)
        self.assertIn("deliveries", doc["summary"])
        self.assertIn("shots", doc["summary"])

    def test_result_carries_the_calibration_block(self):
        """Speed is an estimate and the document says so, next to the number."""
        self.assertTrue(self.document["calibration"]["speed_is_estimate"])
        self.assertFalse(self.document["calibration"]["speed_is_calibrated"])

    def test_reports_one_source_decode(self):
        self.assertEqual(self.document["performance"]["source_decodes"], 1)

    def test_speed_fields_distinguish_estimate_from_measurement(self):
        for d in self.document["deliveries"]:
            b = d["bowling"]
            self.assertIn("speed_kmh", b)
            self.assertIn("speed_calibrated", b)
            self.assertIn("speed_imputed", b)
            self.assertFalse(b["speed_calibrated"])
            if b["speed_kmh"] is None:
                self.assertFalse(b["speed_imputed"])

    def test_the_stored_clip_urls_point_at_the_public_clips_route(self):
        """Never a filesystem path and never a static mount."""
        for d in self.document["deliveries"]:
            url = d.get("clip_url")
            if url is None:
                continue
            self.assertTrue(
                url.startswith(f"/api/analysis/{self.paths.analysis_id}/clips/"),
                url,
            )


if __name__ == "__main__":
    unittest.main()
