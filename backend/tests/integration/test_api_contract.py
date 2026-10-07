"""
test_api_contract.py — The contract the frontend is told to depend on.

Six routes, one spelling each, and nothing else routed. These tests pin the
parts that were previously either absent, duplicated or wrong:

  * the exact route surface — the legacy plural paths, the per-delivery routes,
    the SSE progress route, `/reference` and `/result/internal` are NOT routed;
  * progress that is monotonic, derived from measured units, and published as
    `progress` alone (no legacy `percent`, no unit counters, no event log);
  * the purity gate actually removing impure clips from the delivery list, which
    is what the real run showed was not happening (40 candidates exposed, 3 of
    them MULTIPLE_EVENTS);
  * per-delivery isolation inside the served array — one entry cannot carry
    another delivery's trajectory;
  * uncalibrated analytics reaching the API as null rather than as numbers;
  * uploads rejected at the door, and media that is missing saying so.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import JobRegistry, weighted_progress
from app.api.routes import router
from app.core.config import PipelineConfig, Paths
from app.core.constants import PIPELINE_STAGES, STAGE_PROGRESS_BANDS
from app.pipeline.analysis_pipeline import PipelineResult, UnifiedPipeline
from app.storage.persistence import read_json
from tests.helpers import StubDetector, make_video

W, H, FPS = 320, 180, 25.0

# The whole public surface, as mounted under /api.
DOCUMENTED_ROUTES = {
    "/health",
    "/analyze",
    "/upload",
    "/analysis/{analysis_id}/status",
    "/analysis/{analysis_id}/result",
    "/analysis/{analysis_id}/clips/{name}",
}


def client_for() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


class TestRouteSurface(unittest.TestCase):
    """Exactly the documented routes exist. Not one alias more."""

    def setUp(self):
        self.paths = {
            r.path for r in router.routes if hasattr(r, "methods")
        }

    def test_the_surface_is_exactly_the_documented_six(self):
        self.assertEqual(self.paths, DOCUMENTED_ROUTES)

    def test_removed_routes_are_not_routed(self):
        removed = (
            "/reference",
            "/analyses",
            "/analyses/{analysis_id}/status",
            "/analyses/{analysis_id}/result",
            "/analyses/{analysis_id}/result/internal",
            "/analyses/{analysis_id}/progress",
            "/analyses/{analysis_id}/cancel",
            "/analyses/{analysis_id}/clips/{name}",
            "/analyses/{analysis_id}/overlays/{name}",
            "/analyses/{analysis_id}/tracking",
            "/analysis/{analysis_id}/result/internal",
            "/analysis/{analysis_id}/progress",
            "/analysis/{analysis_id}/progress/json",
            "/analysis/{analysis_id}/cancel",
            "/analysis/{analysis_id}/deliveries",
            "/analysis/{analysis_id}/deliveries/{delivery_id}",
            "/analysis/{analysis_id}/deliveries/{delivery_id}/clip",
            "/analysis/{analysis_id}/deliveries/{delivery_id}/overlay",
            "/analysis/{analysis_id}/trajectory/{delivery_id}",
        )
        for path in removed:
            self.assertNotIn(path, self.paths, path)

    def test_health_publishes_the_stage_vocabulary(self):
        """A client must not have to hardcode stage names to draw a progress bar."""
        body = client_for().get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(tuple(body["stages"]), PIPELINE_STAGES)
        self.assertIn("schema_version", body)
        self.assertIn("pipeline_version", body)


class TestProgressIsMonotonicAndMeasured(unittest.TestCase):
    def test_bands_cover_zero_to_hundred_without_gaps_or_overlap(self):
        ordered = [STAGE_PROGRESS_BANDS[s] for s in PIPELINE_STAGES
                   if s in STAGE_PROGRESS_BANDS]
        self.assertEqual(ordered[0][0], 0.0)
        self.assertEqual(ordered[-1][1], 100.0)
        for (_, prev_high), (next_low, _) in zip(ordered, ordered[1:]):
            self.assertEqual(
                prev_high, next_low,
                "each stage must start where the previous one ended, or a client "
                "sees the bar stall or jump",
            )

    def test_unknown_denominator_reports_the_stage_floor_not_a_guess(self):
        self.assertEqual(weighted_progress("tracking", 5, 0), 5.0)

    def test_progress_follows_the_measured_fraction(self):
        self.assertEqual(weighted_progress("tracking", 0, 100), 5.0)
        self.assertEqual(weighted_progress("tracking", 100, 100), 62.0)
        mid = weighted_progress("tracking", 50, 100)
        self.assertGreater(mid, 5.0)
        self.assertLess(mid, 62.0)

    def test_an_out_of_range_count_is_clamped_not_extrapolated(self):
        """A count beyond the total must not report past the stage's band."""
        self.assertEqual(weighted_progress("tracking", 150, 100), 62.0)
        self.assertEqual(weighted_progress("tracking", -5, 100), 5.0)

    def test_a_real_job_never_goes_backwards_across_stages(self):
        """
        The regression this exists for.

        Each stage counts its own units, so the handoff from tracking (100% of
        frames) to shot classification (0 of N deliveries) computes a *smaller*
        percentage than tracking had reached. Without a high-water mark a client
        drawing `progress` would visibly rewind mid-analysis.
        """
        registry = JobRegistry()
        job = registry.create(source_name="t.mp4")
        job.status = "running"
        cb = registry.progress_cb(job.analysis_id)

        cb("opening", 0, 13565, "loading detector")
        cb("tracking", 13565, 13565, "done")            # band top: 62.0
        self.assertEqual(job.percent, 62.0)
        cb("tracking_complete", 13565, 13565, "verifying")
        self.assertEqual(job.percent, 68.0)
        cb("shot", 0, 40, "classifying")                # would be 68.0, floor
        self.assertEqual(job.percent, 68.0)
        cb("shot", 40, 40, "classified")
        self.assertEqual(job.percent, 86.0)
        cb("overlay", 40, 40, "rendered")
        self.assertEqual(job.percent, 96.0)
        cb("persist", 1, 1, "written")
        self.assertEqual(job.percent, 100.0)
        # …and the published number is the same monotonic value.
        self.assertEqual(job.as_dict()["progress"], 100.0)

    def test_interleaved_work_is_a_substage_not_a_fake_phase(self):
        """
        Segmentation and per-delivery analytics happen inside the pass.

        Reporting them as top-level phases would describe a pipeline with
        separate stages, which this codebase deliberately does not have.
        """
        registry = JobRegistry()
        job = registry.create(source_name="t.mp4")
        job.status = "running"
        cb = registry.progress_cb(job.analysis_id)

        cb("tracking", 500, 13565, "frame 500")
        self.assertEqual(job.stage, "tracking")
        before = job.percent

        cb("delivery_finalise", 3, 3, "delivery 3: analytics")
        self.assertEqual(job.stage, "tracking", "top-level stage must not move")
        self.assertEqual(job.substage, "analytics")
        self.assertEqual(job.percent, before, "a substage cannot move progress")

        cb("purity_validation", 40, 40, "re-segmenting")
        self.assertEqual(job.stage, "tracking")
        self.assertEqual(job.substage, "validation")
        self.assertEqual(job.percent, before)

    def test_the_status_document_publishes_progress_and_nothing_internal(self):
        """
        One number for the bar (`progress`), one vocabulary for the stage, and
        none of the machinery behind them: no legacy `percent`, no unit
        counters, no internal phase names, no event log, no cancel flag.
        """
        registry = JobRegistry()
        job = registry.create(source_name="t.mp4")
        cb = registry.progress_cb(job.analysis_id)
        cb("tracking", 1204, 13565, "frame 1204")
        body = job.as_dict()
        self.assertEqual(
            set(body),
            {
                "analysis_id", "status", "source_name", "created_at",
                "started_at", "finished_at", "stage", "substage", "progress",
                "frames_done", "frames_total", "detail", "error",
            },
        )
        self.assertEqual(body["progress"], 10.1)
        self.assertNotIn("percent", body)
        self.assertNotIn("phase", body)
        self.assertEqual(body["stage"], "tracking")

    def test_status_of_an_unknown_analysis_is_404_on_the_documented_route(self):
        """A typo must not read as 'still processing'."""
        response = client_for().get("/api/analysis/nope/status")
        self.assertEqual(response.status_code, 404)


# ── Purity gate ───────────────────────────────────────────────────────────────
@dataclass
class _FakeRecord:
    delivery_id: int
    classification: str
    reason: str = "because"
    requires_human_review: bool = True


@dataclass
class _FakeValidation:
    records: list = field(default_factory=list)


@dataclass
class _FakeDelivery:
    delivery_id: int
    validation: Optional[dict] = None

    @property
    def delivery_ref(self) -> str:
        return f"delivery_{self.delivery_id:03d}"

    def as_dict(self) -> dict:
        return {"delivery_id": self.delivery_id, "stub": True}


@dataclass
class _FakeTracking:
    deliveries: list = field(default_factory=list)
    validation: Optional[object] = None


class TestPurityGateRemovesImpureClips(unittest.TestCase):
    """
    The real-video run exposed 40 candidates including 3 MULTIPLE_EVENTS.

    Every one of them was served as a delivery. This asserts the gate: impure and
    undecided-by-absence verdicts are quarantined, never served.
    """

    def _partition(self, verdicts, config=None):
        deliveries = [_FakeDelivery(i + 1) for i in range(len(verdicts))]
        tracking = _FakeTracking(
            deliveries=deliveries,
            validation=_FakeValidation(
                records=[
                    _FakeRecord(d.delivery_id, cls)
                    for d, cls in zip(deliveries, verdicts)
                ]
            ),
        )
        pipeline = UnifiedPipeline(config=config or PipelineConfig())
        result = PipelineResult(analysis_id="gate_test")
        return pipeline._partition_by_purity(result, tracking)

    def test_multiple_events_and_no_event_are_quarantined(self):
        accepted, quarantined = self._partition(
            ["SINGLE_EVENT", "MULTIPLE_EVENTS", "AMBIGUOUS", "NO_EVENT"]
        )
        self.assertEqual([d.delivery_id for d in accepted], [1, 3])
        self.assertEqual([q["delivery_id"] for q in quarantined], [2, 4])
        self.assertEqual(
            {q["classification"] for q in quarantined},
            {"MULTIPLE_EVENTS", "NO_EVENT"},
        )

    def test_ambiguous_is_kept_but_flagged_not_claimed_valid(self):
        accepted, quarantined = self._partition(["AMBIGUOUS"])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(quarantined, [])

    def test_an_unvalidated_clip_is_not_silently_accepted(self):
        """Silence is not a pass. No verdict at all means quarantine."""
        pipeline = UnifiedPipeline()
        tracking = _FakeTracking(
            deliveries=[_FakeDelivery(1)], validation=_FakeValidation(records=[])
        )
        accepted, quarantined = pipeline._partition_by_purity(
            PipelineResult(analysis_id="x"), tracking
        )
        self.assertEqual(accepted, [])
        self.assertEqual(len(quarantined), 1)
        self.assertIsNone(quarantined[0]["classification"])
        self.assertTrue(quarantined[0]["requires_human_review"])

    def test_the_gate_honours_a_custom_accepted_set(self):
        """A caller that wants only unambiguous deliveries gets exactly that."""
        accepted, quarantined = self._partition(
            ["SINGLE_EVENT", "AMBIGUOUS"],
            config=PipelineConfig(accepted_validation_classes=("SINGLE_EVENT",)),
        )
        self.assertEqual([d.delivery_id for d in accepted], [1])
        self.assertEqual([q["delivery_id"] for q in quarantined], [2])

    def test_the_quarantine_record_keeps_the_whole_delivery(self):
        """'Why is delivery 7 missing' must be answerable from the artefact."""
        _, quarantined = self._partition(["MULTIPLE_EVENTS"])
        record = quarantined[0]["record"]
        self.assertTrue(record["stub"], "the full delivery must be retained")
        self.assertEqual(quarantined[0]["reason"], "because")
        self.assertTrue(quarantined[0]["requires_human_review"])


class TestQuarantinedRecordsAreNotServedAsDeliveries(unittest.TestCase):
    """End to end: a quarantined id never appears in the served array."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.video = make_video(
            Path(cls._tmp.name) / "source.mp4", n_frames=220, width=W, height=H, fps=FPS
        )
        cls.paths = Paths(analysis_id="quarantine_test").ensure()
        pipeline = UnifiedPipeline(detector=StubDetector([(30, 70)]))
        pipeline._build_classifier = lambda emit, result: None  # type: ignore
        cls.result = pipeline.run(
            source_video=str(cls.video), paths=cls.paths, progress=None
        )
        cls.client = client_for()
        cls.served = cls.client.get(
            f"/api/analysis/{cls.paths.analysis_id}/result"
        ).json()["deliveries"]

    @classmethod
    def tearDownClass(cls):
        import shutil

        from app.core.config import analysis_root

        shutil.rmtree(analysis_root(cls.paths.analysis_id), ignore_errors=True)
        cls._tmp.cleanup()

    def test_the_run_produced_at_least_one_delivery_to_test_against(self):
        self.assertTrue(self.result.deliveries)

    def test_every_served_delivery_passed_validation(self):
        served_ids = [d["id"] for d in self.served]
        self.assertEqual(served_ids, [d.delivery_id for d in self.result.deliveries])
        for d in self.served:
            flags = set(d["quality"]["flags"])
            self.assertNotIn("multiple_events_in_clip", flags, d["id"])
            self.assertNotIn("no_event_in_clip", flags, d["id"])

    def test_a_quarantined_id_is_not_in_the_served_array(self):
        if not self.result.quarantined:
            self.skipTest("this synthetic run quarantined nothing")
        served_ids = {d["id"] for d in self.served}
        for record in self.result.quarantined:
            self.assertNotIn(record["delivery_id"], served_ids)

    def test_the_quarantine_list_survives_in_the_persisted_document(self):
        """Diagnostics stay on disk: complete, and unreachable over HTTP."""
        document = read_json(self.paths.result_json)
        self.assertIn("quarantine", document)
        self.assertIn("quarantined", document["summary"])
        self.assertEqual(
            self.client.get(
                f"/api/analysis/{self.paths.analysis_id}/result/internal"
            ).status_code,
            404,
        )


class TestPerDeliveryResponsesCannotMixDeliveries(unittest.TestCase):
    """The failure mode that produced mixed-up shots and trajectories."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.video = make_video(
            Path(cls._tmp.name) / "source.mp4", n_frames=400, width=W, height=H, fps=FPS
        )
        cls.paths = Paths(analysis_id="isolation_test").ensure()
        pipeline = UnifiedPipeline(detector=StubDetector([(30, 70), (200, 240)]))
        pipeline._build_classifier = lambda emit, result: None  # type: ignore
        cls.result = pipeline.run(
            source_video=str(cls.video), paths=cls.paths, progress=None
        )
        cls.client = client_for()
        cls.served = cls.client.get(
            f"/api/analysis/{cls.paths.analysis_id}/result"
        ).json()["deliveries"]

    @classmethod
    def tearDownClass(cls):
        import shutil

        from app.core.config import analysis_root

        shutil.rmtree(analysis_root(cls.paths.analysis_id), ignore_errors=True)
        cls._tmp.cleanup()

    def test_the_fixture_produced_two_deliveries(self):
        self.assertGreaterEqual(len(self.result.deliveries), 2)

    def test_each_served_delivery_carries_only_its_own_trajectory(self):
        ids = [d.delivery_id for d in self.result.deliveries]
        self.assertEqual([d["id"] for d in self.served], ids)
        for index, body in enumerate(self.served):
            for other_index in range(len(self.served)):
                if other_index == index:
                    continue
                self.assertNotEqual(
                    body["trajectory"],
                    self.served[other_index]["trajectory"],
                    f"delivery {body['id']} carries delivery "
                    f"{self.served[other_index]['id']}'s trajectory",
                )

    def test_a_trajectory_never_references_frames_outside_its_own_clip(self):
        for d in self.result.deliveries:
            if not d.clip:
                continue
            lo, hi = d.clip.clip_start_frame, d.clip.clip_end_frame
            for p in d.trajectory.points:
                self.assertGreaterEqual(p.original_frame, lo)
                self.assertLessEqual(p.original_frame, hi)


class TestUncalibratedAnalyticsReachTheApiAsNull(unittest.TestCase):
    """
    The bug that reached production: a 42-199 km/h range from an assumed pitch.

    The default run has no pitch corners, so nothing derived from the ground
    plane may appear in the served document.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.video = make_video(
            Path(cls._tmp.name) / "source.mp4", n_frames=220, width=W, height=H, fps=FPS
        )
        cls.paths = Paths(analysis_id="uncalibrated_test").ensure()
        pipeline = UnifiedPipeline(detector=StubDetector([(30, 70)]))
        pipeline._build_classifier = lambda emit, result: None  # type: ignore
        cls.result = pipeline.run(
            source_video=str(cls.video), paths=cls.paths, progress=None
        )
        cls.client = client_for()
        cls.served = cls.client.get(
            f"/api/analysis/{cls.paths.analysis_id}/result"
        ).json()["deliveries"]
        cls.document = read_json(cls.paths.result_json)

    @classmethod
    def tearDownClass(cls):
        import shutil

        from app.core.config import analysis_root

        shutil.rmtree(analysis_root(cls.paths.analysis_id), ignore_errors=True)
        cls._tmp.cleanup()

    def test_the_fixture_produced_a_delivery(self):
        self.assertTrue(self.result.deliveries)

    def test_no_served_delivery_carries_a_speed_line_length_or_swing(self):
        for delivery in self.served:
            b = delivery["bowling"]
            self.assertIsNone(b["speed_kmh"], delivery["id"])
            self.assertIsNone(b["length"], delivery["id"])
            self.assertIsNone(b["line"], delivery["id"])
            self.assertIsNone(b["swing"], delivery["id"])
            self.assertFalse(b["calibrated"])

    def test_no_bounce_carries_ground_plane_metres(self):
        for delivery in self.served:
            bounce = delivery["bounce"]
            self.assertIsNone(bounce["ground_x_m"], delivery["id"])
            self.assertIsNone(bounce["ground_y_m"], delivery["id"])
            # Pixels are real and must survive.
            self.assertIsNotNone(bounce["x"], delivery["id"])

    def test_the_summary_reports_a_null_mean_rather_than_guessing(self):
        self.assertIsNone(self.document["summary"]["mean_speed_kmh"])
        self.assertEqual(self.document["summary"]["with_speed"], 0)

    def test_the_persisted_config_records_the_uncalibrated_state(self):
        self.assertFalse(self.document["config"]["geometry_calibrated"])
        self.assertIsNone(self.document["config"]["pitch_corners_px"])

    def test_a_calibrated_config_is_recorded_as_calibrated(self):
        """
        The suppression must be reachable in both directions.

        A result document that understates its own calibration would be as wrong
        as one that overstates it, so the corners have to survive into the
        persisted document.
        """
        cfg = PipelineConfig(pitch_corners_px={
            "TL": (40, 20), "TR": (280, 20), "BL": (90, 160), "BR": (230, 160),
        })
        self.assertTrue(cfg.geometry_calibrated)
        document = self.result.result_document(self.paths, cfg)
        calibration = document["calibration"]
        self.assertTrue(calibration["geometry_calibrated"])
        self.assertTrue(calibration["ground_plane_analytics_available"])
        self.assertEqual(calibration["geometry_source"], "per_video_pitch_corners")
        self.assertEqual(
            calibration["pitch_corners_px"]["TL"], [40.0, 20.0]
        )
        # D2 still applies: measured corners make the ground plane meaningful, but
        # the speed is still a least-squares estimate over a hand-tuned scale, so
        # it must not be advertised as calibrated.
        self.assertFalse(calibration["speed_is_calibrated"])
        self.assertTrue(document["config"]["geometry_calibrated"])
        self.assertEqual(
            document["config"]["pitch_corners_px"]["TL"], [40.0, 20.0]
        )

    def test_partial_corners_do_not_count_as_calibrated(self):
        cfg = PipelineConfig(pitch_corners_px={"TL": (40, 20), "BR": (230, 160)})
        self.assertFalse(
            cfg.geometry_calibrated,
            "a homography from two corners is not a calibration",
        )


class TestUploadValidation(unittest.TestCase):
    """Rejections happen at the door, not minutes into a tracking pass."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.client = client_for()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_non_video_extension_is_rejected(self):
        bogus = self.dir / "notes.txt"
        bogus.write_bytes(b"this is not a video")
        with open(bogus, "rb") as fh:
            response = self.client.post(
                "/api/analyze", files={"video": ("notes.txt", fh, "text/plain")}
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_a_correctly_named_file_that_is_not_a_video_is_rejected(self):
        """
        The extension check alone passed a text file renamed to .mp4, which then
        failed deep inside the tracking pass with an opaque OpenCV error.
        """
        bogus = self.dir / "fake.mp4"
        bogus.write_bytes(b"definitely not an mp4 container")
        with open(bogus, "rb") as fh:
            response = self.client.post(
                "/api/analyze", files={"video": ("fake.mp4", fh, "video/mp4")}
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("could not be opened as a video", response.json()["detail"])

    def test_an_empty_upload_is_rejected(self):
        empty = self.dir / "empty.mp4"
        empty.write_bytes(b"")
        with open(empty, "rb") as fh:
            response = self.client.post(
                "/api/analyze", files={"video": ("empty.mp4", fh, "video/mp4")}
            )
        self.assertEqual(response.status_code, 400)

    def test_a_rejected_upload_leaves_no_analysis_directory_behind(self):
        from app.core.config import RUNS_DIR

        before = {p.name for p in RUNS_DIR.iterdir()} if RUNS_DIR.exists() else set()
        bogus = self.dir / "fake.mp4"
        bogus.write_bytes(b"not a video")
        with open(bogus, "rb") as fh:
            self.client.post(
                "/api/analyze", files={"video": ("fake.mp4", fh, "video/mp4")}
            )
        after = {p.name for p in RUNS_DIR.iterdir()} if RUNS_DIR.exists() else set()
        self.assertEqual(after - before, set(), "a rejected upload left debris")

    def test_the_other_upload_spelling_is_routed_too(self):
        """`POST /api/upload` with a `file=` field is the same handler."""
        bogus = self.dir / "notes.txt"
        bogus.write_bytes(b"not a video")
        with open(bogus, "rb") as fh:
            response = self.client.post(
                "/api/upload", files={"file": ("notes.txt", fh, "text/plain")}
            )
        self.assertEqual(response.status_code, 400)


class TestMediaRoutesAreHonest(unittest.TestCase):
    """A missing artefact says so; it is never substituted."""

    def setUp(self):
        self.client = client_for()

    def test_a_missing_clip_is_404_with_an_explanation(self):
        response = self.client.get(
            "/api/analysis/does_not_exist/clips/delivery_001.mp4"
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("Clip", response.json()["detail"])

    def test_an_overlay_route_does_not_exist(self):
        """Overlays are internal artefacts; the API serves clips only."""
        response = self.client.get(
            "/api/analysis/does_not_exist/overlays/delivery_001.mp4"
        )
        self.assertEqual(response.status_code, 404)

    def test_a_bad_clip_name_is_a_client_error(self):
        for bad in ("delivery_001.mov", "etc_passwd", "delivery_1.txt"):
            response = self.client.get(f"/api/analysis/x/clips/{bad}")
            self.assertEqual(response.status_code, 400, bad)

    def test_path_traversal_never_reaches_the_filesystem(self):
        for bad in ("..%2F..%2Fsecret", "%2e%2e%2fsecret"):
            response = self.client.get(f"/api/analysis/x/clips/{bad}")
            self.assertIn(response.status_code, (400, 404), bad)


if __name__ == "__main__":
    unittest.main()
