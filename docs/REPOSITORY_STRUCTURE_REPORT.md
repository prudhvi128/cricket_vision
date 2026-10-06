# Repository Structure Report

Phase 11 restructuring of the cricket video analysis repository: moving from two
nested source-project wrappers to a flat, domain-separated tree, with no change to
pipeline behaviour.

Date: 2026-10-05

## 1. Motivation

The repository contained three projects in one directory:

- `CricketUnified/backend/` — the production code, itself wrapped in a `CricketUnified/`
  folder that existed only to keep it separate from its two source projects.
- `CricketTracker-main/` — reduced to a frontend and a stale requirements file.
- `CricketShot-Classification-main/` — removed during the preceding cleanup phase.

Inside the production code, all 21 application modules lived in a single
`app/services/` directory with no separation between detection, tracking,
segmentation, shot classification, analytics, video handling, persistence, and
orchestration. `app/api/endpoints.py` alone was 661 lines holding 24 routes and
their error handling. A change to calibration math, an HTTP status code, and a
video probe were all equally likely to appear in the same file, and `utils/io.py`
was a two-purpose grab bag.

## 2. Structure before

```
<root>/
  CricketUnified/
    backend/
      main.py
      app/
        core/{config,constants}.py
        reference.py
        schemas/tracking.py
        utils/io.py
        api/{deps,endpoints}.py            # endpoints.py = 661 lines
        services/
          pipeline.py
          tracking_service.py              # 1457 lines
          analytics.py
          overlay_renderer.py
          ring_buffer.py, clip_writer.py, clip_validator.py
          event_segmenter.py, event_horizons.py, fragment_merger.py
          models.py, shot_classifier.py
          tracker_modules/
            {detector,tracker,trajectory,renderer,utils}.py
      models/
      data/{samples,analyses,uploads}/
      tests/                               # 15 flat modules
      tools/e2e_real_video.py
    tools/
      build_test_fixture.py, real_video_diagnostic.py,
      diagnose_real_deliveries.py, run_real_video_test.py, diagnose_results.py
  CricketTracker-main/
    frontend/ballscribe-insight-main/
    backend/requirements.txt
  docs/
  real_cricket.mp4
```

## 3. Structure after

```
<root>/
  README.md
  requirements.txt
  Dockerfile
  docker-compose.yml
  .gitignore
  real_cricket.mp4
  backend/
    main.py
    models/{ball_detector.pt,shot_classifier.ckpt,README.md}
    app/
      core/{config,constants,logging}.py
      reference.py
      schemas/tracking.py
      pipeline/analysis_pipeline.py
      tracking/{detector,kalman_tracker,tracking_service}.py
      segmentation/{candidate_generation,barriers,fragment_merger,purity_validator}.py
      shot_classification/{model,preprocessing,inference}.py
      analytics/{bowling,trajectory,calibration,pitch_map}.py
      video/{video_io,frame_buffer,clips,overlay,renderer}.py
      storage/persistence.py
      api/
        deps.py, helpers.py
        routes/{system,analysis,deliveries,media}.py
    tests/
      conftest.py    redirects generated output, fixes sys.path
      unit/          10 modules
      integration/   5 modules
      helpers.py
  data/
    README.md
    datasets/source_clips/      41 approved clips + upstream results JSON
    manifests/source_clips.json checksums + provenance
    samples/full_fixture.mp4    generated
    runs/                       per-analysis output, gitignored
    uploads/                    gitignored
  scripts/
    dataset/build_test_fixture.py
    diagnostics/{real_video_diagnostic,diagnose_real_deliveries,run_real_video_test,diagnose_results}.py
    verification/e2e_real_video.py
  frontend/
  docs/
```

## 4. Files moved

| Before | After |
|---|---|
| `CricketUnified/backend/app/services/pipeline.py` | `backend/app/pipeline/analysis_pipeline.py` |
| `CricketUnified/backend/app/services/tracking_service.py` | `backend/app/tracking/tracking_service.py` |
| `.../services/tracker_modules/detector.py` | `backend/app/tracking/detector.py` |
| `.../services/tracker_modules/tracker.py` | `backend/app/tracking/kalman_tracker.py` |
| `.../services/tracker_modules/trajectory.py` | `backend/app/analytics/trajectory.py` |
| `.../services/tracker_modules/renderer.py` | `backend/app/video/renderer.py` |
| `.../services/event_segmenter.py` | `backend/app/segmentation/candidate_generation.py` |
| `.../services/event_horizons.py` | `backend/app/segmentation/barriers.py` |
| `.../services/fragment_merger.py` | `backend/app/segmentation/fragment_merger.py` |
| `.../services/clip_validator.py` | `backend/app/segmentation/purity_validator.py` |
| `.../services/models.py` | `backend/app/shot_classification/model.py` |
| `.../services/shot_classifier.py` | `backend/app/shot_classification/{preprocessing,inference}.py` |
| `.../services/analytics.py` | `backend/app/analytics/bowling.py` |
| `.../services/overlay_renderer.py` | `backend/app/video/overlay.py` |
| `.../services/ring_buffer.py` | `backend/app/video/frame_buffer.py` |
| `.../services/clip_writer.py` | `backend/app/video/clips.py` |
| `.../utils/io.py` | `backend/app/storage/persistence.py` |
| `.../api/endpoints.py` | `backend/app/api/routes/{system,analysis,deliveries,media}.py` + `api/helpers.py` |
| `CricketUnified/backend/tools/e2e_real_video.py` | `scripts/verification/e2e_real_video.py` |
| `CricketUnified/tools/*.py` | `scripts/{dataset,diagnostics}/` |
| `CricketUnified/backend/data/samples/source_clips/` | `data/datasets/source_clips/` |
| `CricketTracker-main/frontend/ballscribe-insight-main/*` | `frontend/*` |
| `CricketTracker-main/backend/requirements.txt` | replaced by root `requirements.txt` |

## 5. Files created

| Path | Purpose |
|---|---|
| `backend/app/core/logging.py` | `configure_logging()`, `CRICKET_LOG_LEVEL` |
| `backend/app/analytics/calibration.py` | homography, `geometry_status`, `calibration_metadata` |
| `backend/app/analytics/pitch_map.py` | `pitch_map_svg()` moved out of the API layer |
| `backend/app/shot_classification/preprocessing.py` | `IMAGE_TRANSFORM`, `sample_indices`, `frames_to_tensor`, `tensor_from_clip_file` |
| `backend/app/video/video_io.py` | `get_fps()`, `is_readable_video()` |
| `backend/app/api/helpers.py` | `parse_delivery_id`, `receive_upload`, `result_or_404`, `find_delivery`, `serve_file` |
| `backend/app/api/routes/{system,analysis,deliveries,media}.py` | 24 routes split by resource |
| `backend/models/README.md` | checkpoint provenance, checksums, input/output contracts |
| `data/README.md` | dataset vs runtime-output distinction |
| `data/manifests/source_clips.json` | 41 clips: size, SHA-256, upstream labels |
| `backend/conftest.py` | session setup: redirects generated output to a temp dir, puts `backend/` on `sys.path` |
| `.gitignore`, `requirements.txt`, `Dockerfile`, `docker-compose.yml` | deployment and dependency management |
| `README.md` | root entry point: architecture, pipeline, setup, API, limitations |
| `docs/REPOSITORY_STRUCTURE_REPORT.md` | this document |

## 6. Files deleted

- `CricketUnified/` and `CricketTracker-main/` — emptied and removed once their
  contents had been relocated.
- `backend/app/services/tracker_modules/__init__.py` — the re-export shim. Nothing
  imports across that boundary any more; each module is imported from its owner.
- `backend/app/utils/` — one module did not justify a package.
- `backend/app/api/endpoints.py` — split into four route modules.

## 7. Configuration changes

`app/core/config.py` now resolves every path through `_env_path()` /
`_env_file()`, which read the environment first and fall back to repository
defaults. A relative value is resolved against the project directory, so a service
started from any working directory finds the same data.

New variables: `CRICKET_DATA_DIR`, `CRICKET_MODELS_DIR`, `CRICKET_UPLOADS_DIR`,
`CRICKET_RUNS_DIR`, `CRICKET_SAMPLES_DIR`, `CRICKET_DATASETS_DIR`,
`CRICKET_MANIFESTS_DIR`, `CRICKET_BALL_MODEL`, `CRICKET_SHOT_MODEL`,
`CRICKET_LOG_LEVEL`.

Two symbols renamed for the new vocabulary: `ANALYSES_DIR` → `RUNS_DIR`,
`ANALYSIS_ROOT()` → `analysis_root()`. `Paths.public_prefix` changed from
`/data/analyses/{id}` to `/data/runs/{id}` to match.

## 8. Behaviour preserved

Unchanged, and covered by the verification below:

- one source decode, one YOLO pass, one Kalman pass per video
- delivery IDs assigned once, never renumbered
- `SINGLE_EVENT` / flagged `AMBIGUOUS` served; `MULTIPLE_EVENTS` / `NO_EVENT`
  / missing verdict quarantined with 404
- uncalibrated analytics reported as `null`, never estimated
- Adarsh shot architecture, 30-frame input, `(B, 3, 224, 224)` per frame,
  repetition for short clips
- all 24 `/api` routes and the legacy plural aliases

## 9. Verification

| Check | Result |
|---|---|
| `python -m compileall` (`app`, `main.py`, `tests`, `scripts`) | clean |
| `python -m pytest tests -q` | **324 passed, 1 skipped**, 82.44s |
| API + service imports | OK |
| Route count | 24 `/api` routes |
| E2E on `real_cricket.mp4` | **47/47 checks** |

E2E detail: 37 validated deliveries, 3 quarantined, 538 detected trajectory
points, one decode, one YOLO pass, one Kalman pass.

One test failed during the work: `assertLogs("app.services.shot_classifier")` in
`tests/integration/test_single_pass.py` referenced a logger name the module split
invalidated. Updated to `app.shot_classification.inference`. No behavioural code was
touched to fix it.

## 10. Outstanding

- Frontend not wired to `/api/*`; legacy aliases preserved but unverified.
- No project-scoped git repository; `git rev-parse --show-toplevel` resolves to
  `C:\Users\LENOVO`.
- `analytics/calibration.py` was reconstructed rather than moved (see below).
- OpenH264 unavailable in this environment; software encoding succeeds, stderr
  noise only.

### Reconstruction note

`services/tracker_modules/utils.py` was deleted with its directory before being
moved. Its contents were recovered from an upstream copy at
`Desktop/try/CricketTracker-main/backend/tracker_modules/utils.py` and rebuilt as
`analytics/calibration.py`, using the surviving call sites in `bowling.py` and
`trajectory.py` to confirm the contract: `build_homography(width, height,
corners_px)`, reading fractions from `HOMOGRAPHY_CORNER_FRACTIONS` rather than
hardcoding them. `HOMOGRAPHY_CORNER_FRACTIONS` holds the same values the upstream
file had inline. `savgol_smooth` and `compute_angle_deg` were reconstructed into
`analytics/trajectory.py` the same way. The E2E run and the analytics unit tests
both pass against the rebuilt versions, but the provenance is worth recording.