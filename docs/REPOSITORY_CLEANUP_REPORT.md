# Repository Cleanup Report

Date: 2026-10-05
Audit performed first: `docs/REPOSITORY_CLEANUP_AUDIT.md`
Scope: `C:\Users\LENOVO\Desktop\New folder`

| | Before | After |
|---|---|---|
| Files | 515 | **207** |
| Directories | 83 | **33** |
| Size | ~1,569 MiB | **172.8 MiB** |

No pipeline logic, threshold, model architecture, or API contract was changed. The cleanup deleted
files and repaired one path reference; it did not modify the stadium pipeline.

---

## 1. Files and folders removed

### 1.1 The Adarsh source project — `CricketShot-Classification-main/` (98 files, 177 MiB)

Removed in full, as instructed. Before deleting it, its one irreplaceable asset was relocated (see §5).

| Removed | Category | Why |
|---|---|---|
| `backend/src/tracking/detector.py`, `tracker.py` | duplicate | Adarsh duplicate ball detector + tracker. CricketTracker is authoritative. Zero imports from `CricketUnified/`. |
| `backend/src/segmentation/auto_clipper.py` | obsolete | Old HSV-ratio segmentation, superseded by `event_segmenter.py` + `event_horizons.py`. |
| `backend/src/pipeline/*` (5 files) | obsolete | Old per-clip standalone pipeline. `efficientnet_transformer.py` superseded by `services/models.py`. |
| `backend/src/inference/*` (2 files) | duplicate | Duplicate inference implementation, superseded by `shot_classifier.py`. |
| `backend/src/visualization/trajectory_video.py` | duplicate | Old trajectory renderer, superseded by `overlay_renderer.py`. |
| `backend/main.py` | obsolete | Prototype API serving `/api/predict`, `/api/progress/{id}`, `/api/cancel/{id}` — **none of these routes exist** in the unified router. |
| `backend/test_*.py` (5 files) | obsolete | Tests for the deleted pipeline; would fail on import. |
| `backend/models/cricket_model_transformer.ckpt` (72,010,842 B) | duplicate | **Byte-identical** to `shot_classifier.ckpt` (MD5 `2974EFDCF449C844…`). `resolve_shot_model_path()` searches only `MODELS_DIR`. |
| `backend/models/best.pt` (6,225,450 B) | duplicate | Byte-identical to `ball_detector.pt` (MD5 `ADDE61B5409FCE5E…`). |
| `backend/models/Ball_Detection_Model.pt` (51,568,261 B) | obsolete | Adarsh duplicate ball detector. Unique hash, never referenced anywhere. |
| `backend/models/Bat_Detection_Model.pt` (40,501,989 B) | obsolete | Bat-pose detector. No bat or pose stage exists in the stadium pipeline. |
| `frontend/*` (17 files) | obsolete | Wired to non-existent routes. Cannot function against the unified backend. |
| `backend/visualization_test/*.mp4` (2 files) | generated | Rendered test videos from the old pipeline. |
| `config.yaml` | obsolete | AutoClipper segmentation values describe the deleted segmenter; shot values already ported to `constants.py:101-102`. |
| `backend/requirements.txt`, `.gitignore` | obsolete | Listed `easyocr`/`pandas`/`requests`/`dotenv`/`tqdm` — none used. |

### 1.2 Superseded duplicates in `CricketTracker-main/backend/` (9 files)

| Removed | Category | Why |
|---|---|---|
| `tracker_modules/` (6 files) | duplicate | Superseded originals. `renderer.py` byte-identical to the vendored copy; the other five were strictly **extended** upstream: `detector.py` 3,174→4,649 B, `tracker.py` 5,845→13,973 B, `trajectory.py` 12,391→14,340 B, `utils.py` 4,933→5,330 B, `__init__.py` 673→2,311 B. Keeping them risked importing the weaker Kalman. |
| `main.py`, `model.py` | obsolete | Old prototype API. Its own frontend calls root `/upload` + `/status`, neither of which exists on the unified router. |
| `models/best.pt` (6,225,450 B) | duplicate | Byte-identical to `ball_detector.pt`. |

### 1.3 Generated output (86 files, ~868 MiB)

| Removed | Category | Why |
|---|---|---|
| `data/analyses/da02f89b00c545d8/` (80 files, 823.7 MiB) | generated | Prior real-video run output: 37 clips (386.7 MiB), 37 overlays (358.8 MiB), a 76.2 MiB copy of the source video, `result.json`, `tracking.json`. Fully regenerable. |
| `data/analyses/4c458118e2f04112/` (80 files, 823.7 MiB) | generated | The post-cleanup smoke run's own artefacts — deleted so the repo does not ship 83 % of its weight as regenerable video. Both the run and its 47/47 result are recorded in §8. |
| `data/analyses/phase3_real/` (82 files, 78.3 MiB) | generated | Output of the synthetic-fixture round. Regenerable by the retained `tools/run_real_video_test.py`. |
| `data/analyses/test_{clip_isolation,clip_purity,event_horizons,single_pass}/` (18 files) | generated | Leaked test artefacts — the tests write `Paths(analysis_id="test_*")` and never clean up. |
| `data/stage3_after.json` (7.2 MiB), `data/real_diagnostic.json` (7.2 MiB) | generated | Default `--out` targets of `tools/real_video_diagnostic.py`; regenerable in one command. |
| `docs/stage4_frames/` (5 PNG, 4.4 MiB) | generated | Rendered diagnostic contact sheets. |

### 1.4 Abandoned experiments and caches

| Removed | Category | Why |
|---|---|---|
| `tools/ab_test_the_in_loop_slowdown.py` | experimental | One-off perf forensics; conclusion recorded in `docs/phase3_report.md`. |
| `tools/why_detection_is_slower_in_the_loop.py` | experimental | Same perf question; conclusion recorded. |
| `tools/near_miss_contact_sheets.py` | experimental | One-shot sheet builder for the stage-4 round; both its input and output removed as generated data. |
| 10 `__pycache__/` dirs, `.pytest_cache/` | generated | Byte-compiled and test-runner caches. |

## 2. Duplicate pipelines removed

The stadium pipeline now has exactly one ball detector, one Kalman tracker, one trajectory builder, one
segmenter, one merger, one purity validator, and one shot classifier. Confirmed by import trace from
`main.py`: all 21 modules under `app/` are reachable, and there are no competing implementations
anywhere in the tree.

## 3. Files and folders retained

**Production (required):** all of `CricketUnified/backend/app/` (31 files), `main.py`,
`tests/` (15 test modules + `helpers.py` + `__init__.py`), `backend/tools/e2e_real_video.py`,
`tools/` (5 diagnostic scripts).

**Models (untouched, verified by hash before and after):**

| File | Bytes | MD8 prefix |
|---|---|---|
| `backend/models/ball_detector.pt` | 6,225,450 | `ADDE61B5409FCE5E` |
| `backend/models/shot_classifier.ckpt` | 72,010,842 | `2974EFDCF449C844` |

**Dataset:** `backend/data/samples/source_clips/` (41 clips + provenance JSON, 10.0 MiB) and
`backend/data/samples/full_fixture.mp4` (10.4 MiB).

**Not part of the pipeline but retained as UNCERTAIN:** `CricketTracker-main/README.md`,
`CricketTracker-main/backend/requirements.txt`, `CricketTracker-main/frontend/` (76 files),
`.claude/settings.local.json`, `real_cricket.mp4`.

**Documentation (14 files):** all prior docs kept — none described an architecture that no longer
exists. Added `CURRENT_SYSTEM.md`, `REPOSITORY_CLEANUP_AUDIT.md`, and this report.

## 4. Important components preserved

Upload/input · single full-video decode · YOLO11 detector · Kalman tracking · ball detection/tracking
state · candidate generation · segmentation · event-horizon/barrier logic · fragment merging · purity
validation · delivery persistence · Adarsh shot classifier · exact Adarsh preprocessing · the 10-class
mapping · trajectory generation · valid analytics · overlay rendering · FastAPI API · status/progress ·
delivery navigation data · all tests · both checkpoints · the 41-clip dataset.

**Verified live, not just by inspection:** YOLO11 loads and reports `task=detect, classes=1`; the shot
model loads on `cuda:0` and completes a forward pass returning logits of shape `(1, 10)`; the
segmenter, horizon tracker, fragment merger, purity validator, detector and tracker all instantiate;
FastAPI exposes **24 `/api` routes** plus the `/data` static mount.

## 5. One reference repaired (Phase 7)

`tools/build_test_fixture.py` read the 41 source clips from
`CricketShot-Classification-main/backend/full_match_output/delivery_clips/`. Deleting the Adarsh project
would have permanently broken it — and with it the reproducibility of `full_fixture.mp4`, which two
Phase 5 keep-list docs cite 17 times.

**Fix:** the 41 clips were copied to `CricketUnified/backend/data/samples/source_clips/` and **verified
byte-identical** (41/41 matching MD5), the builder's `SRC_DIR` was repointed in-project, its stale
docstring updated, and the now-orphaned `WORKSPACE` constant removed.

No other broken references exist. Every remaining mention of a deleted path is a **provenance comment**
(e.g. `constants.py:151` recording where the checkpoint was ported from) or narrative documentation —
never a filesystem path. `tools/diagnose_results.py` reads the deleted `phase3_real` directory, but it
already guards that at line 307 and exits with a message pointing at the retained
`tools/run_real_video_test.py`, which regenerates it.

## 6. Tests executed

```powershell
cd CricketUnified\backend
python -m compileall -q app main.py tools
python -m pytest -q --no-header -p no:randomly
```

## 7. Test results

| Check | Result |
|---|---|
| `compileall` (app, main, tools) | **OK** |
| Import of all API/service/core/schema modules | **OK** |
| YOLO11 load (`ball_detector.pt`) | **OK** — `task=detect`, 1 class |
| Shot model load + forward pass | **OK** — cuda:0, logits `(1, 10)` |
| Component instantiation (segmenter, horizons, merger, validator, detector, tracker) | **OK** |
| FastAPI app import + route enumeration | **OK** — 24 routes + `/data` |
| `TestClient` startup, `/api/health`, `/api/reference`, `/api/analyses`, 404 on unknown id | **OK** |
| **Full pytest suite** | **324 passed, 1 skipped, 1 warning** (187 s) |

324/1 is **identical to the pre-cleanup baseline**, so cleanup broke nothing. The one skip is a
synthetic fixture that produces no quarantine; the gate is covered directly by unit tests instead. The
single warning is a Starlette/anyio deprecation inside `testclient.py`, not project code.

## 8. Stadium pipeline smoke test

```powershell
python tools\e2e_real_video.py          # real_cricket.mp4, in-process HTTP, ~13 min
```

| | |
|---|---|
| Video | 1280×720, 25 fps, 13,565 frames, 542.6 s, 76.2 MiB |
| Candidate events | **40** |
| Accepted deliveries | **37** |
| Quarantined | **3** — ids 17, 19, 22, all `MULTIPLE_EVENTS` (2 distinct tracked events inside one clip) |
| By validation class | `SINGLE_EVENT` 24, `AMBIGUOUS` 13 |
| Multiple-event detections | 3 (all quarantined, none served) |
| No-event detections | 0 |
| Source decodes / YOLO passes / Kalman passes / shot-model loads | **1 / 1 / 1 / 1** |
| Shot labels | 37 of 37, **0 failures** |
| Short clips padded to 30 samples | 4 — deliveries 2 (19 f), 7 (21 f), 14 (21 f), 37 (19 f), all declaring `repeated=True` |
| Trajectory detected points | 538 |
| Speeds / lengths / lines invented | **0 / 0 / 0** — all null, `mean_speed_kmh: null` |
| Calibration | `geometry_calibrated: false`, `speed_is_calibrated: false`, `pitch_corners_px: null` |
| Clip / overlay delivery | 37 clips + 37 overlays served, all decodable, HTTP 206 on ranges |
| Warnings / errors | 96 / **0** |
| **E2E checks** | **47 / 47 passed** |

**Comparison with the previous diagnostic:** identical — 40 candidates, 37 deliveries, 3 quarantined,
538 detected points, same 4 short clips, same 0 uncalibrated fabrications. Cleanup changed no outcome.
No threshold was modified to obtain this result.

OpenH264 (`avc1`) is unavailable in this environment, so OpenCV falls back to software encoding. This
produces repeated `Could not open codec libopenh264` stderr noise but does not affect any result.

## 9. Remaining suspicious / uncertain files

| Item | Status |
|---|---|
| `CricketTracker-main/frontend/` (76 files) | **Kept as UNCERTAIN.** Not wired to the unified API — it calls root `/upload` and `/status`, missing the `/api` prefix. It is the only UI in the workspace and its upload→status→results shape matches the contract, so removing it is a product decision, not a cleanup one. |
| `CricketTracker-main/backend/requirements.txt` | **Kept as UNCERTAIN.** The only dependency manifest in the workspace. Inaccurate for the unified build: lists `filterpy` and `scipy` (neither is imported — the Kalman is hand-rolled pure numpy) and omits `python-multipart` (required for uploads). |
| 16 unreferenced symbols inside live modules | **Kept.** Removing code from inside a working module is a refactor, not file cleanup. Carried over from the prior cleanup report. |
| `full_fixture.mp4` + `diagnose_results.py` | **Kept as UNCERTAIN.** Cited 17× across 4 docs. Its builder documents it as a *synthetic composite with black gaps*, so no accuracy claim should rest on it. |
| `.claude/settings.local.json` | Kept — local tooling config, not project code. |

## 10. Known technical debt

1. **No `requirements.txt`, no `Dockerfile`, no `.gitignore`, no CI in `CricketUnified`.** This is the
   single largest gap. Actual imports are `numpy, cv2, fastapi, torch, torchvision, PIL, ultralytics`
   plus `uvicorn` and `python-multipart`. Not created here — authoring deployment config is
   implementation work this task forbids.
2. **No project-scoped Git repository.** `git rev-parse --show-toplevel` resolves to `C:/Users/LENOVO`,
   the user's home directory. Nothing was committed, and nothing should be committed from this path.
3. **Tests leak artefacts.** Four test modules write into `data/analyses/` and never clean up, so they
   reappear on every run. Confirmed twice during this cleanup.
4. **Legacy routes retained but their consumers are gone.** The plural aliases and `/api/upload` remain
   for compatibility, but both frontends that used them have been removed, so nothing in the repo calls
   them now.
5. **Shot confidences are 0.17–0.56.** The checkpoint is untouched and unattributable-in-practice; this
   cleanup made the model auditable, not better.
6. **13 of 37 deliveries are `AMBIGUOUS`.** Served and flagged rather than suppressed.
7. **Caches regenerate on every run.** `__pycache__` and `.pytest_cache` were cleared; they reappear and
   need a `.gitignore` to stay out of version control.

---

## Status

**CLEANUP STATUS: PASS**

**STADIUM PIPELINE: INTACT** — 47/47 smoke checks, 324 passed / 1 skipped, byte-identical outcome to the
pre-cleanup baseline.

**SHOT MODEL: INTACT** — checkpoint untouched (MD5 `2974EFDCF449C844…`), architecture untouched, loads on
`cuda:0`, forward pass returns `(1, 10)`, 37/37 deliveries classified.

**TRACKING: INTACT** — exactly one source decode, one YOLO pass, one Kalman pass; 538 detected points;
full trajectory provenance.

**SEGMENTATION: INTACT** — 40 candidates → 37 deliveries, 3 `MULTIPLE_EVENTS` quarantined, all
segmentation tests pass, no threshold touched.

**API: INTACT** — 24 `/api` routes plus `/data` static mount; `TestClient` startup, health, reference,
analyses list and 404 behaviour all verified.

### Exact next recommended step

**Create `CricketUnified/backend/requirements.txt` and initialise a project-scoped Git repository,
then commit the cleaned tree.** That is the only remaining blocker to the repo being self-contained and
reproducible, and it is deployment/config work rather than cleanup. Pin the seven runtime dependencies
observed above plus `uvicorn` and `python-multipart`, add a `.gitignore` covering `__pycache__`,
`.pytest_cache`, `data/analyses/`, `data/uploads/` and `*.pyc`, then decide the frontend question:
either rewire `CricketTracker-main/frontend/` to `/api/upload` + `/api/analyses/{id}/status`, or
remove it. Do not begin box-cricket work or model optimisation until those two are settled.