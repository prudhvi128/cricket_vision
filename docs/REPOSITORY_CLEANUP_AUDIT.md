# Repository Cleanup Audit — Unified Cricket Analysis

Audit date: 2026-10-05
Scope: `C:\Users\LENOVO\Desktop\New folder` (515 files, 83 directories before cleanup)
Purpose: establish, **before any deletion**, what the current STADIUM CRICKET pipeline actually requires.

Method: full recursive tree listing, import-graph trace from `main.py`, MD5 hashing of every model file,
byte-comparison of every vendored source module against its upstream original, reference search across
all `.py/.md/.json/.yaml/.tsx/.ts` files, and API-contract comparison of both frontends against the
unified router.

Categories: **A** required · **B** duplicate · **C** dead/unused · **D** experimental ·
**E** obsolete · **F** generated output · **G** uncertain — do not delete.

---

## A. REQUIRED — production pipeline (delete nothing here)

| File / folder | Category | Why required | References / imports found | Action | Risk |
|---|---|---|---|---|---|
| `CricketUnified/backend/app/services/tracking_service.py` | A | Single full-video decode → one YOLO11 pass → one Kalman pass → segmentation → horizons → purity → persistence | imported by `api/deps.py`, `services/pipeline.py` | keep | — |
| `.../services/pipeline.py` | A | Orchestrates stages, purity quarantine, config/calibration snapshot, shot provenance | imported by `api/deps.py:UnifiedPipeline` | keep | — |
| `.../services/tracker_modules/detector.py` | A | CricketTracker YOLO11 `BallDetector` (authoritative ball detection) | `tracking_service.py`, `api/deps.py`, `pipeline.py` | keep | — |
| `.../services/tracker_modules/tracker.py` | A | Ball tracking state + Kalman. Extended upstream 5,845 → 13,973 B | `tracker_modules/__init__.py` → `tracking_service.py` | keep | — |
| `.../services/tracker_modules/trajectory.py` | A | Ball trajectory generation (18 internal refs) | `tracker_modules/__init__.py` → `analytics.py` | keep | — |
| `.../services/tracker_modules/utils.py` | A | Homography, `pixel_to_ground`, Savitzky-Golay (pure numpy), `get_fps` | `trajectory.py`, `analytics.py` | keep | — |
| `.../services/tracker_modules/renderer.py` | A | Overlay rendering primitives; imported directly by `overlay_renderer.py:45-49` | `overlay_renderer.py` | keep | — |
| `.../services/event_segmenter.py` | A | Delivery/event candidate generation + segmentation | `tracking_service.py`, `clip_validator.py`, `fragment_merger.py` | keep | — |
| `.../services/event_horizons.py` | A | Event-horizon / barrier logic | `tracking_service.py` | keep | — |
| `.../services/fragment_merger.py` | A | Fragment merging (`BarrierSet`, `FragmentMerger`) | `tracking_service.py` | keep | — |
| `.../services/clip_validator.py` | A | Delivery purity validation | `tracking_service.py` | keep | — |
| `.../services/clip_writer.py` | A | Writes per-delivery clips + clip↔frame maps | `tracking_service.py` | keep | — |
| `.../services/ring_buffer.py` | A | Frame retention for single-pass decode | `tracking_service.py` | keep | — |
| `.../services/analytics.py` | A | Valid CricketTracker analytics: bounce, swing, angles, calibration gate, homography | `tracking_service.py`, `pipeline.py`, `reference.py` | keep | — |
| `.../services/overlay_renderer.py` | A | Overlay generation | `pipeline.py` | keep | — |
| `.../services/models.py` | A | **Adarsh EfficientNet-B0 + Transformer architecture.** Loaded lazily at `pipeline.py:381` | `pipeline.py` (`from .models import ImprovedSOTAModel`) | keep — do not modify architecture | — |
| `.../services/shot_classifier.py` | A | Adarsh exact preprocessing (30×3×224×224, ImageNet norm), 10-class mapping, sparse frame lookup | `pipeline.py` | keep | — |
| `.../schemas/tracking.py` | A | Delivery / trajectory / shot / bowling / provenance schemas | imported by 7 service modules | keep | — |
| `.../api/endpoints.py` | A | FastAPI routes: analyze, upload, status, progress, deliveries, clip, overlay, trajectory, cancel, health, reference | `main.py:app.include_router` | keep | — |
| `.../api/deps.py` | A | Job registry, model cache, weighted monotonic progress | `endpoints.py` | keep | — |
| `.../core/config.py`, `.../core/constants.py` | A | Single source of filesystem truth + segmentation/shot constants | imported by 11 modules | keep | — |
| `.../reference.py`, `.../utils/io.py`, `.../app/__init__.py`, `main.py` | A | Reference document, atomic JSON IO, package exports, ASGI app | `endpoints.py`, `main.py` | keep | — |
| `.../models/ball_detector.pt` (6,225,450 B) | A | CricketTracker YOLO11 weights, hard-coded `BALL_MODEL_PATH` | `core/config.py:55` | keep — untouched | — |
| `.../models/shot_classifier.ckpt` (72,010,842 B) | A | **Production checkpoint.** MD5 `2974EFDCF449C8447BEF107D94633217` | `core/config.py:56`, `resolve_shot_model_path()` | keep — untouched | — |
| `CricketUnified/backend/tests/` (13 files) | A | Tests for the pipeline. `test_api.py` (4 classes) and `test_api_contract.py` (9 classes) are **complementary, not duplicates** | — | keep all | — |
| `CricketUnified/backend/tools/e2e_real_video.py` | A | Live HTTP stadium smoke test + `--verify-only` mode | Phase 9 | keep | — |
| `real_cricket.mp4` (79,853,450 B) | A | **The only real stadium video.** Unrecoverable external footage; smoke-test input. `CreationTime` ≠ `LastWriteTime` ⇒ copied in, not generated | `tools/diagnose_real_deliveries.py:54` | keep | — |
| `.../data/samples/source_clips/` (41 clips, 41,364,977 B) | A | **Approved dataset.** The only real cricket clip corpus in the workspace; source of the test fixture. Relocated here from the Adarsh repo (see B-9) | `tools/build_test_fixture.py` | keep | — |

## B. DUPLICATE — provably redundant, safe to remove

| File / folder | Category | Why unnecessary | References / imports found | Action | Risk |
|---|---|---|---|---|---|
| `CricketShot-Classification-main/backend/src/tracking/detector.py`, `tracker.py`, `__init__.py` | B | Adarsh duplicate ball detector + tracker. CricketTracker is authoritative | zero imports from `CricketUnified/` | delete | Low |
| `.../src/segmentation/auto_clipper.py`, `__init__.py` | B | Old HSV-based AutoClipper; superseded by `event_segmenter.py` + `event_horizons.py` | zero imports from `CricketUnified/` | delete | Low |
| `.../src/pipeline/` (`match_pipeline.py`, `delivery_analyzer.py`, `run_pipeline.py`, `extract_frames.py`, `efficientnet_transformer.py`) | B | Old per-clip standalone pipeline; `efficientnet_transformer.py` superseded by `services/models.py` | zero imports from `CricketUnified/` | delete | Low |
| `.../src/inference/` (`predict.py`, `batch_predict.py`) | B | Duplicate inference implementation; superseded by `shot_classifier.py` | zero imports from `CricketUnified/` | delete | Low |
| `.../src/visualization/trajectory_video.py` | B | Old duplicate trajectory renderer; superseded by `overlay_renderer.py` | zero imports from `CricketUnified/` | delete | Low |
| `.../backend/main.py` (16,328 B) | B/E | Old prototype API serving `/api/predict`, `/api/progress/{id}`, `/api/cancel/{id}` — **none of these routes exist** in the unified router | no importer | delete | Low |
| `.../backend/test_41_transformer.py`, `test_all_deliveries.py`, `test_delivery_analyzer.py`, `test_full_match_pipeline.py`, `test_transformer_inference.py` | B | Tests for the deleted old pipeline; would fail on import | no importer | delete | Low |
| `.../backend/models/cricket_model_transformer.ckpt` (72,010,842 B) | B | **Byte-identical** to `shot_classifier.ckpt` (same MD5 `2974EFDC…`). `resolve_shot_model_path()` only searches `MODELS_DIR` | no runtime reference | delete | Low |
| `.../backend/models/best.pt` (6,225,450 B) | B | Byte-identical to `ball_detector.pt` (same MD5 `ADDE61B5…`) | no runtime reference | delete | Low |
| `.../backend/models/Ball_Detection_Model.pt` (51,568,261 B) | B/E | Adarsh duplicate ball detector. Unique hash, **never referenced** by the unified pipeline | no reference anywhere | delete | Low |
| `.../backend/models/Bat_Detection_Model.pt` (40,501,989 B) | B/E | Bat-pose detector. No bat/pose stage exists in the stadium pipeline | no reference anywhere | delete | Low |
| `.../frontend/` (17 files, React/Vite) | B/E | Frontend wired to non-existent routes `/api/predict`, `/api/progress/{id}`, `/api/cancel/{id}`. Cannot function against the unified backend | `src/services/api.ts:50,79,112` | delete | Low |
| `CricketTracker-main/backend/main.py` (3,418 B), `model.py` (11,580 B) | B/E | Old prototype API. Its own frontend calls root `/upload` + `/status`, neither of which exists on the unified router | no importer | delete | Low |
| `CricketTracker-main/backend/tracker_modules/` (6 files) | B | **Superseded originals.** `renderer.py` byte-identical to the vendored copy; `detector.py` 3,174→4,649 B, `tracker.py` 5,845→13,973 B, `trajectory.py` 12,391→14,340 B, `utils.py` 4,933→5,330 B, `__init__.py` 673→2,311 B — all strictly extended upstream. Keeping them risks importing the weaker Kalman | no importer | delete | Medium (loses upstream diff base; recoverable from `rohitgandhal/CricketTracker`) |
| `CricketTracker-main/backend/models/best.pt` (6,225,450 B) | B | Byte-identical to `ball_detector.pt` | no runtime reference | delete | Low |
| `CricketShot-Classification-main/config.yaml` | E | AutoClipper segmentation values (`min_duration 3.0`, `max_duration 10.0`, `hsv_green_ratio_threshold 0.3`) describe the deleted segmenter. Shot values (`num_frames 30`, `target_size [224,224]`) already ported to `constants.py:101-102` | no loader anywhere | delete | Low |

## C. DEAD / UNUSED

| File / folder | Category | Why unnecessary | References / imports found | Action | Risk |
|---|---|---|---|---|---|
| 16 unreferenced symbols inside live modules (`renderer.py`: `draw_prediction_arc`, `draw_hud`, `draw_mini_pitchmap`, `_trail_color`, `_glow_line`; `trajectory.py`: `predict_trajectory`; `utils/io.py`: `update_json_inplace`; `api/deps.py`: `JobRegistry.load_result`; …) | C | Genuinely unreferenced, but removing code from inside a live module is a **refactor**, not file cleanup | none | **keep — out of scope** (carried over from the prior cleanup report) | — |
| `CricketUnified/tools/ab_test_the_in_loop_slowdown.py` | D/C | One-off performance forensics answering "why is in-loop detect 4× slower"; its conclusion (VideoWriter/codec contention, not the ring buffer) is already recorded in `docs/phase3_report.md`. Not needed to run or verify the pipeline | no importer; cites `full_fixture.mp4` | delete | Low |
| `CricketUnified/tools/why_detection_is_slower_in_the_loop.py` | D/C | Same one-off perf question (GPU throttling vs thread-pool bleed). Answer already documented | no importer; cites `full_fixture.mp4` | delete | Low |

## D. EXPERIMENTAL — abandoned after the answer was recorded

| File / folder | Category | Why unnecessary | References / imports found | Action | Risk |
|---|---|---|---|---|---|
| `CricketUnified/tools/near_miss_contact_sheets.py` | D/F | One-shot contact-sheet builder for a past tuning round. Reads `stage3_after.json`, writes `docs/stage4_frames/` — both are removed as generated output (F). Regenerable by `real_video_diagnostic.py` | cited only by `STAGE4_NEAR_MISS_DIAGNOSTIC.md` | delete script, **keep the .md narrative** | Low |
| `CricketUnified/backend/data/stage3_after.json` (7,547,427 B) | F | Per-frame diagnostic output of a completed tuning round. Regenerable by re-running `real_video_diagnostic.py --video ..\..\real_cricket.mp4` | read by `near_miss_contact_sheets.py` (deleted), `STAGE3_KALMAN_FIX_REPORT.md`, `STAGE4_NEAR_MISS_DIAGNOSTIC.md` (both kept as narrative) | delete | Medium |
| `CricketUnified/backend/data/real_diagnostic.json` (7,529,356 B) | F | Default `--out` of `real_video_diagnostic.py`; regenerable by the same command | `REAL_VIDEO_DIAGNOSTIC.md`, `STAGE3_KALMAN_FIX_REPORT.md` | delete | Medium |

## E. OBSOLETE

| File / folder | Category | Why unnecessary | References / imports found | Action | Risk |
|---|---|---|---|---|---|
| `CricketUnified/backend/data/analyses/phase3_real/` (82 files, 82,036,729 B) | F | Output of the synthetic-fixture experiment round; superseded by the real-video run | `tools/diagnose_results.py` (kept for its narrative citations) | delete | Low |
| `CricketUnified/backend/.pytest_cache/` (5 files) | F | Test-runner cache | n/a | delete | None |
| All `__pycache__/` (14 dirs) + 61 `.pyc` | F | Byte-compiled caches | n/a | delete | None |
| `CricketShot-Classification-main/backend/visualization_test/*.mp4` (2 files) | F | Rendered test videos from the old pipeline | none | delete | Low |
| `CricketShot-Classification-main/.gitignore` | E | Belongs to the deleted project | none | delete | None |
| `CricketShot-Classification-main/backend/requirements.txt` | E | Lists `easyocr`, `pandas`, `requests`, `dotenv`, `tqdm` — none used by the unified pipeline | none | delete | Low |

## F. GENERATED OUTPUT

| File / folder | Category | Why unnecessary | References / imports found | Action | Risk |
|---|---|---|---|---|---|
| `CricketUnified/backend/data/analyses/test_clip_isolation/` (4 files) | F | Leaked test artifact — `tests/test_clip_isolation.py` writes `Paths(analysis_id="test_clip_isolation")` and never cleans up | regenerable by re-running the test | delete | None |
| `.../analyses/test_clip_purity/` (4 files) | F | Same leak (`test_clip_purity.py:316`) | regenerable | delete | None |
| `.../analyses/test_event_horizons/` (4 files) | F | Same leak (`test_event_horizons.py:317`) | regenerable | delete | None |
| `.../analyses/test_single_pass/` (6 files) | F | Same leak (`test_single_pass.py:33`) | regenerable | delete | None |
| `docs/stage4_frames/` (5 PNG, 4,622,512 B) | F | Rendered diagnostic contact sheets | `STAGE4_NEAR_MISS_DIAGNOSTIC.md` (narrative kept) | delete | Low |
| `CricketShot-Classification-main/backend/full_match_output/match_pipeline_results.json` (105,501 B) | F/G | Provenance record of the 41 kept source clips | the clips it describes | **keep** with the dataset | — |

## G. UNCERTAIN — DO NOT DELETE

| File / folder | Category | Why uncertain | Evidence | Action | Risk |
|---|---|---|---|---|---|
| `CricketUnified/backend/data/samples/full_fixture.mp4` (10,857,411 B) | G | Looks like a disposable generated fixture, but is cited **17×** across 4 docs, two of which are on the Phase 5 keep list (`DELIVERY_PROBLEM_DIAGNOSIS.md`, `DELIVERY_SEGMENTATION_FIX_DESIGN.md`), and is the input of 4 kept tools. Its own builder documents it as a *synthetic composite with black gaps* — so no accuracy claim should rest on it, but the diagnosis docs reference it as the reproduction input | doc citation counts | **keep** | — |
| `CricketUnified/tools/diagnose_results.py`, `run_real_video_test.py`, `build_test_fixture.py` | G | `build_test_fixture.py` is the **only** way to regenerate `full_fixture.mp4`, and it reads the 41-clip dataset. `run_real_video_test.py` verifies clip frame-map round-trips and trajectory provenance — checks `e2e_real_video.py` does not duplicate | cited by 4 keep-list docs | **keep** | — |
| `CricketTracker-main/frontend/ballscribe-insight-main/` (76 files) | G | **Neither frontend is wired to the unified API** — this one calls root `/upload` and `/status`, missing the `/api` prefix. So it is not "actually used". But it is the only UI in the workspace, its upload→status→results shape matches the unified contract, and deleting a frontend is a product decision, not a cleanup one | `src/pages/Upload.tsx:47`, `src/pages/Processing.tsx:29` | **keep** — record the wiring gap | — |
| `CricketTracker-main/backend/requirements.txt` | G | The **only dependency manifest in the workspace** — `CricketUnified` has none. Inaccurate for the unified build (lists `filterpy` and `scipy`, neither of which is imported; omits `python-multipart`) but still the best available record | actual imports: numpy, cv2, fastapi, torch, torchvision, PIL, ultralytics | **keep** | — |
| `CricketTracker-main/README.md` | G | Upstream provenance for the authoritative tracker | — | keep | — |
| `docs/phase3_report.md`, `STAGE3_KALMAN_FIX_REPORT.md`, `STAGE4_NEAR_MISS_DIAGNOSTIC.md`, `CLEANUP_REPORT.md` | G | Not on the Phase 5 keep list, but each describes code that **still exists** (the Kalman fix, the stage-3/4 tuning, the prior cleanup). The rule is to remove documentation only when it describes code that no longer exists | — | keep | — |
| `CricketUnified/backend/data/uploads/` (empty) | A | Declared `UPLOADS_DIR` at `core/config.py`, created by `ensure_base_dirs()` | — | keep | — |
| `.claude/settings.local.json` | G | Local tooling config, not project code, not referenced by the pipeline | — | keep | — |

---

## Cross-cutting findings

1. **No project-scoped Git repository exists.** `git rev-parse --show-toplevel` from the project directory
   resolves to `C:/Users/LENOVO` (the user's home directory). No commit was made and none should be made
   from this directory.
2. **Zero notebooks.** A recursive search found no `.ipynb` file anywhere in the workspace.
3. **`CricketUnified` has no `requirements.txt`, no Dockerfile, no `.gitignore`, and no CI config.**
   This is a real deployment gap — reported as technical debt, **not** created during cleanup, because
   authoring deployment config is implementation work this task forbids.
4. **The unified app import graph has zero dead modules.** Every one of the 21 modules under
   `CricketUnified/backend/app/` is reachable from `main.py`.
5. **Both frontends are unwired to the unified API** (see G). No frontend is currently used by the system.
6. **The dataset carve-out.** Deleting the Adarsh repository outright would destroy the only real cricket
   clip corpus and permanently break `tools/build_test_fixture.py`, which two Phase 5 keep-list docs cite.
   Resolution: relocate the 41 clips to `CricketUnified/backend/data/samples/source_clips/` and repoint the
   builder, then delete the old project.