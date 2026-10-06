# Cleanup Report — Unified Cricket Project

**Date:** 2026-10-05
**Scope of this task:** post-merge cleanup of the unified Cricket project.
**Nature of the task:** cleanup only. No refactor, no redesign, no functionality change.

---

## 0. Executive Summary

The workspace contains four top-level items:

| Path | Nature | Action |
|---|---|---|
| `CricketUnified/` | The unified application (backend + tools) | **Partially cleaned** — dead artifacts removed, production code untouched |
| `CricketTracker-main/` | Source repository #1 (kept whole per instruction) | **Untouched** |
| `CricketShot-Classification-main/` | Source repository #2 / "old project" (kept whole per instruction) | **Untouched** |
| `docs/` | Unified-architecture documentation (5 files) | **Kept in full; report added** |
| `real_cricket.mp4` | Unrecoverable external input footage | **Kept** |
| `.claude/` | Local assistant/tooling config, not project code | **Kept** |

**The critical finding of this audit:** the unified backend at `CricketUnified/backend` is
**completely self-contained**. No module under `backend/app/` or `backend/tests/` reads a single
byte from either source repository at runtime. The only cross-references to the source repos live
in *comments* (`app/core/constants.py:6`) and in a handful of standalone diagnostic scripts under
`CricketUnified/tools/`. Consequently, **every runtime dependency of the unified pipeline is inside
`CricketUnified/` itself**, and nothing required by it was deleted.

**Totals:** 294 files removed, 50 folders removed, 0 production source files removed.

---

## 1. Files and Folders Reviewed

Total inventory before cleanup: **544 files** (excluding `__pycache__`).

### 1.1 `CricketUnified/backend/` — the unified application (production)

| Group | Files | Verdict |
|---|---|---|
| `main.py` | 1 | Keep (ASGI entry point) |
| `app/__init__.py` | 1 | Keep |
| `app/api/` (`__init__`, `deps`, `endpoints`) | 3 | Keep |
| `app/core/` (`__init__`, `config`, `constants`) | 3 | Keep |
| `app/schemas/` (`__init__`, `tracking`) | 2 | Keep |
| `app/services/` (`__init__`, `analytics`, `clip_validator`, `clip_writer`, `event_horizons`, `event_segmenter`, `models`, `overlay_renderer`, `pipeline`, `ring_buffer`, `shot_classifier`, `tracking_service`) | 12 | Keep |
| `app/services/tracker_modules/` (`__init__`, `detector`, `renderer`, `tracker`, `trajectory`, `utils`) | 6 | Keep — **this is the single authoritative YOLO11 + Kalman pipeline** |
| `app/utils/` (`__init__`, `io`) | 2 | Keep |
| `models/` (`ball_detector.pt`, `shot_classifier.ckpt`) | 2 | Keep — **required checkpoints** |
| `tests/` (9 test modules + `helpers.py` + `__init__.py`) | 11 | Keep — **152 tests** |
| `data/samples/`, `data/uploads/` | 1 file / 0 files | Keep |
| `data/analyses/` | 227 | **Remove** — generated output (see §2.1) |
| `.pytest_cache/` | 5 | **Remove** — cache |
| `__pycache__/` (× 9) | 50 | **Remove** — cache |

### 1.2 `CricketUnified/tools/` — standalone diagnostic scripts (18)

All 18 are standalone `__main__` scripts. **No tool is imported by any other tool**, and **no test
imports any tool** (verified: zero matches for `tools` inside `backend/tests/*.py`). They are
developer-facing, not part of the running application.

### 1.3 `docs/` — 5 markdown files

| File | Bytes | Subject |
|---|---|---|
| `UNIFIED_ARCHITECTURE.md` | 45,044 | The current, authoritative unified design |
| `MERGE_AUDIT.md` | 22,199 | Pre-merge audit of the two source repos |
| `phase3_report.md` | 17,697 | Phase 3 backend-integration verification report |
| `DELIVERY_PROBLEM_DIAGNOSIS.md` | 44,557 | Delivery-segmentation root-cause analysis |
| `DELIVERY_SEGMENTATION_FIX_DESIGN.md` | 18,756 | Segmentation fix design addendum |

### 1.4 Reviewed and deliberately left alone

- `CricketTracker-main/` — 100 files. **Not touched** (source repository).
- `CricketShot-Classification-main/` — 98 files. **Not touched** (source repository / "old project").
- `real_cricket.mp4` — 79,853,450 bytes.
- `.claude/settings.local.json` — 212 bytes.

---

## 2. Files and Folders Identified as Unused

### 2.1 Generated analysis outputs — `CricketUnified/backend/data/analyses/**` (227 files, 517.5 MB)

**13 directories removed.** Every one is *pipeline output*, not pipeline input or fixture.

| Directory | Files | Size | Why unused |
|---|---|---|---|
| `real_diag/` | 36 | 378.9 MB | Output of `tools/diagnose_real_deliveries.py`. Zero test references. The tool `shutil.rmtree`s and regenerates it on every run (`diagnose_real_deliveries.py:194-195`). Largest directory in the project. |
| `diag_full/` | 44 | 48.5 MB | Output of `tools/full_diagnostic.py` (a tool that is itself broken — §2.3). Zero test references. |
| `diag_events/` | 46 | 48.6 MB | Output of `tools/diagnose_clip_events.py` / `attribute_clip_events.py` (both removed, §2.3). Zero test references. |
| `seg_final_pipeline/` | 42 | 28.9 MB | **Zero references anywhere in the project** — not in any tool, not in any doc, not in any test. Only self-references inside its own `tracking.json`. Orphan. |
| `seg_final_check/` | 21 | 14.5 MB | **Zero references anywhere in the project.** Orphan. |
| `phase3_real/` | 22 | 22.7 MB | Output of the *kept* `tools/run_real_video_test.py`, which `shutil.rmtree`s and regenerates it (`run_real_video_test.py:97`). Zero test references. |
| `test_single_pass/` | 4 | 68.7 KB | **Test output.** Created by `tests/test_single_pass.py:32` via `Paths(...).ensure()`. |
| `test_event_horizons/` | 4 | 77.3 KB | **Test output.** Created by `tests/test_event_horizons.py:214,317`. |
| `test_clip_isolation/` | 4 | 144.0 KB | **Test output.** Created by `tests/test_clip_isolation.py:100,236,374`. |
| `debug_seg/` | 2 | 24.5 KB | **Zero references** outside its own `tracking.json`. Debug artifact. |
| `debug_seg2/` | 2 | 24.5 KB | **Zero references** outside its own `tracking.json`. Debug artifact. |
| `ab_codec = mp4v only/` | 0 | 0 | Empty orphan left by `tools/ab_test_the_in_loop_slowdown.py:47,90` after an aborted run. Directory name is itself a truncation artifact. |
| `ab_no clip window (pre/` | 0 | 0 | Empty orphan left by the same tool (`:94`); name truncated at the `/` in `pre/post`. |

**Proof that no test depends on any of these — five independent checks:**

1. **Every directory a test writes to is created by that test in the same run.** All creation goes
   through `Paths.ensure()` → `mkdir(parents=True, exist_ok=True)` (`app/core/config.py:326-329`)
   or `io.py:40`. Because `parents=True` is used, deleting the entire `data/` tree cannot break a
   test: the tree is rebuilt from nothing on the next run.
2. **No test reads a pre-existing analysis directory.** The only reads are of files the test itself
   just wrote moments earlier in the same test method.
3. **Tests use a stub detector, never the real model.** `tests/helpers.py:52-84` sets
   `self.model = object()`. `test_api.py:137-139` additionally replaces
   `pipeline._build_classifier` with a no-op. No test contains the strings `ball_detector`,
   `shot_classifier.ckpt`, `BALL_MODEL_PATH`, `SHOT_MODEL_PATH` or `ultralytics`.
4. **`test_api.py` already cleans up after itself** — `shutil.rmtree(ANALYSIS_ROOT("schema_test"))`
   at line 152, which is why `schema_test/` does not exist on disk.
5. **Independent timestamp evidence.** `.pytest_cache/v/cache/nodeids` has LastWriteTime
   `00:47:27`, byte-identical to `data/analyses/test_single_pass/tracking.json` (`00:47:27`);
   `test_clip_isolation/tracking.json` = `00:46:54` and `test_event_horizons/tracking.json` =
   `00:47:08`. One contiguous ~30-second window = one pytest session. These three directories are
   demonstrably the last test run's own output.

`CricketUnified/backend/data/analyses/` **itself is kept** — it is declared at
`app/core/config.py:52` and created by `ensure_base_dirs()` (`config.py:336-338`).

### 2.2 Caches (55 files, 10 folders)

| Item | Files | Why unused |
|---|---|---|
| `CricketUnified/backend/.pytest_cache/` | 5 | pytest's own cache. `.gitignore`, `CACHEDIR.TAG`, `README.md`, `v/cache/lastfailed`, `v/cache/nodeids`. Auto-recreated on the next pytest run. Contains **zero** project information. |
| `__pycache__/` × 9 | 50 | CPython bytecode caches. All `.pyc`, including 15 `cpython-310-pytest-9.1.1.pyc` variants. Auto-recreated. No `.pyc` was the sole copy of anything. |

The 9 `__pycache__` locations: `backend/`, `backend/app/`, `backend/app/api/`, `backend/app/core/`,
`backend/app/schemas/`, `backend/app/services/`, `backend/app/services/tracker_modules/`,
`backend/app/utils/`, `backend/tests/`.

Neither source repository contained a single `__pycache__`, so no cache was removed from them.

### 2.3 Abandoned one-off experimental/debug scripts — `CricketUnified/tools/` (12 files)

**Kept (6)** — the reproducible verification chain documented in `docs/phase3_report.md:337-345`,
each of which is self-sufficient:

| Kept tool | Why kept |
|---|---|
| `build_test_fixture.py` | Documented (`phase3_report.md:340`). Generates `data/samples/full_fixture.mp4`, which is kept. |
| `run_real_video_test.py` | Documented (`:341`). `rmtree`s and regenerates `phase3_real`. Also the only tool that imports `tests.helpers`. |
| `diagnose_results.py` | Documented (`:342`). Reads `phase3_real`, which the immediately-preceding documented step regenerates. |
| `why_detection_is_slower_in_the_loop.py` | Documented (`:343`). Reads only `full_fixture.mp4` (kept). |
| `ab_test_the_in_loop_slowdown.py` | Documented (`:344`). `rmtree`s and regenerates its own `ab_*` dirs; reads only `full_fixture.mp4` (kept). |
| `diagnose_real_deliveries.py` | `rmtree`s and regenerates `real_diag` (`:194-195`); reads `real_cricket.mp4` (kept). Self-sufficient. The only consumer of that video. |

**Removed (12)** — every one is a one-off investigation script pinned to a hard-coded analysis ID
inside the generated output being deleted, and not referenced by any documentation:

| Removed tool | Why unused |
|---|---|
| `full_diagnostic.py` (1,134 lines) | **Provably non-functional:** line 1044 does `from app.main import app`, but `CricketUnified/backend/app/main.py` does not exist — the ASGI app is at `backend/main.py`. Guaranteed `ImportError`. Also writes to a hard-coded machine-specific path `Path.home()/"AppData/Local/Temp/opencode/diag_findings.json"` (line 1126) with an unclosed file handle. Undocumented. Sole producer/consumer of `diag_full`. |
| `diag_followup.py` | **Provably non-functional:** line 262 does `from app.main import app as fastapi_app` — same missing module, same guaranteed `ImportError`. Note the same file imports correctly at line 227 (`import main as backend_main`), proving the module layout makes `app.main` unreachable. Undocumented. Pinned to `diag_full`. |
| `analyze_clip_019.py` | Reads one hard-coded file: `backend/data/analyses/phase3_real/clips/delivery_019.mp4` (line 4). Pinned to a single delivery index. Undocumented, unreferenced. |
| `test_motion_analyzer.py` | Reads `backend/data/analyses/phase3_real/clips/{clip}` (line 36). Also a **pytest collection hazard**: named `test_*.py` outside `tests/`, so a `pytest` invocation rooted at `CricketUnified/` would try to collect it. Undocumented. |
| `scan_all_clips.py` | Reads `backend/data/analyses/phase3_real/clips` (line 5). Read-only consumer of deleted output. Undocumented. |
| `scan_diag_clips.py` | Reads `backend/data/analyses/diag_full/clips` (line 5). Read-only consumer of deleted output. Undocumented. |
| `diagnose_segmentation.py` | Reads `backend/data/analyses/phase3_real/tracking.json` (line 4). Undocumented. |
| `map_fixture_clips.py` | Reads `backend/data/analyses/phase3_real/tracking.json` (line 4). Undocumented. |
| `compare_analyses.py` | Iterates `phase3_real`, `diag_full`, `diag_events` (line 5). All deleted. Undocumented. |
| `diagnose_clip_events.py` | Sole producer/consumer of `diag_events`; hard-coded `ANALYSIS_ID = "diag_events"` (line 53). Undocumented. |
| `attribute_clip_events.py` | Reads `diag_events` (lines 26-27). Undocumented. |
| `classify_regions_by_shot.py` | Writes `diag_events/region_shots.json` (line 188). Undocumented. Only importer of `app/services/models.py` other than the pipeline itself. |

### 2.4 Empty directory (0 files, 1 folder)

| Item | Why unused |
|---|---|
| `CricketUnified/docs/` | **Completely empty (0 items).** The canonical architecture document is `docs/UNIFIED_ARCHITECTURE.md` at the workspace root, which is what the code docstrings actually mean: `app/__init__.py:7` and `app/services/tracker_modules/__init__.py:5` both direct the reader to `docs/UNIFIED_ARCHITECTURE.md`, and `app/core/constants.py:7,150` cites `docs/UNIFIED_ARCHITECTURE.md §4` and `§10 D2` — all of which resolve to the root `docs/` copy, not to this empty folder. Removing an empty directory removes no content. |

### 2.5 Items explicitly requested for removal that were **NOT** removed

These appear on the request's removal list but are **protected by an explicit instruction or by
proof of live use**. Each was investigated rather than assumed.

| Requested-for-removal item | Why it was kept |
|---|---|
| "duplicate ball detectors" / "duplicate Kalman trackers" | The copies in `CricketUnified/backend/app/services/tracker_modules/` **are** the single authoritative pipeline — not duplicates of it. A full import-graph trace from `main.py` shows all 6 modules are production-reachable: `detector.py` ← `tracking_service.py`; `tracker.py` ← `tracking_service.py`; `trajectory.py` + `utils.py` ← `analytics.py` and `tracking_service.py`; `renderer.py` ← `overlay_renderer.py` (imported directly at `overlay_renderer.py:45-49`). **Zero dead modules exist in `app/`.** |
| "old per-clip YOLO tracking" | No per-clip tracking code exists inside `CricketUnified/`. It exists only in `CricketShot-Classification-main/src/tracking/`, which is inside a source repository that must be preserved. |
| "obsolete AutoClipper implementations" | `auto_clipper.py` exists only at `CricketShot-Classification-main/backend/src/segmentation/auto_clipper.py`. The unified segmenter is `app/services/event_segmenter.py`. Again, inside a protected repository. |
| "old trajectory-generation pipelines" | `CricketShot-Classification-main/src/visualization/trajectory_video.py` and `src/pipeline/*`. Protected repository. `app/services/tracker_modules/trajectory.py` is live (§2.5, first row). |
| "unused model files" | **Every model file in the project is either required or protected.** See §3. |
| "duplicate frontend applications" | **Both frontends live inside the two source repositories**, which must not be deleted. See §4.2. |
| "duplicate backend services" | `CricketTracker-main/backend/` and `CricketShot-Classification-main/backend/` are protected. Inside `CricketUnified/backend` there is exactly one service layer and no duplication. |
| "obsolete notebooks" | **Zero `.ipynb` files exist anywhere in the workspace** (verified by recursive search). Nothing to remove. |
| "generated temporary files" | Only the 50 `.pyc` files and the 5 `.pytest_cache` files matched temp-file patterns. No `.tmp`, `.bak`, `.log`, `.swp`, `.orig`, `.part` files exist in `CricketUnified/` or `docs/`. |
| "old debug artifacts" | `debug_seg/`, `debug_seg2/`, `real_diag/`, `seg_final_*/`, `ab_*` all removed (§2.1). The two broken diagnostic scripts removed (§2.3). No other debug artifacts exist. |
| "unused sample outputs" | `data/samples/full_fixture.mp4` **is used** — by 8 kept/removed tools and by `docs/phase3_report.md:55` and `docs/DELIVERY_PROBLEM_DIAGNOSIS.md` (8 citations). Kept. |
| "duplicate documentation that describes the old architecture" | **No document in `docs/` describes a retained old architecture.** See §4.1. |

---

## 3. Model Files Kept (the high-risk area)

No model file was deleted. **Four checkpoints exist across the workspace; all four are either
required or protected.**

| Path | Bytes | Required? | Evidence |
|---|---|---|---|
| `CricketUnified/backend/models/ball_detector.pt` | 6,225,450 | **YES — runtime-critical** | Hard-coded at `app/core/config.py:55` as `BALL_MODEL_PATH = MODELS_DIR / "ball_detector.pt"`. Loaded by the detector in `app/services/tracker_modules/detector.py`. This is the **CricketTracker YOLO11 ball detector**, item #1 of the required architecture. |
| `CricketUnified/backend/models/shot_classifier.ckpt` | 72,010,842 | **YES — runtime-critical** | Hard-coded at `app/core/config.py:56` as `SHOT_MODEL_PATH = MODELS_DIR / "shot_classifier.ckpt"`. Loaded by `app/services/shot_classifier.py`. This is the **Adarsh CricketShot-Classification transformer**, item #2 of the required architecture. |
| `CricketTracker-main/backend/models/best.pt` | 6,225,450 | Protected | Inside source repository #1, which must be preserved whole. Byte-size-identical to `ball_detector.pt` — it is the origin of the unified copy. |
| `CricketTracker-main/backend/models/Best_Detection_Model.pt` / `Ball_Detection_Model.pt` / `Bat_Detection_Model.pt` | — | Protected | Inside source repositories. `Bat_Detection_Model.pt` (40,501,989 B) and `Ball_Detection_Model.pt` (51,568,261 B) exist only in `CricketShot-Classification-main`. They are **not** referenced by the unified pipeline — the unified ball detector is CricketTracker's YOLO11, per `app/core/constants.py:5-7` and `docs/UNIFIED_ARCHITECTURE.md §1`. They were nonetheless **kept**, because they sit inside a source repository that the task forbids deleting or touching. |

**Checks performed on every model file before deciding:**
- Grepped the entire workspace for the filename and for the symbolic path (`BALL_MODEL_PATH`,
  `SHOT_MODEL_PATH`, `ball_detector`, `shot_classifier`, `cricket_model_transformer`, `best.pt`).
- Read `app/core/config.py` (the single source of filesystem truth — lines 46-56) and confirmed
  there are **no** absolute paths and **no** fallback model locations anywhere in `app/`.
- Confirmed the test suite references neither checkpoint (uses `StubDetector`).
- Confirmed no removed tool or doc is the sole consumer of any checkpoint.

**Conclusion: no model file was deleted, and none needed to be. Every model file that the unified
pipeline actually needs was already correctly located inside `CricketUnified/backend/models/`.**

---

## 4. Files and Folders Kept

### 4.1 Documentation — all 5 kept, none deleted

Every document describes the **current** unified architecture or the history that produced it. None
is a duplicate describing an architecture that is still present in the tree.

| Kept doc | Why it is relevant documentation |
|---|---|
| `docs/UNIFIED_ARCHITECTURE.md` | **The authoritative current design.** Cited directly from code: `app/core/constants.py:7` (`§4`), `:150` (`§10 D2`), `app/__init__.py:7`, `app/services/tracker_modules/__init__.py:5`. |
| `docs/phase3_report.md` | Verification report for the implemented backend. **Defines the reproducible verification chain** — the reason 6 tools were kept and the command used to re-verify after cleanup (`python -m unittest discover -s tests -t .`, line 338). Also documents the 53-check real-footage validation and the performance findings. |
| `docs/DELIVERY_PROBLEM_DIAGNOSIS.md` | Root-cause analysis of the delivery-segmentation defect. The defect it describes is **fixed** in the current code (`event_horizons.py`, `clip_validator.py`, `SEG_*` constants in `constants.py:152-250`). Its `SEG_*` thresholds are cited by `app/core/constants.py:14-25`. |
| `docs/DELIVERY_SEGMENTATION_FIX_DESIGN.md` | The design addendum for that fix; specifies `test_clip_isolation.py` (§252), which is kept. |
| `docs/MERGE_AUDIT.md` | **Not dead despite being marked "superseded".** `UNIFIED_ARCHITECTURE.md:6-9` explicitly cross-references it: *"Supersedes: `docs/MERGE_AUDIT.md` §10 ... Audit A§12 risks mostly still stand"*. Deleting it would break that citation. It is the record of what was audited before the merge. |
| `docs/CLEANUP_REPORT.md` | This document (newly added). |

### 4.2 Ambiguous items **NOT** deleted because usage could not be proven unnecessary

Listed here per the requirement to report them explicitly.

| Ambiguous item | Why it was NOT deleted |
|---|---|
| **`real_cricket.mp4`** (79,853,450 bytes) | It **looks** like a leftover generated/temp file: a 76 MB video sitting loose at the workspace root, outside the project's own `data/` layout, covered by no `.gitignore`. **But usage could not be proven unnecessary:** it is referenced by `tools/diagnose_real_deliveries.py:54` as a hard-coded absolute `DEFAULT_VIDEO`, and no script in the project can regenerate it (`build_test_fixture.py` produces `full_fixture.mp4`, a different file). Its `CreationTime` (`05-10 00:47:23`) differs from its `LastWriteTime` (`01-10 19:45:06`), proving it was *copied in*, not produced locally. **It is unrecoverable external footage. Deleting it would permanently destroy input data.** Kept. |
| **Both frontend applications** | `CricketUnified` contains **no frontend at all** — verified. The two frontends therefore live exclusively inside the two source repositories, which the task explicitly forbids deleting. Separately, a contract audit found that **neither frontend is currently wired to the unified API**: `CricketShot-Classification-main/frontend/src/services/api.ts` calls `POST /api/predict`, `GET /api/progress/{id}`, `DELETE /api/cancel/{id}`, whereas the unified router exposes `POST /api/upload`, `GET /api/analyses/{id}/progress`, `POST /api/analyses/{id}/cancel`. `CricketTracker-main/frontend/.../ballscribe-insight-main` calls `http://127.0.0.1:8000/upload` and `/status` — neither route exists on the unified backend. **So requirement #8 ("frontend currently used by the unified system") cannot be satisfied from inside `CricketUnified/`, and deleting either frontend is forbidden by the "do not delete the source repositories" instruction.** No frontend was deleted. This gap is recorded, not fixed, because fixing it would be implementation work, which this task forbids. |
| **`data/samples/full_fixture.mp4`** (10,857,411 bytes) | Generated by `tools/build_test_fixture.py`, so it is reproducible output. However, it **is actively used** as an input by 8 tools and cited in 2 documents, and `SAMPLES_DIR` is a declared path at `app/core/config.py:53`. Not "unused". Kept. |
| **`data/uploads/`** (empty) | Declared at `app/core/config.py:51` as `UPLOADS_DIR` and created by `ensure_base_dirs()`. Zero files. Kept as a declared runtime directory. |
| **`data/analyses/`** (the parent directory) | Declared at `app/core/config.py:52` and created by `ensure_base_dirs()` (`config.py:336-338`). Its *contents* were removed; the directory itself is required. Kept. |
| **Dead symbols inside live modules** | 16 unreferenced symbols were found (e.g. `draw_prediction_arc`, `draw_hud`, `draw_mini_pitchmap`, `_trail_color`, `_glow_line` in `tracker_modules/renderer.py`; `predict_trajectory` in `trajectory.py`; `update_json_inplace` in `utils/io.py`; `JobRegistry.load_result` in `api/deps.py`). **They were NOT deleted** — removing code from inside a module is a refactor, and this task is cleanup of *files and folders* only. Recorded here for a future refactor task. |
| **`.claude/settings.local.json`** | Local assistant/tooling permission config, not project code and not referenced by the pipeline. Outside the scope of "files no longer needed after the merge" for the application. Kept. |
| **Undocumented-but-live tools** | `diagnose_real_deliveries.py` is not cited in any doc, but is provably self-sufficient (regenerates its own output; consumes `real_cricket.mp4`). Kept rather than deleted under the "do not delete what merely looks old" rule. |

### 4.3 Production code kept — `CricketUnified/backend/app/**` (30 modules)

Complete import graph, traced from `main.py`:

```
main.py
 ├─ app.api.endpoints ──── app.reference ──── app.services.analytics
 │   ├─ app.core.config                            app.core.constants
 │   │   └─ app.utils.io
 │   └─ app.api.deps ────── app.services.pipeline
 │                            ├─ app.schemas.tracking
 │                            ├─ app.services.analytics, overlay_renderer,
 │                            │   shot_classifier, tracking_service
 │                            ├─ app.services.tracker_modules
 │                            └─ app.services.models      (function-local import, pipeline.py:218)
 └─ app.core.config
app.services.tracking_service
 ├─ app.services.{clip_validator, clip_writer, event_horizons,
 │                event_segmenter, ring_buffer}, tracker_modules
app.services.overlay_renderer ──── tracker_modules.renderer   (direct, line 45-49)
app.services.clip_validator ──── event_segmenter
app.services.analytics ──── tracker_modules + app.core.constants
```

**Result: zero dead modules.** Two modules that appear dead at first glance are in fact live and
are recorded here so a future reviewer does not delete them:

- `app/services/clip_validator.py` — imported **only** at `tracking_service.py:70`. A graph built
  from `deps.py`/`pipeline.py` alone would miss this and wrongly mark it dead.
- `app/services/models.py` — its only `app`-level importer is a deliberately function-local import
  at `pipeline.py:218` (`# local import: heavy`).

### 4.4 Tests kept — all 11 files (152 tests)

| File | Tests |
|---|---|
| `tests/__init__.py` | – |
| `tests/helpers.py` | – (`StubDetector`, `make_video`, `make_tagged_video`, `decode_grey_means`, `classify_tag`, `CountingVideoCapture`) |
| `tests/test_analytics_decisions.py` | 15 |
| `tests/test_api.py` | 16 |
| `tests/test_clip_isolation.py` | 17 |
| `tests/test_clip_writer.py` | 20 |
| `tests/test_event_horizons.py` | 16 |
| `tests/test_frame_mapping.py` | 16 |
| `tests/test_ring_buffer.py` | 13 |
| `tests/test_single_pass.py` | 28 |
| `tests/test_tracker_provenance.py` | 11 |

---

## 5. Change Manifest

### 5.1 Files removed — 294

| Category | Files |
|---|---|
| Generated analysis outputs (`data/analyses/**`) | 227 |
| `__pycache__` bytecode | 50 |
| `.pytest_cache` | 5 |
| Abandoned diagnostic scripts (`tools/`) | 12 |
| **Total** | **294** |

### 5.2 Folders removed — 50

| Category | Folders |
|---|---|
| Generated analysis directories | 36 (13 top-level + 23 nested `clips/`/`overlays/`) |
| `__pycache__` | 9 |
| `.pytest_cache` | 1 |
| Empty `CricketUnified/docs/` | 1 |
| **Total** | **50** (approx. — nested count includes empty leaf dirs) |

### 5.3 Space reclaimed — ~518 MB

### 5.4 Zero changes to functionality

No `.py`, `.ts`, `.tsx`, `.json`, `.yaml`, `.md` or model file inside `CricketUnified/backend/app`,
`CricketUnified/backend/tests`, `CricketUnified/backend/models`, either source repository, or
`docs/` was modified, moved, or renamed. The only files written in this entire task are
`docs/CLEANUP_REPORT.md` (new) and this manifest.

---

## 6. Verification Plan

| # | Check | Command |
|---|---|---|
| 1 | Full backend test suite | `python -m pytest tests -q` from `CricketUnified/backend` |
| 2 | Tests also pass under stdlib unittest | `python -m unittest discover -s tests -t .` |
| 3 | Python imports resolve | import every module in `app/` + `main` |
| 4 | Backend starts | `uvicorn main:app` + HTTP probe |
| 5 | Frontend builds | `npm run build` in each frontend |
| 6 | Model paths still resolve | assert both checkpoints exist and are non-zero |
| 7 | Unified pipeline still starts | `POST /api/upload` end-to-end via `TestClient` |
| 8 | No required file removed | import graph + test suite + path resolution re-check |

Baseline recorded **before** any deletion: **152 passed** (pytest 9.1.1, Python 3.10.11).

---

## 7. Post-Cleanup Verification Results

All eight checks executed. Environment: Windows / PowerShell 5.1, Python 3.10.11, pytest 9.1.1,
OpenCV 4.10.0, NumPy 2.1.3, FastAPI 0.115.6, Node v24.14.1, npm 11.11.0.

### 7.1 Complete backend test suite — PASS

| Runner | Before | After | Result |
|---|---|---|---|
| `python -m pytest tests -q` | **152 passed** (83.25 s) | **152 passed** (73.39 s) | Identical |
| `python -m unittest discover -s tests -t .` | 152 tests, `OK` | **`Ran 152 tests` → `OK`** (70.53 s) | Identical |

Zero failures, zero errors, zero skips in both runs. The only warning is a pre-existing
`starlette.testclient` `anyio.abc.BlockingPortal` deprecation, present identically before and after.

### 7.2 Python imports — PASS

All **26** modules imported individually with zero exceptions:

`main`, `app`, `app.api.endpoints`, `app.api.deps`, `app.core.config`, `app.core.constants`,
`app.reference`, `app.schemas.tracking`, `app.utils.io`, `app.services.pipeline`,
`app.services.analytics`, `app.services.models`, `app.services.ring_buffer`,
`app.services.event_horizons`, `app.services.event_segmenter`, `app.services.clip_writer`,
`app.services.clip_validator`, `app.services.tracking_service`, `app.services.shot_classifier`,
`app.services.overlay_renderer`, `app.services.tracker_modules`,
`app.services.tracker_modules.{detector,tracker,trajectory,renderer,utils}`.

### 7.3 Backend starts — PASS

`python -m uvicorn main:app --port 8077` started cleanly:

```
CricketUnified 1.0.0 (schema 2.0) starting
Data directory:      ...\CricketUnified\backend\data
Analyses directory:  ...\CricketUnified\backend\data\analyses
Application startup complete.
```

- `GET /api/analyses` → **200** `{"analyses":[]}` (correctly reports an empty analyses dir after
  cleanup — the mount point survived)
- `GET /openapi.json` → **200**, `title=CricketUnified`, `version=1.0.0`
- **All 13 API routes present**: `/api/health`, `/api/reference`, `/api/upload`, `/api/analyses`,
  `…/{id}/status`, `…/{id}/progress`, `…/{id}/cancel`, `…/{id}/result`, `…/{id}/tracking`,
  `…/{id}/deliveries/{delivery_id}`, `…/{id}/trajectory/{delivery_id}`, `…/{id}/clips/{name}`,
  `…/{id}/overlays/{name}`

### 7.4 Model paths still resolve — PASS

| Constant | Resolved path | Exists | Bytes |
|---|---|---|---|
| `BALL_MODEL_PATH` | `backend/models/ball_detector.pt` | ✅ | 6,225,450 |
| `SHOT_MODEL_PATH` | `backend/models/shot_classifier.ckpt` | ✅ | 72,010,842 |

Both **load and execute for real**:
- `BallDetector()` → `🚀 Detector using device: 0` / `✅ YOLO model loaded`, then ran inference on
  a real 1080p frame in 1.34 s.
- `UnifiedPipeline._build_classifier` → `load_shot_model(SHOT_MODEL_PATH, ImprovedSOTAModel)`
  returned a live model; `result.shot_model_available == True`.

### 7.5 Unified processing pipeline still starts — PASS (full end-to-end run)

A real end-to-end run was executed on real footage — 1,500 frames (60 s @ 25 fps, 1280×720) cut from
`real_cricket.mp4` — through the full single-pass pipeline with **both real models loaded**:

```
verification source: 1280x720 @ 25.0 fps, 1500 frames
🚀 Detector using device: 0
✅ YOLO model loaded
📹 Video FPS (from metadata): 25.000
PIPELINE OK in 50.8s
  deliveries       : 2
  ball model       : ball_detector.pt
  shot model       : shot_classifier.ckpt | available: True
  tracking.json    : True
  result.json      : True
  clips written    : 2
  overlays written : 2
  warnings         : 8 (all legitimate: 7 rejected low-evidence events, 1 untracked activity window)
```

Every required stage confirmed live: **event-horizon segmentation** (2 events emitted, 7 rejected
with reasons), **clip generation** (2 clips), **clip validation** (populated `classification`,
`event_count_estimate`, `confidence`, `flags`, `requires_human_review`), **shot classification**
(shot model available), **trajectory/bowling analytics** (`Delivery` carries `trajectory`,
`bowling`, `bounce`, `release_frame`, `speed_samples`, `event`, `quality`, `validation`), and
**overlay rendering** (2 overlays).

The run's output was deleted afterwards, restoring `data/analyses/` to empty.

*Non-defect observation:* one delivery logged `Shot inference skipped: got 19 of 30 frames`. That
is the documented short-clip fallback path in `shot_classifier.py` (a 19-frame event cannot supply
the 30 frames `SHOT_N_FRAMES` requires), not a consequence of cleanup.

### 7.6 Frontend builds — PASS (both)

There is **no frontend inside `CricketUnified`**; both frontends live in the two source
repositories, which this task forbids altering. Each was therefore installed, built, and then
**restored to its exact original state**.

| Frontend | Install | Build | Result |
|---|---|---|---|
| `CricketTracker-main/frontend/ballscribe-insight-main` | `npm ci` → 488 packages | `vite build`, 1722 modules | **✅ built in 24.21 s** → `index.html` 1.13 kB, `index.js` 393.96 kB (gzip 124.71), `index.css` 69.00 kB, `hero-cricket.jpg`, `favicon.ico`, `placeholder.svg`, `robots.txt` |
| `CricketShot-Classification-main/frontend` | `npm ci` → 162 packages | `tsc -b && vite build` (vite 8.3.0), 2869 modules | **✅ built in 4.47 s** → `index.html` 0.46 kB, `index.js` 764.29 kB (gzip 232.57), `index.css` 51.03 kB, `cricket-bg.mp4`, `favicon.svg`, `icons.svg` |

**Source repositories restored exactly** — the generated `node_modules/` and `dist/` were removed
from both:

| Repository | Files before | Files after | Stray artifacts |
|---|---|---|---|
| `CricketTracker-main` | 100 | **100** | 0 |
| `CricketShot-Classification-main` | 98 | **98** | 0 |

### 7.7 No required file accidentally removed — CONFIRMED

| Check | Evidence |
|---|---|
| Every `app/` module still imports | 26/26 imported, §7.2 |
| Test suite unchanged | 152 → 152, §7.1 |
| Both checkpoints present, loadable, executable | §7.4, §7.5 |
| All 13 API routes intact | §7.3 |
| Full pipeline completes a real run | §7.5 |
| Both frontends build | §7.6 |
| Both source repos byte-count identical | §7.6 |
| All 11 test files present | Final-state listing |
| All 5 architecture documents present | `docs/` untouched except the added report |
| `data/analyses/`, `data/uploads/`, `data/samples/` still exist | All `is_dir == True` (§7.2 run) |
| `data/samples/full_fixture.mp4` present | 10,857,411 bytes |

### 7.8 Caches cleared again after verification

Running the suite regenerated 51 `.pyc` files, `.pytest_cache/`, and 3 test-output directories
under `data/analyses/`. All were cleared a second time so the delivered tree is clean:

- 51 `__pycache__` `.pyc` files removed
- `.pytest_cache/` removed
- `data/analyses/{test_single_pass,test_event_horizons,test_clip_isolation}` removed

Final state: `data/analyses/` is empty but **present** (required, `app/core/config.py:52`), and
`data/uploads/` and `data/samples/` are intact.

---

## 8. Notable Findings Recorded, Not Fixed

This task was cleanup only. The following defects and gaps were discovered during the audit and
are **deliberately left in place**. They are listed so they are not lost.

| # | Finding | Location |
|---|---|---|
| 1 | **No frontend is wired to the unified API.** Both frontends call routes that do not exist on the unified backend (`POST /api/predict`, `GET /api/progress/{id}`, `DELETE /api/cancel/{id}`, `http://127.0.0.1:8000/upload`, `/status`). The API contract was changed during the merge but no client was migrated. Requirement #8 of the unified architecture is therefore **unmet**. | `CricketShot-Classification-main/frontend/src/services/api.ts:50,112,79`; `CricketTracker-main/frontend/…/Upload.tsx:47`, `Processing.tsx:29`, `Results.tsx:41` |
| 2 | **No `requirements.txt` / `pyproject.toml` exists for `CricketUnified`.** The only dependency manifests in the workspace belong to the two source repositories. The unified backend's dependencies are therefore undocumented. Creating one would be implementation work, which this task forbids. | `CricketUnified/` (absent) |
| 3 | **No `.gitignore` for `CricketUnified`.** 76 MB `real_cricket.mp4` and all generated `data/` output are untracked-and-unignored. | `CricketUnified/` (absent) |
| 4 | **No `conftest.py` / pytest config.** Tests rely on the CWD being `backend/` because `tests/` is a package importing `from tests.helpers import ...`. | `CricketUnified/backend/` (absent) |
| 5 | **16 dead symbols** inside live modules — largest cluster is ~190 unused lines in `tracker_modules/renderer.py` (`draw_prediction_arc`, `draw_hud`, `draw_mini_pitchmap`, `_trail_color`, `_glow_line`). Removing them is a refactor. | see §4.2 |
| 6 | `analytics.py:165-166` hard-codes `pitch_length_m: 20.12` / `crease_width_m: 2.64` instead of importing `PITCH_LENGTH_M` / `CREASE_WIDTH_M` from `core/constants.py:76-77`. They can drift silently. | `app/services/analytics.py:165-166` |
| 7 | `tracker_modules/__init__.py:16` claims `renderer.py` is "byte-identical to upstream"; it is not — `draw_trajectory_trail` was rewritten. | `app/services/tracker_modules/__init__.py:16` |
| 8 | `main.py:44-45` comment claims the frontends are served from a vite dev server on `:8080`; neither `vite.config.ts` sets `server.port` or a proxy. The real mechanism is `VITE_API_URL` + `allow_origins=["*"]`. | `backend/main.py:44-45` |
| 9 | `app/core/constants.py:24` cites `docs/DELIVERY_SEGMENTATION_DIAGNOSTIC.md`, which **does not exist**. | `app/core/constants.py:24` |
| 10 | `tests/helpers.py:8` cites `test_real.py`, which **does not exist**. No test loads the real models. | `backend/tests/helpers.py:8` |
| 11 | `tools/ab_test_the_in_loop_slowdown.py:47,94` builds analysis IDs from free-text labels, producing directory names truncated at `/` (see `ab_no clip window (pre`). | `tools/ab_test_the_in_loop_slowdown.py` |
| 12 | The containing git repo is the **entire user profile** (`C:/Users/LENOVO`) with **zero commits**, so nothing in this project is version-controlled and no deletion performed here is recoverable via git. | repository state |