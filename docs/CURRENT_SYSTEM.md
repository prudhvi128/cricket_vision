# Current System

The architecture that exists in this repository **right now**. Anything not described here is not
part of the system. For why each file is present, see `docs/REPOSITORY_CLEANUP_AUDIT.md`.

- Pipeline version: **1.1.0** · Result schema: **2.1**
- Entry point: `backend/main.py`
- Runtime root: the project root

## What the system does

```
uploaded cricket stadium video
  → single full-video decode          (one pass, one YOLO11 predict, one Kalman update)
  → YOLO11 ball detection             CricketTracker weights, 1 class
  → Kalman ball tracking              track state, reseed, missed-frame recovery
  → delivery / event segmentation     candidates from confirmed detections
  → event horizons + barriers         activity windows and clip windows
  → fragment merging                  adjacent fragments joined under a merge gate
  → purity validation                 each clip accepted or quarantined
  → individual delivery clips         per-delivery mp4 + clip↔frame map
  → shot classification               Adarsh EfficientNet-B0 + Transformer, 10 classes
  → trajectory / ball analytics       trajectory, bounce, swing, angles
  → overlay rendering                 annotated clip per delivery
  → persistence                       result.json + tracking.json + clips + overlays
  → FastAPI API                       24 routes under /api
```

**The single-pass rule.** The source video is decoded exactly once. YOLO11 runs exactly once and the
Kalman filter updates exactly once per frame. No stage re-decodes the source. Later stages read either
frames retained in memory (`ring_buffer.py`) or the already-written delivery clip. The result document
records this as `performance.source_decodes / yolo_passes / kalman_passes / shot_model_loads`, and the
E2E check asserts each equals 1.

## Modules

| Module | Responsibility |
|---|---|
| `app/tracking/tracking_service.py` | Drives the single pass: decode → detect → track → segment → validate → persist |
| `app/tracking/detector.py` | YOLO11 `BallDetector` |
| `app/tracking/kalman_tracker.py` | Ball track state + Kalman filter |
| `app/analytics/trajectory.py` | Ball trajectory construction |
| `app/analytics/calibration.py` | Homography, `pixel_to_ground`, Savitzky-Golay smoothing, `get_fps` |
| `app/video/renderer.py` | Overlay drawing primitives |
| `app/segmentation/candidate_generation.py` | Delivery/event candidate generation and boundaries |
| `app/segmentation/barriers.py` | `ActivityWindowTracker` — horizon and barrier logic |
| `app/segmentation/fragment_merger.py` | `FragmentMerger`, `BarrierSet` — joins adjacent fragments |
| `app/segmentation/purity_validator.py` | `ClipValidator` — delivery purity verdicts |
| `app/video/clips.py` | Writes clips and the clip↔original-frame map |
| `app/video/frame_buffer.py` | Retains decoded frames for later stages |
| `app/analytics/bowling.py` | `analyse_delivery`, `make_homography`, `calibration_metadata` |
| `app/video/overlay.py` | Per-delivery overlay assembly |
| `app/shot_classification/model.py` | `ImprovedSOTAModel` — the Adarsh shot architecture |
| `app/shot_classification/inference.py` | Adarsh exact preprocessing + 10-class inference |
| `app/pipeline/analysis_pipeline.py` | `UnifiedPipeline` — stage orchestration, quarantine, persistence |
| `app/api/endpoints.py` | All HTTP routes |
| `app/api/deps.py` | Job registry, `ModelCache`, weighted monotonic progress |
| `app/core/config.py` | Paths, tunables, calibration, checkpoint resolution |
| `app/core/constants.py` | Stage vocabulary, progress bands, segmentation + shot constants |
| `app/schemas/tracking.py` | Pydantic result schemas |
| `app/reference.py` | Shot classes, segmentation vocabulary, calibration rules for `/api/reference` |
| `app/utils/io.py` | Atomic JSON read/write |

## Stage vocabulary

`uploading → tracking → validation → shot_classification → overlay → persisting → completed`
(with `failed` as the terminal error state).

Progress is **weighted and monotonic**: each stage owns a fixed percentage band, and a high-water mark
prevents the value dropping when a stage hands over. Sub-stages are emitted from inside tracking
(`delivery_finalise`, `purity_validation`).

## Shot classification

- Architecture: EfficientNet-B0 → spatial attention → 1280→256 projection → sinusoidal positional
  encoding → 2-layer Transformer encoder → temporal attention → 10-class classifier.
- Input: `(B, 30, 3, 224, 224)`, ImageNet normalisation.
- Classes: `Cover, Defense, Flick, Hook, Late Cut, Lofted, Pull, Square Cut, Straight, Sweep`.
- A clip shorter than 30 frames is deterministically resampled with `np.linspace` and the record
  declares it via `frames_sampled`, `unique_frames_sampled`, `frames_repeated`.
- The 72 MB checkpoint is loaded **once per process** and reused across analyses by `ModelCache`.
- Every delivery records which checkpoint produced it.

## Analytics honesty rule

Trajectory is authoritative and every point is explicitly `detected` or `predicted`.

Ground-plane analytics — `speed_kmh`, `length`, `line`, `swing`, `bounce_angle`, and ground-plane
coordinates — are computed **only when the video is calibrated**. Without per-video pitch corners they
are `null`, never estimated from a template. The calibration block in the result states which fields
are suppressed and why. `speed_is_calibrated` is `false` by default; `geometry_calibrated` and
`ground_plane_analytics_available` describe whether the geometry is usable at all.

## Purity and quarantine

The purity gate runs before the shot and overlay stages, so an impure clip is never served.

| Verdict | Behaviour |
|---|---|
| `SINGLE_EVENT` | Served as a delivery |
| `AMBIGUOUS` | Served, flagged |
| `MULTIPLE_EVENTS`, `NO_EVENT`, missing verdict | Quarantined with full diagnostics; **not** served |

Delivery IDs are never renumbered after quarantine, so a quarantined ID returns 404 while the record
survives in the `quarantine` array.

## API

24 routes under `/api`, each also reachable through plural legacy aliases that delegate to the same
handler. Static analysis output is mounted at `/data`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analyze` | Upload and start an analysis (202) |
| POST | `/api/upload` | Legacy upload entry point |
| GET | `/api/analysis/{id}/status` | Terminal status |
| GET | `/api/analysis/{id}/progress` | SSE progress stream |
| GET | `/api/analysis/{id}/progress/json` | Progress snapshot |
| GET | `/api/analysis/{id}/deliveries` | Delivery list |
| GET | `/api/analysis/{id}/deliveries/{ref}` | One delivery |
| GET | `/api/analysis/{id}/deliveries/{ref}/clip` | Delivery clip (range requests supported) |
| GET | `/api/analysis/{id}/deliveries/{ref}/overlay` | Delivery overlay |
| GET | `/api/analysis/{id}/deliveries/{ref}/trajectory` | Delivery trajectory |
| GET | `/api/analysis/{id}/result` | Frontend array of deliveries (clean schema) |
| GET | `/api/analysis/{id}/result/internal` | Full persisted result document |
| GET | `/api/analyses/{id}/tracking` | Per-frame tracking log |
| POST | `/api/analysis/{id}/cancel` | Cancel a running analysis |
| GET | `/api/analyses` | Known analyses |
| GET | `/api/health` | Health, versions, stage vocabulary |
| GET | `/api/reference` | Shot classes, vocabulary, calibration rules |

Plural aliases: `/api/analyses/{id}/status|result|progress|deliveries/{ref}|trajectory/{ref}|cancel` and
`/api/analyses/{id}/clips/{name}`, `/api/analyses/{id}/overlays/{name}`.

Uploads are validated at the door: extension allow-list, size ceiling, non-empty body, and an OpenCV
readability probe, so a mislabelled file is rejected before any model work starts.

## Models

| File | Role |
|---|---|
| `backend/models/ball_detector.pt` (6,225,450 B) | CricketTracker YOLO11 ball detector |
| `backend/models/shot_classifier.ckpt` (72,010,842 B) | Adarsh shot classifier |

Both are referenced from `app/core/config.py` and resolved with no fallback to any other directory.

## Tests

`backend/tests/` — 15 test modules plus `helpers.py`. `python -m pytest -q` from `backend/`.
The suite is hermetic: it drives the real segmenter, merger, validator, writer and API with stub
detectors and synthetic frames, and does not require either checkpoint or a video.

## Running it

```powershell
cd backend
python -m pytest -q                 # tests
python -m uvicorn main:app --port 8000
python tools\e2e_real_video.py      # real-video stadium smoke test (uses ..\..\real_cricket.mp4)
```

## Data layout

| Path | Contents |
|---|---|
| `backend/data/uploads/` | Uploaded source videos |
| `data/runs/<id>/` | `result.json`, `tracking.json`, `source.mp4`, `clips/`, `overlays/` |
| `backend/data/samples/source_clips/` | The 41 real broadcast delivery clips (dataset) |
| `backend/data/samples/full_fixture.mp4` | Synthetic composite fixture built from those clips |
| `docs/` | Audit, architecture, diagnosis and validation records |

## Diagnostic tooling

`scripts/` holds read-only instrumentation that runs the real pipeline and records what
happened per frame. None of it modifies pipeline behaviour or thresholds.

| Script | Purpose |
|---|---|
| `backend/tools/e2e_real_video.py` | Live HTTP smoke test; `--verify-only <id>` re-checks a stored run |
| `tools/real_video_diagnostic.py` | Per-frame stage-by-stage diagnostic |
| `tools/diagnose_real_deliveries.py` | Locates where deliveries are lost |
| `tools/run_real_video_test.py` | Service-level real-video verification |
| `tools/diagnose_results.py` | Explains a fixture run's output |
| `tools/build_test_fixture.py` | Rebuilds `full_fixture.mp4` from `source_clips/` |