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
  → FastAPI API                       6 routes under /api
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
| `app/pitch/detector.py` | Roboflow keypoint `RoboflowPitchDetector` + response parser (optional) |
| `app/pitch/calibration.py` | `derive_corners`, homography to the pitch unit square, `PitchSmoother` |
| `app/pitch/service.py` | `PitchService` — interval-gated, non-blocking, per-frame `delivery.pitch` |
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
| `app/api/routes/` | All HTTP routes (`analysis`, `media`, `system`) |
| `app/api/deps.py` | Job registry, `ModelCache`, weighted monotonic progress |
| `app/core/config.py` | Paths, tunables, calibration, checkpoint resolution |
| `app/core/constants.py` | Stage vocabulary, progress bands, segmentation + shot constants |
| `app/schemas/tracking.py` | Pydantic result schemas |
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
- The result carries `shot.classification` + `shot.confidence` per delivery, and the ball cards
  display both — the confidence is shown, not hidden, so a 23% call reads as a 23% call.

## Analytics honesty rule

Trajectory is authoritative and every point is explicitly `detected` or `predicted`.

Ground-plane analytics — `speed_kmh`, `length`, `line`, `swing`, `bounce_angle`, and ground-plane
coordinates — are computed **only when the geometry was measured on this video**: pitch corners
measured by an operator, or the pitch keypoint model's quad converted to metres (next section).
Without either they are `null`, never estimated from a template. The calibration block in the result
states which fields are suppressed and why. `speed_is_calibrated` is `false` by default;
`geometry_calibrated` and `ground_plane_analytics_available` describe whether the geometry is usable
at all, and `geometry_source` says which measurement it came from.

## Pitch keypoint detection (optional)

`app/pitch/` asks Roboflow for pitch keypoints on one frame every
`PITCH_DETECTION_INTERVAL` frames. Its only job is to say **where the pitch is
in the frame**. It is inert until `ROBOFLOW_API_KEY` is present (see
`.env.example` at the project root); a checkout with no `.env` behaves as if the
feature were never installed.

- The hook is `on_frame(frame_number, frame)` inside the tracking loop: it never
  blocks and never raises. The queue has room for two, drops the oldest, and the
  detector runs on its own daemon thread, so a slow or dead API cannot slow the
  pass or fail the analysis.
- Four accepted keypoints give a quad and a homography. Three give a calibration
  with no quad: `corners: {}`, `homography_available: false`, every
  `pitch_position: null`. A fourth corner is never invented.
- Per-frame states: `calibrated`, `temporary_loss` (last good quad, drawn as
  stale), `no_calibration`, `calibration_expired`.
- A delivery's block is scoped to its own window (`snapshot_for_delivery`): a
  quad measured anywhere inside the delivery describes that footage, so a quad
  that has gone stale by the delivery's LAST frame is reported as
  `temporary_loss` — labelled as previous, still usable — rather than discarded
  as expired. An expired quad is never used.
- Output: `delivery.pitch` on every served delivery, and `pitch_detection` in
  the persisted `result.json` — counters, effective config, last refusal/error.
  The API key is never recorded anywhere (`scrub()`).
- **It can feed analytics — but only when the quad can be oriented.**
  `app/analytics/pitch_geometry.py` scales the quad's long axis to
  `PITCH_LENGTH_M` (20.12 m) and its short axis to `CREASE_WIDTH_M` (2.64 m),
  then uses the direction the ball travelled (release → bounce, cut at the
  deepest image point so the post-bounce rise cannot vote) to decide which end
  of the quad the batter stood at. The result is a pixel→metres homography in
  the SAME convention the operator's corners use, so `speed_kmh`, `length`,
  `line`, `swing`, `bounce_angle` and the ground-plane coordinates are computed
  from it — with `length` now metres from the batting crease
  (`LENGTH_ZONES_M`). The assumption that the quad *is* that rectangle is
  quoted in `geometry_note`, not hidden.
- **Failure stays a reason, never a number.** A roughly square quad, travel too
  short or too even to name a direction, fewer than three usable detections —
  each is refused by name and counted in `pitch_keypoint_geometry.rejections`,
  and the affected delivery keeps `null` for every metre it cannot honestly
  report.
- **The map needs the orientation.** `pitch_position` alone is a fraction of
  the quad, not a compass bearing, so `pitch.orientation` (`axis`,
  `batting_end`, `travel_delta`, `calibration_frame`) is served beside it and
  the pitch map plots nothing without it — a frame-coordinate fallback would
  put bounces at the wrong end of the strip.

## Purity and quarantine

The purity gate runs before the shot and overlay stages, so an impure clip is never served.

| Verdict | Behaviour |
|---|---|
| `SINGLE_EVENT` | Served as a delivery |
| `AMBIGUOUS` | Served, flagged |
| `MULTIPLE_EVENTS`, `NO_EVENT`, missing verdict | Quarantined with full diagnostics; **not** served |

Delivery IDs are never renumbered after quarantine, so a quarantined ID simply
never appears in `deliveries` while the record survives in the `quarantine`
array of `result.json`.

## API

Six routes under `/api` — one spelling per route, no aliases, no static mount.
Full contract: `docs/API_CONTRACT.md`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analyze` | Upload (`video=` field) and start an analysis (202) |
| POST | `/api/upload` | Same handler, `file=` field (the other frontend's spelling) |
| GET | `/api/analysis/{id}/status` | `queued` / `running` / `completed` / `failed` + `progress` |
| GET | `/api/analysis/{id}/result` | The envelope: `{analysis_id, status, pitch, deliveries[]}` |
| GET | `/api/analysis/{id}/clips/{name}` | Delivery clip (range requests supported) |
| GET | `/api/health` | Health, versions, stage vocabulary |

Removed and deliberately absent: the plural `/api/analyses/{id}/…` aliases,
`/result/internal`, `/progress` + `/progress/json`, `/cancel`, every
`/deliveries…` and `/trajectory/{id}` route, the overlay routes, `/tracking`,
`/api/reference`, the `/api/analyses` listing, and the static `/data` mount.
They return 404; the tests assert the absence. The complete `result.json`
(summary, calibration, performance, quarantine) is written to disk and served by
no route.

Uploads are validated at the door: extension allow-list, size ceiling, non-empty
body, and an OpenCV readability probe, so a mislabelled file is rejected before
any model work starts. A failed run reports `status: "failed"` with `error`
set; an unknown id is `404`.

## Models

| File | Role |
|---|---|
| `backend/models/ball_detector.pt` (6,225,450 B) | CricketTracker YOLO11 ball detector |
| `backend/models/shot_classifier.ckpt` (72,010,842 B) | Adarsh shot classifier |

Both are referenced from `app/core/config.py` and resolved with no fallback to any other directory.

## Tests

`backend/tests/` — 21 test modules plus `helpers.py`. `python -m pytest -q` from `backend/`.
The suite is hermetic: it drives the real segmenter, merger, validator, writer and API with stub
detectors and synthetic frames, and does not require either checkpoint or a video.

## Running it

```powershell
cd backend
python -m pytest -q                 # tests
python -m uvicorn main:app --port 8000
python scripts\verification\e2e_real_video.py   # from the project root; real-video smoke test
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
| `scripts/verification/e2e_real_video.py` | Live HTTP smoke test; `--verify-only <id>` re-checks a stored run |
| `scripts/diagnostics/real_video_diagnostic.py` | Per-frame stage-by-stage diagnostic |
| `scripts/diagnostics/diagnose_real_deliveries.py` | Locates where deliveries are lost |
| `scripts/diagnostics/run_real_video_test.py` | Service-level real-video verification |
| `scripts/diagnostics/diagnose_results.py` | Explains a fixture run's output |
| `scripts/dataset/build_test_fixture.py` | Rebuilds `full_fixture.mp4` from `source_clips/` |