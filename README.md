# Cricket Video Analysis

Upload a cricket video; get back delivery clips, tracked trajectories, shot
labels, bowling analytics, and rendered overlays — from a single pass over the
frames.

## What this project does

Given a broadcast clip of a cricket match, it detects the ball in every frame,
tracks it, decides which frames belong to which delivery, cuts those deliveries
into their own clips, classifies the shot, computes bowling analytics, and draws
the result back onto the footage as an overlay. Everything is exposed over an
HTTP API.

The core design constraint is that **the source video is decoded exactly once**.
Detection and tracking each run exactly once per upload; nothing downstream
reopens the source. That is what makes an analysis affordable, and it is why
several code paths that would otherwise look reasonable are deliberately
forbidden.

## Architecture

```
              POST /api/analyze
                    │
                    ▼
        ┌───────────────────────┐
        │  DECODE (once)        │  ← the source is opened here and nowhere else
        └───────────┬───────────┘
                    │  frames + retained ring buffer
                    ▼
        ┌───────────────────────┐
        │  YOLO11 DETECT        │  ← one pass over every frame
        └───────────┬───────────┘
                    │  (x, y, confidence)
                    ▼
        ┌───────────────────────┐
        │  KALMAN TRACK         │  ← one pass; distinguishes detected from
        └───────────┬───────────┘     predicted points
                    │  positions + provenance
                    ▼
        ┌───────────────────────┐
        │  SEGMENTATION         │  candidate generation → activity barriers →
        │                       │  fragment merging → purity validation
        └───────────┬───────────┘
                    │  validated deliveries (IDs assigned once, never renumbered)
                    ▼
        ┌───────────────────────┐
        │  DELIVERY CLIPS       │  + ClipFrameMap (clip frame → source frame)
        └───────────┬───────────┘
                    │
      ┌─────────────┴─────────────┐
      ▼                           ▼
┌───────────────┐         ┌──────────────────┐
│ SHOT LABEL    │         │ BOWLING          │
│               │         │ ANALYTICS        │
└───────┬───────┘         └────────┬─────────┘
        └─────────────┬─────────────┘
                      ▼
        ┌───────────────────────┐
        │  OVERLAY + PERSIST    │  result.json, tracking.json
        └───────────────────────┘
                      │
                      ▼
                GET /api/analysis/{id}/...
```

Stages and who owns them:

| Stage | Module | Owns |
|---|---|---|
| Decode, detect, track | `app/tracking/` | the single pass; ring buffer; clip writing |
| Delivery boundaries | `app/segmentation/` | candidates, barriers, merging, purity gate |
| Shot classification | `app/shot_classification/` | the Adarsh model, unmodified |
| Speed, bounce, length, line | `app/analytics/` | what may be claimed about a delivery |
| Orchestration | `app/pipeline/` | stage order, persistence |
| Frames as pixels | `app/video/` | buffer, clips, overlay, probing |
| HTTP | `app/api/` | routing, validation, error shapes |

Two honesty rules are enforced in code rather than left to the caller:

- **Uncalibrated means null.** The homography is measured from one specific
  broadcast template. Applied to different footage it returns plausible numbers
  that are wrong — the test fixture produced 42–199 km/h with bounces off the end
  of a 20.12 m pitch. Without geometry measured on this video — pitch corners
  measured by an operator, or the pitch keypoint model's quad converted to
  metres — speed, length, line, swing and bounce angle are reported `null`, not
  estimated. Which of those two measured the ground plane is named in
  `calibration.geometry_source`.
- **A quarantined delivery is not a delivery.** `MULTIPLE_EVENTS`, `NO_EVENT`
  and missing verdicts are withheld with a 404 and recorded under `quarantine`
  with the reason. `AMBIGUOUS` deliveries *are* served, with their flag visible.

## Repository structure

```
.
├── backend/
│   ├── main.py                  FastAPI app
│   ├── models/                  ball_detector.pt, shot_classifier.ckpt, README.md
│   ├── app/
│   │   ├── core/                config, constants, logging
│   │   ├── schemas/             persisted contract (trading.py)
│   │   ├── pipeline/            orchestration
│   │   ├── tracking/            detector, kalman_tracker, tracking_service
│   │   ├── segmentation/        candidate_generation, barriers,
│   │   │                        fragment_merger, purity_validator
│   │   ├── shot_classification/ model, preprocessing, inference
│   │   ├── analytics/           bowling, trajectory, calibration
│   │   ├── video/               video_io, frame_buffer, clips, overlay, renderer
│   │   ├── storage/             durable JSON writes
│   │   └── api/                 deps, helpers, routes/{analysis,media,system}
│   └── tests/
│       ├── unit/                14 modules, no I/O beyond the filesystem
│       ├── integration/         7 modules, drive TrackingService end to end
│       └── helpers.py           synthetic video + stub detector
├── data/
│   ├── datasets/source_clips/   approved 41-clip evaluation set
│   ├── manifests/               checksums + provenance
│   ├── samples/                 generated fixtures
│   ├── runs/                    per-analysis output  (gitignored)
│   └── uploads/                 (gitignored)
├── scripts/
│   ├── dataset/                 build_test_fixture.py
│   ├── diagnostics/             real_video_diagnostic, diagnose_real_deliveries,
│   │                            run_real_video_test, diagnose_results
│   └── verification/            e2e_real_video.py
├── frontend/                    BallScribe UI — talks to /api through src/lib/api.ts
└── docs/
```

`docs/REPOSITORY_STRUCTURE_REPORT.md` records why this layout and what moved.

## The pipeline in detail

**Decode once.** `TrackingService` opens the video and reads it linearly. Frames
needed later go into a bounded `FrameRingBuffer` as JPEGs. That buffer is why the
shot classifier can label a delivery without a second decode.

**Detect once.** `BallDetector` wraps YOLO11. `detect_with_confidence()` returns
the confidence alongside the box, because segmentation thresholds against it — a
detector that returned only boxes would force the threshold to be guessed.

**Track once.** `BallTracker` (ported from CricketTracker) runs the Kalman filter
and returns a `TrackResult` marking each point `detected` or `predicted`. That
provenance survives into the API: a trajectory that is mostly Kalman extrapolation
says so, rather than presenting predicted points as measurements.

**Segment.** `EventSegmenter` accumulates evidence frame by frame from
`SignalTimeline` and commits a segment when the activity signal collapses.
`FragmentMerger` rejoins fragments of one delivery split by a brief detection
gap, without merging two real deliveries. `ClipValidator` makes the final call:
`SINGLE_EVENT` and flagged `AMBIGUOUS` are served, everything else is quarantined
with a reason.

**Cut.** `ClipWriterRegistry` writes `delivery_NNN.mp4` plus a `ClipFrameMap` that
maps clip positions back to original source frames. The overlay renderer and the
persisted trajectory read the *same* stored array, so a plotted path and a
rendered path cannot disagree.

**Label.** The Adarsh `ImprovedSOTAModel` takes `(1, 30, 3, 224, 224)` and
returns one of 10 shot classes. Frames already in memory are preferred; the
written clip is the fallback. A clip shorter than 30 frames has its sampled
positions repeated — the model has a fixed temporal length and no shorter path.
Repetition is recorded (`frames_repeated`, `unique_frames_sampled`), never
silent. A failed classification is recorded on the `Shot`, not raised: one
unclassifiable delivery must not fail a valid analysis.

**Analyse.** Speed comes from a least-squares fit of ground position against real
elapsed time, with one outlier-rejection pass, rather than averaging noisy
frame-to-frame differences. Bounce requires a local minimum in both raw pixel Y
and ground-plane Y. Length is METRES from the batting crease
(`LENGTH_ZONES_M`); a bounce that projects off the measured pitch gets `null`,
never the nearest zone. Line classification still takes pixel coordinates plus
the frame dimension — leg and off cannot be derived from a pitch quad, which
says nothing about which way the batter faces.

**Overlay.** Trajectory trail, bounce marker, release marker, prediction arc, and
a pitch map, all in coordinates the tracker actually measured. The map's length
bands are the measured metres on a 20.12 m pitch; its dots appear only for
deliveries whose bounce could be placed on that pitch.

## Models

| File | Purpose | MD5 |
|---|---|---|
| `backend/models/ball_detector.pt` | YOLO11 ball detection | `ADDE61B5409FCE5E…` |
| `backend/models/shot_classifier.ckpt` | Adarsh shot classification | `2974EFDCF449C8447BEF107D94633217` |

Neither is in git. `backend/models/README.md` has the full checksum, the input and
output contract for each, and the verification snippet. Both paths are overridable
via `CRICKET_MODELS_DIR`, `CRICKET_BALL_MODEL`, `CRICKET_SHOT_MODEL`.

## Setup

Requires Python 3.10+ and, for real speed, a CUDA GPU (CPU works, slowly).

```bash
pip install -r requirements.txt
```

The shot model checkpoint and the YOLO detector must both be present before the
first analysis. A missing checkpoint raises `FileNotFoundError` at startup naming
every path tried, rather than failing later as "shot model failed to load".

```bash
# or with Docker
docker compose up --build
```

## Running the backend

```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

`backend/` must be the working directory (or use `--app-dir backend`), because
`main.py` imports `app.*` and `backend/` is the import root.

Then:

```bash
curl http://localhost:8000/api/health
curl -F "video=@match.mp4" http://localhost:8000/api/analyze
curl http://localhost:8000/api/analysis/{id}/result
```

## Running an analysis

```bash
# From the project root. Runs the real pipeline over a real video and records
# what happened at each stage.
python scripts/diagnostics/real_video_diagnostic.py --video real_cricket.mp4

# Full HTTP round trip with the real detector and the real checkpoint, asserting
# the properties this README claims. 66 checks.
python scripts/verification/e2e_real_video.py real_cricket.mp4
```

Rebuild the synthetic fixture:

```bash
python scripts/dataset/build_test_fixture.py
```

## API endpoints

Base path `/api`. Six routes; the full contract is in `docs/API_CONTRACT.md`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/analyze` | upload (`video=` field); returns `202` with an `analysis_id` |
| `POST` | `/api/upload` | same handler, `file=` field (the other frontend's spelling) |
| `GET` | `/api/analysis/{id}/status` | `queued` / `running` / `completed` / `failed`, `progress`, `error` |
| `GET` | `/api/analysis/{id}/result` | `{analysis_id, status, pitch, deliveries[]}` — the envelope the UI renders |
| `GET` | `/api/analysis/{id}/clips/{name}` | clip bytes (range-supported), the route `clip_url` points at |
| `GET` | `/api/health` | liveness, schema + pipeline versions, stage vocabulary |

There is one spelling per route. The plural aliases, `/result/internal`,
`/progress` and `/progress/json`, `/cancel`, every `/deliveries…` and
`/trajectory/{id}` route, the overlay routes, `/tracking`, `/api/reference`, the
`/api/analyses` listing and the static `/data` mount are gone rather than
deprecated — all return 404, and the tests assert that. The complete
`result.json` (summary, calibration, performance, quarantine) is written to disk
and served by no route.

## Testing

```bash
cd backend
python -m pytest tests -q
```

489 passed, 1 skipped. `tests/unit/` covers pure logic;
`tests/integration/` drives `TrackingService` and the HTTP API end to end with a
stub detector, which keeps the suite fast and deterministic while
`scripts/verification/e2e_real_video.py` covers the real models.

`backend/conftest.py` redirects generated output to a temporary directory, so the
suite leaves `data/runs/` untouched and runs from the project root as well as from
`backend/`.

## Configuration

Every path and log level is overridable. See `data/README.md` for the full table.

| Variable | Default |
|---|---|
| `CRICKET_DATA_DIR` | `<project>/data` |
| `CRICKET_MODELS_DIR` | `<project>/backend/models` |
| `CRICKET_RUNS_DIR` | `<CRICKET_DATA_DIR>/runs` |
| `CRICKET_UPLOADS_DIR` | `<CRICKET_DATA_DIR>/uploads` |
| `CRICKET_BALL_MODEL` | `<CRICKET_MODELS_DIR>/ball_detector.pt` |
| `CRICKET_SHOT_MODEL` | `<CRICKET_MODELS_DIR>/shot_classifier.ckpt` |
| `CRICKET_LOG_LEVEL` | `INFO` |

Segmentation thresholds are `SegmentationConfig` in
`backend/app/core/config.py`; detector and tracking constants are in
`backend/app/core/constants.py`.

The optional Roboflow pitch-keypoint detector is configured the same way:
`ROBOFLOW_API_KEY`, `ROBOFLOW_API_URL`, `ROBOFLOW_MODEL_ID` and `PITCH_*`.
`.env.example` at the project root documents each one. With no key the feature
is inert — the server starts, analyses run unchanged, and every delivery reports
`pitch.state = "no_calibration"`.

## Dataset and evaluation

`data/datasets/source_clips/` holds 41 delivery clips from one full match, with
the upstream pipeline's shot label and tracking outcome for each.
`data/manifests/source_clips.json` carries a size and SHA-256 per clip plus
provenance.

Two limits worth stating plainly:

- **The labels are model output, not human annotation.** Fine for regression and
  consistency checks; not a ground-truth accuracy figure.
- **One match, one camera angle.** Enough to catch a regression, not enough to
  characterise performance. No accuracy claim here rests on it alone.

The synthetic fixture in `data/samples/` has synthetic *gaps* between clips. Any
figure derived from it describes that composite, not a real match.

## Known limitations

- **No pitch calibration.** Without per-video pitch-corner measurement, every
  ground-plane quantity is `null`. This is deliberate — see above.
- **Speed is an estimate.** A single fixed-camera homography cannot absorb
  broadcast foreshortening, so a hand-tuned scale and offset are applied and
  labelled `speed_is_estimate` everywhere.
- **One camera angle.** The segmentation thresholds assume a broadcast-style side
  view. Not validated on handheld or drone footage.
- **The UI expects the API on `:8000`.** `frontend/` posts to `/api/analyze`,
  polls `/api/analysis/{id}/status` and reads `/api/analysis/{id}/result`
  through `frontend/src/lib/api.ts`. There is no dev proxy, so a backend on
  another origin has to be given as `VITE_API_URL`.
- **Single-process by design.** One analysis holds the whole video in one
  process. Scale with replicas, not `--workers`.
- **OpenH264 unavailable here.** Clip encoding falls back to software and emits
  FFmpeg warnings on stderr; output is valid.
- **41 clips is a small evaluation set.** See above.

## Development workflow

1. Change the narrowest module that owns the behaviour. If you find yourself
   editing `pipeline/analysis_pipeline.py` to change what a clip *means*, the
   change belongs in `segmentation/`.
2. Add or update a test in `tests/unit/` or `tests/integration/`.
3. `python -m pytest tests -q` — expect 489 passing, 1 skipped.
4. For anything touching detection, tracking, segmentation, or models:
   `python scripts/verification/e2e_real_video.py real_cricket.mp4`.
5. Clean up `data/runs/test_*`.
6. For anything in `frontend/`: `npm run build && npm run lint` (and
   `npm run test`), run from `frontend/`.

Docs to read before changing anything: `docs/CURRENT_SYSTEM.md`,
`docs/UNIFIED_ARCHITECTURE.md`, `docs/API_CONTRACT.md`, and the module docstring
at the top of the file you are about to edit — they carry the reasoning behind the
invariants, not just the rules.