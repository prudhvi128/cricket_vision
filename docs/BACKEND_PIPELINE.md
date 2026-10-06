# Backend Pipeline

How `backend/` actually works, and why it is built this way. This
is the working reference; `API_CONTRACT.md` is what a client codes against.

---

## 1. The constraint that shapes everything

**The source video is decoded exactly once.**

Not "once per stage". Once. For a 9-minute broadcast segment that decode is the
single most expensive thing in the system, and reopening it per stage is how the
old pipeline ended up decoding the same 13,565 frames several times over.

This is enforced structurally, not by convention:

```python
# pipeline/analysis_pipeline.py
def run(self, source_video: str, paths: Paths, progress=None):
    service = TrackingService(self.config)
    tracking = service.analyse(source_video, paths, progress=emit, ...)
    # ... stages 2 and 3 below never receive `source_video`.
```

`run()` takes the source path, uses it only to build stage 1, and does not pass
it onwards. There is no code path from stage 2 or 3 back to the source file.

Stages 2 and 3 read either frames the tracking pass **retained in memory** or the
**clip file it wrote**. Both are permitted; reopening the source is not.

This is measured, not asserted. `tests/helpers.py` provides
`CountingVideoCapture`, which records every path handed to `cv2.VideoCapture`,
so `tests/test_single_pass.py` fails if the count is ever anything but 1.

---

## 2. Stage order

```
                    ┌─────────────────────────────────────────────┐
   source video ───► │ 1. TRACKING  (the only decode, the only pass)│
   13,565 frames     │    YOLO11 detect → gate → Kalman → record    │
                    │    event segmentation (interleaved)           │
                    │    per-delivery analytics (interleaved)       │
                    │    clip windowing + write (interleaved)      │
                    │    → tracking.json                           │
                    └─────────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────┐
                    │ 2. SHOT CLASSIFICATION                       │
                    │    in-memory frames, else the written clip    │
                    │    → never the source                         │
                    └─────────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────┐
                    │ 3. OVERLAY                                   │
                    │    reads the STORED trajectory, draws the clip│
                    └─────────────────────────────────────────────┘
                                          │
                                          ▼
                                          result.json
```

The API's `stage` field reflects these three plus `uploading` / `validation` /
`persisting`. Everything that happens *inside* the tracking pass is reported as
`substage`, because presenting interleaved work as a separate stage would
describe a pipeline this codebase deliberately does not have.

### Model loading is shared

Both models are process-wide singletons (`api/deps.py:ModelCache`), held under a
lock. Loading a 72 MB transformer checkpoint per request would dominate a
40-delivery run. `UnifiedPipeline` loads on first use and `ModelCache` harvests
the weights afterwards, so a second upload reuses them.

`ModelCache` uses an `RLock`, not a `Lock`: `pipeline()` holds the lock while
calling `detector()`, which acquires it again. A plain lock self-deadlocks on the
first request.

---

## 3. Stage 1 — tracking

`app/tracking/tracking_service.py`, orchestrated by `TrackingService._run`.

Per frame:

1. **Decode** — the only decode.
2. **Black-frame / scene-cut detection** — `black_frame_run_barrier` and
   `scene_cut_diff_threshold` are real barriers that stop an extraction window
   spanning a cut.
3. **Detect** — YOLO11, one forward pass.
4. **Gate** — detections below confidence, or failing a plausibility test, are
   rejected. Rejected detections are *evidence*, never measurements.
5. **Kalman** — smoothing between accepted detections, and extrapolation across
   short dropouts. `tracker_modules.py`.
6. **Record** — append the point with explicit provenance (`detected` /
   `predicted`).
7. **Stream into the open clip** — `clip_writer.py` writes straight from the
   frame; frames are never accumulated in RAM waiting for a clip to be decided.

Steps 2–7 all happen in the same loop iteration. That is why segmentation,
analytics and clip writing are substages rather than stages.

### Provenance

Every point carries `source`:

- `detected` — a real YOLO detection that survived the gates.
- `predicted` — a Kalman extrapolation. Real, but derived.

The speed and bounce fits use **detected points only**. Fitting a slope to a
filter's own output would launder an estimate into a measurement.

### Event segmentation

`app/segmentation/candidate_generation.py`. The segmenter owns delivery identity; the
Kalman filter only smooths *within* an event the segmenter has already ruled on.

Every threshold in `SegmentationConfig` is in **seconds or fractions of the
frame**, never in frames and never a delivery count. `resolve(fps, w, h)` converts
them into the specific video's frames and pixels, so the same configuration
segments a 25 fps 640×360 clip and a 50 fps 1080p clip on the same physical
terms.

There is deliberately **no `expected_deliveries` field**. The number of
deliveries is an output. An earlier pixel analysis that counted 41 contiguous
footage regions is diagnostic input, never a target.

Rejection is a decision with a recorded reason, not silence:

```
Discarded event at frames 565-573: only 7 confirmed detections, below the
minimum of 8. Kept in the segmentation evidence as a rejected event, not
emitted as a delivery.
```

The real run emits 96 such warnings. That is the pipeline explaining itself.

### Fragment merging

`app/segmentation/fragment_merger.py`. Two adjacent candidates may be merged only if
they look like one ball: bounded silence (`fragment_merge_max_gap_seconds`,
strictly inside `hard_split_seconds`), a reachable path, and no corridor
conflict. Merging is not an optimisation — an unmerged fragment is a second
delivery for one ball.

### Event horizons

A merged or rejected event still constrains its neighbours, so it is filed as a
*horizon* describing the frame span it occupies and why. A candidate is held
open for `merge_horizon_frames` before it is finalised — exactly the merge gap,
so waiting longer cannot change the outcome.

### Pre-persistence purity gate

Before a clip is written, the window is checked for the one-clip-one-event
invariant. A window that fails is refused **before any file is written**, and
filed as a horizon so the neighbourhood still segments correctly. This is why the
real run's quarantine contains no stray clips from this gate.

### Clip writing

`app/video/clips.py`, with a ring buffer (`clip_writer.py`,
`RingBuffer`) holding the last N frames as JPEGs so pre-roll can be reconstructed
without a second decode.

`ClipFrameMap` is the contract between the clip on disk and the source video.
It records the original-frame number of **every** written frame, so a trajectory
point at clip frame 41 can be stated as original frame 142. `frame_map_valid` is
persisted per clip; a map that could not be built is reported invalid, not
approximated.

### Independent post-pass validation

At end of pass, `clip_validator.py` re-derives how many events each *written*
clip holds, from stored tracking data, rather than trusting the segmentation
that produced it. This catches a merge made inside the segmenter — something the
pre-persistence gate structurally cannot see.

Both verdicts are kept. The accepted ones become deliveries; the rest are
quarantined. See `API_CONTRACT.md` §4.

### Persistence

`tracking.json` is rewritten after **every** delivery, so a crash leaves every
completed delivery recoverable. Writes are atomic (`utils/io.py:atomic_write_json`).

---

## 4. Stage 2 — shot classification

`app/shot_classification/inference.py`.

The model is Adarsh's `ImprovedSOTAModel`, weights
`backend/models/shot_classifier.ckpt` — byte-identical to upstream's
`cricket_model_transformer.ckpt` (MD5 `2974EFDCF449C8447BEF107D94633217`).
`resolve_shot_model_path()` accepts either filename, so the backend runs against
a checkout that kept either.

### The 30-frame contract

The tensor is fixed at `(B, 30, 3, 224, 224)`. Ten classes. This is not
negotiable by the architecture.

`sampled_indices()` returns **exactly 30 positions**, always, using
`np.linspace(0, n-1, 30)`:

| clip frames | sampled | unique | repeated |
| --- | --- | --- | --- |
| 150 | 30 | 30 | no |
| 30 | 30 | 30 | no |
| 21 | 30 | 21 | **yes** |
| 19 | 30 | 19 | **yes** |
| 0 | 0 | 0 | — (no frames to sample) |

The previous code returned whatever positions existed, so a 19-frame clip
produced a 19-frame input and the model received something its architecture
cannot accept. Short clips are now padded by deterministic repetition, and the
record declares it: `frames_sampled: 30`, `unique_frames_sampled: 19`,
`frames_repeated: true`.

Real footage contains 19-frame clips. The invariant is not that every clip is
long enough — it is that a short clip never masquerades as 30 distinct
observations. The repeats are the same ones upstream's inference code produces;
byte-identical weights plus a different sampler would be a silent distribution
shift.

### Frame sources, and why the mapping matters

`in_memory` is preferred: the frames the tracking pass already retained, keyed by
**clip-frame position**. The retained frames are a *sparse* subset of the clip, so
their positions are what tells the sampler which moments to use.

Collapsing the mapping to a dense list first — the original bug — renumbers the
frames and silently changes which 30 moments the model sees. The model still gets
30 frames, just the wrong 30, and the resulting label depends on bookkeeping
rather than pixels. `tests/test_single_pass.py::TestInMemoryFrameLookup` is a
regression guard for exactly this.

If the retained set does not cover a sampled position, the code falls back to
reading the written clip (permitted — the source is not reopened) and says so
via `input_source`. If it cannot, `shot.error` is set and `shot.type` is `null`.

A clip with a hole in its mapping is **not** silently filled with a substitute
frame.

### Failure is per delivery

One unclassifiable delivery does not fail the analysis, and is never reported as
a successful label:

```
Delivery 14: shot classification failed (<reason>); shot is null for this
delivery.
```

---

## 5. Stage 3 — overlay

`app/video/overlay.py`. Draws onto the **written clip** using the
**stored trajectory** — the same array the API serves and the analytics were
fitted to. A plotted path and a rendered path cannot disagree.

A failed overlay leaves the clip and every measurement intact; only the video is
missing, and the reason travels with the record:

```json
"overlay": { "rendered": false, "error": "<reason>" }
```

The route then returns `404` rather than serving the clip as an overlay.

---

## 6. Analytics and the calibration decision

`app/analytics/bowling.py`.

```python
def analyse_delivery(..., geometry_calibrated: bool = False) -> ...
```

**The default is `False`, not `True`.**

`analyse_delivery` is a pure function that receives a homography. It cannot know
whether that homography was built from corners measured on *this* video or copied
from a fixed broadcast template — the latter is what `calibration_metadata()`
describes. Defaulting to calibrated made the suppression branch unreachable, and
that is precisely how a fabricated 42–199 km/h range reached `result.json`.

The caller states the truth via `PipelineConfig.pitch_corners_px`. It is `None`
by default and no API route supplies corners, so every deployment of this backend
is honestly uncalibrated until an operator measures corners on the specific
footage.

**Suppressed when uncalibrated** (`null`, not estimated):

`speed_kmh` · `length` · `line` · `swing` · `bounce_angle` ·
`bounce.ground_x_m` · `bounce.ground_y_m` · `speed_samples[].ground_*_m`

**Unaffected** (pixel space, always real):

`release_frame` · `bounce.x_px` · `bounce.y_px` · `bounce.x_norm` ·
`bounce.y_norm` · `bowling.release_angle` · the whole trajectory

`speed_is_calibrated` is `false` even *with* corners, and stays false
deliberately: the speed is a least-squares fit over a hand-tuned
`scale=2.0 / offset=-20.0`, so it remains an estimate. The signal that the ground
plane is trustworthy is `geometry_calibrated` /
`ground_plane_analytics_available`.

### No imputation

`impute_missing_speeds` defaults to `False`. A delivery whose speed cannot be
computed reports `null`; it is never replaced by a median, by another ball's
speed, or by a plausible-looking constant.

---

## 7. Purity policy

`PipelineConfig.accepted_validation_classes` controls the gate. It is read from
the **live config**, not a fresh default — a caller that tightens the set must
have that honoured.

| Verdict | Delivered? |
| --- | --- |
| `SINGLE_EVENT` | yes |
| `AMBIGUOUS` | yes, flagged; verdict + confidence + `requires_human_review` travel with the record |
| `MULTIPLE_EVENTS` | quarantined |
| `NO_EVENT` | quarantined |
| no verdict | quarantined — silence is not a pass |

`AMBIGUOUS` is kept because the requirement is not to *claim* an undecidable
clip is valid, and it is not claimed. A client wanting only unambiguous
deliveries filters on `validation.classification`. The backend does not silently
drop a third of a real over's deliveries to tidy its output.

Quarantined records are persisted in full under `quarantine`, including the
reason and the complete delivery, so the omission is auditable. **Delivery ids
are not renumbered**, so `delivery_017.mp4` stays unambiguous on disk while the
id 404s on the delivery routes.

---

## 8. Progress that means something

`api/deps.py`.

`progress` is a function of **measured units**, not of elapsed time:

| Stage | Band | Unit |
| --- | --- | --- |
| `uploading` | 0–5% | bytes written |
| `tracking` | 5–62% | frames decoded |
| `validation` | 62–68% | clips re-segmented |
| `shot_classification` | 68–86% | deliveries classified |
| `overlay` | 86–96% | overlays rendered |
| `persisting` | 96–100% | documents written |

Bands are contiguous by construction (`tests/test_api_contract.py` asserts it),
so a client can render a segmented bar without gaps.

A high-water mark makes the value monotonic. Each stage counts its own units, so
the handoff from tracking (13565/13565 → 62.0%) to shot classification (0/37 →
68.0% floor) computes a *smaller* number than tracking had reached; without the
clamp a client drawing `percent` would visibly rewind.

When a stage has begun but does not yet know its denominator, it reports the
**floor of its band**, not a guess. `units` carries the raw counts so a client
can show "3,042 / 13,565 frames" instead of a rounded number.

Only a genuinely finished pipeline reports `100.0`.

---

## 9. Layout

```
backend/
  app/
    api/
      endpoints.py     HTTP routes; one handler per behaviour, aliases delegate
      deps.py          job registry, model cache, progress accounting
    core/
      config.py        paths, PipelineConfig, SegmentationConfig, calibration
      constants.py     every threshold, and why it has the value it has
    schemas/
      tracking.py      Delivery, Trajectory, Shot, Bowling, Bounce
    services/
      pipeline.py      stage orchestration + purity gate
      tracking_service.py   the single pass
      event_segmenter.py    delivery identity
      fragment_merger.py    fragment safety
      clip_validator.py     purity verdicts
      clip_writer.py        ring buffer + clip writing
      shot_classifier.py    30-frame sampling + inference
      overlay_renderer.py   overlay from the stored trajectory
      analytics.py          physics, calibration gate
  models/
    ball_detector.pt
    shot_classifier.ckpt   byte-identical to upstream cricket_model_transformer.ckpt
  data/runs/{id}/
    source.*  clips/  overlays/  tracking.json  result.json
  tests/                 324 tests
  tools/
    e2e_real_video.py    real-video verification over HTTP
```

### Artifacts per analysis

| File | Contents |
| --- | --- |
| `tracking.json` | what the single pass measured. No shot labels. |
| `result.json` | validated deliveries + quarantine + calibration + config snapshot |
| `clips/delivery_NNN.mp4` | the extracted clip |
| `overlays/delivery_NNN.mp4` | clip with the trajectory drawn on it |

---

## 10. Running

```powershell
cd backend
python -m uvicorn main:app --reload --port 8000
```

```powershell
python -m pytest -q --no-header -p no:randomly        # 324 tests
python tools/e2e_real_video.py                       # real video, ~17 min
python tools/e2e_real_video.py --verify-only <id>    # re-check, seconds
```