# Final Backend Validation

Evidence for the claims this backend makes. Every number below was produced by
running the code, not by reading it.

- **Run:** `backend/tools/e2e_real_video.py`, real footage through the real HTTP
  API with the real detector and the real shot checkpoint.
- **Video:** `real_cricket.mp4`, 79.9 MB â€” 13,565 frames, 25 fps, 1280Ã—720,
  542.6 s.
- **Analysis id:** `4c458118e2f04112`
- **Result:** **41 / 41 checks passed.** Unit suite: **324 passed, 1 skipped.**

Artefacts were written to
`data/runs/4c458118e2f04112/` and have since been deleted as generated
output during repository cleanup (823.7 MiB, 80 files — 37 clips, 37 overlays, a
copy of the source video, `result.json`, `tracking.json`). Regenerate them with:

```powershell
cd backend
python tools\e2e_real_video.py          # ~13 min; prints a fresh analysis id
```

The pipeline is deterministic for a fixed input, so a re-run reproduces the same
40 candidates / 37 deliveries / 3 quarantined split. This report records the
measured outcome; it is not re-verified by reading the artefacts back.

---

## 1. Headline

| | Phase 3 (before) | This run |
| --- | --- | --- |
| Candidates produced | 40 | 40 |
| Served as deliveries | **40** | **37** |
| `MULTIPLE_EVENTS` served as deliveries | **3** | **0** |
| Quarantined with a recorded reason | 0 | 3 |
| Clips shorter than the model's 30-frame input | 4 (unhandled) | 4 (padded, declared) |
| Deliveries with a fabricated speed | **33** | **0** |
| Length / line zones on uncalibrated footage | populated | **0** |
| `mean_speed_kmh` | number | **`null`** |
| Source decodes | 1 | 1 |
| YOLO passes / Kalman passes | 1 / 1 | 1 / 1 |

The two defects that mattered â€” impure clips presented as deliveries, and
invented speeds on uncalibrated footage â€” are both closed and both asserted by
tests, so they cannot regress silently.

---

## 2. Single-pass guarantee

Measured, not claimed. `CountingVideoCapture` records every path handed to
`cv2.VideoCapture`.

```
source_decodes      1
yolo_passes         1
kalman_passes       1
shot_model_loads    1
shot_model_file     shot_classifier.ckpt
shot_model_prov     cricket_model_transformer
frames_decoded      13565 / 13565
```

`stage_seconds`: `detect 423.0` Â· `ring_encode 343.8` Â·
`record_and_finalise 97.5` Â· `clip_write 97.5` Â· `decode 61.0` Â· `kalman 0.5`.

`kalman 0.5 s` against `detect 423.0 s` is the shape you would expect if the
Kalman filter is a smoothing step on a sparse detection stream rather than a
second pass over every frame.

---

## 3. Purity validation

| Verdict | Count |
| --- | --- |
| `SINGLE_EVENT` | 24 |
| `AMBIGUOUS` | 13 |
| `MULTIPLE_EVENTS` | **3 â†’ quarantined** |
| Served as deliveries | **37** |

Every served delivery has a verdict. No delivery is `MULTIPLE_EVENTS` or
`NO_EVENT`. The three quarantined ids are **17, 19, 22**:

```
delivery 17  MULTIPLE_EVENTS  quarantined: MULTIPLE_EVENTS â€” clip spans 2 detected ball events
delivery 19  MULTIPLE_EVENTS  quarantined: MULTIPLE_EVENTS â€” clip spans 2 detected ball events
delivery 22  MULTIPLE_EVENTS  quarantined: MULTIPLE_EVENTS â€” clip spans 2 detected ball events
```

Verified over HTTP:

- `GET /api/analysis/{id}/deliveries/17` â†’ **404**
- `GET /api/analysis/{id}/result` â†’ `quarantine` holds the full record + reason
- Delivery ids are **not renumbered**, so `delivery_017.mp4` stays unambiguous on
  disk while id 17 is unreachable as a delivery.

The 13 `AMBIGUOUS` records are served, each carrying its verdict, confidence and
`requires_human_review`, and counted separately in `summary.by_validation`. They
are not presented as valid â€” they are presented as undecided, with the decision
attached.

---

## 4. Calibration honesty

No pitch corners were measured for this footage, and no API route supplies any.

```
0 of 37 deliveries carry a speed_kmh
0 of 37 carry a length or line zone
summary.mean_speed_kmh           null
summary.with_speed               0
config.geometry_calibrated       false
config.pitch_corners_px          null
calibration.ground_plane_available false
```

Suppressed as `null`: `speed_kmh`, `length`, `line`, `swing`, `bounce_angle`,
`bounce.ground_x_m`, `bounce.ground_y_m`, `speed_samples[].ground_*_m`.

Retained as real: the full trajectory, `bounce.x_px/y_px/x_norm/y_norm`,
`release_frame`, `release_angle`.

The calibration block travels in the same payload as any number it qualifies, and
the config snapshot is persisted, so a reader of `result.json` cannot see a speed
without also seeing that the ground plane was never calibrated for this video.

Phase 3 reported 33 deliveries with speeds spanning 42â€“199 km/h from a homography
built from fixed corner fractions measured on a different broadcast template.

---

## 5. The 30-frame contract

The model tensor is fixed at `(B, 30, 3, 224, 224)`.

```
all 37 deliveries sampled exactly 30 frames
every sampled clip position is inside its clip
0 shot failures
```

Four clips in this footage are shorter than 30 frames. They are padded by
deterministic `linspace` repetition, and the padding is declared on the record:

| Delivery | Clip frames | Sampled | Unique | `frames_repeated` |
| --- | --- | --- | --- | --- |
| 2 | 19 | 30 | 19 | `true` |
| 7 | 21 | 30 | 21 | `true` |
| 14 | 21 | 30 | 21 | `true` |
| 37 | 19 | 30 | 19 | `true` |

Sampled positions for delivery 2 (19 frames â†’ 30 samples):
`[0, 0, 1, 1, 2, 3, 3, 4, ...]`

Previously these produced a 19- or 21-frame tensor, which the model's
architecture cannot accept. Byte-identical weights plus a different sampler would
have been a silent distribution shift, so the repeats are the ones upstream's own
inference code produces, asserted against `np.linspace` in the test suite.

The invariant is not that every clip is long enough â€” real footage contains
19-frame clips â€” it is that a short clip never masquerades as 30 distinct
observations.

---

## 6. Shot classification

37 of 37 labelled, 0 failures. Checkpoint identified as
`shot_classifier.ckpt`, provenance name `cricket_model_transformer`, and every
delivery's `provenance.shot_model` matches.

| Shot | Count |
| --- | --- |
| Late Cut | 9 |
| Lofted | 7 |
| Flick | 6 |
| Sweep | 6 |
| Cover | 4 |
| Square Cut | 3 |
| Straight | 2 |

Confidence range **0.1705 â€“ 0.5636**. Reported as-is, with the full class
probability distribution. No threshold is applied and no confidence is
rounded up; a low-confidence label is surfaced with its confidence so a client
can decide, rather than hidden by the backend.

Frame sources: 31 `in_memory`, 6 `clip_file` (frames not retained in memory, so
the written clip was re-read â€” permitted, and the source was never reopened).
`tracking_source` is `single_pass` for all 37; no delivery fell back to
re-tracking from a clip.

---

## 7. Media and per-delivery integrity

```
37 / 37 clips served          37 / 37 overlays served
37 / 37 clips decodable      byte-range request â†’ 206
```

Every trajectory point lies inside its own clip's frame window â€” verified for all
37. Trajectory provenance is explicit: **538 detected points**, plus predicted
points, each tagged. Detected and predicted points are never presented
indistinguishably.

Cross-delivery isolation, over HTTP:

- repeating a delivery request returns a byte-identical response;
- a delivery response contains only its own id's fields;
- two deliveries yield two different trajectories;
- `/deliveries/3` and `/deliveries/delivery_003` return the same record;
- a quarantined id is not reachable as a delivery.

There is no endpoint that mixes two deliveries, so a client cannot assemble a
shot from one ball and a trajectory from another.

---

## 8. Progress

523 polls over the run.

```
5.0% â†’ 62.0%   tracking              frames 0 â†’ 13565
              substages interleaved: analytics, validation
     62.0% â†’ 68.0%  validation
     68.0% â†’ 86.0%  shot_classification  0 â†’ 37 deliveries
     86.0% â†’ 96.0%  overlay              0 â†’ 37 overlays
     96.0% â†’ 100.0%  persisting
100.0%          completed
```

- **Never moved backwards.** Asserted over all 523 samples and in the event log.
- **Never overstated.** A stage that has begun but does not yet know its
  denominator reports its band floor.
- `progress` and `percent` are the same value, so the existing frontend and the
  documented contract cannot disagree.
- Interleaved work appears as `substage` and never moves `progress`. Presenting it
  as a top-level stage would describe a pipeline this backend does not have.

---

## 9. Test suite

```
324 passed, 1 skipped, 1 warning in 309.03s
```

Baseline before this work: 280 passed. The 44 new tests cover the parts that
were added or deliberately changed:

| Area | What is asserted |
| --- | --- |
| `test_api_contract.py` | documented routes exist; legacy plural routes survive; both `delivery_id` spellings |
| | progress bands are contiguous 0â†’100; monotonic across stage handoff; substages do not move progress |
| | purity gate quarantines `MULTIPLE_EVENTS`/`NO_EVENT`/no-verdict; keeps `AMBIGUOUS` flagged; honours a custom accepted set |
| | per-delivery responses cannot mix deliveries; trajectory frames nest inside their clip |
| | uncalibrated analytics reach the API as `null`; a calibrated config *is* recorded as calibrated |
| | upload validation rejects non-video content at the door and leaves no debris |
| | missing clip/overlay report honestly rather than substituting |
| `test_single_pass.py` | short clips yield exactly 30 positions, deterministically, matching upstream's sampler |
| `test_analytics_decisions.py` | the uncalibrated default suppresses ground-plane analytics and keeps pixel-space results |

The skip is the synthetic fixture that produces no quarantine case; the gate
itself is covered by the direct unit tests.

---

## 10. Deliberate limitations

Stated plainly rather than left to be discovered.

1. **No calibrated geometry.** No pitch-corner measurement procedure and no API
   input for one. Every run is honestly uncalibrated; speed, length, line, swing
   and all ground-plane metres are `null`. Pixel-space results are unaffected.
   Supplying `PipelineConfig.pitch_corners_px` with measured corners activates
   them, and the result document then records itself as calibrated â€” asserted in
   both directions so the document can neither overstate nor understate.

2. **Shot confidences are low.** 0.17â€“0.56 on this footage. Reported verbatim
   with the full distribution. This backend does not improve the checkpoint; it
   makes the checkpoint's output attributable, reproducible and auditable, and
   makes a failure visible instead of silent.

3. **`AMBIGUOUS` is served.** 13 of 37 deliveries are undecided by the
   validator. They are exposed because suppressing them would silently discard a
   third of a real over's deliveries; they are flagged so a client can filter.

4. **Throughput is 14.6 fps.** 13,565 frames in 926 s on CPU, dominated by YOLO
   inference (423 s) and ring-buffer JPEG encoding (344 s). No GPU acceleration
   and no hardware codec: OpenCV logs
   `Could not open codec libopenh264` and falls back to a software encoder for
   clip writing. Total pipeline time 1048 s for 542 s of footage, i.e. roughly
   0.5Ã— realtime. This is a correctness-first pipeline, not a fast one.

5. **Progress is in-memory.** The job registry holds transient progress keyed by
   `analysis_id`. Results persist to disk and survive a restart; progress does
   not, and `/status` reconstructs a `completed` response from disk for an
   analysis this process did not run.

---

## 11. Reproducing

```powershell
cd backend

# unit suite
python -m pytest -q --no-header -p no:randomly

# real video, end to end, ~17 min
python tools/e2e_real_video.py

# re-verify an existing analysis in seconds (needs its artefacts on disk;
# run the command above first, since generated artefacts are not retained)
python tools\e2e_real_video.py --verify-only <analysis_id>
```

The verification script is a script rather than a pytest test on purpose: it
takes minutes, loads both models, and is meant to be run deliberately against
real footage instead of as part of the unit suite.

---

## 12. Files

| File | Purpose |
| --- | --- |
| `backend/tools/e2e_real_video.py` | the 41 checks in this document |
| `backend/tests/test_api_contract.py` | the new contract, asserted |
| `backend/app/api/endpoints.py` | routes |
| `backend/app/api/deps.py` | job registry, model cache, progress |
| `backend/app/pipeline/analysis_pipeline.py` | stage orchestration, purity gate |
| `backend/app/tracking/tracking_service.py` | the single pass |
| `backend/app/shot_classification/inference.py` | 30-frame sampling, inference |
| `backend/app/analytics/bowling.py` | physics, calibration gate |
| `docs/API_CONTRACT.md` | what a client codes against |
| `docs/BACKEND_PIPELINE.md` | how it works and why |