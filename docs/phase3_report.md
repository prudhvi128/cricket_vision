# Phase 3 Report — Backend Integration

**Status:** complete. Awaiting sign-off. Phase 4 not started.
**Scope:** the single-pass ball-processing backend only. No frontend work.
**Date:** 2026-10-04

---

## 1. Verdict

The backend does what the architecture specifies, and it does it on real footage with the real
models. Every architectural invariant that can be checked mechanically was checked and passed.

Three results are **not** good news and are reported here rather than buried:

1. Every shot was labelled `Lofted`. This is a **test-fixture artifact**, not a classifier result —
   68% of the frames the classifier sampled are synthetic black filler. Provenance in §5.1.
2. **12 of 17 speed estimates are outside the physically plausible range for cricket**
   (42–199 km/h). This is measurement, not opinion, and it confirms that D2's calibration is not
   merely a caveat — on this footage the numbers are wrong. Provenance in §5.2.
3. **21 deliveries were found, not 41.** D3 predicted this; it still needs a side-by-side
   accuracy comparison before sign-off.

Everything else — the single decode, frame mapping, provenance, D1, clip validity, overlay
validity — passed.

---

## 2. What was actually tested

### 2.1 The unit and integration suite

`backend/tests/`, stdlib `unittest` (pytest is not available in this environment).

```
python -m unittest discover -s tests -t .
Ran 119 tests in 54.690s
OK
```

`httpx` was installed solely to satisfy FastAPI's `TestClient`. No other dependency was added.

| File | Covers |
|---|---|
| `test_ring_buffer.py` | byte cap, frame cap, oversize-frame refusal, eviction order |
| `test_frame_mapping.py` | `original_frame → clip_frame` round-trip, INVARIANT B |
| `test_tracker_provenance.py` | detected vs predicted is never fabricated away |
| `test_analytics_decisions.py` | D1 (no imputation by default), D2 (calibration disclosed) |
| `test_clip_writer.py` | pre/post-roll windowing, codec fallback, `isOpened()` gate |
| `test_single_pass.py` | one decode, one detector call/frame, no third pass, shot sampling |
| `test_api.py` | schemas, SSE progress, `/cancel`, calibration in the payload |

### 2.2 The real-video run

`tools/run_real_video_test.py` against `backend/data/samples/full_fixture.mp4`.

**The fixture must be described honestly, because it determines what the results mean.**
It is 2,870 frames, 640×360, 25 fps, 114.8 s, built by `tools/build_test_fixture.py` from
**41 genuine delivery clips** taken from the Shot repository's `full_match_output/`, separated by
**1,845 synthetic black frames**. So:

- the cricket pixels are real;
- the *container* is synthetic, and the camera framing is discontinuous between deliveries;
- the ball is genuinely absent from the gaps, which is why segmentation works as intended.

Three of the 41 embedded clips are not detectable as deliveries (see §5.3).

### 2.3 Environment

- Both `best.pt` files were byte-identical — SHA256
  `8ECD4F8B03602A49BE4AB910BD1EE41545E9EB641A544EC9F9DFBEB6B7E6380F`, 6,225,450 bytes.
  CricketTracker's copy is authoritative. Copied to `backend/models/ball_detector.pt`; both
  originals re-hashed after the copy to confirm they were untouched.
- The shot checkpoint loads on CUDA, `strict=True`, 5,953,959 params, forward output `(1, 10)`.
- Detector model classes: `{0: 'ball'}`.
- OpenCV logs `Failed to load OpenH264 library` / `Could not open codec libopenh264` on every
  clip write. **This is benign and the codec fallback chain never engages** — measured directly:
  `ClipWriter` reports `codec_used='avc1'`, `codec_errors=[]`, and all 21 written clips report
  container FourCC `h264`. OpenCV's ffmpeg backend tries the OpenH264 DLL, fails, and falls back
  *internally* to a working H.264 encoder. The `isOpened()` check at open time is what established
  this correctly; a hardcoded `avc1` with no check would have looked identical.

---

## 3. Results

### 3.1 Headline

```
Source decodes .............. 1 (counted: 1)      <- the central claim, instrumented
Frames decoded ............... 2870
Wall time .................... 80.0s
  tracking ................... 62.4s
  shot ....................... 4.3s
  overlay .................... 13.3s
Processing rate .............. 35.9 fps
Deliveries found ............. 21
```

The source video was opened **exactly once**, proven by a `cv2.VideoCapture` wrapper that records
every path opened — not by assertion.

### 3.2 Clips and the frame map

```
Clips written ................ 21
  codecs ..................... ['avc1']            (all 21 confirmed h264 on disk)
  frame maps valid / invalid  21 / 0
  clip frames ................. min 72, median 172, max 183, total 3530
```

3,530 clip frames were written from a 2,870-frame source. The clip total exceeds the source
because each delivery's window overlaps its neighbours' — pre-roll reaches back before the
detection and post-roll runs on after ball-loss. That is by design, and it is the reason clips are
streamed rather than seeked.

`frame_map_valid: true` for all 21 deliveries. For every delivery, `frames_written ==
frame_count`, and every `original_frame → clip_frame` mapping round-trips with **0 mismatched
points**.

### 3.3 Provenance and decisions

```
Trajectory points ............ 485 (244 detected, 241 predicted)
Trajectory gaps ............. 0 reported, 0 filled
Speeds present ............... 17/21  (null: 4)
Speeds imputed ............... 0      <- D1
Tracking source .............. {'single_pass': 21}
Bounce (scored / fallback) ... 13 / 8
Shots labelled ............... 21/21  ({'in_memory': 21})
Overlays rendered ............ 21/21
```

- **D1 holds.** Four deliveries have `speed_kmh: null` and **zero** speeds were imputed. Not from
  the median, not from a sibling delivery. `speed_imputation_enabled_by_default: false` is exposed
  in the payload.
- **Provenance holds.** 244 detected and 241 predicted points, each tagged. 241 predicted points
  is nearly half the trajectory and none of it is passed off as measured.
- **D2 holds.** The payload discloses
  `"speed_is_estimate": true, "speed_is_calibrated": false, "speed_scale": 2.0,
  "speed_offset_kmh": -20.0`, plus the formula, the sanity bounds, and an explicit note that the
  corner fractions "are NOT a general property of cricket video."
- **No re-tracking.** All 21 deliveries are `single_pass`. The three documented `clip_fallback`
  re-track paths exist but were never needed, so no delivery is tagged `clip_fallback`.

### 3.4 The 53 mechanical checks

```
53/53 checks passed
```

These assert, among others: one source decode; detector runs once per frame; no third pass from
any stage; `frames_written == frame_count`; exact `clip_frame` round-trip; D1 not violated; every
bounce is a real tracked point; every trajectory point has an integer source frame and recorded
provenance; every overlay length matches its clip; shot inference preferred the in-memory path;
D2 disclosed in the API payload.

---

## 4. Performance

Measured **inside** the tracking loop, so the numbers add up to the stage total rather than
approximating it:

| Stage | Seconds | Share | Per frame |
|---|---:|---:|---:|
| YOLO detection | 39.2 | 63% | 13.67 ms |
| Record + finalise (analytics, clip open, JSON) | 13.1 | 21% | 4.56 ms |
| Ring buffer encode | 7.9 | 13% | 2.74 ms |
| Video decode | 1.1 | 2% | 0.38 ms |
| Clip write | 0.9 | 2% | 0.33 ms |
| Kalman update | 0.1 | 0% | 0.04 ms |
| **In-loop total** | **62.3** | 100% | |
| Outside the loop | 0.0 | | model load, homography, final JSON |

The pass is detector-bound, which is the correct place for the cost to sit: one detector call per
frame is the floor for this design, and the Kalman filter is free (0.04 ms/frame).

**Run-to-run variance, disclosed.** An earlier run of the identical pass measured detection at
40.79 ms/frame and the tracking stage at 175.7 s. It did not reproduce. Three controlled
experiments were run to find the cause:

| Experiment | Result |
|---|---|
| Detection alone, sustained, 6×200 frames | 10.59 ms/frame, first-to-last drift **−5%** — no thermal throttling |
| Detection immediately after ring-buffer encode | 11.40 ms/frame (**+8%**) — the ring buffer is not interfering |
| Full tracking pass × 4 variants (shipped / mp4v / no clip window / no usable codec) | detect 14.13 / 17.09 / 14.44 / 13.30 ms — the codec and the open writer are not the cost |

Every structural hypothesis was ruled out, and the profile is flat across variants, so the slow
run is attributed to **host contention** rather than to the code. Reported throughput is therefore
**~36 fps for 640×360**; do not extrapolate it linearly to 1080p without re-measuring, since
detection cost scales with input pixels.

The codec-fallback experiment also confirms the chain is not a hidden cost: removing the writer
entirely changes detection by under 1 ms/frame.

---

## 5. The three results that need interpretation

### 5.1 All 21 shots are `Lofted` — the fixture, not the classifier

Measured per clip, the black-pixel fraction of the 30 frames the classifier actually sampled:

```
min 60%   mean 68%   max 73%
clips where >50% of sampled frames are synthetic black: 21/21
```

A clip window is 4 s of pre-roll + the delivery + 2 s of post-roll ≈ 172 frames, of which only
~25 frames are real cricket. The synthetic gaps therefore fill ~85% of the window by duration, and
**68% of the linspace-sampled frames are black**. The classifier is being asked to describe a
sequence that is two-thirds black and has no visible bat stroke. `Lofted` is a reasonable answer
to that input.

**This tells us nothing about shot-classification accuracy.** Shot confidence bottoms out at
0.223, consistent with a model guessing at uninformative input rather than being confident.

The pipeline behaviour it *does* validate is real: shot inference ran 21/21 with
`input_source: {'in_memory': 21}`, meaning the in-memory primary path worked on every delivery and
**no clip was re-decoded** for shot sampling. Getting a *correct* shot label out of this pipeline
needs a fixture built from continuous match footage, not a concatenation of isolated clips.

### 5.2 The speed numbers are not physically valid on this footage

This is the most important finding in the report, so it is stated without softening.

**The assumed pitch quad does not match this footage.** The fixed corner fractions place the pitch
at 42–57% across and 27–66% down a 640×360 frame:

```
TL (267.5,  96.5)   TR (358.4,  96.5)
BL (276.5, 235.8)   BR (366.1, 235.8)
```

Mapping the detected bounces through that homography puts them **off the pitch entirely**:

```
bounce ground x: -1.36 .. 3.59 m   (crease is 0 .. 2.64 m)
bounce ground y:  7.23 .. 25.59 m  (pitch is  0 .. 20.12 m)
detected ball positions inside the assumed quad: 156/244 (64%)
```

**The speeds inherit that error:**

```
n = 17   mean 109.7   median 106.0   min 42.5   max 198.5 km/h
outside the plausible 100-165 km/h cricket range: 12/17
```

No bowler delivers at 198 km/h. The `*3.6*2.0 - 20.0` calibration cannot repair a wrong
homography — it is a fixed affine map applied *after* the geometry is already incorrect, so it
cannot know which deliveries are wrong.

**What this means for the design.** D2's wording ("keep it for now, label it as an estimate") is
**necessary but not sufficient**. Labelling these numbers as estimates is honest; *using* them as
speed is not defensible. Per-video pitch-corner calibration is a **prerequisite** for the bowling
analytics to mean anything, not a future improvement. Phase 4 must not present these as speeds.

The same defect affects `length` and `line`, which are derived from the same geometry. Note
`line` returned `Middle` for 19 of 21 deliveries — a near-constant output is what a miscalibrated
homography produces, and it should be treated as a symptom, not as a result.

### 5.3 21 deliveries, not 41

D3 predicted the count would differ and that this would be correct, since the boundaries are now
physics-based rather than heuristic. Measured:

- 41 real delivery clips were embedded.
- 21 deliveries were produced.
- Detected points per delivery: min 8, median 10, max 19.

The fixture's 1-second clips make this a **harder** case than real footage, not an easier one: the
ball is on screen for ~25 frames and the model fires on only 8–19 of them. D3 explicitly requires
a side-by-side accuracy comparison on the same video before Phase 4 sign-off, and **that comparison
has not been done.** Until it is, the correct statement is "21 deliveries were found and none were
discarded for the debounce", not "the count is correct".

---

## 6. Decision status

| # | Decision | Status |
|---|---|---|
| D1 | `impute_missing_speeds = false`; unreliable ⇒ `null` | **Implemented and verified.** 4 nulls, 0 imputations. |
| D2 | Keep the calibration, label it homography-derived, expose it | **Implemented and verified** — and §5.2 shows the label is necessary but not sufficient. |
| D3 | Delivery count will not be 41 | **Measured: 21.** Still needs the side-by-side accuracy comparison. |
| D4 | Speed-fitting window (first 12 detected frames) | **Still open.** Deferred pending measurement, as D4 instructs. Note §5.2 means a better window will not fix the speeds; the geometry must be fixed first. |
| D5 | Do not trust `CAP_PROP_FRAME_COUNT` for numbering | **Implemented; please ratify.** Frame numbering comes from the decode-loop counter. `CAP_PROP_FRAME_COUNT` is used only for progress reporting and `clip.end_frame` clamping. `test_single_pass.py` and the 53 checks assert the invariant. |

---

## 7. Bugs found and fixed during Phase 3

Real defects, found by testing rather than review:

1. **Shot sampling used the wrong frames.** `pipeline.run` collapsed the position-keyed capture
   dict into a dense 30-element list, so `sample_indices(172)` addressed positions 0–29 instead of
   0–171. The classifier still received 30 frames — just the wrong 30, silently. It surfaced as
   `In-memory frames insufficient (30 frames, need index 166); falling back to the clip file`,
   meaning every delivery was paying a redundant clip decode. Fixed by passing the position-keyed
   mapping through and looking frames up **by position**; a mapping with a hole now falls back
   rather than substituting a neighbour. 5 regression tests added.
2. **`Trajectory.gaps` off-by-one** — a gap run ended at `prev` instead of `prev - 1`.
3. **Ring-buffer byte cap could be exceeded** by a single frame larger than `max_bytes`; such a
   frame is now refused, so the ceiling is a real ceiling.
4. **`ClipWriter.open_for` referenced undefined `width`/`height`/`ring`.**
5. **Clip drain skipped to end-of-window on holes**, marking every future post-roll frame missing.
   `open_for` now drains only to `min(end_frame, ring.last_frame)`.
6. **`last_delivery_frame` was seeded at `0`** (inherited from upstream), so any delivery ending
   before frame 60 was silently discarded — including anything in the first 2.4 s, which is exactly
   where a first over appears. Upstream had this bug. Now seeded `None`.
7. **Delivery `id` incremented before acceptance**, producing gaps in the id sequence.
8. **Cooldown was checked after analytics** ran; moved before.
9. **`t_stage_start` NameError** — introduced by the §4 instrumentation itself and caught by the
   existing suite on the next run. Worth recording as evidence the suite is load-bearing.

---

## 8. Known limitations

1. **Speeds, lengths and lines are not physically valid on this footage** (§5.2). The homography
   needs per-video calibration.
2. **Shot accuracy is untested** (§5.1). Needs continuous match footage.
3. **Delivery accuracy is unverified** (§5.3). D3's comparison is outstanding.
4. **The ring buffer's byte cap was never exercised by this run** — peak was 3.6 MiB against a
   512 MiB ceiling. That logic is covered by unit tests only. At 1080p the cap will actually bite.
5. **`line` returns `Middle` for 19/21** — treat as a symptom of (1).
6. **No re-track fallback was exercised on real data.** All three `clip_fallback` paths are
   unit-tested; none triggered here.
7. **Throughput is measured at 640×360 only** and is detector-bound; 1080p has not been run.

---

## 9. Reproducing this

```powershell
cd CricketUnified\backend
python -m unittest discover -s tests -t .        # 119 tests
python ..\tools\build_test_fixture.py           # rebuild the fixture
python ..\tools\run_real_video_test.py          # 53/53 checks
python ..\tools\diagnose_results.py             # §5.1, §5.2
python ..\tools\why_detection_is_slower_in_the_loop.py   # §4 throttling/bleed
python ..\tools\ab_test_the_in_loop_slowdown.py          # §4 codec/writer A/B
```

Artifacts from the run are kept in `backend/data/analyses/phase3_real/`: `clips/`,
`overlays/`, `tracking.json` (tracker output only, no shot labels) and `result.json` (the
combined payload the API serves).

Note: both original repositories remain untouched — nothing committed, staged or pushed in
either.

---

## 10. Recommendation

Phase 3 is complete and its invariants hold. Before Phase 4:

1. **Answer D4** — or explicitly defer it again, now knowing it cannot fix §5.2 on its own.
2. **Ratify or reject D5** as implemented.
3. **Decide how bowling analytics will present speed** given §5.2: suppress it, or gate it behind
   a per-video calibration step. Presenting 42–199 km/h as speeds is not defensible.
4. **Commission the D3 side-by-side** on continuous footage, and rebuild the fixture that way so
   shot classification can be evaluated at all.

Then stop for approval.
