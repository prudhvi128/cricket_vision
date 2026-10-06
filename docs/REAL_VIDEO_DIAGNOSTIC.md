# Real Video Diagnostic — `real_cricket.mp4`

**Date:** 2026-10-05
**Question:** the pipeline yields ~20 delivery clips on `real_cricket.mp4`; a manual review
estimates ~47 batting events. Which pipeline stage loses them?

**Verdict:** **Category A — code defect.** The dominant loss is Stage 3 (Kalman association
gate) feeding Stage 4 (confirmed-frame counting). Threshold tuning (B) is a real but
secondary lever. The model is not the limit (C), and the behaviour is not expected (D).

---

## 1. Scope and method

### Constraints honoured

| Constraint | Status |
|---|---|
| Diagnose only — no fixes applied | Honoured |
| No changes to segmentation logic | Honoured |
| No changes to YOLO thresholds | Honoured |
| No changes to Kalman parameters | Honoured |
| No manual video splitting | Honoured |
| No per-clip YOLO or per-clip Kalman | Honoured — one decode, one tracker pass |
| Single authoritative tracking pass | Honoured |

### Harness

`scripts/diagnostics/real_video_diagnostic.py` is the only file added. It wraps production
objects and delegates to production methods; it does not alter pipeline behaviour:

- `DiagDetector(BallDetector)` calls the unmodified
  `BallDetector.detect_with_confidence()` and returns that result unchanged.
- A **shadow** `model.predict(conf=0.01)` runs on the same already-decoded frame purely to
  account for boxes the production threshold discarded. It is never returned to the pipeline.
  This is a second inference on an *already-decoded* frame — not a second video pass, and not
  per-clip inference.
- Wrappers record `BallTracker.update/reseed/reset_delivery`,
  `EventSegmenter.observe/close_open_event`, and `SignalTimeline.record`.

Raw output: `data/real_diagnostic.json`.

### Measurement

| Property | Value |
|---|---|
| Resolution | 1280x720 |
| FPS | 25.0 |
| Reported frames | 13,565 |
| Decoded frames | 13,565 (exact match — no decode shortfall) |
| Duration | 542.60 s (9.04 min) |
| Pass wall time | 577.6 s (23.5 frames/s, shadow pass included) |

---

## 2. Headline

| Quantity | Value |
|---|---|
| **Delivery clips emitted** | **34** |
| Manual estimate | ~47 (comparison only, **not** treated as ground truth) |
| Event candidates built by the segmenter | 91 |
| Candidates delivered (`confirmed >= 8`) | 34 |
| **Candidates silently discarded** | **57** |
| YOLO accepted frames | 843 |
| **Kalman gate-rejected frames** | **195 (23.1% of accepted detections)** |
| Clip-validation `AMBIGUOUS` | 12 of 34 |
| Untracked activity windows | 29 |

**The pipeline is not failing to find events. It finds 91 candidates and throws 57 away.**

---

## 3. Stage-by-stage

### Stage 1 — Decode / ring buffer — **healthy**

13,565 reported = 13,565 decoded. No frame loss, no seek failures. Not a factor.

### Stage 2 — YOLO detection — **healthy (not the bottleneck)**

| Metric | Value |
|---|---|
| Frames with >=1 class-0 box (conf >= 0.01) | 2,556 (18.8%) |
| Frames with a box at conf >= 0.30 | 883 |
| **Accepted frames** (post size filter) | **843** |
| Total class-0 boxes (any conf) | 5,291 |
| Sub-threshold boxes (conf < 0.30) | 4,355 |
| Boxes rejected by the **size** filter | 45 (on 44 frames) |
| Boxes with wrong class | 0 |

Confidence distribution of accepted detections:

| min | p25 | median | p90 | max |
|---|---|---|---|---|
| 0.3006 | 0.4633 | 0.6026 | 0.7402 | 0.7993 |

Two independent checks rule detection out as the primary cause:

1. **The size filter is not eating balls.** All 45 size-rejected boxes had diagonals of
   80.2–101.4 px — *larger* than `DETECTOR_MAX_DIAG_PX = 80`. These are oversized blobs
   (batsman, fielder, crowd), not small balls. A ball being lost to the size gate would show
   diagonals *below* `DETECTOR_MIN_DIAG_PX = 4`; that never happened.
2. **The threshold is not sitting on a wall of real signal.** Of the 4,355 sub-threshold
   boxes, **62.2% score below 0.05** — noise floor:

   | band | boxes | share of sub-threshold |
   |---|---|---|
   | 0.00–0.05 | 2,710 | 62.2% |
   | 0.05–0.10 | 775 | 17.8% |
   | 0.10–0.15 | 356 | 8.2% |
   | 0.15–0.20 | 238 | 5.5% |
   | 0.20–0.25 | 160 | 3.7% |
   | 0.25–0.30 | 116 | 2.7% |

   Only 276 boxes (6.3%) sit in the 0.20–0.30 band. Dropping the threshold to 0.20 would
   raise accepted frames from 843 to at most 1,077 (+22%), and most of that is noise-adjacent.

### Stage 3 — Kalman tracking — **PRIMARY DEFECT**

| Outcome | Frames |
|---|---|
| Confirmed (`source = detected`) | 843 |
| Predicted (coasted) | 1,122 |
| **Gate-rejected** (`gated_out`) | **195** |
| No position (lost) | 11,795 |
| `reseed()` calls | 0 |

`GATE_RADIUS = 120` px, applied in **full-frame 1280x720 coordinates** — the detector returns
full-frame centres (`detector.py:116-117` rescales by `sx = 1280/640 = 2.0`). Deriving the gate
displacement from logged data (gated frames log the post-`predict()` position, which is exactly
what the gate compares against):

| min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|
| 120.1 | 165.4 | 229.5 | **387.1** | 602.8 | **838.6** | 1584.8 |

**The refused detections are not marginal misses — they are 3.2x the gate radius at the median
and 7.0x at p90.** For scale, the median step between consecutive confirmed detections is only
22.7 px/frame (p90 = 54.3). A genuine continuation of the same ball never approaches a 387 px
innovation.

The refused detections are **high-confidence**, and they are high-confidence *after already
passing the 0.30 threshold*:

- median confidence of gated detections: **0.565**
- 128 of 195 (66%) score **>= 0.50**
- max 0.773

So this is high-quality evidence being discarded by association logic, not by the threshold.

**Mechanism.** A new ball becomes visible while the filter is still extrapolating the previous
one. The prediction is stale and far away, so every real detection is refused. This produces
runs of consecutive refusals lasting over a second:

| run | frames | duration | confidence range |
|---|---|---|---|
| f9867–f9892 | 26 | 1.04 s | 0.33–0.72 |
| f9230–f9243 | 14 | 0.56 s | 0.50–0.77 |
| f10135–f10148 | 14 | 0.56 s | 0.32–0.69 |
| f7780–f7792 | 13 | 0.52 s | 0.32–0.74 |

`missed_frames` does advance on a refused measurement (`tracker.py:183`, a deliberate and
correctly-documented fix), so the filter does eventually report "lost" — but `kf.initialized`
stays `True`, so `update()` keeps taking the predict-and-gate branch instead of re-initialising.
Recovery only arrives when the open event hits its coast ceiling and `reset_delivery()` clears
the filter (`tracker.py:255`).

**Knock-on effect — events end by timeout, not by the next ball.** Because a newly arrived ball
can never be *confirmed*, it can never terminate the previous event. The boundary has to be
forced by the coast ceiling instead:

| close reason | count | share |
|---|---|---|
| `coast_ceiling` | 84 | **92%** |
| `unreachable_displacement` | 7 | 8% |

A real next delivery being visible is precisely the signal that *should* close an event, and it
is being downgraded to a non-event.

### Stage 4 — Event segmentation — **where the loss materialises**

`tracking_service.py:554-555` sets `confirmed = (source == "detected")` and
`candidate = gated_out`. A refused detection is therefore **never** counted toward an event's
confirmed-frame total. Against `MIN_DELIVERY_FRAMES = 8`:

**Discarded candidates (57), by confirmed-frame count:**

| confirmed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| candidates | 22 | 11 | 3 | 4 | 8 | 4 | 5 |

- **40** candidates have <= 4 confirmed frames — genuinely hopeless, correctly discarded.
- **17** candidates have **5–7** confirmed frames — **one to three frames short of acceptance.**

Representative warnings emitted by the pipeline:

```
Discarded event at frames 115-130: only 5 confirmed detections, below the minimum of 8.
Discarded event at frames 338-350: only 7 confirmed detections, below the minimum of 8.
Discarded event at frames 565-573: only 7 confirmed detections, below the minimum of 8.
```

**Counterfactual (computed from logged per-frame data only; nothing was re-run or modified).**
Adding each discarded candidate's own gate-refused detections back into its confirmed count:

| candidate | t (s) | confirmed | + refused | would-be total |
|---|---|---|---|---|
| f115–f130 | 4.6 | 5 | 4 | 9 |
| f338–f350 | 13.5 | 7 | 2 | 9 |
| f606–f623 | 24.2 | 7 | 1 | 8 |
| f4922–f4958 | 196.9 | 5 | 10 | 15 |
| f7760–f7790 | 310.4 | 3 | 15 | 18 |
| f8145–f8175 | 325.8 | 1 | 16 | 17 |
| f9213–f9247 | 368.5 | 5 | 16 | 21 |
| f10150–f10183 | 406.0 | 6 | 9 | 15 |

**8 more candidates clear the threshold: 34 -> 42 deliveries.** The three worst
(f7760–f7790, f8145–f8175, f9213–f9247) had 1–5 confirmed frames alongside 15–16 refusals —
the starvation signature of the Stage 3 defect in its purest form.

### Stage 5 — Clip emission — **minor secondary defect (over-splitting)**

Two pairs of accepted events are separated by exactly **1 frame** — a single real delivery
counted twice:

| first | gap | second |
|---|---|---|
| f11139–f11166 (11 confirmed) | **1 frame** | f11167–f11193 (11 confirmed) |
| f11284–f11298 (8 confirmed) | **1 frame** | f11299–f11330 (20 confirmed) |

Clip statistics:

| Metric | Value |
|---|---|
| Clips written | 34 |
| Codec | avc1 |
| Frame maps valid | 33 |
| **Frame maps invalid** | **1** |
| Write errors | none |

One clip carries a broken frame map — worth a separate fix, unrelated to event count.

### Stage 6 — Clip validation — **corroborates over-splitting**

| Classification | Clips |
|---|---|
| `SINGLE_EVENT` | 22 |
| **`AMBIGUOUS`** | **12** |
| `MULTIPLE_EVENTS` | 0 |
| `NO_EVENT` | 0 |

**12 of 34 clips (35%) are `AMBIGUOUS`**, and **zero** clips contain multiple events. If
over-splitting were the dominant story we would expect `MULTIPLE_EVENTS`; instead the failures
are *uncertain boundaries*, which is what a coast-ceiling close (92%) produces. The clip layer
is not losing deliveries — it never receives them.

---

## 4. Missing-event analysis — exact stage

Every one of the 13,565 frames resolves to exactly one bucket, and the buckets reconcile
against the totals:

| Fate of frame | Frames | Share |
|---|---|---|
| Accepted and confirmed into an event | 648 | 4.8% |
| **Accepted but gate-refused** | **195** | **1.4%** |
| Sub-threshold only (conf < 0.30) | 1,673 | 12.3% |
| No class-0 box at all | 11,049 | 81.5% |
| **Total** | **13,565** | 100% |

Bookkeeping check: 648 confirmed + 195 gate-refused = 843 = total accepted detections. The
648 confirmed split into 480 inside the 34 delivered events and 168 inside the 57 discarded
ones. No detection is unaccounted for.

**Therefore the loss chain is:**

```
Stage 2  YOLO detects the ball correctly (conf 0.30-0.77) ......... OK
             |
             v
Stage 3  Kalman gate refuses it: 387 px median innovation
         vs GATE_RADIUS = 120 ................................... LOSS HERE
         (195 frames, 23.1% of accepted detections, conf >= 0.30)
             |
             v
         marked candidate, never confirmed (tracking_service.py:554)
             |
             v
Stage 4  confirmed_count never reaches MIN_DELIVERY_FRAMES = 8 .. LOSS MATERIALISES
             |
             v
         57 of 91 candidates discarded  (17 of them by only 1-3 frames)
```

Plus a smaller, independent Stage 5 defect where one delivery is split into two accepted clips.

**Largest inter-delivery gaps, with the detection activity inside them:**

| gap | duration | frames with any box | sub-threshold | best conf | refused cand. inside (confirmed) |
|---|---|---|---|---|---|
| f3104–f4018 | 36.6 s | 516 | 498 | 0.762 | 5 (1,1,3,1,2) |
| f6446–f7320 | 35.0 s | 144 | 136 | 0.663 | 2 (5,2) |
| f4760–f5611 | 34.0 s | 276 | 236 | 0.766 | 5 (5,3,6,1,4) |
| f8024–f8679 | 26.2 s | 140 | 109 | 0.765 | 4 (1,1,4,2) |

These are not empty stretches of video. Each contains a valid 0.66–0.77-confidence detection
that never became a confirmed frame.

---

## 5. Reconciling with the manual estimate

The manual ~47 is a sanity target, not ground truth, so the honest statement is a bracket:

- **Floor (conservative):** 34 emitted, minus 2 duplicate fragments from the over-split
  pairs, plus 8 recovered by the counterfactual = **~40 real deliveries**.
- **Ceiling (permissive):** 34 emitted plus all 17 near-miss candidates (5–7 confirmed) =
  **~51 real deliveries**.

**~40–51 brackets the manual ~47.** The pipeline's real coverage is much closer to the manual
count than the 34 emitted clips suggest; the shortfall is dominated by one association-logic
defect rather than by 13 independent failures.

Note the arithmetic asymmetry: 34 emitted is *inflated* by over-splitting and *deflated* by
Stage 3/4 discards, which partially cancel. The emitted count is not a reliable measure of
coverage in either direction.

---

## 6. Conclusion

### Category A — code defect — **PRIMARY**

The dominant cause is a logic defect in **Stage 3 (Kalman association)**, surfacing in
**Stage 4 (confirmation counting)**:

1. `GATE_RADIUS = 120` px is compared in **full-frame 1280x720** coordinates while the ball's
   typical inter-frame step is 22.7 px (median) / 54.3 px (p90). Real ball motion after a bounce,
   off the bat, or a camera pan produces innovations of 387 px (median) and 839 px (p90) — so the
   gate rejects genuine new balls rather than just outliers.
2. A gate-refused detection is marked `candidate` and **can never** count toward the
   `MIN_DELIVERY_FRAMES = 8` total, so a real delivery detected at 0.5–0.77 confidence for
   13–26 consecutive frames is discarded for having 1–5 confirmed frames.
3. `kf.initialized` remains `True` after a track is reported lost, so the filter keeps
   predict-and-gating against a stale state instead of re-initialising; recovery only occurs when
   the open event hits its coast ceiling. This is why **92% of events close by timeout**
   instead of by the arrival of the next delivery.

Two smaller code defects:

4. **Over-splitting (Stage 5):** two deliveries are emitted as two clips each, separated by a
   single frame; this also explains the 12 `AMBIGUOUS` validations.
5. **Broken clip frame map (Stage 5):** 1 of 34 clips has an invalid frame map.

### Category B — threshold too strict — **SECONDARY, real but lower yield**

`MIN_DELIVERY_FRAMES = 8` is what converts a partially-tracked real ball into a discarded event,
and 17 candidates miss it by only 1–3 frames — so the constant is genuinely load-bearing.

But the evidence says tuning it is *not* the main fix:

- Lowering the YOLO threshold 0.30 -> 0.20 adds at most 194 frames (+22%), and 62.2% of the
  sub-threshold population scores below 0.05, so most of that is noise.
- The 195 gate-refused frames carry **median confidence 0.565** and are **already above the
  threshold**. Fixing the threshold cannot recover a single one of them; only fixing Stage 3 can.

Lowering `MIN_DELIVERY_FRAMES` alone would also promote the 40 noise candidates
(1–4 confirmed frames), trading 13 missing deliveries for a large rise in false positives.

### Category C — model limitation — **RULED OUT**

The detector finds the ball reliably when it is given a chance: 843 accepted frames at median
confidence 0.603, max 0.799, zero wrong-class boxes, and only 45 size rejections — all of them
oversized blobs rather than small balls. The 81.5% of frames with no box are genuine non-ball
frames (crowd, ground, players at rest); the ball is simply not in frame most of the time.

### Category D — expected behaviour — **RULED OUT**

Discarding a candidate requires `confirmed_frames` to be empty or fewer than 8
(`emit_event`). That rule is correct in intent, but here it fires on candidates backed by
15–16 high-confidence detections. Emitting ~20 clips for ~47 real deliveries is not expected
behaviour of a correct implementation.

### Recommended fix order (not applied — diagnosis only)

1. **Stage 3, primary:** make the association gate scale with the ball's expected motion rather
   than a fixed 120 px radius, and/or clear `kf.initialized` when the filter reports "lost" so
   the next real detection re-initialises instead of being gated against a stale state.
2. **Stage 4:** let a strongly corroborated `candidate` run contribute toward the
   `MIN_DELIVERY_FRAMES` total, or gate the threshold on corroboration count.
3. **Stage 5:** prevent single-frame boundary splits (the f11139/f11167 and f11284/f11299 pairs).
4. **Stage 5:** repair the single invalid clip frame map.
5. **Stage 6:** re-check that the 12 `AMBIGUOUS` clips resolve to `SINGLE_EVENT` once 1–3 are fixed.

---

## Appendix A — Timeline of the 34 delivered events

| start frame | t (s) | duration (s) | confirmed |
|---|---|---|---|
| 839 | 33.6 | 1.00 | 11 |
| 1372 | 54.9 | 1.00 | 14 |
| 1621 | 64.8 | 0.80 | 13 |
| 1858 | 74.3 | 0.48 | 11 |
| 2157 | 86.3 | 0.40 | 9 |
| 2775 | 111.0 | 0.44 | 10 |
| 3086 | 123.4 | 0.76 | 11 |
| 4018 | 160.7 | 0.72 | 16 |
| 4254 | 170.2 | 1.08 | 17 |
| 4497 | 179.9 | 0.68 | 16 |
| 4741 | 189.6 | 0.80 | 18 |
| 5611 | 224.4 | 0.64 | 15 |
| 6069 | 242.8 | 0.56 | 11 |
| 6431 | 257.2 | 0.64 | 15 |
| 7320 | 292.8 | 2.28 | 19 |
| 7595 | 303.8 | 1.48 | 13 |
| 8005 | 320.2 | 0.80 | 18 |
| 8679 | 347.2 | 1.28 | 25 |
| 8884 | 355.4 | 0.48 | 12 |
| 9041 | 361.6 | 1.36 | 19 |
| 9452 | 378.1 | 0.56 | 14 |
| 9597 | 383.9 | 1.20 | 15 |
| 9847 | 393.9 | 1.84 | 10 |
| 10120 | 404.8 | 1.20 | 8 |
| 10245 | 409.8 | 1.28 | 21 |
| 10529 | 421.2 | 0.60 | 13 |
| 10680 | 427.2 | 1.24 | 14 |
| 10943 | 437.7 | 1.00 | 15 |
| 11139 | 445.6 | 1.12 | 11 |
| 11167 | 446.7 | 1.08 | 11 |
| 11284 | 451.4 | 0.60 | 8 |
| 11299 | 452.0 | 1.28 | 20 |
| 11868 | 474.7 | 1.64 | 18 |
| 12034 | 481.4 | 0.40 | 9 |

## Appendix B — The 17 near-miss candidates (1–3 frames short of delivery)

| start–end | t (s) | duration (s) | confirmed | gap from previous delivered event |
|---|---|---|---|---|
| f115–f130 | 4.6 | 0.64 | 5 | first event |
| f338–f350 | 13.5 | 0.52 | 7 | first event |
| f565–f573 | 22.6 | 0.36 | 7 | first event |
| f606–f623 | 24.2 | 0.72 | 7 | first event |
| f4922–f4958 | 196.9 | 1.48 | 5 | 162 f (6.5 s) |
| f5148–f5160 | 205.9 | 0.52 | 6 | 388 f (15.5 s) |
| f5775–f5792 | 231.0 | 0.72 | 5 | 149 f (6.0 s) |
| f5928–f5942 | 237.1 | 0.60 | 5 | 302 f (12.1 s) |
| f6234–f6247 | 249.4 | 0.56 | 6 | 152 f (6.1 s) |
| f6589–f6599 | 263.6 | 0.44 | 5 | 143 f (5.7 s) |
| f9213–f9247 | 368.5 | 1.40 | 5 | 139 f (5.6 s) |
| f9308–f9314 | 372.3 | 0.28 | 7 | 234 f (9.4 s) |
| f9947–f9956 | 397.9 | 0.40 | 7 | 55 f (2.2 s) |
| f10150–f10183 | 406.0 | 1.36 | 6 | 1 f (0.0 s) |
| f11730–f11745 | 469.2 | 0.64 | 5 | 400 f (16.0 s) |
| f12545–f12566 | 501.8 | 0.88 | 5 | 502 f (20.1 s) |
| f12739–f12763 | 509.6 | 1.00 | 6 | 696 f (27.8 s) |

## Appendix C — Segmenter decision tally (all 13,565 frames)

| decision | frames |
|---|---|
| `idle` | 10,437 |
| `coasting` | 2,305 |
| `same_event` | 461 |
| `new_event` | 91 |
| `reacquired` | 96 |
| `gated_continuation` | 91 |
| `event_expired` (coast ceiling) | 84 |

## Appendix D — Untracked activity windows (29)

29 windows contain motion/activity with no confirmed ball detection. Notable entries, with the
best detection confidence found inside each:

| window | t (s) | length | accepted dets | best conf |
|---|---|---|---|---|
| f2925–f3070 | 117.0 | 5.84 s | 0 | 0.256 |
| f6248–f6282 | 249.9 | 1.40 s | 2 | 0.620 |
| f6601–f6734 | 264.0 | 5.36 s | 0 | 0.257 |
| f7543–f7574 | 301.7 | 1.28 s | 0 | 0.402 |
| f7790–f7983 | 311.6 | 7.76 s | 3 | 0.737 |
| f10713–f10769 | 428.5 | 2.28 s | 2 | 0.562 |
| f11401–f11651 | 456.0 | 10.04 s | 0 | 0.433 |
| f12564–f12632 | 502.6 | 2.76 s | 1 | 0.527 |

Every accepted detection inside these windows was gate-refused (`gated_out=True`,
`source="predicted"`), confirming Stage 3 as the responsible stage.