# Stage 3 Kalman Recovery Fix — Report

**Date:** 2026-10-05
**Scope:** Stage 3 only — Kalman association/recovery. Diagnosis: `docs/REAL_VIDEO_DIAGNOSTIC.md`.
**Source:** `CricketUnified/backend/app/services/tracker_modules/tracker.py` (one condition changed)

---

## 1. Root cause

### The state machine as it stood

```
BallTracker.update(detection, frame_no)
│
├─ detection present
│  └─ kf.initialized ?
│     ├─ no  → kf.init(detection); missed=0 → "detected"          ← cold start
│     └─ yes → predict() → dist > 120 ?
│               ├─ yes → missed += 1 → "predicted", gated_out=True   ← REFUSE
│               └─ no  → kf.update(); missed=0 → "detected"
│
└─ no detection
   └─ missed += 1 → initialized and missed <= 8 ? → "predicted" : (None, None)  ← LOST
```

Three state variables drove this: `kf.initialized` (filter has a state vector),
`missed_frames` (consecutive non-observations), and `in_delivery`.

### The bug

`update()` declared a track **lost** in one place and kept **gating on it** in
another:

- `update()` returns `(None, None)` once `missed_frames > MAX_MISSED_FRAMES` —
  the filter's own verdict that it has no ball.
- But `kf.initialized` stayed `True`, so the next detection took the
  **gate** branch: predict from the disowned state, compare, and refuse the
  detection as a false positive.

The filter simultaneously asserted *"I have no ball"* and *"I have a ball over
here and this new detection is a false positive."* Both cannot be true.

### Why it was a livelock, not a rejection policy

A refusal also advanced `missed_frames` (`tracker.py:216`, itself a deliberate
earlier fix), and `missed_frames` only resets on an **accepted** detection —
precisely what the refusal blocked. The counter therefore ratcheted one way with
no escape:

| Evidence | Baseline |
|---|---|
| Refusals occurring **after** the filter already reported itself lost | 123 / 195 (63%) |
| Peak `missed_frames` reached | **40** (threshold is 8) |
| `gated → accepted` transitions on the very next frame | **0 / 116** |
| Refusal median innovation | 387 px vs a 120 px gate (3.2×) |
| Longest refusal runs | 26 frames (f9867–f9892) |

Zero recoveries. The only exit was `reset_delivery()`, reachable solely from a
coast-ceiling close — which is why 92% of events closed by timeout: a newly
arrived ball could never be *confirmed*, so it could never *terminate* the
previous event.

---

## 2. Code change

One condition, in `BallTracker.update()`:

```python
# before
if self.kf.initialized:

# after
if self.kf.initialized and self.missed_frames <= MAX_MISSED_FRAMES:
```

A track the filter has already disowned no longer gates. Control falls through
to the **pre-existing** `else: self.kf.init(detection)` cold-start branch — the
same path used at frame 1 of any video. No new branch, no new state, no new
constant.

**Explicitly not done:** `GATE_RADIUS` unchanged (still 120),
`MIN_DELIVERY_FRAMES` unchanged (still 8), no segmentation redesign, no second
decode/YOLO/Kalman pass, no fabricated trajectory points.

### Why this is safe, not a loosening of the gate

1. **Healthy tracking is bit-identical.** While `missed_frames <= 8` the gate
   runs exactly as before at the same radius.
2. **It is not blind acceptance.** It applies *only* where the filter already
   returns "no ball" for every empty frame. Gating against a prediction you have
   disowned is the bug, not the protection.
3. **Provenance stays truthful.** The seed comes from a real measured box:
   `source` is `"detected"` and the point enters `delivery_positions`, which
   contains detections only. `kf.init()` sets velocity to 0, so nothing is
   invented and a lone false positive cannot fling the filter.
4. **Downstream corroboration still guards it.** One recovered detection cannot
   become a delivery — `MIN_DELIVERY_FRAMES = 8` still applies. If the seed was
   spurious, `missed_frames` resets to 0, the track dies again within 8 frames,
   and the state is recoverable once more.

---

## 3. Regression tests

New file `tests/test_tracker_recovery.py` (24 tests) plus 2 corrected tests in
`test_single_pass.py`. **Full suite: 176 passed, 0 failed.**

| Group | Tests | Covers |
|---|---|---|
| **A. Healthy tracking** | 4 | `detection → prediction → detection` still associates; gate still refuses a distant impostor while healthy; association holds right up to `missed_frames == MAX_MISSED` |
| **B. Genuine loss** | 3 | Track does become lost after `MAX_MISSED`; still predicts within coast; refusals alone also drive loss |
| **C. Recovery** | 6 | Lost track re-initialises; `missed_frames` ratchet reset; recovery immediate not deferred; recovered point is a real measurement; no fabricated trail points; repeated loss/recovery cycles |
| **D. Stale prediction** | 4 | Track follows the *new* ball not the old ghost; gate still refuses impostor while healthy; spurious seed cannot reach `MIN_DELIVERY_FRAMES`; spurious seed does not poison the filter |
| **E. Delivery behaviour** | 6 | Delivery accumulation detections-only; `reset_delivery` clears all; cold start unaffected; silence invents nothing; **`GATE_RADIUS` still 120**; **`MIN_DELIVERY_FRAMES` still 8** |

The last two are deliberate guard-rails: they fail if the fix is ever "improved"
by widening the gate or lowering the threshold.

### One pre-existing test was corrected, not deleted

`test_debounce_suppresses_a_close_second_delivery` asserted that windows
`(30,60)` and `(70,95)` produce **1** delivery. It now passes with **2**.

Investigation showed the test's stated mechanism did not exist.
`MIN_FRAMES_BETWEEN_DELIVERIES` is read by **no logic** — `config.py:268`
documents it as *"DEPRECATED and unused as a segmentation rule. Retained so the
value upstream used is still visible in the run record."* The only occurrences
are the constant, the config field, a docstring, and the test itself.

What actually collapsed the second window was the Stage 3 livelock. Measured by
replaying the stub detector against both tracker versions:

| | gate-refused frames in window 2 | confirmed in window 2 |
|---|---|---|
| pre-fix | **26 / 26** | **0** |
| post-fix | 0 / 26 | 26 |

Window 2 never became an event at all pre-fix. **The test was pinning the
defect.** It is now `test_close_second_window_is_tracked_not_suppressed`, with
the history recorded in its docstring, and a companion
`test_debounce_constant_is_not_a_segmentation_rule` documents the deprecation.

This is a behaviour change and is called out deliberately rather than buried.

---

## 4. Before / after metrics

Full `real_cricket.mp4`, 13,565 frames, one decode, one YOLO pass, one Kalman
pass, same harness as the baseline.

| Metric | Baseline | After | Δ |
|---|---|---|---|
| **Gate-refused detections** | 195 | **106** | **−89 (−46%)** |
| Accepted detections | 843 | 843 | 0 (unchanged, as required) |
| Confirmed into events | 648 | **737** | **+89** |
| Kalman-predicted frames | 1,122 | 1,184 | +62 |
| Event candidates | 91 | 97 | +6 |
| Discarded candidates | 57 | 61 | +4 |
| **Deliveries emitted** | **34** | **36** | **+2** |
| Confirmed frames in delivered events | 480 | **548** | **+68** |
| Max confirmed in a delivered event | 25 | **35** | +10 |
| `coast_ceiling` closures | 84 (92%) | 84 (**87%**) | share −5pp |
| `unreachable_displacement` closures | 7 | 13 | +6 |
| Untracked activity windows | 29 | 27 | −2 |
| Validation `SINGLE_EVENT` | 22 | **24** | +2 |
| Validation `AMBIGUOUS` | 12 | 12 | 0 |
| Validation `NO_EVENT` | 0 | 0 | 0 |
| Validation `MULTIPLE_EVENTS` | 0 | 0 | 0 |
| Frame maps valid / invalid | 33 / 1 | 35 / 1 | invalid unchanged |

Frame budget reconciles exactly in both runs:
`confirmed + gated_out = accepted` (648+195=843, 737+106=843). Nothing is
double-counted or lost.

### The 89 recovered detections are confirmed real

The fix moved **89 detections from refused to confirmed**, and produced **6 more
boundary decisions**, yet only **+2 deliveries**. The gap is now explained rather
than assumed:

- `unreachable_displacement` rose 7 → 13. The segmenter received recovered
  detections at genuinely far positions, judged the displacement physically
  impossible, and **split the event instead of extending it** — correct
  behaviour. f4922–f4958 went 5 → 2 confirmed as a result: the recovered
  evidence arrived after a split point, not because it was lost.
- The 8 counterfactual candidates did not all flip as predicted: f9213 **did**
  deliver (25 confirmed), f10150 was **superseded by a split**, and f8145 rose
  only 1 → 6. The original counterfactual counted gated frames *within a
  candidate's fixed span*; recovery moves boundaries, which redistributes
  evidence. The prediction was directionally right, quantitatively optimistic.

### Recovered events are the ones the diagnostic predicted

| New delivery | Start | Duration | Confirmed | Ambiguous |
|---|---|---|---|---|
| 1 | f7781 (311.2 s) | 0.48 s | 12 | No |
| 2 | f9213 (368.5 s) | 1.68 s | 25 | No |

Both overlap the longest gate-refusal runs the baseline identified in advance —
f7780–f7792 (13 frames, entirely inside delivery 1) and f9230–f9243 (14 frames,
inside delivery 2, which opens 17 frames earlier at f9213). That they
materialise with 12 and 25 corroborating detections is independent confirmation
the gate was rejecting true balls.

### Remaining 106 refusals are legitimate

| | Count |
|---|---|
| Refusals while the track was **healthy** (`missed <= 8`) | **92** |
| Refusals while the track was **lost** (`missed > 8`) | 14 |

The gate is doing its job on a track it still stands behind. The 14 residual
lost-state refusals occur on the first frame after loss, before a recovery
opportunity exists; they are not a livelock because the next frame can recover.

---

## 5. New false positives

**None detected on any axis.** Clip count alone proves nothing, so quality was
checked independently:

| Signal | Baseline | After |
|---|---|---|
| Delivered events with the bare minimum of 8 confirmations | 2 | 2 |
| Delivered events flagged `AMBIGUOUS` | 0 | 0 |
| Validation `NO_EVENT` | 0 | 0 |
| Validation `MULTIPLE_EVENTS` | 0 | 0 |
| Delivered events under 0.4 s (implausibly short) | 0 | 0 |
| **Baseline-delivered events lost** | — | **NONE** |

Every one of the 34 baseline deliveries is still delivered; the 2 additions are
strictly new. Delivered events carry *more* evidence, not less: total confirmed
frames 480 → 548, max per event 25 → 35. A false-positive fix inflates clip count
while degrading evidence per clip; the opposite happened here.

---

## 6. Remaining problems

**Stage 3 is fixed and proven; the delivery shortfall is not resolved.** Stated
plainly: **34 → 36 deliveries is not a fix to the 20-vs-47 complaint.** Raising
the clip count is not evidence of correctness.

1. **The dominant loss mechanism is untouched.** 61 of 97 candidates are still
   discarded for having fewer than 8 confirmed frames. 39 have ≤4 (noise,
   correctly dropped); **17 remain near-misses at 5–7** — unchanged from
   baseline. These are now limited by Stage 4 confirmation counting, not Stage 3.
   The next diagnostic question is why a delivery yields only 5–7 confirmed
   detections, not why detections are being refused.

2. **`AMBIGUOUS` is unchanged at 12 of 36 (33%).** Uncertain boundaries persist.
   The validator still cannot certify a third of the output.

3. **Stage 4 boundary splits now do more work.** `unreachable_displacement`
   rose 7 → 13 as recovered detections legitimately trigger splits. Whether the
   kinematics model is calibrated for true post-bounce displacement needs
   separate review — not attempted here.

4. **Over-splitting is not fixed.** The diagnostic found two accepted event pairs
   separated by exactly 1 frame (f11139/f11167, f11284/f11299). That is a Stage 5
   defect and is untouched by design.

5. **1 invalid clip frame map** persists (1 of 36). Pre-existing, unrelated.

6. **The rejected-test question deserves review.** A test that encoded the
   defect was corrected. That the "debounce" was never implemented — while a
   docstring in `clip_writer.py:33` still asserts
   *"Only one delivery can be open at a time (MIN_FRAMES_BETWEEN_DELIVERIES
   debounce)"* — is a documentation defect worth correcting separately.

7. **The dedicated re-acquisition path never fired.** `event_segmenter.py:751`
   and `:816` emit a `reseed_position`, and `tracking_service.py:595` calls
   `tracker.reseed(detection, ...)` — a mechanism that re-acquires a returning
   ball *without* cutting the delivery in two. **It fired zero times in both
   runs.** Its trigger requires the segmenter to open a *new* event first, which
   is precisely what the livelock prevented. Worth re-examining now that
   recovery works, since `unreachable_displacement` splits rose 7 → 13 and may
   leave `reseed_position` set at boundaries it did not before.

8. **Visual confirmation still outstanding.** The 89 recovered detections are
   verified by confidence, trajectory, and the validator, not by eye. Dumping
   crops of the recovered frames would close this gap.

---

## 7. Verification artifacts

| Item | Path |
|---|---|
| Changed source | `CricketUnified/backend/app/services/tracker_modules/tracker.py` |
| New tests | `CricketUnified/backend/tests/test_tracker_recovery.py` |
| Corrected tests | `CricketUnified/backend/tests/test_single_pass.py` |
| Baseline data | `CricketUnified/backend/data/real_diagnostic.json` |
| After-fix data | `CricketUnified/backend/data/stage3_after.json` |
| Harness | `CricketUnified/tools/real_video_diagnostic.py` |
| Full suite | 176 passed, 0 failed |

`GATE_RADIUS = 120` and `MIN_DELIVERY_FRAMES = 8` are both asserted unchanged by
tests, so the fix cannot be silently traded for a threshold change later.