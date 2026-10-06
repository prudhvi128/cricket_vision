# Stage 4 Near-Miss Diagnostic — the 17 candidates with 5–7 confirmed frames

**Date:** 2026-10-05
**Status:** DIAGNOSIS ONLY. No fix implemented. No threshold changed.
**Input:** `CricketUnified/backend/data/stage3_after.json` (post-Stage-3-fix full run)
**Predecessor:** `docs/STAGE3_KALMAN_FIX_REPORT.md`

---

## 0. Scope, and one correction to the brief

Nothing was changed. `MIN_DELIVERY_FRAMES` stays 8, the YOLO cut stays 0.30,
`GATE_RADIUS` stays 120, Stage 3 is untouched.

**Correction:** the brief and the Stage 3 report both say *"39 have ≤4 confirmed
frames"*. The actual count is **44**. The discarded total is 61, and
44 + 17 = 61 checks out; 39 + 17 = 56 does not. The histogram is
`{1: 19, 2: 17, 3: 3, 4: 5, 5: 3, 6: 6, 7: 8}`. This changes nothing material —
44 noise candidates are still noise — but the report figure was wrong.

### A limitation, stated up front

**I could not visually verify any of this.** This model has no image input, so I
cannot look at a frame. Labelled contact sheets for all 17 candidates were
generated so *you* can adjudicate them:

```
docs/stage4_frames/sheet_1.png .. sheet_5.png      (4 candidates per sheet)
CricketUnified/tools/near_miss_contact_sheets.py   (regenerates them)
```

Each sheet shows 4 frames per candidate spanning its confirmed detections, with
a caption carrying the candidate id, frame range, confirmed count, prediction
count and close reason. My classifications below are derived from logged
kinematics, spatial distribution and temporal structure — **not** from looking.
Where the data cannot settle a question, I say so instead of guessing.

---

## 1. The full table, all 17 candidates

`Conf` = confirmed detections. `Pred` = Kalman-predicted frames (correctly *not*
counted as evidence). `Refused` = `gated_continuation_count`, detections the
gate rejected while the event continued. `cmean` = mean confidence of the
confirmed detections (min–max). `medSpd` = median px/frame between consecutive
confirmed detections. `lng/gap` = longest confirmed run / largest internal gap.
`inPitch` = share of confirmed points inside the pitch corridor
(x 384–896, y 144–576 — the corridor `tracking_service.py:527` uses).
`Δprev` = frames after the previous delivery's last confirmed detection.

| # | Frames | Conf | Pred | Refused | cmean | Dur (s) | Path (px) | medSpd | lng/gap | inPitch | Δprev | Close reason | Class |
|---|--------|------|------|---------|-------|---------|-----------|--------|---------|---------|-------|---------------|-------|
| 1 | f115–130 | 5 | 8 | 4 | 0.59 (0.39–0.69) | 0.64 | 205.6 | 17.8 | 3/3 | 100% | none | coast_ceiling | **A** |
| 2 | f338–350 | 7 | 5 | 2 | 0.50 (0.35–0.61) | 0.52 | 114.2 | 17.5 | 7/0 | 100% | none | coast_ceiling | **A** |
| 3 | f565–573 | 7 | 10 | 0 | 0.49 (0.38–0.69) | 0.36 | 232.7 | 29.4 | 5/2 | **0%** | none | coast_ceiling | **B** |
| 4 | f606–623 | 7 | 10 | 1 | 0.54 (0.42–0.69) | 0.72 | 191.0 | 20.5 | 4/4 | 100% | none | coast_ceiling | **A** |
| 35 | f4937–4956 | 7 | 7 | 1 | 0.71 (0.62–0.77) | 0.80 | 211.8 | 26.6 | 6/4 | 100% | 177 | unreachable_displacement | **C** |
| 38 | f5148–5160 | 6 | 15 | 0 | 0.60 (0.46–0.76) | 0.52 | 271.6 | 31.3 | 3/4 | 100% | 388 | coast_ceiling | **A** |
| 42 | f5775–5792 | 6 | 18 | 0 | 0.48 (0.31–0.72) | 0.72 | 308.9 | 24.2 | 3/10 | 100% | 149 | coast_ceiling | **A** |
| 43 | f5928–5944 | 5 | 10 | 1 | 0.61 (0.39–0.73) | 0.68 | 141.9 | 21.4 | 3/3 | 100% | 302 | unreachable_displacement | **C** |
| 48 | f6234–6247 | 6 | 9 | 1 | 0.50 (0.34–0.69) | 0.56 | 183.7 | 24.5 | 4/4 | 100% | 152 | coast_ceiling | **A** |
| 50 | f6589–6599 | 5 | 14 | 0 | 0.57 (0.39–0.66) | 0.44 | 193.1 | 26.5 | 4/6 | 100% | 143 | coast_ceiling | **A** |
| 60 | f8145–8175 | 6 | 19 | 6 | 0.62 (0.57–0.69) | 1.24 | **933.0** | **45.1** | 2/10 | 16.7% | 121 | coast_ceiling | **B** |
| 68 | f9308–9314 | 7 | 8 | 0 | 0.48 (0.41–0.57) | 0.28 | **5.2** | **1.0** | 7/0 | 100% | 54 | coast_ceiling | **B** |
| 76 | f9947–9956 | 7 | 11 | 0 | 0.57 (0.37–0.74) | 0.40 | 176.8 | 24.2 | 5/3 | 100% | 55 | coast_ceiling | **A** |
| 78 | f10144–10160 | 6 | 2 | 0 | 0.49 (0.33–0.57) | 0.68 | 301.7 | 31.1 | 5/11 | 66.7% | 10 | unreachable_displacement | **C** |
| 91 | f11730–11754 | 7 | 18 | 1 | 0.53 (0.35–0.64) | 1.00 | 349.1 | 21.7 | 3/13 | 100% | 400 | coast_ceiling | **A** |
| 96 | f12545–12566 | 6 | 22 | 0 | 0.56 (0.41–0.70) | 0.88 | 300.9 | 16.2 | 3/10 | 100% | 502 | coast_ceiling | **A** |
| 97 | f12739–12763 | 7 | 22 | 0 | 0.57 (0.44–0.67) | 1.00 | 463.8 | 22.4 | 2/12 | 100% | 696 | coast_ceiling | **A** |

### Reference classes, for calibration

The 36 confirmed deliveries and the 44 ≤4-frame candidates are the two labelled
classes available. Everything below is judged against them.

| Property | Delivered (36) | Near-miss (17) | Noise ≤4 (44) |
|---|---|---|---|
| Confirmed frames | 8–35, med 14 | 5–7 | 1–4 |
| Duration (frames) | 10–57, med 24 | 7–31 | — |
| Median speed (px/frame) | 0.74–32.4, med 20.6 | 1.0–45.1, med 24.2 | 1.5–146.7 |
| Mean confidence | med **0.64** | med **0.56** | med **0.43** |
| **In pitch corridor** | **93.2%** | **86.9%** | **61.0%** |
| x-range of points | 214–1137 | 390–1036 | 48–1247 |

Near-misses sit **between** the two classes on confidence and sit **with** the
delivered class on speed and spatial location. That is the whole story of this
document: these are not noise, and they are not cleanly deliveries either.

---

## 2. Why the segmenter is not closing these too early

The first hypothesis to test was *"the segmenter is cutting real events short."*
**It is not.** For all 14 `coast_ceiling` candidates the tail between the last
confirmed detection and the close was measured frame by frame:

| Tail content | Meaning |
|---|---|
| 14–20 of ~21 frames have **zero boxes of any confidence** | the detector saw nothing at all |
| 1–14 frames hold a **sub-threshold** box | ball present, below the 0.30 cut |
| 0–6 frames were **gate refusals** | filter rejected a real box |

The segmenter waited its full `max_coast_frames` = 20 frames (0.80 s) every
single time before closing. Closing early would mean closing on a frame where
the ball was still visible; that never happened. `coast_ceiling` is doing
exactly what it is designed to do, and its own docstring is right that it is
*"the NORMAL, confident end of a delivery."*

**So `MIN_DELIVERY_FRAMES` is not being starved by premature closure.**

---

## 3. Category A — why only 5–7 confirmed frames

There are two distinct mechanisms, and they are not the same problem.

### 3.1 Mechanism 1 — intermittent visibility with low detector confidence (11 candidates)

This is the dominant cause and it is **not a Stage 4 logic fault**.

Across the 11 category-A candidates there are 92 internal gap frames — frames
*inside* the candidate span where the ball was not confirmed. Contents:

| Gap frame content | Count |
|---|---|
| **Sub-threshold box present** (ball there, below the 0.30 cut) | **64** |
| Completely empty (no box at any confidence) | 28 |
| Gate refusal of a real box | 3 |

The critical follow-up: are those sub-threshold boxes actually the ball, or
random detections that happen to fall near it? Measuring each box against the
linearly interpolated position between its neighbouring confirmed detections:

- **127 of 138 sub-threshold boxes (92%) lie within 120 px of the ball's own path.**

They are on the trajectory. They are the ball. YOLO is simply unsure about it —
median confidence 0.02–0.13 in these frames, with only 9 of 138 reaching 0.20
and **none** reaching the 0.30 cut.

That is the physical story: a small, fast, often motion-blurred ball at range.
It is detected in bursts of 2–4 frames and dropped in between. The confirmed
frames are real; they are simply not *eight* of them, because the detector will
not commit to eight.

Two candidates show this most clearly:

- **#97** (f12739–12763): 7 confirmed in **five** separate runs (`12739-40`,
  `12745`, `12747-48`, `12750`, `12763`) — longest run 2, largest internal gap
  12 frames. 18 of its gap frames hold an on-path box.
- **#96** (f12545–12566): 6 confirmed, longest run 3, and **all 16** gap frames
  hold an on-path sub-threshold box.

### 3.2 Mechanism 2 — `unreachable_displacement` over-splitting (3 candidates)

Candidates #35, #43 and #78 were not truncated by the coast ceiling; they were
**cut in half by a boundary decision**, and in two cases the cut lands on a real
trajectory that was already delivered or would have been.

Fragment chains — consecutive candidate events separated by ≤20 frames:

| Chain | Confirmed per event | Sum | Outcome |
|---|---|---|---|
| f4922–4965 | #34 = 2, **#35 = 7**, #36 = 2 | **11** | merged clears 8 → **a real delivery is being destroyed** |
| f10120–10192 | **#77 = 8 (DELIVERED)**, **#78 = 6**, #79 = 4 | **18** | already delivered once as #77; #78 is a **duplicate fragment** |
| f5928–5945 | #43 = 5, #44 = 1 | 6 | #43 split off a trajectory it was still part of |

`#77` is the uncomfortable one: it cleared the delivery threshold with **exactly
8** confirmed frames and was then truncated by the split at f10144, handing the
remainder to #78. A delivery sitting on the minimum and getting chopped is the
signature of an over-eager split.

The specific boundaries, straight from the logged kinematics:

| Cand | Displacement | Reachable | Speed ratio | Dir change | Note |
|---|---|---|---|---|---|
| #35 | 163.7 px | 86.3 px | 23.4 | 28.4° | unreachable (164 vs 86) and speed ratio 23.45 |
| #43 | 180.4 px | 129.7 px | 4.67 | 33.2° | unreachable (180 vs 130) and speed ratio 4.67 |
| #78 | 93.2 px | 89.9 px | 8.76 | 90.2° | unreachable (93 vs 90) and speed ratio 8.76 |

#78 is the marginal one — 93 px against a 90 px reachable radius, a 3% overrun,
with a 90° direction change. That is a boundary decided on a rounding-scale
overshoot.

---

## 4. Category B — noise and non-delivery activity (3)

**#68 (f9308–9314) — certainly noise.** Total path **5.2 px**. Vertical span
1 px. Median speed **1.0 px/frame**. Seven "confirmed" frames all inside a
5×1 pixel neighbourhood. Nothing travels 1 px per frame; this is a static false
positive trembling on a bat, a shoe or a stump. Its 7 frames are the single
clearest argument in this document that a *count* of frames is a weak test of
evidence.

**#3 (f565–573) — non-delivery motion.** **0%** of its confirmed points fall in
the pitch corridor (all other candidates are 100%). Vertical span 14 px against
a horizontal span of 229 px: a fast, flat, purely horizontal streak at constant
height. A bowled ball travels *toward* the batter, which is vertical motion in
this camera. 29.4 px/frame horizontally is fast for a ball and unremarkable for
anything else crossing the frame.

**#60 (f8145–8175) — non-delivery motion or detector artefact.** Path **933 px**,
more than any of the 36 deliveries (max 792). Median **45.1 px/frame**, faster
than the fastest delivery in the whole video (32.4). Only 16.7% in the corridor,
646 px of horizontal travel in 102 px of vertical, longest confirmed run of 2,
and 6 gate refusals — the filter was repeatedly rejecting measurements it could
not reconcile. A coherent 646 px lateral traverse at 45 px/frame is a different
object from a struck cricket ball.

---

## 5. `reseed_position` — why it fired zero times

**It is unreachable dead code, and the Stage 3 report's explanation for it was
wrong.**

The call site, `tracking_service.py:590`:

```python
if (
    decision.continued_same_event
    and decision.reseed_position is not None
    and track.position is None
):
    track = tracker.reseed(detection, frame_number)
```

`reseed_position` is set in only two places, `event_segmenter.py:751` and
`:816`, both inside the **continuation** branch — the branch that runs when a
confirmed measurement `(px, py)` has arrived and the segmenter has decided this
is the same ball after a gap. So `reseed_position is not None` implies
**this frame has a detection**.

But `track` was already updated with that same detection 20 lines earlier, at
`tracking_service.py:545`. When a detection is present, `BallTracker.update()`
returns a `TrackResult` whose `.position` is always populated — either the
smoothed measurement (`kf.update`/`kf.init`) or the predicted point on the
refusal path (`tracker.py:219`). `.position` is `None` **only** on the
no-detection path (`tracker.py:246`).

Therefore:

> `reseed_position is not None` ⇒ there was a detection ⇒ `track.position` is
> not None ⇒ `track.position is None` is False.

The three-way conjunction can never be satisfied. `tracker.reseed()` has never
run and cannot run, in any video, with or without the Stage 3 fix.

**Two consequences:**

1. **It cost nothing on this video.** All 120 `reacquired` decisions had
   `kalman_source = "detected"` and `kalman_missed = 0` — the measurement was
   already accepted and the filter already healthy. Reseed would have been
   redundant on every one of them. Its absence is a latent maintenance hazard
   and a misleading comment, not a source of lost deliveries.

2. **This corrects `STAGE3_KALMAN_FIX_REPORT.md` §6.7**, which speculated that
   *"its trigger requires the segmenter to open a new event first, which is
   precisely what the livelock prevented"*, and that rising
   `unreachable_displacement` counts *"may leave `reseed_position` set at
   boundaries it did not before."* Both claims are wrong. The livelock had no
   bearing on it, and the counts did rise without reseed becoming reachable.

Note the mechanism the comment at `tracking_service.py:587` describes — *"a
short dropout inside ONE event… restart the filter on the ball that is actually
visible"* — is currently served **only** by the Stage 3 fix, and only for the
already-lost case. For a *healthy-but-lagging* filter that refuses a real
measurement, nothing re-seeds it. That path is the reason the guard was written,
and it remains unserved.

---

## 6. Temporal context — where the 17 sit

| Placement | Count | Candidates |
|---|---|---|
| Before the first delivery (first delivery opens f839 / 33.6 s) | 4 | #1, #2, #3, #4 |
| Inside a long (>6 s) inter-delivery gap | 10 | #35, #38, #42, #43, #48, #50, #60, #68, #76, #91 |
| After the last delivery, near end of video | 2 | #96, #97 |
| Inside a *normal* short inter-delivery gap | 1 | #78 — and it is a fragment of an already-delivered event |

**This cuts against the category-A reading, and I am not going to hide it.** 16
of 17 fall in windows containing no detected delivery: the pre-match opening,
overs breaks (up to 36.6 s), and the end of play. That is where warm-up throws,
field-setting and players tossing the ball around happen. It is equally where
*undetected* deliveries would be.

The counter-evidence is that 14 of 17 have **100%** of their confirmed points
inside the pitch corridor, versus 61% for the ≤4 noise class — real deliveries
do not wander to the boundary rope.

### Two signals I tested that turned out to be useless

Reported so nobody re-runs them:

- **Camera motion has no discriminative power.** `motion_max` median: delivered
  33.13, noise 33.28. The best single split keeps only 16/36 deliveries while
  dropping 21/44 noise. `pitch_motion` likewise (delivered med 5.60, noise 6.92).
- **Directional spread does not separate the classes.** Delivered events run to
  159° of spread, because a delivery includes the ball before and after the
  strike plus bounces, which reverse direction.

---

## 7. Conclusions

### Counts

| Category | Count | Candidates |
|---|---|---|
| **A — consistent with a real delivery** | **11** | #1, #2, #4, #38, #42, #48, #50, #76, #91, #96, #97 |
| **B — noise / non-delivery activity** | **3** | #3, #60, #68 |
| **C — mis-segmentation artefacts, not independent candidates** | **3** | #35, #43, #78 |

**Category A is "consistent with a real delivery", not "proven to be one."** The
classification rests on kinematics (16–31 px/frame, matching the delivered
class), 100% pitch-corridor residency (delivered class 93.2%, noise 61%),
and confidence 0.48–0.60 against a delivered median of 0.64. No batting-event
signal exists to confirm it — the pipeline records
`batting_event_timing: null` and states plainly that it will not infer it from a
classifier that runs after segmentation.

One coincidence worth recording without over-reading it: **11 candidates in
category A, and 47 − 36 = 11 deliveries unaccounted for** in the original
complaint. That is consistent with the shortfall being fully explained by
`MIN_DELIVERY_FRAMES = 8` rejecting intermittently-visible balls. It is
suggestive, not proof. The 16-of-17 placement in non-delivery windows is real
counter-evidence against it.

### The exact Stage 4 mechanism responsible

**For the 11 real candidates: no Stage 4 mechanism is responsible. The limit is
detector confidence.**

- `MIN_DELIVERY_FRAMES` counts *total* confirmed frames. These candidates have
  5–7 because YOLO commits to only 5–7 measurements, in bursts of 2–4, with
  low-confidence on-path boxes in between.
- 64 of 92 internal gap frames hold a sub-threshold box; 92% of those sit within
  120 px of the ball's own path; **none reach the 0.30 cut**.
- The segmenter behaved correctly throughout: full 20-frame coast ceiling,
  truthful provenance, no fabricated confirmations, no premature closure.

So this is a **threshold-shaped** problem wearing a segmentation costume. The
honest framing is that `MIN_DELIVERY_FRAMES = 8` on raw counts is a weak
evidence test — it rejects a 7-frame ball at 24 px/frame in the pitch corridor
while accepting #68's 7 frames of 5 px jitter. **A total-count threshold cannot
distinguish those two.** That is the design finding, and it holds regardless of
where the threshold is set.

**For the 3 category-C candidates: `unreachable_displacement` is the mechanism.**
It splits one trajectory into fragments, and two of the three fragments are
counted as separate candidates. On f4922–4965 it destroyed a delivery that had
11 confirmed frames; on f10120–10192 it truncated a delivered event at exactly
its 8-frame minimum and produced #78 as a duplicate.

### What I did not do, and what I would want before changing anything

Not implemented, per instruction: no threshold moved, no confirmation logic
changed, no candidate promoted to a delivery.

Before any fix, two things are worth settling, because they point at different
code:

1. **Fragment merging is a separable, lower-risk fix with a known payoff.**
   Merging chains at gap ≤20 frames recovers exactly **one** delivery (the
   f4922–4965 chain, 2+7+2 = 11) and removes the #78 double-count. That is a
   bounded, verifiable change to over-splitting, and it needs no threshold
   change at all. #78 is a genuine false-positive risk in the *current* output:
   a trajectory already delivered as #77 is being counted again as a rejected
   candidate.

2. **The 11 category-A candidates are a threshold question, not a logic
   question.** For scale, and explicitly *not* as a recommendation to act now:
   counting on-path sub-threshold boxes at ≥0.15 would lift #1, #4, #76, #91,
   #96 and #97 over 8 — six deliveries — while #38, #42, #48 and #50 would
   still fall short, and it would require moving the YOLO cut, which is
   out of scope. It also would not admit #68, whose problem is that it moves at
   1 px/frame.

   A coherence test (path length and pitch-corridor residency, which separate
   the classes cleanly — 933 px and 100% corridor versus 5.2 px) discriminates
   far better than a frame count, and would not require touching the detector at
   all. That is the direction I would investigate next, but it is a Stage 4
   design change and it is not made here.

### Still open

- **Visual adjudication is the missing step.** `docs/stage4_frames/sheet_1..5.png`
  will settle category A definitively — 11 minutes of your attention, and it
  converts "consistent with" into "confirmed".
- Whether the 4 pre-match candidates (#1–#4, all before f839) are warm-up throws
  or genuine missed deliveries is the single highest-value question, since 4 of
  the 11 category-A candidates sit there.