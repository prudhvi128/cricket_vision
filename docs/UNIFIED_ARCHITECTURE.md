# Unified Architecture — CricketShot-Classification + CricketTracker

**Status:** Design only — **nothing implemented.** Phase 3 has not been started.
**Date:** 2026-10-04
**Note:** Historical design record. Module paths and directory layouts below predate
the Phase 11 restructure and are left as written; the current layout is in
`docs/REPOSITORY_STRUCTURE_REPORT.md` and `README.md`. The design reasoning and the
endpoint contract still hold.
**Supersedes:** `docs/MERGE_AUDIT.md` §10 (proposed end-to-end data flow). That flow re-ran ball
detection on every delivery clip — see §2 for what it actually cost and why it is rejected. Audit §12
risks mostly still stand; the one this design **eliminates** rather than mitigates is *"Trajectory
overlay sync (**High**)"*.

---

## 1. Core Architectural Decision: One Ball-Processing Pass

**CricketTracker's YOLO11 detector + Kalman filter is the single, authoritative ball-processing pass
over the uploaded full video. It runs exactly once.**

The CricketShot-Classification shot model (EfficientNet-B0 + Transformer) is a **second, independent
model that never touches ball pixels**. It consumes only the delivery clips produced by the tracking
pass, and its output is *merged into* the already-computed tracking analysis — never recomputed from.

### 1.1 The flow

```
FULL VIDEO (single upload)
    │
    ├──────────────────────────────────────────────────────────────────────┐
    │  ONE FRAME-BY-FRAME PASS  (this is the ONLY decode of the video)    │
    │                                                                      │
    │  per frame:                                                          │
    │    YOLO11 ball detection        (CricketTracker tracker_modules)    │
    │      → Kalman filter (x, y, vx, vy) + gating + occlusion handling    │
    │    → push into ring buffer (pre-roll)                                │
    │    → push into active clip writer (if a delivery is open)            │
    │                                                                      │
    │  on delivery finalise (ball lost, ≥ MIN_DELIVERY_FRAMES):           │
    │    compute bounce / speed / line / length / swing / angles           │
    │    compute delivery start & end frames                                │
    │    PERSIST tracking record  →  tracking.json  (atomic, incremental)   │
    │    on clip close (end + POST_ROLL):                                  │
    │        write delivery clip            (no seek, no re-decode)         │
    │        run shot model on in-memory frames (30 sampled)               │
    │        render + write trajectory-overlay clip                          │
    └──────────────────────────────────────────────────────────────────────┘
    │
    ↓
PERSISTED TRACKING RESULT  (data/analyses/{analysis_id}/tracking.json)
    │      detections · tracked positions · start/end frames · trajectory
    │      bounce · speed · line · length · swing · angles
    │
    ↓  (read from disk — not recomputed, not re-derived from video)
COMBINE  shot prediction  ⊕  stored tracking analysis
    │
    ↓
UNIFIED DELIVERY RESULT  →  API  →  Frontend
```

### 1.2 Hard constraints

| # | Constraint | Enforcement point |
|---|---|---|
| C1 | The uploaded full video is decoded **exactly once**. | All clip/overlay/shot work happens inside the tracking loop from an in-memory frame buffer. No stage after the pass opens the source video. |
| C2 | YOLO + Kalman run **exactly once per uploaded video**, not once per delivery. | `BallDetector` / `BallTracker` are instantiated in the tracking service only. The shot pipeline has no access to them. |
| C3 | Every downstream consumer reads the **same stored trajectory**. | `DeliveryRecord.trajectory.points` is the single source of truth. Overlay renderer, bounce marker, pitch map, bowling analytics and the API all read it. There is no second trajectory anywhere. |
| C4 | No re-processing of the original video per delivery. | There is no `process_delivery(video_path)` function in the codebase. |
| C5 | A clip-level re-track is permitted **only** under the enumerated conditions in §8, and is always flagged in the output. | `delivery.tracking_source` ∈ {`single_pass`, `clip_fallback`} + `delivery.fallback_reason`. |

---

## 2. Why the previous flow is rejected

The flow in `MERGE_AUDIT.md` §10 was:

> for EACH delivery clip: (a) shot classification, (b) **run YOLO11 + Kalman on clip**

That design has three defects, in increasing order of severity:

**2.1 Duplicate ball processing.** `delivery_analyzer.track_ball()` re-runs YOLO + Kalman on every
extracted clip. Every ball pixel is therefore inferred twice: once during the full-video pass that
produced the delivery boundaries, and once again on the clip. This is pure waste, and it scales with
delivery count.

**2.2 A second, contradictory ball pipeline.** `CricketShot-Classification` also ships an
`AutoClipper` (`src/segmentation/auto_clipper.py`) which opens the full video a *second* time and runs
**two** additional YOLO models (`Ball_Detection_Model.pt`, `Bat_Detection_Model.pt`) on every 5th frame
— 7,200 dual-inferences for an 18,000-frame video — plus EasyOCR on every candidate frame. So the
original design was, in fact, **three** ball passes over the full video, plus one per clip.

**2.3 Two coordinate spaces that cannot be reconciled (correctness, not just speed).** This is the
serious one. The full-video pass produces a trajectory in **full-resolution source pixel space**, with
**original-video frame numbers**. The per-clip re-track produces a trajectory in **clip pixel space
with clip frame numbers**. The overlay must be drawn on the clip using the *tracker* trajectory, but
the *shot result* is computed from a *different* tracker run over the same frames. If the two runs
disagree by even one detection — and they will, because `BallDetector` picks the highest-confidence
box and `BallTracker` gates at `GATE_RADIUS = 120` px against a differently-seeded Kalman state — the
overlay will not match the shot. The audit already flagged this as risk *"Trajectory overlay sync —
**High**"*. The single-pass design removes the entire class of bug rather than mitigating it.

---

## 3. What is kept, what is dropped

### 3.1 Kept from CricketTracker (authoritative)

| Component | Source | Role in unified system |
|---|---|---|
| `BallDetector` (YOLO11, conf 0.30, 640×360 inference, diag 4–80 px filter) | `tracker_modules/detector.py` | The one and only detector. Singleton, loaded once. |
| `BallTracker` + `KalmanBallTracker` (4-state, `MAX_MISSED=8`, `GATE_RADIUS=120`) | `tracker_modules/tracker.py` | The one and only tracker. **Needs one surgical change** — see §7.1. |
| `build_homography` / `pixel_to_ground` / `savgol_smooth` / `get_fps` | `tracker_modules/utils.py` | Used unchanged. Fractions + source resolution must be persisted (§5.3). |
| `detect_bounce`, `compute_release_speed`, `estimate_swing`, `compute_release_angle`, `compute_bounce_angle`, `classify_length`, `classify_line` | `tracker_modules/trajectory.py` | All bowling analytics, unchanged. |
| `draw_trajectory_trail`, `draw_bounce_marker`, `draw_release_marker` | `tracker_modules/renderer.py` | Overlay drawing primitives, reused on the clip. |

### 3.2 Kept from CricketShot-Classification

| Component | Source | Role in unified system |
|---|---|---|
| `ImprovedSOTAModel` + `SpatialAttention` + `PositionalEncoding` + `TemporalAttention` | `src/pipeline/efficientnet_transformer.py` | Shot classifier. Architecture must match the checkpoint **exactly**. |
| `load_shot_model`, `_extract_state_dict`, `IMAGE_TRANSFORM` | `src/pipeline/delivery_analyzer.py` | Checkpoint loading with `module.`/`model.` prefix stripping and the `class_weights` buffer fix. |
| `predict_shot` (softmax → argmax + per-class probabilities) | same | Inference. Batch dimension added (§8.3). |
| `SHOT_CLASSES` (10 classes) | same | Canonical class list, served to the frontend via the API to kill client-side enum duplication. |

### 3.3 Dropped / replaced

| Dropped | Reason | Replacement |
|---|---|---|
| `src/segmentation/auto_clipper.py` | Second full-video decode + 2 extra YOLO models + EasyOCR | Delivery boundaries come from the tracking pass (§4.3). |
| `src/tracking/detector.py`, `src/tracking/tracker.py` | Duplicate ball pipeline | CricketTracker's versions only. |
| `delivery_analyzer.track_ball()` | Per-clip YOLO+Kalman re-run | Stored trajectory (§5.2). |
| `src/visualization/trajectory_video.py` `generate_trajectory_video()` | Re-runs YOLO+Kalman to draw a trail we already have | Overlay service reading stored trajectory (§6). |
| `analyze_delivery_clip()` as-is | Bundles shot + ball tracking | Split: `classify_shot(clip_or_frames)` + `read_tracking(delivery_id)`, merged by the combiner (§7). |

### 3.4 Ball-detection model identity — must be resolved in Phase 3

Both repos ship a `backend/models/best.pt` at **exactly 6,225,450 bytes**. Same size is not proof of
same content. Phase 3 step 1 must hash both files and record the result. CricketTracker's
`best.pt` is the authoritative one under the single-pass rule; if the hashes differ, the CricketShot
copy is dropped and only CricketTracker's is loaded.

---

## 4. Stage A — The Single Tracking Pass

### 4.1 Service boundary

```
services/tracking_service.py
  run_tracking_pass(video_path, analysis_dir, progress_cb) -> TrackingResult
```

Responsibilities, in order:

1. Open the video. Read `total_frames`, `fps`, `width`, `height`. Build `H = build_homography(w, h)`.
   Persist the homography's provenance (§5.3).
2. Instantiate `BallDetector` (singleton, process-wide) and one `BallTracker`.
3. Initialise the **frame ring buffer** (§4.4) and the **clip writer registry** (§4.5).
4. Run the frame loop. Per frame: `detect → tracker.update → analyse → persist → render → write`.
5. On each delivery finalisation: compute analytics, build the `DeliveryRecord`, **flush
   `tracking.json` atomically**.
6. On each clip close: write the clip, run the shot model, render + write the overlay, then flush.
7. Return the final `TrackingResult`.

### 4.2 The frame loop

Pseudocode — this is the hot path and the only place the video is read.

```python
frame_number = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_number += 1                      # 1-based, original-video frame number

    progress(track_stage, frame_number / total_frames)

    detection = detector.detect(frame)                  # YOLO11 — ONCE per frame, ever
    position, source = tracker.update(detection, frame_number)   # source ∈ {detected, predicted}

    ring.push(frame_number, frame)                      # JPEG-compressed, byte-capped
    if position is not None:
        persist_frame_observation(frame_number, position, source)   # in-memory until flush
        if source == "detected":
            raw_detections.append({frame, x, y, confidence})

    # ── delivery segmentation (unchanged CricketTracker logic) ────────────────
    if tracker.is_lost and tracker.in_delivery:
        _finalise_delivery(...)      # §4.3
        tracker.reset_delivery()

    # ── close any clip whose window has elapsed ──────────────────────────────
    clips.tick(frame_number)          # §4.5
```

Note the ordering: `clips.tick()` runs **after** segmentation, so a delivery finalised on frame *N*
immediately registers its clip and the very next frame begins streaming into it.

### 4.3 Delivery segmentation and boundary definition

CricketTracker's rule is preserved exactly:

| Constant | Value | Meaning |
|---|---|---|
| `MIN_DELIVERY_FRAMES` | 8 | Minimum detected positions before a delivery is real. |
| `MIN_FRAMES_BETWEEN_DELIVERIES` | 60 (2 s @ 30 fps) | Debounce. A delivery finalising sooner than this after the previous one is discarded. |

> **Cleanup noted in passing:** `model.py:247` hardcodes the literal `60` instead of referencing
> `MIN_FRAMES_BETWEEN_DELIVERIES`. Fix during the port; do not carry the magic number forward.

Four frame numbers are recorded per delivery, and they mean different things:

| Field | Definition |
|---|---|
| `detection_start_frame` | `frame_no` of the **first** entry in `delivery_positions` — the first frame with a *confirmed* detection. |
| `detection_end_frame` | `frame_no` of the **last** entry in `delivery_positions`. |
| `ball_lost_frame` | The `frame_number` at which `tracker.is_lost` became true. Typically `detection_end_frame + MAX_MISSED + 1`. This is when the *ball* disappeared, not when the *shot* finished. |
| `release_frame` | `detection_start_frame + detect_release_frame(delivery_positions)` (quarter-window max-displacement heuristic, `trajectory.py:209`). |

`delivery_positions` contains **detection-confirmed frames only** — in `BallTracker.update()` the
`delivery_positions.append(...)` call sits inside the `if detection is not None` branch, so
Kalman-extrapolated frames never enter it. That is intentional and must be preserved: the physics
(`compute_release_speed`) must not be fitted to extrapolated points. It also means
`detection_start_frame` is genuinely the first *observed* ball, not the first predicted one.

### 4.4 The frame ring buffer

Delivery boundaries are only known **after** the delivery has ended, but the clip must start *before*
it. Hence a rolling buffer of recent frames.

| Property | Value | Rationale |
|---|---|---|
| Encoding | JPEG, quality 92, BGR | 1080p raw is 5.93 MiB/frame; a 180-frame raw ring is 1.04 GiB. JPEG q92 is ~250–400 KB/frame → **~45 MiB**. |
| Capacity | `PRE_ROLL_SECONDS + POST_ROLL_SECONDS` ≈ 4 s + 2 s = **180 frames @ 30 fps** | Only the pre-roll must be *retained*; once a clip writer is open, frames stream out and are immediately evictable. |
| Hard cap | `RING_BUFFER_MAX_BYTES` (default 512 MiB) | Backstop for unusually large sources. |
| Eviction | Oldest first. On byte-cap eviction, **log a warning** and record the *achieved* pre-roll in the delivery record. | Never silently deliver a shorter-than-configured clip. |

Frames are decoded from JPEG immediately before being written to the clip writer. At ~2 ms per 1080p
JPEG decode and ~41 writes per match, this is negligible.

### 4.5 Clip extraction — streaming, never seeking

**Clips are written from the ring buffer as frames stream past. The source video is never seeked.**

This is the crux of C1 and it eliminates a whole class of risk:

- `AutoClipper._extract_clip()` used `reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)` followed by
  sequential reads. On H.264 with B-frames, `CAP_PROP_POS_FRAMES` seeks to the nearest preceding
  keyframe and the decoder may return the wrong frame or land off-by-N. With variable frame rate it
  is worse. Because `clip_frame` numbering *must* be exactly right for the overlay to land on the
  right frame, **seeking is not acceptable for clip extraction.** Sequential decode sidesteps it.
- Only one delivery can be open at a time (deliveries are ≥60 frames apart by the debounce), so a
  single active writer suffices.

Clip window:

```
clip.start_frame = max(1, detection_start_frame - PRE_ROLL_FRAMES)
clip.end_frame   = min(total_frames, ball_lost_frame + POST_ROLL_FRAMES)
```

**`POST_ROLL` is not cosmetic.** The shot is played *after* the ball reaches the batter, which is
after the last tracked ball frame. A clip ending at `ball_lost_frame` would show the ball travelling
and nothing else — the shot classifier would be fed footage in which the stroke has not yet begun.
`POST_ROLL_FRAMES` (default 2 s) is what makes the shot label meaningful.

Codec: `VideoWriter` fourcc is resolved through a fallback chain, because the current code hardcodes
`avc1`, which commonly fails to open on Windows without OpenH264 — and `model.py` never checks
`out.isOpened()`, so that failure only surfaces much later as *"Output video was not created"* after
the entire tracking pass has been wasted.

```
avc1  (H.264, best browser support)
  → mp4v (MPEG-4 Part 2, always available via OpenCV)
  → raise PipelineError("no writable video codec available")
```

The resolved codec is recorded in `clip.codec`.

### 4.6 Frame-mapping invariant

For every clip, these must hold. They are asserted at write time, not assumed:

```
clip.frame_offset = clip.start_frame - 1
clip_frame_1based(original_frame) = original_frame - clip.frame_offset
original_frame(clip_frame_1based) = clip_frame_1based + clip.frame_offset

INVARIANT A:  clip.start_frame ≤ original_frame ≤ clip.start_frame + clip.frame_count - 1
INVARIANT B:  frames_written == clip.frame_count
```

If either fails (a writer that silently drops frames, or a pre-roll that could not be fully
retained), the delivery is marked:

```
clip.frame_map_valid = false
clip.frame_map_error = "wrote 68 of 70 frames"   // or "pre-roll truncated: 2.1s of 4.0s"
```

and `tracking_source` is escalated per §8.1. A clip with `frame_map_valid == false` must **never** be
overlaid from the stored trajectory — the mapping is exactly what cannot be trusted, and drawing a
trajectory onto the wrong frames is worse than drawing none.

---

## 5. Persisted Tracking Result

### 5.1 Storage layout

```
backend/data/
  uploads/                                  {analysis_id}/source.mp4
  analyses/
    {analysis_id}/
      tracking.json                         ← §5.2. THE reusable artifact.
      clips/         delivery_001.mp4 …      ← raw extracted clips
      overlays/      delivery_001.mp4 …      ← trajectory-overlay clips
      result.json                          ← unified per-delivery results (stage B)
```

`tracking.json` is written with **temp-file + atomic rename**, after every delivery finalises and
again after every clip closes. A crash mid-video therefore leaves a valid, partial-but-consistent
`tracking.json` — which is also what makes the artifact genuinely *reusable*: shot classification,
overlay rendering or API serving can be re-run from `tracking.json` without ever touching the source
video again.

### 5.2 `tracking.json` schema

`schema_version` is `"2.0"` (deliberately distinct from any legacy `analysis.json`).

```jsonc
{
  "schema_version": "2.0",
  "analysis_id": "3f9c1e42",
  "created_utc": "2026-10-04T11:20:31Z",
  "pipeline_version": "1.0.0",

  "video": {
    "source_path": "uploads/3f9c1e42/source.mp4",
    "original_filename": "match.mp4",
    "width": 1920, "height": 1080,
    "fps": 29.97, "total_frames": 17982, "duration_sec": 600.07
  },

  // Persisted so that every downstream metric is reproducible and auditable.
  "calibration": {
    "homography_corner_fractions": {
      "TL": [0.418, 0.268], "TR": [0.560, 0.268],
      "BL": [0.432, 0.655], "BR": [0.572, 0.655]
    },
    "calibrated_for_resolution": [1920, 1080],
    "pitch_length_m": 20.12, "crease_width_m": 2.64,
    "note": "Corner fractions are measured off one specific broadcast template. Re-measure if camera framing changes; speed is wrong otherwise."
  },

  "detector": {
    "model": "best.pt", "sha256": "…", "arch": "yolo11",
    "conf_threshold": 0.30, "infer_size": [640, 360],
    "min_diag_px": 4, "max_diag_px": 80, "device": "cuda:0"
  },
  "tracker": { "MAX_MISSED": 8, "GATE_RADIUS": 120, "MAX_TRAIL": 120,
               "states": ["x", "y", "vx", "vy"] },

  // Raw YOLO output, all frames. Kept separate from tracked positions so that
  // "what the model saw" is never conflated with "what the filter believed".
  "detections": [
    { "frame": 100, "x": 1188, "y": 214, "confidence": 0.71 }
  ],

  "deliveries": [ /* §5.2.1 */ ],

  "totals": {
    "frames_decoded": 17982, "detections": 12406, "delivery_count": 41,
    "deliveries_with_speed": 38, "deliveries_with_bounce": 41,
    "degraded_deliveries": 3, "clip_fallback_deliveries": 0
  },
  "timings": { "tracking_pass_sec": 214.6, "shot_classification_sec": 41.2,
               "clip_write_sec": 18.9, "overlay_render_sec": 63.4 }
}
```

#### 5.2.1 `deliveries[]` — the per-delivery record

```jsonc
{
  "delivery_id": 1,

  // ── boundaries (original-video frame numbers, 1-based) ───────────────────
  "detection_start_frame": 100,
  "detection_end_frame": 145,
  "ball_lost_frame": 154,
  "release_frame": 103,

  // ── the artifact ─────────────────────────────────────────────────────────
  "clip": {
    "path": "clips/delivery_001.mp4",
    "url": "/data/analyses/3f9c1e42/clips/delivery_001.mp4",
    "start_frame": 82,          // == max(1, detection_start_frame - 72)
    "end_frame": 214,           // == min(total_frames, ball_lost_frame + 60)
    "frame_count": 133,
    "frame_offset": 81,         // == start_frame - 1
    "fps": 29.97, "width": 1920, "height": 1080,
    "codec": "avc1",
    "frame_map_valid": true,
    "frame_map_error": null,
    "pre_roll_actual_sec": 0.60,  // clamped by start of video
    "post_roll_frames": 60
  },

  // ── THE trajectory. Single source of truth for C3. ────────────────────────
  "trajectory": {
    "points": [
      { "original_frame": 100, "clip_frame": 19, "x": 1188, "y": 214,
        "source": "detected", "confidence": 0.71 },
      { "original_frame": 101, "clip_frame": 20, "x": 1196, "y": 219,
        "source": "predicted", "confidence": null }
      // …
    ],
    "point_count": 54,
    "detected_count": 46,
    "predicted_count": 8,
    "continuity": "continuous",      // continuous | gapped
    "gaps": []                       // [[start_frame, end_frame], …] where nothing was tracked
  },

  // ── physics inputs, persisted so the analytics are auditable ─────────────
  "speed_samples": [                 // the exact list fitted by compute_release_speed
    { "frame": 100, "x": 1188, "y": 214, "ground_x_m": 1.31, "ground_y_m": 0.94, "t_sec": 3.34 }
  ],

  // ── bounce ───────────────────────────────────────────────────────────────
  "bounce": {
    "detected": true,
    "original_frame": 128, "clip_frame": 47,
    "x_px": 1421, "y_px": 618,
    "x_norm": 0.740, "y_norm": 0.572,   // for the pitch map; strictly in [0,1]
    "ground_x_m": 1.98, "ground_y_m": 11.42,
    "method": "score"                     // score | fallback_max_y
  },

  "bowling": {
    "speed_kmh": 137.4,
    "speed_estimated": false,             // true only if §9 imputation is enabled
    "length": "Good Length",
    "line": "Off Side",
    "swing": "3.2° outswing",
    "release_angle": 12.3,
    "bounce_angle": 8.7
  },

  "shot": null,                          // filled in §7 once the clip closes
  "overlay": {
    "path": "overlays/delivery_001.mp4",
    "url": "/data/analyses/3f9c1e42/overlays/delivery_001.mp4",
    "rendered": true,
    "trajectory_source": "single_pass"   // provenance of what was drawn
  },

  "tracking_source": "single_pass",      // single_pass | clip_fallback
  "fallback_reason": null,
  "quality": { "trajectory_valid": true, "analytics_complete": true, "flags": [] }
}
```

### 5.3 Normalised coordinates: the pitch-map contract

`PitchMap.tsx` consumes `bounce_x` / `bounce_y` as **normalised 0–1** values and applies its own
perspective transform. Two rules follow, and both must be enforced by the backend:

1. **Emit normalised values only.** `bounce.x_norm = round(x_px / video_width, 3)`,
   `bounce.y_norm = round(y_px / video_height, 3)`. Because `classify_length` / `classify_line`
   already classify on image fractions, these are resolution-independent.
2. **Omit, never `null`.** The existing frontend gates on `d.bounce_x !== undefined` and then
   non-null-asserts. A `null` passes the gate and `null * 160` coerces to `0`, silently plotting a
   dot at the top-left of the pitch. If no bounce was detected, the key must be **absent** from the
   API payload.

`x_norm` is a left→right fraction across the frame; `y_norm` is a top→bottom fraction down the frame.
Because the two homography regimes (broadcast side-on vs. down-the-pitch) invert that relationship,
the perspective template is recorded in `calibration.homography_corner_fractions` and the frontend
must select its pitch-map projection from it. Shipping two named templates
(`broadcast_side_on`, `down_the_pitch`) and letting the API select is the clean fix.

---

## 6. Stage B — Clip-Frame Overlay Rendering

### 6.1 Input

The stored `delivery.trajectory.points` and `delivery.clip.*`. **The source video is not opened.**

### 6.2 Rendering algorithm

For each clip frame index `i` (0-based), the original frame is `original = clip.frame_offset + i + 1`,
and the rendered trail is every stored point with `original_frame ≤ original`:

```python
for i in range(clip.frame_count):
    frame  = decode(clip_path, i)
    origin = clip.frame_offset + i + 1
    trail  = [p for p in points if p.original_frame <= origin]     # progressive reveal

    if len(trail) >= 2:
        draw_trajectory_trail(frame, [(p.x, p.y) for p in trail])

    if bounce.detected and origin >= bounce.original_frame:
        draw_bounce_marker(frame, (bounce.x_px, bounce.y_px))

    if origin >= release_frame:
        draw_release_marker(frame, (points[release_index].x, points[release_index].y))

    writer.write(frame)
```

`trail` is maintained as a running prefix list (append-only, `bisect` on `original_frame`) rather
than recomputed per frame — an O(n·m) rescan would be needlessly quadratic.

### 6.3 Resolution identity is mandatory

Trajectory coordinates are in **full-source-resolution pixel space** (`BallDetector.detect()` scales
back by `sx = w_full / RESIZE_W`, `sy = h_full / RESIZE_H`), and the clip is written at the source
resolution without resizing. Therefore the mapping is the **identity** on (x, y) and is valid only
while `clip.width == video.width and clip.height == video.height`. Assert this before rendering; if a
future change ever rescales a clip, it must rescale the stored coordinates by the same factor or the
overlay will be silently misaligned.

---

## 7. Shot Classification and Merge

### 7.1 One required change to `BallTracker`

`BallTracker.update()` currently returns a bare position for both confirmed detections and
Kalman-extrapolated frames. The unified schema needs provenance, because `trajectory.points[].source`
is what distinguishes real measurement from prediction in the overlay, the continuity reporting and
the frontend's honesty about what it is showing. The change is non-breaking — return the source
alongside the position, or expose it as an attribute — and it **must not** change
`delivery_positions` semantics (§4.3).

### 7.2 Feeding the model — no seeks

`delivery_analyzer.load_30_frames()` seeks 30 times via `cap.set(CAP_PROP_POS_FRAMES, index)`. Under
the single-pass rule that is 30 × 41 = 1,230 seeks, each a potential decode stall and a
frame-accuracy risk, all of it avoidable: **the frames are already in memory.** Sample
`np.linspace(0, len(clip_frames) - 1, 30)` from the in-memory clip frames, apply `IMAGE_TRANSFORM`
unchanged (BGR→RGB, resize 224, ImageNet normalisation), and batch.

Sampling from the ring buffer is byte-identical to sampling from the written clip **iff**
`frames_written == expected`, which is exactly what INVARIANT B checks. When it does not hold, the
shot model falls back to reading the written clip (§8.2) and the result is tagged
`shot.input_source: "clip_file"` rather than `"ring_buffer"`.

The checkpoint must be loaded with `strict=True` and the `class_weights` buffer fix intact — this is
an architecture/compatibility contract, not a nicety, and a silent `strict=False` would produce
confident nonsense across all 41 deliveries.

### 7.3 The merge

```jsonc
"shot": {
  "type": "Cover",
  "confidence": 0.87,
  "class_probabilities": { "Cover": 0.87, "Defense": 0.08, "…": 0.01 },
  "input_source": "ring_buffer",       // ring_buffer | clip_file
  "frames_sampled": 30,
  "sampled_clip_frames": [0, 4, 9, 13, 18, 22, 27, 31, 36, 40, 45, 49, 54, 58,
                           63, 67, 72, 76, 81, 85, 90, 94, 99, 103, 108, 112,
                           117, 121, 126, 132]
}
```

The combiner does **not** compute anything. It joins `shot.*` onto `bowling.*`, `bounce.*`,
`trajectory.*` and `clip.*` from the same `DeliveryRecord`, and adds no derived numbers. Per C3, one
trajectory object serves the overlay video, the bounce marker, the pitch map, the bowling analytics
and this response.

---

## 8. Documented Fallback Cases

Re-running YOLO + Kalman on a delivery clip is permitted **only** when the predicate below fires.
Each is checked in order; the first match wins. Every fallback is recorded as
`tracking_source: "clip_fallback"` with a `fallback_reason`, and the UI is expected to surface the
flag. A delivery never silently mixes pass-1 and fallback values.

### 8.1 Fallback table — re-track the clip (YOLO + Kalman on the clip)

| ID | Predicate | Why the stored data is unusable |
|---|---|---|
| **F1** | `tracking.json` has no record for this `delivery_id`, or the record is unreadable (failed parse, truncated write, schema mismatch). | There is no stored trajectory to render. |
| **F2** | `clip.frame_map_valid == false`. | The clip's frame numbering does not match the source numbering, so the stored trajectory **cannot** be mapped onto the clip. Re-tracking the clip is the only way to obtain a trajectory that is self-consistent with the file being rendered. |
| **F3** | `trajectory.point_count == 0` **or** `detection_count < MIN_DELIVERY_FRAMES`. | The single pass never acquired a usable track for this delivery, though a clip window exists. |

F2 is the case that most directly justifies the exception: the failure is *in the mapping*, so the
fix has to be a trajectory measured in the clip's own frame space.

### 8.2 Clip-level fallbacks — re-read the *clip*, never the source video

| ID | Predicate | Action |
|---|---|---|
| **F4** | `frames_written != frame_count` (the clip file is short or long). | Re-read the **clip** for shot-model frame sampling. Source video untouched. Tags `shot.input_source: "clip_file"`. |
| **F5** | Shot model checkpoint failed to load / inference raised. | Record `shot: null, shot.error: "…"`. Bowling analytics are unaffected. Shot classification must never be able to fail a delivery's tracking data. |
| **F6** | Pre-roll truncated at video start, or post-roll truncated at video end. | Note it in the record. The clip is still correct, just shorter — **not** a reason to re-track. |

### 8.3 Explicitly *not* fallbacks

- Clip window too tight to show the stroke → widen `PRE_ROLL`/`POST_ROLL` and re-extract. **No re-track.**
- Speed is `null` or outside the 40–200 km/h plausibility band → the answer is `null`, not a re-track.
  `compute_release_speed` deliberately returns `None` rather than clamping ("never fabricated").
- Bounce fell back to `fallback_max_y` → recorded as `bounce.method: "fallback_max_y"` and flagged.
  Re-tracking will not improve a genuinely bounce-less delivery.
- Shot confidence low → reported as-is with the confidence value. No re-run, no threshold fudging.

### 8.4 Batching

Shot classification for the N deliveries may be batched (`ImprovedSOTAModel.forward` accepts a batch
dimension `B > 1`; only `predict_shot` currently asserts `B == 1`). Batching touches no tracking data
and is purely a throughput optimisation — it does not weaken C1–C5.

---

## 9. Unified API Contract

### 9.1 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/analyses` | Upload the full video. Returns `{analysis_id, status: "queued"}`. |
| `GET` | `/api/analyses/{id}` | Status + progress + errors. |
| `GET` | `/api/analyses/{id}/events` | SSE progress stream. See §9.1.1 — neither repo's progress channel currently works. |
| `GET` | `/api/analyses/{id}/result` | Unified delivery result (§9.2). |
| `GET` | `/api/analyses/{id}/tracking` | The raw `tracking.json`. Independently consumable — this is the "reusable artifact" endpoint. |
| `POST` | `/api/analyses/{id}/shots` | Re-run shot classification from persisted clips/tracking only. **Never touches YOLO.** |
| `POST` | `/api/analyses/{id}/overlays` | Re-render overlays from the stored trajectory. |
| `GET` | `/api/reference` | `SHOT_CLASSES`, `LENGTH_ZONES`, `LINE_ZONES`, pitch-map templates, colour map. Kills the 4-way client-side enum duplication. |
| `GET` | `/data/…` | Static clips + overlays. |

No global mutable `processing_status` singleton (`CricketTracker-main/backend/main.py:29`). State
lives per `analysis_id`, so concurrent uploads cannot corrupt each other.

#### 9.1.1 Both existing progress channels are broken

Neither repo's frontend↔backend progress contract actually functions. Recorded here because it is
easy to assume SSE already works and build on top of it:

| Repo | Frontend expects | Backend provides | Status |
|---|---|---|---|
| CricketShot | `EventSource` on `/api/progress/{task_id}` (`frontend/src/services/api.ts:105-181`) | A plain `JSONResponse` from a normal route (`backend/main.py:369-377`). Not `text/event-stream`. | **Mismatch.** `EventSource` cannot parse a plain JSON body; the stream errors immediately. |
| CricketShot | `DELETE /api/cancel/{task_id}` (`api.ts:77-100`) | Route does not exist. | **404.** |
| CricketShot | `result.predictedClass`, `result.timeline[]` | `analyze_delivery_clip()` returns `result["shot"]["prediction"]` and no `timeline` at all. | **Field mismatch.** |
| CricketTracker | `setInterval(…, 1000)` polling of `/status` (`Processing.tsx:29`) | `/status` returns JSON. | **Works**, but polls a global singleton and only ever reports `0 / 5 / 100`. |

The unified contract above therefore specifies SSE *and* keeps `GET /api/analyses/{id}` working, so
the frontend degrades to polling rather than breaking. Under the single-pass design the backend can
finally report **real** per-stage progress (§10 note: the Tracker's decorative 5-stage UI array was
only ever fed three distinct values), because there are now six genuinely distinct, long-running
stages: upload → track → clip write → shot classify → overlay render → complete.

### 9.2 `result.json`

```jsonc
{
  "analysis_id": "3f9c1e42",
  "status": "completed",
  "processing_time_sec": 338.1,
  "video": { "original_filename": "match.mp4", "duration_sec": 600.07,
             "width": 1920, "height": 1080, "fps": 29.97, "total_frames": 17982 },
  "reference": {
    "shot_classes": ["Cover", "Defense", "Flick", "Hook", "Late Cut",
                     "Lofted", "Pull", "Square Cut", "Straight", "Sweep"],
    "length_zones": ["Beamer", "Bouncer", "Short", "Good Length", "Full", "Yorker"],
    "line_zones":   ["Wide Leg", "Leg Side", "Middle", "Off Side", "Wide Off"],
    "pitch_map_template": "broadcast_side_on"
  },
  "summary": {
    "total_deliveries": 41,
    "with_shot": 41, "with_speed": 38, "with_bounce": 41,
    "avg_speed_kmh": 134.8, "max_speed_kmh": 148.2,
    "degraded_deliveries": 3, "clip_fallback_deliveries": 0
  },
  "deliveries": [
    {
      "delivery_id": 1,

      "clip_url":        "/data/analyses/3f9c1e42/clips/delivery_001.mp4",
      "overlay_video_url": "/data/analyses/3f9c1e42/overlays/delivery_001.mp4",

      "shot":    { "type": "Cover", "confidence": 0.87, "class_probabilities": {…} },
      "bowling": { "speed_kmh": 137.4, "speed_estimated": false,
                   "length": "Good Length", "line": "Off Side",
                   "swing": "3.2° outswing",
                   "release_angle": 12.3, "bounce_angle": 8.7 },

      "bounce": { "detected": true, "original_frame": 128, "clip_frame": 47,
                  "x": 1421, "y": 618, "x_norm": 0.740, "y_norm": 0.572,
                  "method": "score" },

      // Same array the overlay video was rendered from. §5.2 / C3.
      "trajectory": {
        "points": [ { "original_frame": 100, "clip_frame": 19,
                      "x": 1188, "y": 214,
                      "source": "detected", "confidence": 0.71 } ],
        "point_count": 54, "detected_count": 46, "predicted_count": 8,
        "continuity": "continuous", "gaps": []
      },

      "mapping": { "clip_start_frame": 82, "clip_frame_count": 133,
                   "frame_offset": 81, "frame_map_valid": true },

      "tracking_source": "single_pass",
      "quality": { "trajectory_valid": true, "analytics_complete": true, "flags": [] }
    }
  ]
}
```

**`original_frame → delivery_id → clip_frame → (x, y)` is fully reconstructible** from
`mapping.frame_offset` + `trajectory.points[]`, in both directions, with no reference to the source
video. That is the persistence requirement, satisfied.

### 9.3 Values are never fabricated

`speed_kmh`, `release_angle`, `bounce_angle`, `swing` are `null` when not computable. Nothing clamps
a number into a "realistic" range to make the UI look complete.

One deliberate exception requiring sign-off — see §10.

---

## 10. Open Decisions (need a call before Phase 3)

**D1 — Median speed imputation.** `model.py:182-188` fills every `null` speed with the median of the
balls that *did* compute, setting `speed_estimated: true`. That is a fabricated number in a product
whose stated rule is *"use actual values only; `null` if unavailable"* (`MERGE_AUDIT.md` §11).
**Recommendation:** make it opt-in via `impute_missing_speeds` (default **false**), keep the flag in
the payload so the UI can style imputed values differently if enabled. Do not ship it on by default.

**D2 — Speed calibration fudge factor.** `trajectory.py:110`:
`kmph = speed_mps * 3.6 * 2.0 - 20.0` — a 2× scale plus a 20 km/h offset, tuned because
homography-derived speed reads 60–80 km/h under broadcast perspective. This is a calibration hack, not
a measurement. **Recommendation:** keep it for now (removing it would drop every speed to
nonsense), but persist `calibration.speed_scale` / `speed_offset` alongside the values and label the
speed as homography-derived in the UI. A proper fix needs per-video pitch-corner calibration.

**D3 — Delivery count.** The Tracker repo's flow produced ~6 deliveries on a test video; the Shot
repo's `full_match_output/` contains 41 clips from its AutoClipper. Under the single-pass design the
delivery count is whatever the ball-loss segmentation finds. **Expect it to differ from 41.** This is
correct — the boundaries are now physics-based rather than ball+bat+OCR-heuristic-based — but it
will look like a regression unless it is called out in advance. Needs a side-by-side accuracy
comparison on the same video before Phase 4 sign-off.

**D4 — Speed fitting window.** `compute_release_speed` is fitted on the first
`MAX_EARLY_POSITIONS = 12` *detected* frames. For a fast delivery that may already include the
bounce. Consider fitting only up to `release_frame + N`. Defer; measure first.

**D5 — `TOTAL_FRAMES` trust.** `CAP_PROP_FRAME_COUNT` is metadata and is routinely wrong on
variable-frame-rate files. The pass uses it only for progress reporting and for `clip.end_frame`
clamping — **never** for frame numbering, which comes from the decode loop counter. Confirm this
invariant survives the port.

---

## 11. Performance Accounting

Worked example: 18,000-frame 1080p video, 41 deliveries, GPU available.

| Work item | Rejected flow | Single-pass flow | Saved |
|---|---|---|---|
| Full-video decodes | **2** (tracking + AutoClipper) | **1** | 1 full decode |
| YOLO inferences over the full video | 18,000 (`best.pt`) + 7,200 (`Ball_Detection_Model.pt`) = 25,200 | 18,000 (`best.pt`) | 7,200 |
| Bat-model inferences | 3,600 (`Bat_Detection_Model.pt`) | 0 | 3,600 |
| EasyOCR scoreboard reads | up to 3,600 | 0 | up to 3,600 |
| Per-clip YOLO + Kalman | ~1,230 | 0 | ~1,230 |
| Random seeks for shot sampling | ~1,230 | 0 | ~1,230 |
| **YOLO + Kalman runs on ball pixels** | **4 distinct pipelines** | **1** | **3 pipelines retired** |
| Shot-model inferences (30 frames × 41) | 41 | 41 | — |
| GPU-bound work | 4 passes' worth | 1 pass' worth | ~3–4× less |

**Honest caveat:** for a single 10-minute video the wall-clock saving is dominated by the eliminated
AutoClipper pass (its 7,200 dual-inferences plus OCR on a 40 MB model is the largest single item).
The per-clip re-track is comparatively cheap in absolute terms because the clips are short. The
real, durable wins are:

1. **It scales.** The saving grows linearly with video length and delivery count. A 90-minute
   broadcast would be ~9× this table.
2. **Correctness.** One coordinate space, one frame space, one trajectory — the "trajectory overlay
   sync" *High* risk from the audit is eliminated structurally rather than mitigated.
3. **Bounded memory.** No full-video frame buffer; the ring buffer is capped at ~45 MiB.

---

## 12. Module Layout (Phase 2 scaffold — not yet created)

```
CricketUnified/
├── backend/
│   ├── app/
│   │   ├── api/{endpoints.py, deps.py}
│   │   ├── core/{config.py, constants.py}
│   │   ├── models/            # best.pt (ball), cricket_model_transformer.ckpt (shot)
│   │   ├── schemas/{analysis.py, delivery.py, tracking.py}
│   │   ├── services/
│   │   │   ├── tracking_service.py    # §4 — the ONE pass
│   │   │   ├── ring_buffer.py         # §4.4
│   │   │   ├── clip_writer.py         # §4.5
│   │   │   ├── tracker_modules/       # ported from CricketTracker, unmodified except §7.1
│   │   │   ├── shot_classifier.py     # §7.2
│   │   │   ├── overlay_renderer.py    # §6
│   │   │   ├── analytics.py           # bounce/speed/line/length/swing/angles
│   │   │   └── combiner.py            # §7.3
│   │   └── utils/
│   ├── data/{uploads,analyses}/
│   ├── main.py
│   └── requirements.txt
├── frontend/                 # base: CricketTracker's ballscribe-insight (React 18 + shadcn/ui)
└── docs/
```

Frontend changes required by this architecture:

| Component | Change |
|---|---|
| `TrajectoryChart.tsx` | Currently **entirely fake** — hardcoded SVG bezier, zero props, not mounted anywhere. Replace with a real chart over `trajectory.points`. |
| `PitchMap.tsx` | Normalised-coord contract (§5.3); select projection from `reference.pitch_map_template`; line-zone labels currently say `"Leg"`/`"Off"` but the backend emits `"Leg Side"`/`"Off Side"` — take the enum from `/api/reference`. Gate on key **absence**, not `!== null`. |
| `BallAnalysisSection.tsx` | Delete the dead `fetcher` branch; delete `ALL_LENGTHS`/`ALL_LINES` duplication (feed from `/api/reference`); fix the `|| true` no-op filter on line 76. |
| `SummaryStats.tsx` | Guard `Math.max(...speeds)` spread for very large arrays. |
| `Results.tsx` | Drop the `location.state` → `localStorage` relay and the `.replace("127.0.0.1", "localhost")` string hack; fetch from the API. |
| `Processing.tsx` | Replace the decorative 5-stage array with the real stages, which are now genuinely reportable; fix `data.progress \|\| 10`; stop toasting on every failed 1 s poll. |
| New | `DeliveryViewer` — overlay video + shot/bowling detail + Previous/Next with disabled boundaries. |

---

## 13. Phase Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository audit | ✅ `docs/MERGE_AUDIT.md` |
| 2 | Unified architecture | 🔄 **This document.** Scaffold + models not yet copied. |
| 3 | Backend integration | ⛔ **Not started — explicitly deferred.** |
| 4 | Video/trajectory output | ⛔ Not started |
| 5 | Frontend integration | ⛔ Not started |
| 6 | End-to-end testing | ⛔ Not started |

### Phase 3 entry checklist

1. Hash both `best.pt` files and confirm identity (§3.4).
2. Port `tracker_modules/*` from CricketTracker verbatim, plus the §7.1 provenance change.
3. Persist `delivery_positions` frame numbers — `model.py` currently discards them at the end of
   `_finalise_delivery`, and everything in §4.3 depends on them.
4. Build `ring_buffer.py` and `clip_writer.py`; make `clips.tick()` part of the loop (§4.2).
5. Add the `VideoWriter.isOpened()` check and the `avc1 → mp4v` fallback chain (§4.5).
6. Implement `tracking.json` with atomic incremental writes (§5).
7. Replace `load_30_frames`' 30 seeks with in-memory sampling (§7.2).
8. Build the overlay renderer over stored trajectory (§6).
9. Resolve **D1** and **D2** (§10) before any speed reaches the UI.