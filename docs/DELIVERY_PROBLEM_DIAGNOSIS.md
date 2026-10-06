# Delivery Segmentation Diagnosis — Root Cause Analysis & Validation Architecture

**Date:** 2026-10-04  
**Project:** Cricket Unified (CricketShot-Classification + CricketTracker)  
**Status:** DIAGNOSIS ONLY — Awaiting user sign-off before implementation.  
**Target:** `docs/DELIVERY_PROBLEM_DIAGNOSIS.md`  

---

## 1. Executive Summary

During Phase 3 integration testing, the single-pass architecture succeeded in executing exactly one video decode and single YOLO11+Kalman tracking pass across the uploaded video. However, rigorous forensic inspection of the generated delivery clips (`backend/data/analyses/phase3_real/clips/delivery_*.mp4`) revealed a **critical architectural failure in delivery segmentation**:

> **Critical Defect:**  
> Generated delivery clips frequently contain **multiple batting events** and **synthetic black frame gaps** merged into the same delivery clip.  
> In the Phase 3 run against `full_fixture.mp4`, **21 out of 21 generated delivery clips (100%) violated the core invariant: ONE DELIVERY CLIP MUST CONTAIN EXACTLY ONE BATTING EVENT.**

### Headline Forensic Findings
1. **The Example Clip (`delivery_019.mp4`):**  
   - Contains **30 black frames** (frames 1–30, source 2211–2240)
   - Followed by **Event 1** (frames 31–55, source 2241–2265): the entire batting event of **Delivery 18** (Fixture Clip 33)
   - Followed by **45 black frames** (frames 56–100, source 2266–2310): synthetic gap
   - Followed by **Event 2** (frames 101–125, source 2311–2335): the batting event of **Delivery 19** (Fixture Clip 34)
   - Followed by **44 black frames** (frames 126–169, source 2336–2379): synthetic gap
2. **Every Single Clip in Phase 3 is Polluted:**  
   Every one of the 21 clips generated in Phase 3 contains multiple distinct cricket content regions separated by 45-frame black gaps.
3. **The Black Frames are Not Codec Glitches:**  
   Black frames originate **100% from the test fixture container** (`full_fixture.mp4`), where 45 synthetic black frames were inserted between 25-frame clips. They enter the pipeline through normal video decode into the ring buffer, and are written into clips because the clip extraction windows are blindly oversized.
4. **Why Multiple Events Merge:**  
   The primary failure is that clip extraction applies a static, unconstrained window (`PRE_ROLL = 4.0s = 100 frames`, `POST_ROLL = 2.0s = 50 frames` at 25 fps, total ~170 frames) around ball tracking detections. In a video where deliveries or camera cuts occur closer together than 170 frames (in the fixture, every 70 frames), an unconstrained 170-frame window spans across multiple events.
5. **Why Inter-Clip Clamping Failed in Subsequent Attempts:**  
   In later diagnostic runs (`diag_full`), sequential clamping against the previous delivery's last frame was attempted. However, **20 of the 41 embedded fixture clips had no ball detected by YOLO**. When an intervening delivery was missed by the ball tracker, there was no delivery record to clamp against; the subsequent delivery's pre-roll reached backward across the black gap and swallowed the untracked batting event anyway (e.g. `delivery_003.mp4` swallowed untracked Fixture Clip 03).

---

## 2. Exact Current Segmentation Algorithm

The delivery segmentation and extraction pipeline consists of three interconnected layers:
1. **Ball Tracking Filter (`BallTracker` in `tracker_modules/tracker.py`)**
2. **Event Boundary Segmenter (`EventSegmenter` in `event_segmenter.py` and `tracking_service.py`)**
3. **Clip Window Registry (`ClipWriterRegistry` in `clip_writer.py`)**

### 2.1 Complete Flow Pipeline
```
SOURCE VIDEO (full_fixture.mp4, 2870 frames @ 25 fps)
    ↓
VIDEO DECODE (cv2.VideoCapture, exactly 1 decode pass)
    ↓ frame_number (1-based authoritative index)
RING BUFFER (FrameRingBuffer, JPEG q92, capacity 180 frames)
    ↓
YOLO11 DETECTOR (BallDetector, conf 0.30, 640x360 inference)
    ↓ detection: (cx, cy) or None
KALMAN TRACKER (BallTracker: 4-state [x, y, vx, vy], GATE_RADIUS=120px)
    ↓ track: TrackResult(pos, source, gated_out)
EVENT SEGMENTER (EventSegmenter.observe(FrameEvidence))
    ↓ SegmentDecision(kind, opened, closed, boundary)
DELIVERY FINALISER (_finalise in tracking_service.py)
    ↓ DeliveryRecord (detection_start, detection_end, ball_lost, trajectory)
CLIP WINDOW COMPUTATION:
    desired_start = max(1, detection_start - pre_roll_frames)   [pre_roll = 4.0s = 100f]
    desired_end   = ball_lost_frame + post_roll_frames          [post_roll = 2.0s = 50f]
    ↓
CLIP WRITER REGISTRY (ClipWriterRegistry.open_for)
    ↓ drain from FrameRingBuffer, stream forward via tick(frame_no, frame)
GENERATED DELIVERY CLIP (delivery_XXX.mp4)
```

### 2.2 Mathematical Specifications of the Current Bounds
- `fps` = 25.0
- `pre_roll_frames` = $\text{round}(4.0 \times 25) = 100$ frames
- `post_roll_frames` = $\text{round}(2.0 \times 25) = 50$ frames
- Desired clip span for a delivery with $N$ detected frames:
  $$\text{Span} = 100 + N + \text{coast} + 50 \approx 170 \text{ frames (6.8 seconds)}$$
- Periodicity of deliveries in `full_fixture.mp4`:
  $$\text{Period} = 25 \text{ (cricket)} + 45 \text{ (black gap)} = 70 \text{ frames (2.8 seconds)}$$
- **Mathematical Ratio:**
  $$\frac{\text{Clip Window}}{\text{Delivery Period}} = \frac{170}{70} = 2.43$$
  **An unconstrained 170-frame window mathematically guarantees that every clip will contain between 2 and 3 delivery events.**

---

## 3. Full State Transition Analysis

The tracking and segmentation pipeline operates as a composite state machine:

```
[IDLE / SEARCHING]
       │
       │ YOLO detection confirmed
       ▼
 [DELIVERY START] ─── (first confirmed detection frame)
       │
       │ Subsequent frames with detection
       ▼
   [TRACKING] ◀────────────────────────┐
       │                               │
       │ No detection                  │ Detection within GATE_RADIUS
       ▼                               │ and kinematically reachable
   [COASTING] ─────────────────────────┘
  (Kalman Predicts)
       │
       ├────────────────────────────────────────┐
       │ coast > max_coast (20 frames / 0.8s)   │ New detection arrives:
       │ or missed_frames > MAX_MISSED (8f)     │ Kinematics UNREACHABLE / REVERSED
       ▼                                        ▼
 [DELIVERY END]                           [DELIVERY END & SPLIT]
 (ball_lost_frame marked)                 (close prev, open new)
       │                                        │
       ▼                                        ▼
 [CLIP WINDOW EXTENSION]                  [CLIP WINDOW BOUNDED]
 (open clip writer:                       (post-roll must truncate
  pre-roll 100f + post-roll 50f)           before new event start)
```

### State Definitions & Trigger Matrix

| State | Entry Condition | Exit Condition | Next State | Failure Mode |
|---|---|---|---|---|
| **IDLE** | System startup or previous event finalized | Confirmed YOLO detection | **TRACKING** | False positive starts ghost event |
| **TRACKING** | Confirmed YOLO detection in gate | No detection on current frame | **COASTING** | Detection loss on occlusion |
| **COASTING** | Detection is `None` while filter initialized | Confirmed detection within reachable radius | **TRACKING** (Reacquired) | Extrapolation continues past reality |
| **COASTING** | Coast gap > `max_coast_frames` (20f) | Gap timeout | **DELIVERY_END** | Event stays open too long |
| **COASTING** | Detection outside reachable radius / reversed | Kinematic discontinuity | **DELIVERY_END & SPLIT** | If kinematics fail to detect jump, events merge |
| **DELIVERY_END** | Ball confirmed lost or gap exceeded | Detections evaluated for `MIN_DELIVERY_FRAMES` | **FINALIZED** or **REJECTED** | Debounce discarding valid delivery |
| **CLIP_WRITING** | Delivery finalized | Frames drained from ring + streamed | **CLIP_CLOSED** | Clip window reaches into neighboring events |

---

## 4. Delivery Start Logic

### 4.1 What event starts a delivery?
In the current implementation:
1. `EventSegmenter._open(ev, px, py)` is invoked on the **first confirmed YOLO ball detection** (`track.source == "detected"`).
2. It is **NOT** ball release.
3. It is **NOT** Kalman-confirmed history (it opens on the very first detection, frame $N$).
4. It does **NOT** observe bowler run-up or bowler arm action.

### 4.2 Can a delivery start too early?
**YES.**
- If YOLO fires a false positive detection on a player's shoe, white sightscreen, or umpire hat prior to the ball being bowled, an event starts immediately.
- If pre-roll (100 frames / 4.0s) is subtracted from `detection_start_frame`, the clip window reaches 100 frames backward regardless of whether those 100 frames contain a bowler running up, a completely different delivery, or synthetic black filler.

### 4.3 Can a delivery remain active after the batting event has ended?
**YES.**
- When the ball is struck or fielded, detection drops.
- Kalman extrapolation predicts forward for up to 8 frames (`MAX_MISSED`).
- In `EventSegmenter`, the event remains open in `COASTING` state for up to `max_coast_seconds = 0.8s` (20 frames).
- If a fielder throws the ball or a new ball is detected within the reachable kinematic envelope, `reseed()` re-attaches the track and keeps the delivery alive.

---

## 5. Delivery End Logic

### 5.1 How is delivery end determined?
Delivery end is triggered by ball loss:
1. **In `tracker_modules/tracker.py`:**  
   `missed_frames` increments each frame without detection. When `missed_frames > MAX_MISSED (8)`, `tracker.is_lost` becomes `True`.
2. **In `event_segmenter.py`:**  
   When no detection occurs for `gap > max_coast_frames (20 frames)`, `EventSegmenter._close()` is called with `reason = BoundaryReason.COAST_CEILING`.
3. **Four Recorded Frame Numbers:**
   - `detection_start_frame`: First confirmed detection frame.
   - `detection_end_frame`: Last confirmed detection frame.
   - `ball_lost_frame`: $\max(\text{end\_frame}, \min(\text{closed\_at} - 1, \text{end\_frame} + \text{max\_coast}))$.
   - `release_frame`: Heuristic displacement point within early detections.

### 5.2 The Danger: Does Prediction Keep a Delivery Alive?
- In original CricketTracker, if a detection occurred outside `GATE_RADIUS`, `missed_frames` was **not incremented**. That allowed a stream of foreign detections from a second ball to keep the first delivery alive indefinitely.
- In `CricketUnified`, this was patched in `BallTracker.update()` so gated-out detections increment `missed_frames`.
- However, `EventSegmenter` still allows up to 20 frames (0.8s) of coasting before closing. If a second event begins within that 20-frame window, it is evaluated by kinematics. If the kinematic check fails (e.g. low reference speed), it can be misclassified as a continuation.

---

## 6. Ball Loss / Reacquisition Behavior

### 6.1 State Machine: DETECTED → LOST → PREDICTING → REACQUIRED
```
Frame 0: Confirmed detection at (x0, y0) -> DETECTED
Frame 1-7: No detection -> PREDICTING (Kalman state extrapolated)
Frame 8: No detection -> Filter declares ball LOST (missed_frames > 8)
Frame 9-20: No detection -> Segmenter COASTING (waiting for max_coast_frames=20)
Frame 21: No detection -> Segmenter CLOSES event (COAST_CEILING)
```

### 6.2 Condition Deciding "Same Delivery" vs "New Delivery"
When a detection appears after a gap of $G$ frames:
1. If $G > \text{hard\_split\_frames}$ (50 frames / 2.0s):
   $\rightarrow$ **Unconditionally NEW DELIVERY** (`BoundaryReason.LONG_SILENCE`).
2. If $20 < G \le 50$ (Ambiguous Band):
   $\rightarrow$ Kinematic reachability determines the split:
   $$\text{reach\_radius} = v_{\text{ref}} \cdot G + \text{base\_slack} + \text{rate\_slack} \cdot G$$
   If $\text{displacement} > \text{reach\_radius}$ $\rightarrow$ **NEW DELIVERY** (`BoundaryReason.UNREACHABLE`).
   If direction reverses $\ge 120^\circ$ $\rightarrow$ **NEW DELIVERY** (`BoundaryReason.DIRECTION_REVERSAL`).
   If speed ratio $> 4.0$ or $< 0.15$ $\rightarrow$ **NEW DELIVERY** (`BoundaryReason.SPEED_JUMP`).
   Otherwise $\rightarrow$ Continues same event (`ambiguous_band_continuous`).
3. If $G \le 20$ (Short Band):
   $\rightarrow$ Assumed tracking interruption unless two kinematic tests fail simultaneously.

---

## 7. Cooldown Logic

### 7.1 Upstream CricketTracker Cooldown Defect
In `CricketTracker-main/backend/model.py:246-248`:
```python
frames_since = frame_number - last_delivery_frame
if frames_since < 60:
    return
```
- Upstream checked cooldown **after** analytics were computed, and simply returned without appending to `deliveries`.
- **Consequence:** Any legitimate delivery ending within 60 frames (2.4s at 25 fps) of a previous delivery was **silently deleted from the output**.
- Furthermore, `last_delivery_frame` was initialized to `0`, causing any delivery in the first 60 frames of a match to be deleted.

### 7.2 Cooldown in Unified Architecture
- In `CricketUnified`, `MIN_FRAMES_BETWEEN_DELIVERIES` was marked deprecated in `constants.py` because fixed frame thresholds are rate-dependent and delete real deliveries.
- However, the concept of a minimum time between deliveries was replaced by event segmentation.
- **Critical Risk:** Without an explicit debounce, if a single delivery has an internal tracking dropout that gets split by the segmenter, it produces two back-to-back delivery records for the same ball.

---

## 8. Ring Buffer Behavior

### 8.1 Implementation Details (`FrameRingBuffer` in `ring_buffer.py`)
- Capacity: 180 frames.
- Hard byte ceiling: 512 MiB (`RING_BUFFER_MAX_BYTES`).
- Storage: Frames encoded as JPEG (quality 92, BGR).
- Memory footprint: ~3.6 MiB for 180 frames at 640×360.
- Eviction policy: Oldest frame evicted when frame count reaches capacity or byte limit is approached.

### 8.2 Does the Ring Buffer Cause Foreign or Black Frames?
- **Foreign frames from buffer reuse? NO.**  
  Frames are stored in a dictionary keyed strictly by 1-based source frame number `_frames[frame_number] = enc`. `get(frame_no)` retrieves the exact frame or returns `None`.
- **Frame corruption? NO.**  
  Decoded JPEG images are identical to the source except for minor standard JPEG quantization noise.
- **Where do black frames enter the buffer?**  
  The ring buffer faithfully ingests whatever `cap.read()` yields. When the source video contains black frames (source frames 26–70, 96–140, etc.), those black frames are encoded and stored in the ring buffer.
- When `ClipWriterRegistry.open_for()` requests `desired_start = detection_start - 100`, the ring buffer drains those stored black frames into the clip!

---

## 9. Pre-Roll / Post-Roll Behavior

This is the **primary direct mechanism** of clip pollution.

### 9.1 Window Definition
$$\text{clip.start\_frame} = \max(1, \text{detection\_start\_frame} - \text{PRE\_ROLL\_FRAMES})$$
$$\text{clip.end\_frame} = \text{ball\_lost\_frame} + \text{POST\_ROLL\_FRAMES}$$

Where `PRE_ROLL_FRAMES` = 100 (4.0s @ 25 fps) and `POST_ROLL_FRAMES` = 50 (2.0s @ 25 fps).

### 9.2 The Structural Conflict
- In continuous live broadcast footage, an over consists of 6 deliveries separated by 30–60 seconds of dead time (bowler walking back). In that environment, a 4.0s pre-roll and 2.0s post-roll is safe.
- **BUT in any edited match, highlights package, test fixture, or fast-paced game:**
  - The interval between events can be 1 to 3 seconds.
  - In `full_fixture.mp4`, the interval between deliveries is exactly **45 frames (1.8 seconds)**.
- When the interval between events is 1.8 seconds:
  - Event A ends at frame $T$.
  - Gap runs from $T+1$ to $T+45$ (1.8s).
  - Event B starts at $T+46$.
  - Event A's post-roll extends 50 frames: from $T+1$ to $T+50$.
    $\rightarrow$ **Event A's clip swallows all 45 black gap frames AND the first 5 frames of Event B!**
  - Event B's pre-roll reaches 100 frames backward: from $(T+46) - 100 = T - 54$.
    $\rightarrow$ **Event B's clip swallows Event A's batting stroke, all 45 black frames, and its own batting event!**

### 9.3 Separation of Defects
- **Defect A: Bad Delivery Segmentation**  
  Did YOLO+Kalman merge two events into one tracking record?  
  *Finding:* In the Phase 3 run, YOLO+Kalman and `event_segmenter` created **21 separate delivery records** for the 21 detected events. Delivery 18 and Delivery 19 had distinct delivery IDs, distinct trajectory points, and distinct detection spans.
- **Defect B: Bad Clip Extraction Window**  
  Did the clip extractor combine two separate deliveries into one MP4?  
  *Finding:* **YES.** The clip extractor blindly expanded the pre-roll and post-roll windows, causing the clip files to cross event boundaries.
- **Defect C: Untracked Event Absorption**  
  What happened when an intervening delivery was missed by the detector?  
  *Finding:* When Fixture Clip 03 was missed by YOLO, the gap between Delivery 02 and Delivery 03 was 116 frames. Delivery 03's 100-frame pre-roll reached back across the black gap and swallowed the entire untracked Fixture Clip 03!

---

## 10. Clip Writer Behavior

### 10.1 Streaming vs Seeking
- `ClipWriter` streams sequentially from the ring buffer and the live decode loop (`tick()`).
- It never seeks using `cap.set(CAP_PROP_POS_FRAMES)`. This guarantees that frame numbers are strictly monotonic and frame dropping does not occur.

### 10.2 Truncation & Clamping Mechanics
In `ClipWriterRegistry`:
- `floor = self.next_free_frame()` ($\text{\_last\_written\_frame} + 1$).
- `effective_start = max(requested_start, floor)`.
- `close_before(frame_no)` cuts the active clip's window at `frame_no - 1` when a new delivery begins.

### 10.3 Why Clamping Failed in Phase 3
In the code that produced `phase3_real`:
- There was **NO** inter-clip clamping active!
- In `docs/phase3_report.md` §3.2, the author stated:
  > *"The clip total exceeds the source because each delivery's window overlaps its neighbours' — pre-roll reaches back before the detection and post-roll runs on after ball-loss. That is by design..."*
- Because overlapping was considered "by design", `ClipWriterRegistry` permitted overlapping clip windows.
- In subsequent diagnostic runs (`diag_full`), `floor` clamping was introduced. But as proven in Section 14, **clamping to the previous delivery fails when an untracked batting event sits between them**.

---

## 11. Frame Mapping Analysis

### 11.1 The Invariant
For every clip, the unified architecture defines:
$$\text{clip.frame\_offset} = \text{clip.start\_frame} - 1$$
$$\text{clip\_frame\_1based}(\text{original\_frame}) = \text{original\_frame} - \text{clip.frame\_offset}$$
$$\text{original\_frame}(\text{clip\_frame\_1based}) = \text{clip\_frame\_1based} + \text{clip.frame\_offset}$$

### 11.2 Verification of Mathematical Correctness
- `frames_written == clip.frame_count` was verified across all 21 clips.
- There are no off-by-one errors in `assign_clip_frames`.
- Round-trip mapping `original_frame → clip_frame → original_frame` succeeds with 0 mismatches.

### 11.3 The Semantic Error in the Mapping
While the math is internally consistent, the **meaning of the frames is distorted**:
- In `delivery_019.mp4`, `clip.start_frame` is 2211, `frame_offset` is 2210.
- Delivery 19's ball detections occurred at original frames 2311–2320.
- Therefore, Delivery 19's trajectory appears at clip frames:
  $$2311 - 2210 = 101 \quad \text{to} \quad 2320 - 2210 = 110$$
- Meanwhile, clip frames 31–55 (original frames 2241–2265) show the batter playing **Delivery 18's shot**, but with **no trajectory overlaid**!
- Then clip frames 56–100 show pitch-black frames.
- Then at clip frame 101, a completely different bowler suddenly bowls a different delivery, and the trajectory overlay appears!
- This makes downstream shot classification impossible: when the shot model uniformly samples 30 frames across the 169-frame clip, 68% of the sampled frames are black, and the remaining frames sample two different batsmen!

---

## 12. Black-Frame Origin Analysis

A forensic trace of black frames across the entire pipeline:

```
Step 1: Test Fixture Assembly (tools/build_test_fixture.py)
   41 delivery clips (25 frames each) concatenated with 45 frames of np.zeros((360, 640, 3), dtype=np.uint8)
   Total fixture: 41 * 25 + 41 * 45 = 1025 + 1845 = 2870 frames.
   Source frames with mean pixel brightness < 2.0: exactly 1845 frames (64.3% of the entire video).
   ↓
Step 2: Video Decode (cv2.VideoCapture)
   cap.read() decodes the exact BGR pixels. Frame 26 through 70 have pixel values (0, 0, 0).
   ↓
Step 3: Ring Buffer (FrameRingBuffer)
   push(frame_number, frame) encodes (0,0,0) as JPEG bytes (~4 KB per frame) into ring buffer.
   ↓
Step 4: Delivery Tracking (tracking_service.py)
   YOLO runs on (0,0,0) -> 0 detections.
   Kalman filter predict-coasts for 8 frames, then is_lost = True.
   EventSegmenter coasts for 20 frames, then COAST_CEILING close.
   ↓
Step 5: Clip Extraction Window Selection
   Delivery 19 detection starts at frame 2311.
   Window start calculated as 2311 - 100 = 2211.
   Frames 2211-2240 (30 frames) in ring buffer are black gap frames -> COPIED TO CLIP.
   Frames 2241-2265 (25 frames) are Delivery 18 real frames -> COPIED TO CLIP.
   Frames 2266-2310 (45 frames) in ring buffer are black gap frames -> COPIED TO CLIP.
   Frames 2311-2335 (25 frames) are Delivery 19 real frames -> COPIED TO CLIP.
   Frames 2336-2379 (44 frames) live streamed via tick() are black gap frames -> COPIED TO CLIP.
```

### Conclusion on Black Frames
- **Black frames are NOT generated by OpenCV or ClipWriter.**
- **Black frames are NOT codec initialization artifacts.**
- **Black frames are NOT ring buffer eviction padding.**
- **Black frames are real source video pixels from `full_fixture.mp4`**, pulled into the clip because the pre-roll and post-roll logic treats all source frames as valid cricket action and has no mechanism to halt at black gaps or scene boundaries.

---

## 13. Automatic Delivery Validator Design

To ensure that no clip with multiple batting events, black gaps, or no batting event ever passes into production or shot classification, we propose the **Multi-Signal Automatic Delivery Validator**.

### 13.1 Architectural Principles
1. **Never Trust the Segmenter Alone:** Re-examine the generated clip and its frame evidence independently.
2. **Arbitrary Video Compatibility:** Do not hard-code 41, 47, or 21 events. Do not rely on black-frame presence (must detect back-to-back deliveries with no black gap).
3. **Four Exhaustive Classifications:**
   - `SINGLE_EVENT`: Exactly one continuous batting event, properly framed within the clip.
   - `MULTIPLE_EVENTS`: More than one batting event detected (separated by black gaps, camera cuts, or activity lulls).
   - `NO_EVENT`: Zero confirmed ball detections or zero meaningful motion activity.
   - `AMBIGUOUS`: Single event candidate, but boundaries sit too close to clip edges, or activity profile indicates truncation.

### 13.2 Multi-Signal Inspection Engine
The validator combines five independent signals:

```
                             GENERATED CLIP
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
 [Luminance & Gap]         [Motion Energy]          [Shot Boundary / Cut]
  - Mean brightness         - Inter-frame diff       - HSV histogram diff
  - Black frame runs        - Temporal variance      - Edge change ratio
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                       [Activity Region Parser]
                        Identifies distinct
                        Action Windows [W1, W2..]
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
 [Ball Tracking Overlay]                            [Kinematic Continuity]
  - Trajectory frame span                            - Teleportation check
  - Gaps within trajectory                           - Speed & direction jump
         │                                                   │
         └─────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
                         [COMBINED CLASSIFIER]
                 SINGLE_EVENT | MULTIPLE_EVENTS |
                      NO_EVENT | AMBIGUOUS
```

#### Signal 1: Black / Near-Black Frame Detection
- $\text{Luminance}(t) = \frac{1}{W \times H} \sum_{x,y} Y(x, y, t)$.
- A frame is `BLACK` if $\text{Luminance} < 3.0$ (on 0–255 scale).
- Contiguous runs of $\ge 5$ black frames are registered as `BLACK_GAP(start, end)`.

#### Signal 2: Motion / Activity Energy
- Grayscale absolute frame difference:
  $$D(t) = \frac{1}{W \times H} \sum_{x,y} |I(x, y, t) - I(x, y, t-1)|$$
- A frame is `ACTIVE` if $D(t) > 1.8$ and not `BLACK`.
- Smooth $D(t)$ with a 5-frame moving average.
- Identifies **Activity Regions**: contiguous intervals where motion exceeds background noise.

#### Signal 3: Scene Cut / Shot Boundary Detection
- Broadcast cricket cameras cut between wide pitch view, batsman close-up, and bowler return.
- Detected via Chi-Square color histogram distance $\chi^2(H_t, H_{t-1}) > \theta_{\text{cut}}$.
- Any camera cut within a delivery clip is a strong indicator of multiple events or non-delivery footage.

#### Signal 4: Trajectory & Tracking Cluster Analysis
- Cross-references the delivery's trajectory points mapped into clip coordinates.
- If trajectory points exist only in frames 101–110, but an Activity Region exists at frames 31–55, the clip contains an **untracked foreign event**.
- If two distinct clusters of confirmed detections exist separated by $> 20$ frames, the clip contains **two tracked events**.

### 13.3 Diagnostic Output Schema
```json
{
  "delivery_id": 19,
  "classification": "MULTIPLE_EVENTS",
  "confidence": 0.98,
  "reason": "Two distinct activity/tracking regions separated by 45-frame black gap",
  "evidence": {
    "regions": [
      {
        "type": "BLACK_GAP",
        "clip_frames": [1, 30],
        "source_frames": [2211, 2240],
        "mean_luminance": 0.0
      },
      {
        "type": "ACTIVITY_EVENT_1",
        "clip_frames": [31, 55],
        "source_frames": [2241, 2265],
        "mean_motion": 9.8,
        "tracked_points": 0,
        "note": "Corresponds to Delivery 18 batting event"
      },
      {
        "type": "BLACK_GAP",
        "clip_frames": [56, 100],
        "source_frames": [2266, 2310],
        "mean_luminance": 0.0
      },
      {
        "type": "ACTIVITY_EVENT_2",
        "clip_frames": [101, 125],
        "source_frames": [2311, 2335],
        "mean_motion": 10.6,
        "tracked_points": 10,
        "note": "Corresponds to Delivery 19 batting event"
      },
      {
        "type": "BLACK_GAP",
        "clip_frames": [126, 169],
        "source_frames": [2336, 2379],
        "mean_luminance": 0.0
      }
    ]
  }
}
```

---

## 14. Evidence from Actual Generated Clips

### 14.1 Full Scan of All 21 Clips in `backend/data/analyses/phase3_real/clips/`
Every clip was decoded and analyzed frame-by-frame for content and black regions:

| Clip Name | Total Frames | Detected Regions Breakdown | Verdict |
|---|---|---|---|
| `delivery_001.mp4` | 72 | CONTENT[1-25] (25f), BLACK[26-70] (45f), CONTENT[71-72] (2f) | **MULTIPLE_EVENTS** |
| `delivery_002.mp4` | 146 | CONTENT[1-25] (25f), BLACK[26-70] (45f), CONTENT[71-95] (25f), BLACK[96-140] (45f), CONTENT[141-146] (6f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_003.mp4` | 169 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-169] (44f) | **MULTIPLE_EVENTS** |
| `delivery_004.mp4` | 167 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-167] (42f) | **MULTIPLE_EVENTS** |
| `delivery_005.mp4` | 169 | BLACK[1-28] (28f), CONTENT[29-53] (25f), BLACK[54-98] (45f), CONTENT[99-123] (25f), BLACK[124-168] (45f), CONTENT[169] (1f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_006.mp4` | 179 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f), CONTENT[171-179] (9f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_007.mp4` | 173 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f), CONTENT[171-173] (3f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_008.mp4` | 169 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-169] (44f) | **MULTIPLE_EVENTS** |
| `delivery_009.mp4` | 182 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f), CONTENT[171-182] (12f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_010.mp4` | 170 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f) | **MULTIPLE_EVENTS** |
| `delivery_011.mp4` | 182 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f), CONTENT[171-182] (12f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_012.mp4` | 177 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f), CONTENT[171-177] (7f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_013.mp4` | 178 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f), CONTENT[171-178] (8f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_014.mp4` | 172 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f), CONTENT[171-172] (2f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_015.mp4` | 183 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f), CONTENT[171-183] (13f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_016.mp4` | 179 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f), CONTENT[171-179] (9f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_017.mp4` | 167 | BLACK[1-28] (28f), CONTENT[29-53] (25f), BLACK[54-98] (45f), CONTENT[99-123] (25f), BLACK[124-167] (44f) | **MULTIPLE_EVENTS** |
| `delivery_018.mp4` | 177 | BLACK[1-29] (29f), CONTENT[30-54] (25f), BLACK[55-99] (45f), CONTENT[100-124] (25f), BLACK[125-169] (45f), CONTENT[170-177] (8f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_019.mp4` | 169 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-169] (44f) | **MULTIPLE_EVENTS** |
| `delivery_020.mp4` | 181 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-170] (45f), CONTENT[171-181] (11f) | **MULTIPLE_EVENTS** (3 events!) |
| `delivery_021.mp4` | 169 | BLACK[1-30] (30f), CONTENT[31-55] (25f), BLACK[56-100] (45f), CONTENT[101-125] (25f), BLACK[126-169] (44f) | **MULTIPLE_EVENTS** |

**Summary: 21 / 21 clips failed (100% failure rate).**

### 14.2 Evidence from `diag_full` (Why Naive Clamping Failed)
In `diag_full`, clipping was clamped to `last_written_frame + 1`. The results:
- `delivery_002.mp4`, `007.mp4`, `008.mp4`, `013.mp4`, `015.mp4`, `017.mp4`, `018.mp4`, `019.mp4` had only 1 content region.
- **BUT 11 clips STILL contained multiple events** (`001`, `003`, `006`, `009`, `010`, `011`, `012`, `014`, `016`, `020`, `021`).
- **Why?** Whenever an embedded delivery clip had no ball detected by YOLO, there was no delivery record created. The next delivery's pre-roll reached backward across the black gap and swallowed the untracked delivery!

---

## 15. Detailed Timeline for delivery_019.mp4

### 15.1 Physical Provenance of Frames
- **Source Video:** `backend/data/samples/full_fixture.mp4`
- **Clip File:** `backend/data/analyses/phase3_real/clips/delivery_019.mp4`
- **Container Properties:** 169 frames, 640×360, 25.0 fps, AVC1/H.264
- **Delivery Record in `tracking.json`:**
  - `delivery_id`: 19
  - `detection_start_frame`: 2311
  - `detection_end_frame`: 2320
  - `ball_lost_frame`: 2329
  - `clip.start_frame`: 2211
  - `clip.end_frame`: 2379
  - `clip.frame_offset`: 2210

### 15.2 Frame-by-Frame Timeline

```
Clip Frame   Source Frame   Luminance   Motion Energy   Tracking State      Physical Meaning
───────────────────────────────────────────────────────────────────────────────────────────────────────
1 - 30       2211 - 2240    0.0         0.0             IDLE                Synthetic Black Gap (after Fixture Clip 32)
31 - 55      2241 - 2265    116.9       9.8             IDLE (no points!)   EVENT 1: Delivery 18 (Fixture Clip 33)
                                                                            Batsman plays shot for Delivery 18!
56 - 100     2266 - 2310    0.0         2.8             IDLE                Synthetic Black Gap (after Fixture Clip 33)
101 - 110    2311 - 2320    113.2       10.6            TRACKING (detected) EVENT 2: Delivery 19 (Fixture Clip 34)
                                                                            Ball bowled, 10 tracked points!
111 - 119    2321 - 2329    113.5       11.2            COASTING (predict)  Ball passes bat, Kalman extrapolates
120 - 125    2330 - 2335    113.0       9.4             LOST / POST-ROLL    Batsman completes stroke for Delivery 19
126 - 169    2336 - 2379    0.0         2.4             POST-ROLL           Synthetic Black Gap (after Fixture Clip 34)
```

### 15.3 Visual Timeline Diagram
```
delivery_019.mp4:
┌─────────────────┬─────────────────────┬───────────────────────────┬──────────────────────┬──────────────────────────┐
│  30 Black Frames│ 25 Content Frames   │ 45 Black Frames           │ 25 Content Frames    │ 44 Black Frames          │
│  (source gap)   │ (DELIVERY 18 EVENT) │ (synthetic gap)           │ (DELIVERY 19 EVENT)  │ (synthetic gap)          │
│  Clip f: 1..30  │ Clip f: 31..55      │ Clip f: 56..100           │ Clip f: 101..125     │ Clip f: 126..169         │
└─────────────────┴─────────────────────┴───────────────────────────┴──────────────────────┴──────────────────────────┘
                  ▲                                                 ▲
                  │                                                 │
            Wrong batting event                               Target delivery
            (no overlay trail)                                (overlay trail drawn here)
```

---

## 16. Root Causes (Ranked by Confidence)

### 1. CONFIRMED: Blind Static Pre-Roll / Post-Roll Windowing
- **Confidence:** 100% (proven mathematically and verified on disk).
- **Mechanism:** Hardcoding `PRE_ROLL = 4.0s (100f)` and `POST_ROLL = 2.0s (50f)` without checking whether the expanded window crosses black gaps, camera cuts, or neighboring event boundaries.
- **Impact:** Forces clip length to ~170 frames. In any footage where deliveries or camera cuts occur within 170 frames, foreign frames and black gaps are unconditionally merged into the clip.

### 2. CONFIRMED: Black Frames Originate in the Source Container and are Ingested via Decoder
- **Confidence:** 100% (proven by byte comparison and timeline mapping).
- **Mechanism:** The test fixture contains 1,845 synthetic black frames. The ring buffer stores them, and the clip writer writes them because the pre/post-roll windows request them.
- **Impact:** Clips contain 45-frame black pauses in the middle and edges of delivery videos.

### 3. CONFIRMED: Inter-Clip Clamping Fails on Missed/Untracked Events
- **Confidence:** 100% (proven by `diag_full` scan).
- **Mechanism:** Clamping `clip_start` to `previous_delivery_end + 1` only works if every delivery is detected. 20 of 41 fixture clips had no ball detected. Pre-roll from Delivery $N$ reached across the black gap and swallowed the untracked Delivery $N-1$.
- **Impact:** Naive boundary clamping does not prevent multiple events in clips.

### 4. LIKELY: Incomplete Implementation & NameError in `tracking_service.py`
- **Confidence:** 95% (verified by unit test execution).
- **Mechanism:** Line 431 calls `_tracker_state(track, tracker)` which is undefined, causing `NameError` on execution. Line 714 references undefined `detected`.
- **Impact:** The streaming event segmenter cannot run to completion in the current backend state.

### 5. LIKELY: YOLO11 Ball Detector Sensitivity on Fast/Occluded Deliveries
- **Confidence:** 90% (proven by 20/41 missed clips in fixture).
- **Mechanism:** YOLO11 with `conf = 0.30` and size filtering (4–80 px) misses balls on 20 clips where the ball is motion-blurred, occluded, or too small.
- **Impact:** Causes untracked deliveries to linger as "stealth content" between tracked deliveries.

### 6. POSSIBLE: Cooldown Logic Suppressing Legitimate Close Deliveries
- **Confidence:** 70%.
- **Mechanism:** Upstream 60-frame debounce deleted deliveries that completed within 2.4s of a previous delivery.
- **Impact:** Silently dropped valid balls in rapid succession.

### 7. UNSUPPORTED: Codec / VideoWriter Encoding Failure
- **Confidence:** 0% (Disproven).
- **Finding:** OpenCV VideoWriter using AVC1/H.264 writes every requested frame accurately (`frames_written == frame_count`). The black frames are real decoded image pixels, not encoding corruption.

### 8. UNSUPPORTED: Frame Ring Buffer Corruption or Dropped Indexing
- **Confidence:** 0% (Disproven).
- **Finding:** `FrameRingBuffer` unit tests pass completely. Frames are retrieved by exact integer key.

---

## 17. What is NOT the Root Cause

1. **NOT the Video Codec:** AVC1 and MP4V both encode the requested frames faithfully.
2. **NOT OpenCV VideoWriter:** No frames are dropped by the writer backend.
3. **NOT Frame Mapping Math:** The affine formula `clip_frame = original_frame - frame_offset` is mathematically exact.
4. **NOT the Shot Classifier:** The shot classifier runs downstream on already extracted clips; it has no feedback into segmentation.
5. **NOT Hardware Throttling or Host Contention:** Run-to-run variance affects execution speed, not frame boundaries.

---

## 18. Recommended Fix Strategy

### Phase A: Content-Aware & Activity-Bounded Pre/Post-Roll (The Direct Fix)
Instead of blind static addition (`start = det - 100`, `end = lost + 50`), clip boundaries must be **dynamically bounded by scene content during the single tracking pass**:

1. **Black Gap Barrier:**
   - Maintain a running count of black frames in the decode loop.
   - Pre-roll must **never extend backward across a black gap** ($\ge 5$ consecutive black frames).
   - Post-roll must **never extend forward into a black gap**.
   - If a black gap is encountered during pre-roll search, pre-roll stops immediately at the gap boundary.
2. **Activity / Motion Boundary Barrier:**
   - Compute inter-frame motion energy $D(t)$ during the single decode pass (0.2 ms cost).
   - Pre-roll extends backward until motion falls to background level (bowler at mark) or hits a scene cut, clamped to a maximum of 4.0s.
   - Post-roll extends forward until batsman stroke completion / ball dead, clamped to a maximum of 2.0s.
3. **Hard Clamping Against Any Intervening Content:**
   - Any camera cut or prolonged silence terminates the clip window.

### Phase B: Repair State Machine Bugs in `tracking_service.py`
1. Define `_tracker_state(track, tracker)` properly:
   ```python
   def _tracker_state(track: TrackResult, tracker: BallTracker) -> str:
       if not tracker.kf.initialized:
           return "untracked"
       if tracker.is_lost:
           return "lost"
       if track.source == "predicted":
           return "coasting"
       return "tracking"
   ```
2. Fix `len(detected)` reference in line 714 to `len(segment.confirmed_frames)`.

### Phase C: Implement the Standalone Automatic Validator
Integrate the 5-signal validator designed in Section 13 into the test suite and pipeline report. Every analysis run must emit `validation` in `tracking.json` and flag any clip that is not `SINGLE_EVENT`.

---

## 19. Tests That Must Be Added Before Implementation

1. **`test_clip_never_crosses_black_gap`:**  
   Feeds a video with [Content A (25f)] -> [Black Gap (45f)] -> [Content B (25f)]. Asserts that Clip A contains 0 black frames and Clip B contains 0 frames from Content A and 0 black frames.
2. **`test_clip_bounded_when_intervening_delivery_untracked`:**  
   Feeds a video with [Delivery 1 (tracked)] -> [Delivery 2 (untracked)] -> [Delivery 3 (tracked)]. Asserts that Delivery 3's clip does not contain Delivery 2.
3. **`test_clip_isolation_with_no_black_gap`:**  
   Feeds a continuous video where Delivery A and Delivery B abut with no black frames. Asserts that Clip A ends before Delivery B begins, and Clip B contains no frame from Delivery A.
4. **`test_validator_classifications`:**  
   Runs the automatic validator on known synthetic fixtures and asserts correct labeling (`SINGLE_EVENT`, `MULTIPLE_EVENTS`, `NO_EVENT`, `AMBIGUOUS`).

---

## 20. Risks and Regressions to Watch For

1. **Over-Truncation Risk:**  
   Truncating pre-roll too aggressively could cut off the bowler's release or run-up in real broadcast video.  
   *Mitigation:* Only black gaps ($\ge 5$ frames), scene cuts, and confirmed prior deliveries should hard-truncate; motion energy should gently bound pre-roll.
2. **Shot Classifier Starvation:**  
   The shot classifier samples 30 frames via `np.linspace(0, N-1, 30)`. If a clip is truncated to fewer than 30 frames, `ClipFrameMap` must handle small $N$ gracefully without crashing.
3. **Performance Overhead:**  
   Scene cut and motion energy detection must be lightweight (sub-millisecond) to preserve the ~35 fps processing rate of the single pass.

---

## 21. Summary & Diagnostic Verdict

| Issue | Finding |
|---|---|
| **Root Cause** | Static, unconstrained pre-roll (100f) and post-roll (50f) expanding blindly across temporal boundaries, swallowing neighboring events and synthetic black filler |
| **Secondary Cause** | Inability of sequential delivery clamping to handle intervening untracked cricket deliveries (20 of 41 fixture clips missed by YOLO) |
| **Black Frames Origin** | 100% decoded from the source fixture container (`full_fixture.mp4`), ingested via ring buffer and written due to unconstrained window sizing |
| **Why Multiple Events Merged** | A 170-frame clip window superimposed on a 70-frame delivery period forced 2–3 delivery events into every single clip |
| **Recommended Fix** | Content-aware, black-gap-bounded and scene-cut-bounded dynamic pre/post-roll windowing + Automatic Multi-Signal Validator QA |
| **Files That Must Change** | `backend/app/services/clip_writer.py`, `backend/app/services/tracking_service.py`, `backend/app/services/clip_validator.py`, `backend/app/core/constants.py` |
| **Tests Required** | Black gap barrier tests, untracked event isolation tests, seamless continuous delivery isolation tests, validator unit tests |
