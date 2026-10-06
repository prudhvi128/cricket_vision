# Delivery Segmentation Fix Design Addendum

**Date:** 2026-10-04  
**Project:** Cricket Unified (CricketShot-Classification + CricketTracker)  
**Status:** DESIGN ADDENDUM — Awaiting user sign-off before implementation.  
**File:** `docs/DELIVERY_SEGMENTATION_FIX_DESIGN.md`  

---

## Executive Overview

This design addendum reviews and solidifies the segmentation fix strategy to satisfy the invariant:
$$\textbf{ONE DELIVERY CLIP = EXACTLY ONE BATTING EVENT}$$
**A clip is strictly invalid if it contains even one frame belonging to another batting event.**

This document specifically resolves the hard edge cases identified in diagnosis:
1. **The Untracked Intervening Delivery:**  
   $$\text{Tracked Delivery A} \longrightarrow \text{Untracked Delivery B} \longrightarrow \text{Tracked Delivery C}$$
   Preventing Delivery C's clip from swallowing Delivery B.
2. **The Seamless Consecutive Delivery:**  
   $$\text{Delivery A} \longrightarrow \text{Delivery B}$$
   Separating two deliveries with **no black frames** and **no camera cuts**.
3. **Arbitrary Video Robustness:**  
   Operating without assuming black gaps, without relying naively on general motion, and strictly preserving:
   - Exactly **one** source decode pass
   - Exactly **one** YOLO11 pass
   - Exactly **one** Kalman tracking pass
   - **No** per-clip YOLO re-detection pass
   - Shot classifier remains a pure downstream consumer of in-memory frames

---

## 1. What Defines the TRUE Delivery Boundaries?

A cricket delivery is fundamentally a physical biomechanical cycle, not a static timestamp offset. In video footage, a single delivery consists of three contiguous chronological phases:

```
[PHASE 1: RUN-UP & DELIVERY STRIDE] ──▶ [PHASE 2: BALL IN FLIGHT] ──▶ [PHASE 3: STROKE & FOLLOW-THROUGH]
  (Bowler approaches, gathers,          (Ball released, pitches,       (Bat impact or leave, stroke
   enters crease, releases ball)         travels toward stumps)         execution, play conclusion)
```

### 1.1 The Physical Boundary Anchors
1. **True Delivery Start ($T_{\text{start}}$):**
   - The moment the bowler begins the final delivery approach / gathers into the delivery stride, or when the video cuts to the bowler's run-up / batsman in stance.
   - **Nominal bounds:** 1.5s to 3.0s before ball release.
   - **Hard cutoff:** Must not extend earlier than the dead-ball reset / stance setup of the current delivery, and must **never** reach into any prior ball's follow-through or prior camera cut.
2. **True Delivery End ($T_{\text{end}}$):**
   - The completion of the batsman's stroke, follow-through, and immediate ball path (or wicketkeeper take / stumps struck), before the ball goes dead and players reset.
   - **Nominal bounds:** 1.0s to 2.0s after ball loss / bat impact.
   - **Hard cutoff:** Must not extend later than the completion of the stroke reaction, and must **never** reach into the next delivery's run-up or subsequent camera cut.

---

## 2. How to Prevent an Untracked Delivery from Being Swallowed

### 2.1 The Hard Scenario
$$\text{TRACKED DELIVERY A} \longrightarrow \text{UNTRACKED DELIVERY B} \longrightarrow \text{TRACKED DELIVERY C}$$

Why Delivery B may be untracked:
- YOLO11 ball detector fired fewer than `MIN_DELIVERY_FRAMES (8)` detections (e.g., our diagnostic run proved Fixture Clip 03 had 6 detections, Clip 07 had 5 detections, Clip 08 had 3 detections—all real deliveries, but dropped by the threshold).
- Or YOLO11 had 0 ball detections due to severe motion blur or occlusion behind the bowler's body.

If Delivery C's pre-roll blindly extends backward by 4.0s (100 frames), it crosses the boundary between B and C and swallows Delivery B into `delivery_003.mp4`.

### 2.2 The Solution: The Dual-Sentinel Event Horizon
During the single tracking pass, every decoded frame generates data that establishes **Event Horizons**. An Event Horizon is an impermeable temporal boundary that pre-roll and post-roll can **never cross**, even if no confirmed delivery was finalized.

```
                    DELIVERY TIMELINE
─────────────────────────────────────────────────────────────
[Delivery A] ──▶ [RESET VALLEY 1] ──▶ [Event B] ──▶ [RESET VALLEY 2] ──▶ [Delivery C]
 (tracked)         Activity min        (UNTRACKED)    Activity min        (tracked)
                        │              YOLO 3-6 dets       │
                        │              Pitch motion        │
                        ▼                   │              ▼
                 HORIZON 1                  │        HORIZON 2
           (Clip A cannot cross)            │   (Clip C cannot cross)
                                            ▼
                                  EVENT HORIZON BARRIER
                     (Delivery C pre-roll STOPS at Horizon 2;
                      it CANNOT reach Event B)
```

There are two sentinels evaluated per frame in the single pass:

#### Sentinel 1: The Candidate / Sub-Threshold Ball Sentinel
- As demonstrated in our diagnosis, "untracked" deliveries almost always contain **1 to 7 raw YOLO detections**.
- Even if an event fails the quality bar for bowling analytics (`confirmed_count < 8`), the raw detections are **authoritative evidence of a ball on screen**.
- Any rejected event or sub-threshold cluster is tagged as a `REJECTED_EVENT_CANDIDATE`.
- **Invariant:** A delivery clip's pre-roll or post-roll search **terminates immediately upon encountering any frame belonging to a `REJECTED_EVENT_CANDIDATE`**.

#### Sentinel 2: The Pitch-Corridor Inactivity Valley Sentinel
- Even when ball detections are completely zero (0 detections in Delivery B):
  Between Delivery B and Delivery C, there is always an **Inactivity Valley** (the dead-ball reset):
  - Delivery B's stroke ends $\rightarrow$ players stop intense motion $\rightarrow$ bowler walks back $\rightarrow$ batsman taps bat.
  - Motion energy within the pitch quad drops to a local minimum ($D_{\text{pitch}} < \theta_{\text{quiet}}$).
  - Delivery C begins $\rightarrow$ bowler approaches crease $\rightarrow$ motion energy rises ($D_{\text{pitch}} > \theta_{\text{active}}$).
- **Invariant:** When searching backward from Delivery C's ball release:
  Pre-roll stops at the **first local minimum (inactivity valley)** preceding Delivery C. It cannot cross an activity peak that occurred prior to that valley. This physically prevents Delivery C from reaching back into Delivery B's stroke.

---

## 3. How to Separate Consecutive Deliveries with No Black Gap

### 3.1 The Scenario
$$\text{DELIVERY A} \longrightarrow \text{DELIVERY B} \quad (\text{continuous match footage, single camera, no cuts, no black frames})$$

### 3.2 Distinguishing the Action Windows
In continuous footage, the single-pass loop computes the **Pitch Corridor Activity Profile** $E_p(t)$ on downsampled $160 \times 90$ frame differences masked by the pitch quad:

$$E_p(t) = \frac{1}{|ROI_{\text{pitch}}|} \sum_{(x,y) \in ROI_{\text{pitch}}} |I_{\text{gray}}(x, y, t) - I_{\text{gray}}(x, y, t-1)|$$

The profile over consecutive deliveries exhibits a distinct morphological signature:

```
Energy
  ▲
  │       Delivery A                                Delivery B
  │     (Ball & Stroke)                           (Ball & Stroke)
  │         ┌───┐                                     ┌───┐
  │        ┌┘   └┐                                   ┌┘   └┐
  │       ┌┘     └┐          RESET VALLEY           ┌┘     └┐
  │      ┌┘       └┐        (Dead ball / setup)    ┌┘       └┐
  │  ────┘         └───────────────┬───────────────┘         └────
  └────────────────────────────────┼──────────────────────────────▶ Time
                                   │
                           TRUE BOUNDARY (Split)
                        Argmin of Inactivity Valley
                         Clip A Ends | Clip B Starts
```

### 3.3 Boundary Decision Rule
1. **Anchor Points:**
   - Delivery A has confirmed ball flight from $T_{\text{rel}}^A$ to $T_{\text{lost}}^A$.
   - Delivery B has confirmed ball flight from $T_{\text{rel}}^B$ to $T_{\text{lost}}^B$.
2. **Optimal Partition Point ($T_{\text{split}}$):**
   In the interval $[T_{\text{lost}}^A, T_{\text{rel}}^B]$, find:
   $$T_{\text{split}} = \arg\min_{t \in [T_{\text{lost}}^A + \Delta_{\text{stroke}}, T_{\text{rel}}^B - \Delta_{\text{stride}}]} E_p(t)$$
   Where $\Delta_{\text{stroke}} = 1.0\text{s}$ (minimum stroke follow-through) and $\Delta_{\text{stride}} = 1.5\text{s}$ (minimum delivery stride).
3. **Partition Invariant:**
   - $\text{Clip A.end\_frame} = T_{\text{split}}$
   - $\text{Clip B.start\_frame} = T_{\text{split}} + 1$
   - Neither clip contains a single frame of the other's action window.

---

## 4. Signal Hierarchy: Authoritative vs. Supporting

To prevent ambiguity, signals must have a strict precedence order:

| Hierarchy Tier | Signal | Source | Role & Authority |
|---|---|---|---|
| **Tier 1: Authoritative Core** | **YOLO11 + Kalman Ball Trajectory** | `BallDetector` + `BallTracker` | Determines delivery existence, physics, release point, bounce point, and core flight frames. |
| **Tier 1: Authoritative Hard Barriers** | **Prior Delivery End / Next Delivery Start** | Tracking loop state | Hard barrier: clips must be strictly non-overlapping. |
| **Tier 1: Authoritative Hard Barriers** | **Black / Blank Frames ($\ge 3$ frames)** | Pixel luminance ($Y < 3.0$) | Hard barrier: pre-roll and post-roll never cross black gaps. |
| **Tier 1: Authoritative Hard Barriers** | **Scene Cuts / Camera Switches** | Histogram diff ($\chi^2 > 0.45$) | Hard barrier: delivery clip cannot span across camera cuts. |
| **Tier 2: Semi-Authoritative Sentinels** | **Sub-threshold Ball Detections** | YOLO raw boxes ($1 \le N < 8$) | Marks untracked deliveries as impermeable event horizons. |
| **Tier 2: Semi-Authoritative Sentinels** | **Pitch Inactivity Valleys** | $E_p(t)$ in Pitch ROI | Defines exact partition point between consecutive events. |
| **Tier 3: Supporting Signals** | **Full-frame Motion Energy** | Inter-frame diff ($D_{\text{full}}$) | Fallback when homography/pitch quad is poorly calibrated. |
| **Tier 3: Supporting Signals** | **Kalman Gating Rejections** | `track.gated_out` | Indicates ball is near the gate but occluded or deflected. |

---

## 5. Handling Completely Missing Ball Detections

### 5.1 When an Event Has 0 Detections
If an entire delivery occurs with 0 ball detections (e.g. ball hidden by bowler's body, bowler bowled wide outside frame, or extreme blur):
1. **Tracking Output:**  
   The pipeline produces **no DeliveryRecord** for this interval.  
   *(Under C2/C3 architectural constraints, we do not fabricate trajectory points or invent deliveries without ball tracking data).*
2. **Clip Output:**  
   The pipeline produces **no clip** for this interval.
3. **Containment Protection:**  
   The motion profile $E_p(t)$ identifies the activity cluster as an **Untracked Activity Window**.
   - An untracked activity window acts as a **Clip Exclusion Zone**.
   - Delivery C's pre-roll search detects the exclusion zone and halts before it.
   - Delivery A's post-roll search detects the exclusion zone and halts before it.
   - **Result:** Delivery B is omitted from the deliveries list, but **never swallowed** into Delivery A or Delivery C.

### 5.2 When the Entire Uploaded Video Has 0 Detections
- `TrackingResult.deliveries = []`
- `warnings.append("No sustained ball trajectory detected in uploaded video.")`
- Pipeline returns cleanly with zero delivery clips written.

---

## 6. Calculation of Pre-Roll and Post-Roll (Dynamic Bounding)

Pre-roll and post-roll must not be static additions. They must be **outward-bounded searches** starting from the delivery's core flight window:

```
                  ┌───────────────────────────────┐
                  │ CORE FLIGHT WINDOW            │
                  │ [first_detected .. ball_lost] │
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[BACKWARD PRE-ROLL SEARCH]                        [FORWARD POST-ROLL SEARCH]
Starts at: first_detected                         Starts at: ball_lost
Target: Run-up & stride (up to 3.0s)              Target: Stroke & follow-through (up to 1.8s)

Halts IMMEDIATELY if:                             Halts IMMEDIATELY if:
1. Black gap frame reached                        1. Black gap frame reached
2. Scene cut detected                             2. Scene cut detected
3. Previous delivery boundary reached             3. Next delivery boundary reached
4. Sub-threshold ball candidate reached           4. Sub-threshold ball candidate reached
5. Inactivity valley reached                      5. Inactivity valley reached
6. Max pre-roll cap (3.0s) reached                6. Max post-roll cap (1.8s) reached
         │                                                 │
         ▼                                                 ▼
 clip.start_frame                                  clip.end_frame
```

### 6.1 Mathematical Formulation
Let $T_{\text{first}}$ be the first confirmed detection frame, and $T_{\text{lost}}$ be the ball lost frame.

$$\text{clip.start\_frame} = \max \Big( T_{\text{first}} - \text{PRE\_ROLL\_MAX}, \; B_{\text{left}} + 1 \Big)$$
Where $B_{\text{left}}$ is the highest frame index before $T_{\text{first}}$ satisfying any of:
- Frame $B_{\text{left}}$ is a black frame ($Y < 3.0$)
- Frame $B_{\text{left}}$ is a scene cut
- Frame $B_{\text{left}}$ belongs to a prior delivery (confirmed or sub-threshold)
- Frame $B_{\text{left}} = \arg\min_{t \in [T_{\text{first}} - \text{PRE\_ROLL\_MAX}, T_{\text{first}}]} E_p(t)$ (inactivity valley)

$$\text{clip.end\_frame} = \min \Big( T_{\text{lost}} + \text{POST\_ROLL\_MAX}, \; B_{\text{right}} - 1 \Big)$$
Where $B_{\text{right}}$ is the lowest frame index after $T_{\text{lost}}$ satisfying any of:
- Frame $B_{\text{right}}$ is a black frame
- Frame $B_{\text{right}}$ is a scene cut
- Frame $B_{\text{right}}$ belongs to a subsequent delivery
- Frame $B_{\text{right}} = \arg\min_{t \in [T_{\text{lost}}, T_{\text{lost}} + \text{POST\_ROLL\_MAX}]} E_p(t)$ (inactivity valley)

---

## 7. Generalization to Arbitrary Uploaded Videos

The algorithm must generalize across four completely different types of uploaded cricket video:

| Video Source Type | Typical Structure | How This Design Handles It |
|---|---|---|
| **1. Broadcast TV Match** | Camera cuts after delivery, outfield replay, return to pitch cut | **Scene Cut Sentinel** locks clip to the pitch-camera segment. Pre-roll stops at the cut to the bowler; post-roll stops at the cut to outfield/replay. |
| **2. Edited Highlights / Composite** | Deliveries separated by 0.5s–2.0s transitions or black frames | **Black Gap Sentinel & Rapid Event Horizon** truncates roll windows at transition edges. Deliveries cannot swallow transitions or neighbors. |
| **3. Club / Amateur Cricket** | Single fixed camera behind bowler, uninterrupted 20-minute rolling video | **Pitch Inactivity Valley Sentinel** identifies the natural pause when bowler walks back and batsman resets, placing the cut cleanly in the dead-ball valley. |
| **4. Net Practice Video** | Continuous deliveries bowled every 6–10 seconds | **Activity Profile Partitioning** splits deliveries during the setup/stride interval; neither batsman stroke nor bowler run-up is merged. |

---

## 8. Automated Tests Proving the Strict Invariant

To guarantee that no regression ever re-introduces multi-event clips, the following automated tests must be implemented before final sign-off:

### Test Suite 1: Synthetic Tagged Frame Ground-Truth (`test_clip_isolation.py`)
Each frame is injected with an exact ground-truth identity tag (pixel mean):
- `TAG_DELIV_A` (value 210)
- `TAG_DELIV_B` (value 90)
- `TAG_GAP` (value 30)

1. **`test_untracked_intervening_delivery_isolation`:**  
   Setup: `[A: 25f] -> [Gap: 45f] -> [B (untracked): 25f] -> [Gap: 45f] -> [C: 25f]`  
   Assertion: Clip A contains 0 frames of B/C. Clip C contains 0 frames of A/B.
2. **`test_no_black_gap_consecutive_deliveries`:**  
   Setup: `[A: 60f] -> [B: 60f]` (seamless, no gap, no cut).  
   Assertion: Clip A ends exactly at the activity valley. Clip B starts at valley + 1. Zero overlapping frames.
3. **`test_black_gap_hard_barrier`:**  
   Setup: `[A: 30f] -> [Black Gap: 20f] -> [B: 30f]`.  
   Assertion: Neither clip contains a single black frame ($Y < 3.0$).
4. **`test_scene_cut_hard_barrier`:**  
   Setup: Scene cut 10 frames before ball release.  
   Assertion: Clip start is clamped to the cut frame, not earlier.

### Test Suite 2: Automated Multi-Signal Validator Integration
1. Run the validator (§13 of Diagnosis) across all written clips on the full fixture.
2. Assert:
   $$\text{len}(\text{validator.by\_classification}(\text{"MULTIPLE\_EVENTS"})) == 0$$
   $$\text{len}(\text{validator.by\_classification}(\text{"NO\_EVENT"})) == 0$$
   Every clip must be classified as `SINGLE_EVENT` (or explicitly `AMBIGUOUS` if boundary confidence is borderline).

---

## 9. Deliverables & Next Steps

This design addendum resolves the architectural gap. 

### Proposed Implementation Plan (Pending Approval):
1. **Step 1:** Implement the per-frame light metrics (black frame flag, scene cut flag, and pitch-quad motion energy $E_p(t)$) in `tracking_service.py` inside the single decode loop.
2. **Step 2:** Update `ClipWriterRegistry.open_for()` and boundary search in `clip_writer.py` to use dynamic barrier-bounded pre/post-roll search instead of static offset subtraction.
3. **Step 3:** Implement the candidate ball sentinel to preserve `REJECTED_EVENT_CANDIDATE` horizons.
4. **Step 4:** Implement the automated multi-signal `ClipValidator`.
5. **Step 5:** Run the new test suite and verify that all clips produced on `full_fixture.mp4` pass as `SINGLE_EVENT`.

*(No code has been modified. Awaiting your approval before proceeding with implementation.)*
