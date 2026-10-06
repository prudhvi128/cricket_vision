# Merge Audit - CricketShot-Classification + CricketTracker

## Overview
This document provides a comprehensive audit of both repositories before merging into one unified cricket analysis application.

**Date:** 2026-10-04  
**Repositories in scope:**
1. `CricketShot-Classification-main` - Shot classification (EfficientNet-B0 + Transformer), ball tracking basics (YOLO + Kalman), delivery analysis
2. `CricketTracker-main` - Ball tracking/bowling analysis (YOLO11 + Kalman), trajectory, bounce, speed, line/length, swing, angles

---

## 1. Repository Structure Comparison

### CricketShot-Classification-main
```
CricketShot-Classification-main/
├── backend/
│   ├── full_match_output/
│   ├── models/
│   ├── src/
│   │   ├── inference/        # predict.py, batch_predict.py
│   │   ├── pipeline/         # delivery_analyzer.py, efficientnet_transformer.py, match_pipeline.py, run_pipeline.py, extract_frames.py
│   │   ├── segmentation/      # auto_clipper.py
│   │   ├── tracking/          # detector.py (YOLO), tracker.py (Kalman)
│   │   └── visualization/
│   ├── visualization_test/
│   ├── main.py               # FastAPI app (async task-based)
│   ├── requirements.txt
│   ├── test_*.py             # various test scripts
├── frontend/                 # React + Vite + TS + Tailwind v4 + Three.js + framer-motion
├── config.yaml
└── .gitignore
```

### CricketTracker-main
```
CricketTracker-main/
├── backend/
│   ├── models/
│   ├── tracker_modules/
│   │   ├── __init__.py
│   │   ├── detector.py       # YOLO11 ball detector
│   │   ├── homography.py     # pitch homography
│   │   ├── renderer.py       # trajectory rendering
│   │   ├── tracker.py        # Kalman tracker (more sophisticated)
│   │   ├── trajectory.py     # speed, bounce, angles, swing, line/length classification
│   │   └── utils.py          # utilities
│   ├── main.py               # FastAPI app (simple upload/status)
│   ├── model.py              # main processing pipeline (orchestrator)
│   ├── requirements.txt
│   └── analysis.json         # example output
└── frontend/
    └── ballscribe-insight-main/  # React + Vite + TS + shadcn/ui + Tailwind
        └── [src/components, pages, etc.]
```

---

## 2. Architecture Comparison

| Aspect | CricketShot-Classification | CricketTracker |
|---|---|---|
| **Purpose** | Shot classification focused (identify cricket shot type from clip) | Ball tracking & bowling analysis focused (trajectory, speed, bounce, line/length, swing) |
| **Backend** | FastAPI (`main.py`) with task queue, async processing, status/progress endpoints. Also has `predict` for single clip. | FastAPI (`main.py`) with simple upload/status flow; processing runs in background thread, returns JSON with deliveries + video URL. |
| **Frontend** | React 19 + Vite + TypeScript + Tailwind CSS v4 + framer-motion + Three.js (glossy landing page, scroll effects). Uploads to backend, streams progress via SSE. | React 18 + Vite + TypeScript + shadcn/ui + Tailwind CSS. Clean, professional UI (navbar, upload, processing, results). Shows video with trajectory rendered by backend, ball-by-ball table, pitch map. |
| **Video Processing** | Can analyze delivery clips; also has segmentation/clipping utilities (`auto_clipper.py`, `match_pipeline.py`). API has `/api/predict` for single clip and task-based flow. | Full match video processing end-to-end: detects deliveries implicitly via ball tracking loss, finalizes delivery when ball lost, renders trajectory trail onto output video. |
| **Models** | - Shot: EfficientNet-B0 + Transformer (`cricket_model_transformer.ckpt`) - Ball: YOLO (`best.pt`) for detection - Kalman for tracking | - Ball: YOLO11 (`best.pt`) for detection - Kalman filter (more detailed implementation) - Homography for pitch mapping |
| **Output** | Returns prediction, confidence, class probabilities, ball tracking info (trajectory points with frame,x,y). | Returns `analysis.json` with `trajectory` (full frame log) and `deliveries` array: speed, length (6 zones), line (5 zones), swing, release_angle, bounce_angle, bounce_x,y (normalized). Also outputs rendered MP4 with trajectory trail. |

---

## 3. Pipeline Comparison

### CricketShot-Classification Flow
1. Upload clip/video via API
2. If clip: `analyze_delivery_clip()` - samples 30 frames for shot classification (uniform sampling), runs YOLO ball detector + Kalman tracker, returns combined result
3. Shot classification: load 30 frames → transform → EfficientNet-B0+Transformer → softmax → prediction + confidence + probabilities
4. Ball tracking: YOLO detect on frames (resized 640x360), map back to original coords, Kalman filter, collect trajectory points
5. Returns unified result structure

### CricketTracker Flow
1. Upload full match video → `/upload` starts background processing
2. `process_video()`: frame-by-frame loop
3. For each frame: detect ball (YOLO11), update Kalman tracker, append to trajectory if tracked
4. When ball lost AND `in_delivery` true: finalize delivery (need >= MIN_DELIVERY_FRAMES=8). Compute analytics using `tracker_modules/trajectory.py`
5. Bounce detection via vertical velocity/minima in world/pixel space with homography
6. Speed from early positions (first N frames) using homography to get meters
7. Length classified by bounce y (6 zones), line by bounce x (5 zones)
8. Swing estimation, release/bounce angles
9. Render trajectory trail on frame, write output video
10. Save `analysis.json` and return to API

---

## 4. Models Used

### CricketShot-Classification
- **Shot model**: `backend/models/cricket_model_transformer.ckpt` (PyTorch checkpoint)
  - Architecture: `ImprovedSOTAModel` (EfficientNet-B0 backbone + Transformer) - see `src/pipeline/efficientnet_transformer.py`
  - Classes: 10 - ["Cover", "Defense", "Flick", "Hook", "Late Cut", "Lofted", "Pull", "Square Cut", "Straight", "Sweep"]
  - Input: 30 frames, 224x224
- **Ball detector**: `backend/models/best.pt` (Ultralytics YOLO) - used by `src/tracking/detector.py`
  - Detects ball, filters by size/diagonal, returns center
- **Dependencies**: torch, torchvision, ultralytics, opencv-python, fastapi, uvicorn, numpy, etc.

### CricketTracker
- **Ball detector**: `backend/models/best.pt` (YOLO11) - same model filename, used by `tracker_modules/detector.py`
  - Similar filtering logic (CONF 0.30, size constraints)
- **Homography/pitch logic**: implemented in `tracker_modules/homography.py`, `trajectory.py`
- **Dependencies**: ultralytics, opencv-python, numpy, fastapi, uvicorn, python-multipart

**Note**: Both use `models/best.pt` for ball detection. Need to check if same model file. Shot model only in Shot-Classification repo.

---

## 5. What to Keep from CricketShot-Classification

**Keep these valuable components:**

1. **Shot classification pipeline** - The EfficientNet-B0 + Transformer model and inference logic is the core shot prediction capability. This is absent from CricketTracker.
2. **Delivery analysis module** (`src/pipeline/delivery_analyzer.py`) - Combines shot prediction + ball tracking into unified result structure. Good base for unified data model.
3. **Model loading & inference** (`src/pipeline/efficientnet_transformer.py`, `load_30_frames`, `predict_shot`) - Well-structured, handles checkpoint loading with compatibility logic.
4. **Segmentation/clipping utilities** (`src/segmentation/auto_clipper.py`, `src/pipeline/match_pipeline.py`, `extract_frames.py`) - CricketShot has more explicit clipping logic. Useful if we need to extract delivery clips from full video.
5. **Task-based API design** - Async task tracking with progress updates (good UX). CricketTracker is simpler but functional.
6. **Frontend (CricShot AI)** - Has polished landing page, animations (framer-motion), three.js effects. However UI requirements specify minimal/purposeful animations - may need to tone down per requirements.
7. **Test scripts** - Good reference for expected behavior.

---

## 6. What to Take from CricketTracker

**Take these critical components:**

1. **Superior ball tracking/traj analysis** - `tracker_modules/trajectory.py` computes speed (km/h), bounce detection, release/bounce angles, swing estimation, line/length classification (6 length zones, 5 line zones) - much more detailed than Shot-Classification which has stubs.
2. **Homography module** (`tracker_modules/homography.py`) - Maps pixel coords to real-world (meters) for accurate speed calculation. Critical for realistic bowling metrics.
3. **Robust Kalman tracker** - `tracker_modules/tracker.py` uses proper Kalman filter (4-state: x,y,vx,vy) with gating, occlusion handling, missed-frame tolerance. More sophisticated than the simple implementation in Shot-Classification.
4. **Trajectory rendering** - `tracker_modules/renderer.py` draws trajectory trail; backend renders video with overlay (`draw_trajectory_trail`). This directly addresses requirement 5 (ball trajectory rendered continuously on clipped delivery video).
5. **Delivery finalization logic** - Smart logic: finalize on ball loss, debounce between deliveries (MIN_FRAMES_BETWEEN_DELIVERIES=60), require MIN_DELIVERY_FRAMES>=8. Good for full-match processing.
6. **Frontend (BallScribe Insight)** - Cleaner results UI: video player, ball-by-ball table, pitch map visualization, summary stats. More aligned with "data-driven" display. Also uses shadcn/ui which is professional.
7. **Analysis JSON structure** - The `deliveries` array with speed/length/line/swing/angles/bounce coords is closer to what we need.

---

## 7. What Must Be Rewritten/Adapted

1. **Unified API contract** - Need single API that: uploads video → extracts deliveries (clips) → runs BOTH pipelines per delivery → returns unified delivery results with prev/next navigation support.
2. **Delivery extraction/clipping** - CricketTracker extracts deliveries implicitly during tracking; CricketShot expects pre-clipped delivery videos. We need a robust clipper that identifies delivery boundaries and outputs individual delivery clips.
3. **Per-delivery processing** - For each delivery clip: run shot prediction (Shot-Classification logic) AND run ball tracking/bowling analysis (CricketTracker logic). Then merge.
4. **Unified data model** - Combine shot info + bowling/traj info into one schema. Need to handle nulls when values can't be determined (per requirements).
5. **Trajectory overlay on delivery clips** - CricketTracker renders overlay on full video. We need to render overlay on each delivery clip, synchronized frame-by-frame using trajectory points for that delivery.
6. **Frontend - Unified delivery viewer** - Need viewer with video+trajectory at top, shot/bowling details below, Previous/Next. Current frontends don't do this combined view. Pick best base (recommend CricketTracker's shadcn/ui for cleaner UI) and adapt.
7. **Integration of shot model into Tracker's pipeline** - Or vice versa. Need to load both models. Avoid duplicate model loading.
8. **File organization** - Avoid duplicate files (detector.py, tracker.py exist in both). Choose best implementations.
9. **Port/config conflicts** - Both might run on same ports (8000). Need clear config.
10. **Static file serving** - Need to serve delivery clips and overlay videos.

---

## 8. Dependency Conflicts

| Package | Shot-Classification | CricketTracker | Notes |
|---|---|---|---|
| **React** | ^19.2.8 | ^18.3.1 | Major version difference. CricketTracker frontend uses React 18 (compatible with many libs). Shot frontend uses React 19. |
| **framer-motion** | ^13.4.0 | (not used) | Only in Shot frontend. Adds animations - but requirements say lightweight/purposeful only. |
| **Three.js** | ^0.186.0 (+@react-three/fiber/drei) | (not used) | Heavy 3D libs in Shot frontend - likely overkill for our needs. |
| **shadcn/ui ecosystem** | (not present) | Many @radix-ui/* packages | Cleaner component system in Tracker frontend. |
| **ultralytics** | Present (for YOLO in tracking) | Present (YOLO11) | Similar, should be compatible. |
| **torch/torchvision** | Present (heavy for training/inference) | Not present | Shot needs torch for shot model. Tracker doesn't need torch if using ultralytics? ultralytics can run without torch in some cases, but better with torch. |
| **opencv-python** | Present | Present | Same. |
| **fastapi/uvicorn** | Present | Present | Same. |

**Resolution:** For frontend, better to use React 18 + shadcn/ui (from CricketTracker) as base - more stable, cleaner, easier to build data-driven UI per requirements. Tweak to be minimal. Keep torch/ultralytics as needed.

---

## 9. Proposed Final Folder Structure

Create a new unified project (don't overwrite originals). Suggest structure:

```
CricketUnified/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── endpoints.py      # unified endpoints (upload, process, delivery details, etc.)
│   │   │   └── deps.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── constants.py
│   │   ├── models/
│   │   │   ├── shot/             # shot model files (.ckpt)
│   │   │   └── ball/             # ball detector (.pt)
│   │   ├── schemas/
│   │   │   └── delivery.py       # unified delivery schema
│   │   ├── services/
│   │   │   ├── clipper.py        # delivery extraction/clipping
│   │   │   ├── shot_classifier.py # shot prediction service
│   │   │   ├── ball_tracker.py    # ball tracking + bowling analysis (adapted from Tracker)
│   │   │   ├── trajectory_overlay.py # render overlay on clips
│   │   │   └── pipeline.py       # orchestrate full flow
│   │   └── utils/
│   ├── data/
│   │   ├── uploads/              # uploaded videos
│   │   ├── clips/                # extracted delivery clips
│   │   ├── overlays/             # clips with trajectory overlay
│   │   └── results/              # analysis JSON
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DeliveryViewer.tsx # main viewer (video+traj+details+nav)
│   │   │   ├── VideoPlayer.tsx
│   │   │   └── ui/                # shadcn/ui components
│   │   ├── lib/
│   │   ├── pages/
│   │   │   ├── Home.tsx
│   │   │   ├── Upload.tsx
│   │   │   └── Analysis.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   └── types/
│   ├── package.json
│   └── vite.config.ts
└── docs/                         # audit, architecture, etc.
```

**Rationale:** Modular services (shot_classifier, ball_tracker) keep concerns separate. Clear data dirs for artifacts. Frontend focused on delivery viewer.

---

## 10. Proposed End-to-End Data Flow

```
1. USER uploads full cricket video (MP4/AVI/MOV)
   ↓
2. BACKEND (clipper service)
   - Identify delivery boundaries using ball tracking logic (from CricketTracker)
   - Extract N delivery clips (start_frame..end_frame) as MP4 files
   - Save clips to data/clips/delivery_001.mp4, etc.
   ↓
3. For EACH delivery clip (in parallel where possible):
   a) SHOT CLASSIFICATION (Shot-Classification logic)
      - Load 30 frames uniformly
      - Run EfficientNet-B0+Transformer
      - Get shot type, confidence, class probs
   b) BALL TRACKING + BOWLING ANALYSIS (CricketTracker logic)
      - Run YOLO11 + Kalman on clip
      - Get trajectory points (frame, x, y) for THIS clip
      - Compute bounce, speed (if computable), length, line, swing, angles
   c) TRAJECTORY OVERLAY GENERATION
      - Render trajectory trail on clip frames using detected points
      - Output overlay video: data/overlays/delivery_001_overlay.mp4
   ↓
4. Build UNIFIED DELIVERY RESULTS array
   - Each item has: delivery_id, video (clip URL), overlay_video (URL), shot{...}, bowling{...}, trajectory{points, continuous, has_gaps?}, bounce{...}
   - All values from actual pipeline; null if unavailable
   ↓
5. API returns results to frontend
   ↓
6. FRONTEND DELIVERY VIEWER
   - Display overlay video (or clip) with trajectory visible
   - Show shot type, bowling type, confidence, other metrics
   - Previous/Next buttons to navigate deliveries
   - All fields update together
```

---

## 11. Proposed API Response

```json
{
  "deliveries": [
    {
      "delivery_id": 1,
      "video_url": "/data/clips/delivery_001.mp4",
      "overlay_video_url": "/data/overlays/delivery_001_overlay.mp4",
      "shot": {
        "type": "Cover",
        "confidence": 0.87,
        "class_probabilities": { "Cover": 0.87, "Defense": 0.08, ... }
      },
      "bowling": {
        "type": "Good Length",  // or derived from length/line combo
        "length": "Good Length",
        "line": "Off Stump",
        "speed": 132.5,
        "speed_estimated": false,
        "confidence": 0.72,
        "swing": "Inswing",
        "release_angle": 12.3,
        "bounce_angle": 8.7
      },
      "trajectory": {
        "points": [
          { "frame": 1, "x": 245, "y": 320, "confidence": 0.85, "source": "detected" },
          ...
        ],
        "continuous": true,
        "interpolated_count": 0,
        "gaps": []
      },
      "bounce": {
        "frame": 23,
        "x": 285,
        "y": 412,
        "detected": true
      },
      "metadata": {
        "total_frames": 42,
        "fps": 30.0,
        "frames_processed": 42
      }
    }
  ],
  "total": 6,
  "video_info": {
    "original_filename": "match.mp4",
    "duration": 124.5
  }
}
```

**Notes:**
- Use actual values only. If speed can't be computed reliably, set `speed: null`, `speed_estimated: false`. Don't fabricate.
- Trajectory points must come from detected/tracked positions. If gaps exist (missed frames where we only predicted), mark source appropriately or track interpolated_count.
- Provide both clip and overlay video URLs for flexibility.

---

## 12. Risks/Problems Found

| Risk | Severity | Mitigation |
|---|---|---|
| **React version mismatch** | High | Choose CricketTracker frontend (React 18) as base. Don't mix. |
| **Duplicate detector/tracker code** | Medium | Pick best implementation. CricketTracker's Kalman is more complete; adapt. Shot's simpler one is fine but less robust. |
| **Model file sharing** | Medium | Both use `models/best.pt` (ball detector). Check if identical. Copy shot model separately. |
| **Clipping accuracy** | High | Need to identify delivery start/end correctly. If clipper is wrong, shot prediction and bowling analysis suffer. Test on real videos. |
| **Speed calculation needs homography** | High | Homography requires pitch corners/viewpoint. If auto-homography fails or viewpoint varies, speed may be null - this is correct per "don't fabricate". |
| **GPU/CPU compatibility** | Medium | Torch + ultralytics need proper deps. May have conflicts. Test in env. |
| **Frontend animation overload** | Low | Shot frontend has heavy animations (framer-motion everywhere). Per requirements, strip to minimal purposeful ones. |
| **Large video files** | Medium | Processing can be slow. Need progress updates. Consider chunking. |
| **Trajectory overlay sync** | High | Overlay must align with clip frames exactly. Frame numbers must match. |
| **API contract confusion** | Medium | Shot uses task-based SSE; Tracker uses simple upload/status. Need unified clean API. |
| **Port conflicts** | Low | Run unified backend on 8000, frontend on 5173 as usual. |

---

## 13. Exact Implementation Order

Follow the phased execution strictly:

### Phase 1 - Repository Audit (CURRENT)
- [x] Inspect both repos - structure, backend, frontend, models
- [x] Trace pipelines
- [x] Identify deps/models/conflicts
- [x] Produce this MERGE_AUDIT.md

**STOP after audit - await approval.**

### Phase 2 - Unified Architecture (After approval)
1. Create `CricketUnified/` directory
2. Copy models: `CricketShot-Classification-main/backend/models/cricket_model_transformer.ckpt` and `best.pt` to unified `backend/app/models/`
3. Create module structure as proposed
4. Write `docs/UNIFIED_ARCHITECTURE.md` with detailed design
5. Define unified schemas

### Phase 3 - Backend Integration
1. Port/adapt ball detector + tracker + trajectory from CricketTracker (robust)
2. Port shot classifier from CricketShot-Classification
3. Implement clipper service (extract deliveries from full video)
4. Implement trajectory overlay service (render on clips)
5. Implement unified pipeline orchestrator
6. Create unified API endpoints
7. Wire static file serving for clips/overlays/results

### Phase 4 - Video/Trajectory Output
1. Test clip extraction on sample video
2. Test shot prediction on clips
3. Test ball tracking on clips
4. Generate overlay videos and verify sync
5. Validate browser playback (MP4/H.264)

### Phase 5 - Frontend Integration
1. Start from CricketTracker frontend (shadcn/ui, React 18) - cleaner base
2. Build `DeliveryViewer` component per UI spec
3. Implement video player with trajectory
4. Display shot/bowling details
5. Add Previous/Next navigation
6. Apply minimal purposeful animations only
7. Use custom professional cricket/sports color palette
8. Ensure responsive layout

### Phase 6 - End-to-End Testing
- [ ] Upload full video → clipping works
- [ ] Each delivery processed by both pipelines
- [ ] Unified results returned
- [ ] Frontend displays all data correctly
- [ ] Previous/Next works, boundaries disabled
- [ ] No hardcoded values
- [ ] Videos play in browser
- [ ] No console/API errors
- [ ] Responsive on different screens

---

## Next Steps

**This completes Phase 1.** The audit is done. Do not modify code yet. 

Please review this plan and approve to proceed to Phase 2 (create unified architecture and structure).