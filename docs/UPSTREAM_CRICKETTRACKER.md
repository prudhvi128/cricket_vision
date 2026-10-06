# CrickTrack — IPL-Style Cricket Ball Tracking

A professional cricket ball tracking system with **Hawk-Eye-style broadcast graphics** built with YOLO11, Kalman Filter, and OpenCV.

---

## Folder Structure

```
cricket_project-main/
├── backend/
│   ├── model.py                  # Main pipeline orchestrator
│   ├── main.py                   # FastAPI server
│   ├── requirements.txt
│   ├── analysis.json             # Output analysis (auto-generated)
│   ├── models/
│   │   └── best.pt               # YOLO11 cricket ball weights
│   ├── uploads/                  # Temporary input videos
│   ├── outputs/                  # Processed output videos
│   └── tracker_modules/          # ◄ Modular architecture
│       ├── __init__.py
│       ├── utils.py              # Homography, smoothing, helpers
│       ├── detector.py           # YOLO11 ball detector
│       ├── tracker.py            # Kalman Filter tracker
│       ├── trajectory.py         # Physics, speed, bounce, classification
│       └── renderer.py           # IPL-style broadcast graphics
└── frontend/
    └── ballscribe-insight-main/  # React + Vite + TypeScript UI
        └── src/
            ├── components/
            │   ├── PitchMap.tsx          # Pitch map (all 6 zones)
            │   ├── BallAnalysisCard.tsx  # Per-delivery card
            │   └── BallAnalysisSection.tsx # Filter + grid
            └── pages/
                └── Results.tsx
```

---

## Module Descriptions

### `tracker_modules/utils.py`
Shared helpers used across all modules:
- `build_homography(w, h)` — Builds a pixel → ground-plane homography (metres)
- `pixel_to_ground(pt, H)` — Maps pixel to real-world (x_m, y_m)
- `savgol_smooth(values)` — Savitzky-Golay smoother (no scipy dep)
- `get_fps(cap)` — Reads FPS from metadata, falls back safely

### `tracker_modules/detector.py`
YOLO11-based ball detector:
- Downscales each frame to 640×360 for fast inference
- Scales bounding boxes back to full resolution
- Filters detections by size (rejects logos/crowd noise)
- Returns the highest-confidence ball centre per frame

### `tracker_modules/tracker.py`
Kalman Filter tracker (4-state: x, y, vx, vy):
- **Prediction-only** for up to 8 missed frames (handles occlusion)
- **Gating**: detections more than 120 px from the estimate are rejected (false-positive filter)
- Manages the trajectory buffer and per-delivery position accumulator
- `reset_delivery()` cleanly resets state between deliveries

### `tracker_modules/trajectory.py`
All physics and analytics:
- `compute_release_speed()` — Median speed from homography-corrected ground distances
- `detect_bounce()` — Savitzky-Golay + vertical velocity sign change + horizontal speed drop
- `detect_release_frame()` — Highest-displacement frame heuristic
- `estimate_swing()` — Lateral drift (metres) → inswing/outswing degrees
- `compute_release_angle()` / `compute_bounce_angle()` — Angle at key events
- `classify_length()` — **6 zones**: Beamer / Bouncer / Short / Good Length / Full / Yorker
- `classify_line()` — **5 zones**: Wide Leg / Leg Side / Middle / Off Side / Wide Off
- `predict_trajectory()` — Ballistic (gravity-affected) arc prediction

### `tracker_modules/renderer.py`
IPL broadcast-quality graphics (pure OpenCV):
- `draw_trajectory_trail()` — Yellow→orange→red gradient neon trail with Gaussian glow passes
- `draw_bounce_marker()` — Glowing green circle + "BOUNCE" label
- `draw_release_marker()` — Glowing blue circle + "RELEASE" label
- `draw_prediction_arc()` — Dashed cyan future-path arc
- `draw_hud()` — Semi-transparent bottom bar with speed (colour-coded), length, line, swing, angles
- `draw_mini_pitchmap()` — Corner inset showing all bounce points

### `model.py`
Orchestrates all modules per-frame:
1. Detect (YOLO) → Track (Kalman) → Render (trail/markers/HUD/pitch inset)
2. On ball loss: compute speed + bounce + angles + classification → append to deliveries
3. Write frame to output video; save `analysis.json`

---

## Setup & Run

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend/ballscribe-insight-main
npm install
npm run dev
```

Then open `http://localhost:5173` and upload a cricket video.

---

## Key Improvements Over v1

| Area | Before | After |
|------|--------|-------|
| Architecture | Single 600-line file | 5 focused modules |
| Tracker | Simple deque + velocity extrapolation | Kalman Filter + gating + occlusion handling |
| Speed | Sometimes unreliable | Homography-corrected ground-plane distances, median of early segment |
| Bounce detection | Basic pixel-Y threshold | SG smoothed + sign-change + horizontal-speed cross-check |
| Length zones | 4 (Short/Good/Full/Yorker) | **6** (Beamer/Bouncer/Short/Good Length/Full/Yorker) |
| Line zones | 3 | **5** (Wide Leg / Leg / Middle / Off / Wide Off) |
| Rendering | Simple coloured line | Neon gradient trail + glow + motion blur + bounce/release markers + prediction arc |
| HUD | Basic text | IPL-style bar with colour-coded speed, angles, swing |
| Pitch map | Static dots | Interactive filter chips, zone bands, hover tooltips, count badges |
| Filters | Length showed only Short/Full | **All 6 lengths always shown** |

---

## Tips for Best Results

- Use a side-on broadcast angle (camera parallel to the pitch).
- Ensure the full pitch is visible in frame.
- Higher FPS video → more accurate speed and bounce detection.
- If speed readings seem off, tune the `frac` values in `utils.py → build_homography()` to match your camera angle.
