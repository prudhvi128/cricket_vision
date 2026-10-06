# API Contract

The contract a frontend should code against. It is the one the backend is tested
against (`backend/tests/test_api_contract.py`) and the one the real-video
verification exercises over HTTP (`backend/tools/e2e_real_video.py`).

Base path: `/api`. All bodies are JSON except the two media routes, which stream
`video/mp4`.

- **Schema version:** `2.1`
- **Pipeline version:** `1.1.0`
- **Check:** `GET /api/health` returns both, plus the stage vocabulary, so a
  client never has to hardcode stage names.

> Every `analysis_id` in this document is an illustrative example taken from a
> real run. Analysis artefacts are **not** retained in the repository — they are
> generated output. Produce your own id with `POST /api/analyze`, or run
> `python tools\e2e_real_video.py` against `real_cricket.mp4`.

---

## 1. Start an analysis

```http
POST /api/analyze
Content-Type: multipart/form-data

video=<file>
```

`202 Accepted`:

```json
{
  "analysis_id": "4c458118e2f04112",
  "status": "queued",
  "progress": 0.0,
  "stage": "queued",
  "bytes": 79853450,
  "status_url": "/api/analysis/4c458118e2f04112/status",
  "deliveries_url": "/api/analysis/4c458118e2f04112/deliveries",
  "result_url": "/api/analysis/4c458118e2f04112/result",
  "progress_url": "/api/analysis/4c458118e2f04112/progress",
  "cancel_url": "/api/analysis/4c458118e2f04112/cancel"
}
```

The upload returns immediately. Processing runs on a background thread; poll or
stream `/status` or `/progress`.

Accepted extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`. Ceiling 2 GB.

### Rejections happen here, not minutes later

| Status | Cause |
| --- | --- |
| `400` | Unsupported extension, or a correctly-named file that OpenCV cannot open as video |
| `400` | Empty upload |
| `413` | Over 2 GB |

The extension check alone used to accept a text file renamed `.mp4`, which then
failed deep inside the tracking pass with an opaque OpenCV error. Both checks now
run at the door, and a rejected upload leaves no analysis directory behind.

### Legacy alias

`POST /api/upload` with a `file=` form field is the same handler and the same
`analysis_id` space. The merged CricketShot-Classification frontend posts there;
it is kept rather than deleted. Use `/api/analyze` in new code.

---

## 2. Poll or stream progress

```http
GET /api/analysis/{analysis_id}/status
```

```json
{
  "analysis_id": "4c458118e2f04112",
  "status": "running",
  "stage": "tracking",
  "substage": "analytics",
  "progress": 27.4,
  "percent": 27.4,
  "units": { "tracking": { "done": 3042, "total": 13565 } },
  "frames_done": 3042,
  "frames_total": 13565,
  "detail": "frame 3042 (22%), 12 deliveries",
  "error": null,
  "cancel_requested": false,
  "created_at": "...", "started_at": "...", "finished_at": null
}
```

`GET /api/analysis/{analysis_id}/progress` is the same payload as a genuine
`text/event-stream`:

```
event: progress
data: {"stage": "tracking", "progress": 27.4, ...}

event: done
data: {"status": "completed", "progress": 100.0, ...}
```

`GET /api/analysis/{analysis_id}/progress/json` is the pollable equivalent.

> The merged frontend opened an `EventSource` against a route that returned plain
> JSON, so progress could never have worked. `/progress` is now a real event
> stream.

### Statuses

`queued` â†’ `running` â†’ `completed` | `failed` | `cancelled`

### Stages, and the bands they own

`progress` is derived from **measured units**, never from an assumed fraction of
time. Each stage counts its own real units and maps them through its band:

| Stage | Band | Units counted |
| --- | --- | --- |
| `uploading` | 0â€“5% | bytes written |
| `tracking` | 5â€“62% | frames decoded |
| `validation` | 62â€“68% | clips re-segmented |
| `shot_classification` | 68â€“86% | deliveries classified |
| `overlay` | 86â€“96% | overlays rendered |
| `persisting` | 96â€“100% | documents written |
| `completed` | â€” | terminal |
| `failed` | â€” | terminal |

Guarantees:

- **`progress` never decreases.** Each stage counts its own units, so the
  handoff from tracking (13565/13565) to shot classification (0/37) computes a
  *smaller* number than tracking had reached. A high-water mark prevents the bar
  visibly rewinding mid-analysis.
- **It never overstates.** A stage that has begun but does not yet know its
  denominator reports the floor of its band, not a guess.
- Only a genuinely finished pipeline reports `100.0`. A cancelled run reports how
  far it actually got.
- `percent` and `progress` are the same value. `percent` is kept because the
  existing frontend reads that field.

### `substage` is not a fake phase

Segmentation, per-delivery analytics and purity validation happen *inside* the
single tracking pass. Reporting them as top-level stages would describe a
pipeline with separate stages, which this backend deliberately does not have.
They are surfaced as `substage` (`analytics`, `validation`, `segmentation`,
`clip_writing`) and never move `progress`.

### Unknown analysis

`GET /api/analysis/{id}/status` returns `404`. A typo must not read as "still
processing". The legacy plural route `/api/analyses/{id}/status` keeps its old
lenient `200 {"status": "unknown"}` contract for the existing frontend.

---

## 3. Read the results

```http
GET /api/analysis/{analysis_id}/result              # frontend: array of deliveries
GET /api/analysis/{analysis_id}/result/internal     # diagnostics: full document
GET /api/analysis/{analysis_id}/deliveries          # all validated deliveries
GET /api/analysis/{analysis_id}/deliveries/3        # one delivery, complete
```

### `/result` is a clean array

The frontend contract. A flat array of self-contained deliveries, so
`deliveries[currentIndex]` is all Previous/Next needs:

```json
[
  {
    "delivery_id": 22,
    "delivery_ref": "delivery_022",
    "video": {
      "clip_url": "/data/runs/.../clips/delivery_022.mp4",
      "start_frame": 7986, "end_frame": 8081, "fps": 25.0,
      "width": 1280, "height": 720, "duration_seconds": 3.84
    },
    "trajectory": {
      "points": [
        { "frame": 8005, "x": 781, "y": 290,
          "source": "detected", "confidence": 0.4544 },
        { "frame": 8006, "x": 786, "y": 301,
          "source": "predicted", "confidence": null }
      ],
      "detected_points": 18, "predicted_points": 10,
      "continuity": "continuous"
    },
    "bounce": {
      "detected": true, "frame": 8013, "x": 799, "y": 459,
      "ground_x_m": null, "ground_y_m": null
    },
    "bowling": {
      "speed_kmh": null, "line": null, "length": null, "swing": null,
      "release_angle": -95.3, "bounce_angle": null, "calibrated": false
    },
    "shot": { "classification": null, "confidence": null },
    "quality": {
      "trajectory_valid": true, "tracking_confidence": 0.6306,
      "requires_human_review": true,
      "flags": ["pitch_calibration_required"]
    }
  }
]
```

Rules this response holds to:

- **Same schema for every entry**, always all eight top-level keys.
- **No internal detail.** `event`, `validation`, `provenance`, `boundary`,
  `evidence`, `clip_window`, merge checks, speed samples, clip frame maps,
  Kalman/tracker internals and filesystem paths are not exposed. `clip_url` is
  a URL; `clip_path` never appears.
- **`source` survives on every point** — `detected` is a real detector
  measurement, `predicted` is a Kalman prediction. `frame` is the SOURCE frame,
  so a point lands inside the clip at `frame - video.start_frame + 1`.
- **Null means "could not be computed"**, never a guess: `speed_kmh`, `line`,
  `length`, `swing`, `bounce_angle` and the whole `shot` block are null on a run
  with no pitch calibration or no classifier.
- **Verbose validation becomes flags.** `quality.flags` carries concise strings
  (`multiple_events_in_clip`, `ambiguous_event`, `unvalidated_clip`,
  `pitch_calibration_required`, …) plus the one boolean
  `requires_human_review`. The full verdict is on `/result/internal`.
- **`calibrated`** is the per-video pitch-geometry calibration. Speed is an
  estimate even when it is `true` (Phase 3 D2).

`/result/internal` serves the complete persisted `result.json` — summary,
calibration, performance, warnings, quarantine and the full per-delivery
records with `event`/`validation`/`provenance`. That is the diagnostics route;
the frontend must call `/result`.

`/deliveries` response (internal delivery records, richer than `/result`):

```json
{
  "analysis_id": "...",
  "schema_version": "2.1",
  "total_deliveries": 37,
  "summary": { "...": "..." },
  "calibration": { "...": "..." },
  "deliveries": [ { "...": "..." } ]
}
```

### One delivery per request

Every delivery route takes a `delivery_id` and returns only that delivery's
fields. There is no endpoint that mixes two deliveries, so a client cannot
assemble a shot from one ball and a trajectory from another.

`delivery_id` accepts either spelling:

```
/deliveries/3
/deliveries/delivery_003
```

Anything else is `400`, returned before the analysis lookup â€” a malformed id is
malformed whether or not the analysis exists.

### Delivery record

The record below is what `/deliveries` and `/deliveries/{id}` serve, and what
`/result/internal` keeps per delivery — the internal, diagnostic form. `/result`
serves the cleaned frontend projection of it (§3 above), so a browser never sees
`validation`, `provenance`, `clip_path` or the frame-map audit.

```json
{
  "delivery_id": 3,
  "delivery_ref": "delivery_003",
  "index": 3,
  "clip_url": "/api/analysis/{id}/deliveries/3/clip",
  "overlay_url": "/api/analysis/{id}/deliveries/3/overlay",
  "clip": {
    "start_frame": 101, "end_frame": 195, "frame_count": 95,
    "frame_offset": 100, "fps": 25.0, "width": 1280, "height": 720,
    "codec": "avc1", "frames_written": 95, "frame_map_valid": true,
    "file_name": "delivery_003.mp4"
  },
  "trajectory": {
    "points": [
      { "original_frame": 101, "clip_frame": 1, "x": 612, "y": 401,
        "source": "detected" },
      { "original_frame": 102, "clip_frame": 2, "x": 618, "y": 409,
        "source": "predicted" }
    ]
  },
  "bounce": {
    "detected": true, "original_frame": 118,
    "x_px": 640, "y_px": 512, "x_norm": 0.5, "y_norm": 0.711,
    "ground_x_m": null, "ground_y_m": null,
    "method": "score"
  },
  "bowling": {
    "speed_kmh": null, "speed_method": null,
    "speed_calibrated": false, "speed_imputed": false,
    "length": null, "line": null, "swing": null,
    "release_angle": 74.1, "bounce_angle": null,
    "geometry_calibrated": false
  },
  "shot": {
    "type": "Flick", "confidence": 0.4086,
    "class_probabilities": { "Flick": 0.4086, "...": 0.0 },
    "input_source": "in_memory",
    "frames_sampled": 30, "unique_frames_sampled": 30,
    "frames_repeated": false,
    "sampled_clip_frames": [0, 3, 6, "..."],
    "sampled_original_frames": [100, 103, 106, "..."],
    "error": null
  },
  "overlay": { "rendered": true, "error": null, "...": "..." },
  "validation": {
    "classification": "SINGLE_EVENT", "confidence": 0.82,
    "requires_human_review": false, "reason": "..."
  },
  "quality": { "...": "..." },
  "tracking_source": "single_pass",
  "provenance": {
    "tracking_source": "crickettracker",
    "tracking_pass": "single_pass",
    "trajectory_source": "stored_tracking",
    "shot_model": "cricket_model_transformer",
    "shot_model_input_frames": 30,
    "source_video_decodes": 1,
    "yolo_passes": 1,
    "kalman_passes": 1
  }
}
```

### Trajectory provenance is never ambiguous

Every point carries `source`:

- `detected` â€” a real YOLO detection that survived the tracker gates.
- `predicted` â€” a Kalman extrapolation. Real, but derived.

The fit for speed and bounce uses **detected points only**. Fitting a slope to a
filter's own output would launder an estimate into a measurement.

### Shot records

`shot.type` is `null` when classification failed, with `shot.error` explaining
why. A failed label is never reported as a successful one, and one failure does
not fail the analysis.

The model's tensor is fixed at `(B, 30, 3, 224, 224)`. A clip shorter than 30
frames is padded by deterministic repetition, and the padding is declared:

```json
{ "frames_sampled": 30, "unique_frames_sampled": 19, "frames_repeated": true }
```

Real footage contains 19-frame clips. The invariant is not that every clip is
long enough â€” it is that a short clip never masquerades as 30 distinct
observations.

---

## 4. Purity validation, and what "delivery" means

Two gates judge a clip, and both verdicts are kept.

1. **Pre-persistence gate** (inside the pass) refuses a window it can show is
   impure *before* writing it.
2. **Post-pass validator** (end of pass) re-derives how many events each written
   clip holds from stored tracking data, rather than trusting the segmentation
   that produced it. This catches a merge made inside the segmenter.

| Verdict | Served as a delivery? |
| --- | --- |
| `SINGLE_EVENT` | yes |
| `AMBIGUOUS` | **yes, flagged** â€” verdict, confidence and `requires_human_review` all travel with the record, and `summary.by_validation` counts them separately |
| `MULTIPLE_EVENTS` | no â€” quarantined |
| `NO_EVENT` | no â€” quarantined |
| *no verdict at all* | no â€” quarantined. Silence is not a pass |

`AMBIGUOUS` is included deliberately. The requirement is not to *claim* an
undecidable clip is valid, and it is not claimed. A client that wants only
unambiguous deliveries filters on `validation.classification`; the backend does
not silently drop a third of a real over's deliveries to tidy its own output.

### Quarantine

Refused clips are never presented as deliveries and never deleted:

```json
{
  "summary": { "deliveries": 37, "quarantined": 3,
               "by_validation": { "SINGLE_EVENT": 24, "AMBIGUOUS": 13 } },
  "quarantine": [
    { "delivery_id": 17, "delivery_ref": "delivery_017",
      "classification": "MULTIPLE_EVENTS",
      "reason": "clip spans 2 detected ball events",
      "requires_human_review": true,
      "record": { "...": "the full delivery" } }
  ]
}
```

`GET /api/analysis/{id}/deliveries/17` returns `404`. The record survives in
`quarantine` so "why is delivery 17 missing" is answerable from the artefact
alone.

Delivery ids are **not renumbered** after quarantine, so ids stay stable and
`delivery_017.mp4` remains unambiguous in the filesystem.

---

## 5. Calibration travels with the data

Every payload that contains a speed also carries the calibration block. A client
cannot render an estimate without also receiving the facts that make it an
estimate.

```json
{
  "speed_is_estimate": true,
  "speed_is_calibrated": false,
  "speed_scale": 2.0,
  "speed_offset_kmh": -20.0,
  "geometry_calibrated": false,
  "geometry_source": "template_corner_fractions",
  "ground_plane_analytics_available": false,
  "suppressed_when_uncalibrated": [
    "speed_kmh", "length", "line", "swing", "bounce_angle",
    "bounce.ground_x_m", "bounce.ground_y_m",
    "speed_samples[].ground_x_m", "speed_samples[].ground_y_m"
  ],
  "geometry_note": "Pitch-corner calibration is required for this video. The available homography is a fixed set of corner fractions measured from a different broadcast template, so its ground-plane coordinates, and every speed, length, line, swing and bounce angle derived from them, do not describe this footage."
}
```

**With no per-video pitch corners, these are `null`:**

`speed_kmh` Â· `length` Â· `line` Â· `swing` Â· `bounce_angle` Â·
`bounce.ground_x_m` Â· `bounce.ground_y_m` Â· `mean_speed_kmh`

They are withheld rather than reported as an estimate. An estimate implies the
number is roughly right; a homography fitted to the wrong corners is not roughly
right, it is confidently wrong. A previous version inherited
`geometry_calibrated=True`, which made the suppression branch unreachable and put
a 42â€“199 km/h range into `result.json`.

**Pixel-space results are unaffected and still computed:** `release_frame`,
`bounce.x_px`, `bounce.y_px`, `bounce.x_norm`, `bounce.y_norm`,
`bowling.release_angle`, and the whole trajectory.

`speed_is_calibrated` is `false` even when corners *are* supplied, and stays
false deliberately. Measured corners make the ground plane meaningful, but the
speed is still a least-squares fit over a hand-tuned scale/offset, so it is an
estimate. The signal for "the ground plane is trustworthy" is
`geometry_calibrated` / `ground_plane_analytics_available`.

---

## 6. Media

```http
GET /api/analysis/{analysis_id}/deliveries/{delivery_id}/clip
GET /api/analysis/{analysis_id}/deliveries/{delivery_id}/overlay
```

Both stream `video/mp4` and support HTTP byte ranges (`206`), which is what
`<video>` needs to seek. An overlay that was not rendered is `404`, never a copy
of the clip â€” a client can then show the clip without a trajectory instead of
drawing one that is not there.

Legacy name-based routes, kept for the existing frontend:

```http
GET /api/analyses/{analysis_id}/clips/delivery_003.mp4
GET /api/analyses/{analysis_id}/overlays/delivery_003.mp4
```

Names that are not `delivery_*.mp4` are rejected.

---

## 7. Trajectory for charting

```http
GET /api/analysis/{analysis_id}/deliveries/{delivery_id}/trajectory
```

Returns the stored points, bounce, bowling, clip, quality, the calibration
block, and a self-describing `pitch_map` SVG.

This reads the same array the overlay renderer drew and the analytics were
fitted to, so a plotted path and a rendered path cannot disagree.

The SVG is a plot of real pixel measurements, **not a to-scale pitch**. With no
calibrated homography there are no metres, and no axis is labelled with a
distance. `ground_x_m` / `ground_y_m` are `null`.

---

## 8. Cancel

```http
POST /api/analysis/{analysis_id}/cancel
```

```json
{ "analysis_id": "...", "cancel_requested": true,
  "detail": "Cancellation requested; it takes effect after the current delivery." }
```

Cancellation is checked between deliveries, so a clip is never left half-written
with its frame map claiming otherwise. Reports honestly whether the request was
accepted; a finished or unknown analysis answers `false`.

Upstream's frontend called `/api/cancel`, which did not exist. Cancelling was
impossible; it works now.

---

## 9. Reference data

```http
GET /api/reference
```

Publishes the authoritative enums and calibration facts: the 10 shot class
strings, length/line zone labels, pitch dimensions, and the provenance
vocabularies. Read them from here rather than hardcoding them â€” the two merged
frontends disagreed on both (`Leg Side` vs `Leg`, `Off Side` vs `Off`).

---

## 10. Legacy route map

Every documented route has a plural alias that delegates to the same handler, so
there is one implementation and one behaviour.

| Legacy | Documented |
| --- | --- |
| `POST /api/upload` | `POST /api/analyze` |
| `GET /api/analyses/{id}/status` | `GET /api/analysis/{id}/status` |
| `GET /api/analyses/{id}/result` | `GET /api/analysis/{id}/result` |
| `GET /api/analyses/{id}/result/internal` | `GET /api/analysis/{id}/result/internal` |
| `GET /api/analyses/{id}/progress` | `GET /api/analysis/{id}/progress` |
| `POST /api/analyses/{id}/cancel` | `POST /api/analysis/{id}/cancel` |
| `GET /api/analyses/{id}/deliveries/{id}` | `GET /api/analysis/{id}/deliveries/{id}` |
| `GET /api/analyses/{id}/trajectory/{id}` | `GET /api/analysis/{id}/deliveries/{id}/trajectory` |
| `GET /api/analyses/{id}/tracking` | (no singular equivalent â€” tracker output only) |
| `GET /api/analyses/{id}/clips/{name}` | `GET /api/analysis/{id}/deliveries/{id}/clip` |

`/api/analyses/{id}/tracking` returns the tracker output with **no shot labels
attached** â€” what the single pass actually measured, before inference ran on top
of it. Useful for auditing.

---

## 11. Running it

```powershell
cd backend
python -m uvicorn main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

Results persist under `data/runs/{analysis_id}/`, so an analysis
survives a restart as far as the filesystem is concerned. The in-memory registry
holds only *progress*, which is transient; `/status` reconstructs a `completed`
response from disk for an analysis this process did not run.