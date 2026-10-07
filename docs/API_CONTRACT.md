# API Contract

The contract a frontend should code against. It is the one the backend is tested
against (`backend/tests/integration/test_api_contract.py`) and the one the
real-video verification exercises over HTTP
(`scripts/verification/e2e_real_video.py`).

Base path: `/api`. All bodies are JSON except the clip route, which streams
`video/mp4`.

- **Schema version:** `2.1`
- **Pipeline version:** `1.1.0`
- **Check:** `GET /api/health` returns both, plus the stage vocabulary, so a
  client never has to hardcode stage names.

> Every `analysis_id` in this document is an illustrative example taken from a
> real run. Analysis artefacts are **not** retained in the repository — they are
> generated output. Produce your own id with `POST /api/analyze`, or run
> `python scripts\verification\e2e_real_video.py` against `real_cricket.mp4`.

---

## 0. The whole surface — six routes

| # | Route | Purpose |
| --- | --- | --- |
| 1 | `POST /api/upload` | accept a video (`file=` field) |
| 2 | `POST /api/analyze` | accept a video (`video=` field) |
| 3 | `GET /api/analysis/{analysis_id}/status` | queued / running / completed / failed |
| 4 | `GET /api/analysis/{analysis_id}/result` | the envelope the UI renders |
| 5 | `GET /api/analysis/{analysis_id}/clips/{name}` | clip bytes, byte-range capable |
| 6 | `GET /api/health` | liveness, versions, stage vocabulary |

There is **one spelling per route**. Routes 1 and 2 are the same handler under
the two form-field names the two frontends use. Nothing else is routed: no
plural aliases, no per-delivery routes, no Server-Sent Events, no cancellation,
no static file mount. §9 lists what was removed and why.

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
  "result_url": "/api/analysis/4c458118e2f04112/result"
}
```

The upload returns immediately. Processing runs on a background thread; poll
`/status`.

`POST /api/upload` with a `file=` form field is the same handler and the same
`analysis_id` space — it is the spelling the other frontend posts to.

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

---

## 2. Poll progress

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
  "frames_done": 3042,
  "frames_total": 13565,
  "detail": "frame 3042 (22%), 12 deliveries",
  "error": null,
  "source_name": "worldcup_final.mp4",
  "created_at": "…", "started_at": "…", "finished_at": null
}
```

That is the whole document: thirteen keys, nothing else. There is no second
progress route to keep in step with it — no `/progress`, no `/progress/json`, no
event stream.

### Statuses

`queued` → `running` → `completed` | `failed`

### Stages, and the bands they own

`progress` is derived from **measured units**, never from an assumed fraction of
time. Each stage counts its own real units and maps them through its band:

| Stage | Band | Units counted |
| --- | --- | --- |
| `uploading` | 0–5% | bytes written |
| `tracking` | 5–62% | frames decoded |
| `validation` | 62–68% | clips re-segmented |
| `shot_classification` | 68–86% | deliveries classified |
| `overlay` | 86–96% | overlays rendered |
| `persisting` | 96–100% | documents written |
| `completed` | — | terminal |
| `failed` | — | terminal |

Guarantees:

- **`progress` never decreases.** Each stage counts its own units, so the
  handoff from tracking (13565/13565) to shot classification (0/37) computes a
  *smaller* number than tracking had reached. A high-water mark prevents the bar
  visibly rewinding mid-analysis.
- **It never overstates.** A stage that has begun but does not yet know its
  denominator reports the floor of its band, not a guess.
- Only a genuinely finished pipeline reports `100.0`.
- `progress` is the only percentage. There is no legacy `percent` field, no
  `units` counter, no `phase` vocabulary and no `events` log — the machinery
  behind the number stays server-side.

### `substage` is not a fake phase

Segmentation, per-delivery analytics and purity validation happen *inside* the
single tracking pass. Reporting them as top-level stages would describe a
pipeline with separate stages, which this backend deliberately does not have.
They are surfaced as `substage` (`analytics`, `validation`, `segmentation`,
`clip_writing`) and never move `progress`.

### Failure and unknown ids

- A failed run answers `status: "failed"` with `error` set to the reason.
- An unknown analysis is `404` with a `detail` message. A typo must not read as
  "still processing", and there is no lenient `{"status": "unknown"}` spelling
  left to confuse a client.

---

## 3. Read the results

```http
GET /api/analysis/{analysis_id}/result
```

### One envelope

```json
{
  "analysis_id": "4c458118e2f04112",
  "status": "completed",
  "pitch": {
    "state": "no_calibration",
    "detected": false, "calibrated": false, "confidence": null,
    "using_previous_calibration": false,
    "keypoints": {}, "corners": {}, "center": null,
    "homography_available": false, "orientation": null
  },
  "deliveries": [
    {
      "id": 22,
      "video": {
        "clip_url": "/api/analysis/4c458118e2f04112/clips/delivery_022.mp4",
        "start_frame": 7986, "end_frame": 8081, "fps": 25.0,
        "width": 1280, "height": 720, "duration_seconds": 3.84
      },
      "trajectory": {
        "points": [
          { "frame": 8005, "x": 781, "y": 290,
            "source": "detected", "confidence": 0.4544,
            "pitch_position": null },
          { "frame": 8006, "x": 786, "y": 301,
            "source": "predicted", "confidence": null,
            "pitch_position": null }
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
      },
      "pitch": {
        "state": "no_calibration",
        "detected": false, "calibrated": false, "confidence": null,
        "using_previous_calibration": false,
        "keypoints": {}, "corners": {}, "center": null,
        "homography_available": false, "orientation": null
      }
    }
  ]
}
```

- **`analysis_id` and `status`** say which analysis this is and that it
  finished, so a page that was reloaded can label itself without guessing.
- **`pitch` at the top level** is the calibration of the last delivery that had
  one — the state of the analysis as a whole. Every delivery also carries its
  own `pitch` block, which is what an overlay drawn over a specific clip uses.
- **`deliveries` is an array**, so `deliveries[currentIndex]` is all
  Previous/Next needs: video, trajectory, bounce, bowling analytics, shot,
  quality and pitch in one object.

Rules this response holds to:

- **Same schema for every entry** — always exactly the eight keys `id`, `video`,
  `trajectory`, `bounce`, `bowling`, `shot`, `quality`, `pitch`.
- **No internal detail.** `event`, `validation`, `provenance`, `boundary`,
  `evidence`, `clip_window`, merge checks, speed samples, clip frame maps,
  Kalman/tracker internals and filesystem paths are not exposed. `clip_url` is a
  route; `clip_path` never appears.
- **Nothing internal is served anywhere else either.** The complete `result.json`
  — summary, calibration, performance, quarantine, warnings and every
  per-delivery record — is written to disk under `data/runs/{id}/` and is
  reachable by no route. Read it with a file, not an HTTP client.
- **`source` survives on every point** — `detected` is a real detector
  measurement, `predicted` is a Kalman prediction. `frame` is the SOURCE frame,
  so a point lands inside the clip at `frame - video.start_frame + 1`.
- **Null means "could not be computed"**, never a guess: `speed_kmh`, `line`,
  `length`, `swing`, `bounce_angle` and the whole `shot` block are null on a run
  whose inputs cannot support them — no measured ground plane for the first five
  (§ 6), no classifier for the last. `length` also goes null, with the
  `length_unavailable` flag, when the bounce projects outside the measured
  20.12 m pitch: a ball that cannot be placed on the pitch has no length, and
  clamping it to an edge zone would turn "we do not know" into a confident
  wrong answer.
- **Verbose validation becomes flags.** `quality.flags` carries concise strings
  (`multiple_events_in_clip`, `ambiguous_event`, `unvalidated_clip`,
  `pitch_calibration_required`, …) plus the one boolean
  `requires_human_review`.
- **`calibrated`** is the per-video pitch-geometry calibration. It is the signal
  a client has for whether the ground-plane numbers exist at all. Speed is an
  estimate even when it is `true` (Phase 3 D2).
- **`pitch` is the optional pitch-keypoint detector's own record, in FRAME
  pixels.** `state` is one of `calibrated`, `temporary_loss`,
  `no_calibration`, `calibration_expired`; `keypoints`, `corners` and `center`
  are `[x, y]` pairs in source-pixel coordinates; `homography_available` is
  derived from the stored matrix rather than trusted from a flag. Three
  keypoints are enough for a calibration but not for a quad — in that case
  `corners` is `{}`, `homography_available` is `false` and every
  `pitch_position` is `null`, because a 4th corner is never invented. With no
  `ROBOFLOW_API_KEY` configured the detector is inert and the block reads
  `state: "no_calibration"` with empty containers. **`orientation`** is the
  quad's batting end — `axis` (which way the pitch runs through the frame),
  `batting_end` (0 or 1 along it), `travel_delta` (how far the ball travelled
  to justify that) and `calibration_frame` — or `null` when no quad could be
  oriented (§ 6).
- **`pitch_position` is a point projected onto the pitch, 0..1, and only when a
  calibration existed** for that frame — it is what puts a bounce on the pitch
  map. It is a fraction of the QUAD, not a compass bearing: which end of that
  quad was the batting end is `pitch.orientation`, and a top-down map drawn
  without it can place a bounce at the bowler's end while the length label
  beside it says yorker. It never feeds `speed_kmh`, `length`, `line`, `swing`
  or `bounce_angle` — those come from a measured ground plane (§ 6), and the
  frame-coordinate fallback a map used before the pitch model existed has been
  removed rather than allowed to invent a position.

### One delivery per entry

Each entry is built from a single delivery record, so there is no endpoint that
mixes two deliveries: a client cannot assemble a shot from one ball and a
trajectory from another.

### Errors

| Status | Meaning |
| --- | --- |
| `404` | No such analysis, or no result written yet for one that never ran |
| `409` | The analysis is still `queued`/`running` — come back later |

---

## 4. Health

```http
GET /api/health
```

```json
{
  "status": "ok",
  "schema_version": "2.1",
  "pipeline_version": "1.1.0",
  "analyses_tracked_in_memory": 1,
  "stages": ["uploading", "tracking", "validation", "shot_classification",
             "overlay", "persisting", "completed", "failed"]
}
```

---

## 5. Purity validation, and what "delivery" means

Two gates judge a clip, and both verdicts are kept.

1. **Pre-persistence gate** (inside the pass) refuses a window it can show is
   impure *before* writing it.
2. **Post-pass validator** (end of pass) re-derives how many events each written
   clip holds from stored tracking data, rather than trusting the segmentation
   that produced it. This catches a merge made inside the segmenter.

| Verdict | Served as a delivery? |
| --- | --- |
| `SINGLE_EVENT` | yes |
| `AMBIGUOUS` | **yes, flagged** — `quality.flags` carries `ambiguous_event` and `requires_human_review` travels with the record |
| `MULTIPLE_EVENTS` | no — quarantined (`multiple_events_in_clip` never appears on a served entry) |
| `NO_EVENT` | no — quarantined |
| *no verdict at all* | no — quarantined. Silence is not a pass |

`AMBIGUOUS` is included deliberately. The requirement is not to *claim* an
undecidable clip is valid, and it is not claimed: the flag and the human-review
boolean both say so.

### Quarantine

Refused clips are never presented as deliveries and never deleted. The full
record, its verdict and the reason survive in `result.json`:

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

Delivery ids are **not renumbered** after quarantine, so ids stay stable and
`delivery_017.mp4` remains unambiguous in the filesystem — and a quarantined id
simply does not appear in `deliveries`.

---

## 6. Calibration travels with the data

The `calibration` block is persisted with every analysis, next to the numbers it
explains:

```json
{
  "speed_is_estimate": true,
  "speed_is_calibrated": false,
  "speed_scale": 2.0,
  "speed_offset_kmh": -20.0,
  "speed_scale_applies_to": "The template homography ONLY, where it corrects a fixed set of corner fractions …",
  "geometry_calibrated": false,
  "geometry_source": "template_corner_fractions",
  "ground_plane_analytics_available": false,
  "pitch_keypoint_geometry": {
    "deliveries": 37, "deliveries_with_geometry": 0,
    "rejections": { "pitch_orientation_ambiguous": 12, "no_pitch_calibration": 25 }
  },
  "length_zones_m": [
    ["Yorker", 0.0, 1.5], ["Full", 1.5, 4.0], ["Good Length", 4.0, 7.0],
    ["Short", 7.0, 10.0], ["Very Short", 10.0, 20.12]
  ],
  "suppressed_when_uncalibrated": [
    "speed_kmh", "length", "line", "swing", "bounce_angle",
    "bounce.ground_x_m", "bounce.ground_y_m",
    "speed_samples[].ground_x_m", "speed_samples[].ground_y_m"
  ],
  "geometry_note": "Pitch-corner calibration is required for this video. …"
}
```

`geometry_source` is one of three, and it says *which* measurement the ground
plane came from:

| Source | Meaning |
| --- | --- |
| `per_video_pitch_corners` | pitch corners measured on this video |
| `pitch_keypoints` | the keypoint model's quad, converted to metres (§ below) |
| `template_corner_fractions` | nothing measured on this video — everything below is `null` |

**With no measured geometry, these are `null`:**

`speed_kmh` · `length` · `line` · `swing` · `bounce_angle` ·
`bounce.ground_x_m` · `bounce.ground_y_m` · `mean_speed_kmh`

**`length` is metres from the batting crease**, classified against
`length_zones_m` (coaching-standard thresholds, not values fitted to this
footage): Yorker 0–1.5 m, Full 1.5–4 m, Good Length 4–7 m, Short 7–10 m,
Very Short 10–20.12 m. Outside that range the label is `null`, never the
nearest zone. `line` remains the ported image-space rule — leg and off cannot
be derived from a quad, which says nothing about which way the batter faces.

They are withheld rather than reported as an estimate. An estimate implies the
number is roughly right; a homography fitted to the wrong corners is not roughly
right, it is confidently wrong. A previous version inherited
`geometry_calibrated=True`, which made the suppression branch unreachable and put
a 42–199 km/h range into `result.json`.

What a *served* delivery therefore looks like when uncalibrated: every one of
those fields is `null` and `bowling.calibrated` is `false`. The block itself,
with the scale/offset and the note, stays in `result.json` on disk — the API
serves the decision, not the arithmetic behind it.

**Pixel-space results are unaffected and still computed:** `release_frame`,
`bounce.x`, `bounce.y`, `bowling.release_angle`, and the whole trajectory.

`speed_is_calibrated` is `false` even when geometry *is* supplied, and stays
false deliberately: a least-squares fit over a ball still in the air,
projected onto the ground plane, is an estimate even with perfect corners.
What changes with measured geometry is the hand-tuned `speed_scale` /
`speed_offset_kmh` — those correct a fixed set of template corner fractions,
are NOT applied to geometry measured on this video, and `speed_scale_applies_to`
says so in the same block. The signal for "the ground plane is trustworthy" is
`geometry_calibrated` / `ground_plane_analytics_available`, and on the wire it
is `bowling.calibrated`.

**Two things are called calibration, and they are not the same thing:**

| | Analytics geometry | Pitch keypoints |
| --- | --- | --- |
| Where | `calibration` block in `result.json` | `pitch` on every served delivery, plus the envelope's top-level `pitch` |
| Comes from | operator-measured pitch corners | optional Roboflow detector; inert without `ROBOFLOW_API_KEY` |
| Units | metres on the ground plane | pixels in the frame, 0..1 pitch coordinates for drawing, and metres once the quad is oriented |
| Unlocks | `ground_x_m`, `ground_y_m`, and the fields in `suppressed_when_uncalibrated` | the quad/keypoints drawn over the clip, `pitch_position` for the pitch map — and, when the ball's travel orients the quad, the same ground-plane fields |
| What the client sees | `bowling.calibrated` | `pitch.state`, `keypoints`, `corners`, `homography_available`, `orientation` |

The keypoint detector reports *where the pitch is in the frame*, which on its
own is pixels rather than metres. The detected quad, taken to be the 22-yard
pitch (long axis 20.12 m, short axis 2.64 m, the edge nearest the batter the
batting crease) and oriented by the direction the ball was travelling, IS a
measurement of this video's ground plane — `geometry_source` becomes
`pitch_keypoints` and exactly the same fields are re-enabled. That assumption
is quoted in `geometry_note` whenever it applies (`pitch_keypoint_geometry`
carries the per-delivery counts and the reasons the rest were refused). When a
quad cannot be oriented, nothing is re-enabled: the fields stay `null` and the
reason is on the record.

---

## 7. Media

```http
GET /api/analysis/{analysis_id}/clips/delivery_022.mp4
```

Streams `video/mp4` with HTTP byte-range support (`206`), which is what
`<video>` needs to seek. This is the route `video.clip_url` points at — there is
no static mount, so a client never touches `data/` directly.

- A name that is not `delivery_*.mp4` is `400`, checked before it reaches the
  filesystem.
- A clip that was never written (or an unknown analysis) is `404` with a
  `detail` message.

Overlays are written to `data/runs/{id}/overlays/` as an internal artefact and
are deliberately **not** served: the browser draws its own trajectory overlay
from `trajectory.points`, which is why there is no overlay route to keep in step
with it.

---

## 8. Trajectory data for charting

There is no separate trajectory endpoint. `deliveries[].trajectory` in the
result envelope is the array the overlay renderer drew and the analytics were
fitted to, so a plotted path and a rendered path cannot disagree: one document,
one source.

Points are pixel measurements. With no calibrated homography
`pitch_position` is `null` on every point and `bounce.ground_x_m` /
`bounce.ground_y_m` are `null` — no axis is labelled with a distance that was
never measured.

---

## 9. What was removed, and why

The surface used to carry aliases and diagnostics routes that no single client
needed, each of which had to be kept in step with the real one. All of them are
gone rather than hidden from the docs:

| Removed | Why |
| --- | --- |
| `GET /api/analyses/{id}/…` (plural) | a second spelling of every route; one canonical API |
| `GET /api/analysis/{id}/result/internal` | served the whole `result.json` to a browser: filesystem paths, provenance, model internals |
| `GET /api/analysis/{id}/progress`, `/progress/json` | a duplicate of `/status`, plus an SSE stream nothing consumed |
| `POST /api/analysis/{id}/cancel` | the frontend has no cancel control, and the flag it set was only read after the run finished — it never actually stopped anything |
| `GET /api/analysis/{id}/deliveries…`, `/trajectory/{id}` | one array entry already contains the whole delivery; the SVG plot route went with them |
| `GET /api/analysis/{id}/…/overlay`, `/overlays/{name}` | overlays are drawn client-side from `trajectory.points` |
| `GET /api/analyses/{id}/tracking` | raw tracker output, superseded by reading `result.json` on disk |
| `GET /api/reference` | the enums are already in `app/core/constants` and the frontend hardcodes its own labels; nothing fetched it |
| `GET /api/analyses` | an in-memory job listing no client used |
| static mount at `/data` | exposed the entire runs directory; clips now leave through route 5 |

Anything left that a client could stumble into returns `404`, and the tests
assert the absence (`TestRouteSurface`, `TestRemovedSurfaceStaysRemoved`).

---

## 10. Running it

```powershell
cd backend
python -m uvicorn main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs` list exactly the six routes
above.

Results persist under `data/runs/{analysis_id}/`, so an analysis survives a
restart as far as the filesystem is concerned. The in-memory registry holds only
*progress*, which is transient; `/status` reconstructs a `completed` response
from disk for an analysis this process did not run, and `/result` reads
`result.json` directly.
