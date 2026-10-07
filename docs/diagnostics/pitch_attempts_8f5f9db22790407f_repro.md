# Pitch detection diagnosis — baseline `8f5f9db22790407f`

The baseline run predates the attempt log, so its per-frame telemetry is
gone: only the five counters below survive in `result.json`. This report
reproduces THAT video (`data/runs/8f5f9db22790407f/source.mp4`, 13565 frames
at 25 fps) through THAT configuration with the pipeline unmodified, and
instruments the running process. Reproduced run: `e991206283574d2d`.

## Configuration actually reproduced

| field | baseline (recorded) | reproduction |
|---|---|---|
| model_id | `cricketpitchkeypointdetections/3` | `cricketpitchkeypointdetections/3` |
| interval_frames | 5 | 5 |
| sampling_mode | (not recorded — predates the field; behaviour is interval) | interval |
| min_confidence | 0.5 | 0.5 |
| min_keypoints | 3 | 3 |
| min_area_fraction | 0.01 | 0.01 |
| request_timeout_seconds | 15.0 | 15.0 |
| max_input_size | 1024 | 1024 |
| queue capacity | (not recorded) | 2 |

## Totals

### Baseline (as recorded — this is the target)

| counter | value |
|---|---:|
| SUBMITTED | 2713 |
| QUEUE DROPS | 1850 |
| FAILURES | 799 |
| REFUSALS | 40 |
| DETECTIONS | 24 |
| attempts (= failures + refusals + detections) | 863 |

Identity: `2713 = 1850 + 863` → `1850 + 799 + 40 + 24` = 2713 ✓ (exact).

### Reproduction (`e991206283574d2d`) — per-frame telemetry

| counter | value | share of SUBMITTED |
|---|---:|---:|
| SUBMITTED (frames that entered the queue) | 2713 | 100.0% |
| QUEUE DROPS (evicted before inference) | 1599 | 58.9% |
| PROCESSED ATTEMPTS | 1114 | 41.1% |
| Roboflow HTTP requests (inference POSTs) | 1114 | — |
| HTTP 429 | 0 | — |
| QUOTA / RATE LIMIT | 0 | — |
| TIMEOUTS | 0 | — |
| NETWORK FAILURES | 0 | — |
| NO KEYPOINTS (model answer) | 1035 | 38.1% |
| LOCAL VALIDATION REFUSALS | 52 | 1.9% |
| VALID DETECTIONS | 27 | 1.0% |
| OTHER FAILURES | 0 | — |
| LEFT IN QUEUE AT END | 0 | — |

Reconciliation: 2713 = 1599 + 1114, and 1114 = 1035 + 52 + 27 = 1114 ✓.

### Why the reproduction's split differs from the baseline's

- **SUBMITTED is identical (2713)**: it is deterministic — the gate is
  `interval` mode, every 5th frame of 13565, and both runs took that path
  (gated by spacing: 10852 frames).
- **QUEUE DROPS vs PROCESSED is a machine-load split, not a config one**:
  the queue holds 2 frames and the worker takes ~0.45–1.3 s per request.
  Any frame arriving while the queue is full is evicted (latest wins).
  The baseline dropped 1850/2713; this reproduction dropped
  1599/2713 because the model answered faster on this run
  (mean 476.14 ms vs the 1334 ms mean of instrumented run
  `2275118ad45d410c`). Both are the same mechanism.
- **Attempt outcomes are a model-content split**: detections/refusals/
  no-keypoints depend on what the model answers for a given frame, which
  is why five same-config runs of this video disagree:
  24/18/28/23/15 detections and 799/862/1053/1008/862 failures.

## Same-configuration runs of the same video (evidence set)

| analysis_id | submitted | detections | refusals | failures | queue drops | attempts | sum check |
|---|---:|---:|---:|---:|---:|---:|---|
| `8f5f9db22790407f` (baseline) | 2713 | 24 | 40 | 799 | 1850 | 863 | 1850+863=2713 ✓ |
| `894756f17d3f41a8` | 2713 | 18 | 46 | 862 | 1787 | 926 | 1787+926=2713 ✓ |
| `c4c1e3a5da9f4c60` | 2713 | 28 | 52 | 1053 | 1580 | 1133 | 1580+1133=2713 ✓ |
| `13266b54d6264fcd` | 2713 | 23 | 61 | 1008 | 1621 | 1092 | 1621+1092=2713 ✓ |
| `3b9c4d7632ca464a` | 2713 | 15 | 46 | 862 | 1789 | 923 | 1789+923=2712 (one residual in queue) |
| `2275118ad45d410c` (instrumented) | 2713 | 10 | 15 | 379 | 2309 | 404 | 2309+404=2713 ✓ |
| `e991206283574d2d` (this report) | 2713 | 27 | 52 | 1035 | 1599 | 1114 | 1599+1114=2713 ✓ |

Every one of them submits exactly 2713 frames and then splits them
between queue drops and attempts according to how fast the model answers
that day. The five counters are therefore one pool partitioned two ways,
and 2713 = queue drops + attempts holds in each run.

## Request rate, latency, concurrency

- submissions: **4.941/s** (every 5th frame of a 25 fps video = 5 frames/s offered → 5 submissions/s), gate never idle
- processed: **2.026/s** over 549.454 s
- inference latency: mean **476.14 ms**, p50 454.36 ms, p95 607.56 ms, max 1491.62 ms (n=1114)
- queue wait: mean 286.25 ms, p50 275.47 ms, p95 449.47 ms, max 1674.25 ms
- concurrency: **1** request in flight (single worker thread, queue capacity 2)
- offered frames: 13565, gated by `spacing`: 10852

## HTTP evidence

- status histogram: {"200": 1114} — **all 1114 responses are 200**
- HTTP 429: **0**
- quota/rate-limit messages: **0**
- timeouts: **0** (service category `timeout`), transport timeout errors: 0
- network failures: **0**, transport network errors: 0
- retries observed (more than one HTTP request for one detector call): **0**

The SDK does retry 429/503/504 (`inference_sdk.http.utils.executors.
RETRYABLE_STATUS_CODES = {429, 503, 504}`), so a rate limit would show up
either as a non-200 status in this histogram or as `rate_limited` in the
service's failure categories. Neither appears.

## Attempt outcomes (reproduction)

| outcome | attempts |
|---|---:|
| no_keypoints | 1035 |
| refused | 52 |
| detected | 27 |

### Local validation refusals (keypoints came back, validator said no)

| reason | attempts |
|---|---:|
| keypoint_hull_too_small | 29 |
| only_2_usable_keypoints | 15 |
| only_0_usable_keypoints | 4 |
| only_1_usable_keypoints | 4 |

### Calibration state at the attempts

| state | attempts |
|---|---:|
| calibration_expired | 1060 |
| calibrated | 29 |
| temporary_loss | 16 |
| no_calibration | 9 |

## One clearly visible pitch frame, in isolation

- frame: **341** at 00:13.640 ([1280, 720])
- submitted? **yes** — one direct detector call, HTTP [200] in [1763.48] ms
- Roboflow response: **ok**, 1 request(s), no error
- keypoints: **4** (confidence 0.8002): `{"bottomleftwidepoint": [561.0, 571.0, 0.9227], "bottomrightwidepoint": [811.0, 570.0, 0.9988], "topleftwidepoint": [593.0, 265.0, 0.9696], "toprightwidepoint": [792.0, 261.0, 0.9975]}`
- local validation: **accepted** (min_keypoints=3, min_confidence=0.5, min_area_fraction=0.01)
- final result: **calibration built (pitch drawn)**

→ Roboflow detects this pitch and local validation accepts it. Local
validation is not the blocker on a frame where the model can see the pitch.

## Root cause

Ranked by how much of the 2713-frame pool each one costs:

1. **D — excessive frame submission** (structural, 2713 frames): the gate
   offers every 5th frame unconditionally, ~4.9/s, for a video that is
   mostly not showing a pitch.
2. **E — queue eviction** (1599 of 2713 in this run,
   1850 in the baseline): a queue of 2 cannot absorb
   4.941/s arrivals against 2.026/s service. This is where the headline
   `queue_drops` number comes from, and it is a direct consequence of (1).
3. **B — model returns no keypoints** (1035 of the
   1114 attempts actually run = 92.9%):
   every one of those answers is HTTP 200 with an empty prediction set —
   the scene, not the transport.
4. **C — local validation** (52 refusals =
   4.7% of attempts): real but small; the isolated clear-pitch frame passes.
5. **A — Roboflow rate limiting: NOT present** — 0 × HTTP 429, 0 quota
   messages, 0 retries, 0 `rate_limited` verdicts out of 1114 requests.
6. **F — timeout/network: NOT present** — 0 timeouts, 0 network errors.

So the dominant root cause is **G in the sense of D+E being one mechanism**
(submit far more frames than the single-threaded queue can drain), with B
as the second bottleneck among the frames that do get an answer. A and F
are ruled out by measurement.

## Artifacts

- `docs/diagnostics/pitch_attempts_8f5f9db22790407f_repro.csv` — one row per
  submitted frame with every field requested (frame, video timestamp,
  submission time, queue entry/wait, dropped + drop reason, HTTP status,
  transport error, keypoints returned, keypoint count and confidence,
  local validation verdict, refusal reason, calibration state, inference
  latency).
- `docs/diagnostics/pitch_totals_8f5f9db22790407f_repro.json` — machine totals.
- `docs/diagnostics/pitch_single_frame_8f5f9db22790407f.json` — the isolated
  clear-pitch frame.

Reproduced run directory: `data/runs/e991206283574d2d/` (the run's own `pitch_detection.diagnostics` has the same attempts).
