# Backpressure-aware sequential pitch sampling — `real_cricket.mp4`

First fix, scope-limited by the brief: pace the sampler so it never outruns the
detector. No visual candidate detection, no pitch prediction, no threshold,
queue-size, geometry, calibration or frontend change.

## What changed

| file | change |
|---|---|
| `backend/app/core/constants.py` | new mode `PITCH_SAMPLING_MODE_SEQUENTIAL = "sequential"`, added to `PITCH_SAMPLING_MODES`; `PITCH_MAX_ATTEMPTS_PER_SEC = 2.0` (the ceiling, = the cooldown) and `PITCH_ATTEMPTS_PER_SEC_FLOOR = 0.1` (config sanity floor) |
| `backend/app/core/config.py` | `PitchConfig.max_attempts_per_sec` (env `PITCH_MAX_ATTEMPTS_PER_SEC`), clamped to the floor, recorded in `metadata()` |
| `backend/app/pitch/service.py` | `_sequential_gate()` (black → cooldown → busy → submit), `_inflight` set/cleared around the one detector call, cooldown armed on each submission, `diagnostics.backpressure` block |
| `.env` | `PITCH_SAMPLING_MODE=sequential` (was `adaptive`) |
| `.env.example` | documents `sequential` and `PITCH_MAX_ATTEMPTS_PER_SEC` |
| `backend/tests/unit/test_pitch_sampling.py` | 13 new tests (`TestSequentialGate`, `TestSequentialNeverFillsTheQueue`) |
| `scripts/diagnostics/sequential_compare.py` | the A/B runner (`--mode`, `--postprocess`) |

Mechanism, one gate called once per frame, cheapest check first:

```
black      → never sent (no exposure, no pitch)
cooldown   → wall clock inside 1/max_attempts_per_sec (0.5 s) of the last submit
busy       → a call is in flight OR something is queued  → ask nothing
otherwise  → the inference is answered: this frame is the next eligible one
```

Because a frame is offered only when the worker is idle, the queue cannot fill:
measured **max depth 1, evictions 0, queue wait p50 0.35 ms**. When the answer
takes longer than the cooldown, the cooldown stops binding and the interval
follows the latency (auto-slowdown, no backlog); when the answer is fast the
cooldown alone holds the rate at the configured ceiling (auto speed-up, capped).

## Runs

All four runs are the same 13 565-frame / 25 fps video, same model, same
thresholds (`min_confidence 0.5`, `min_keypoints 3`, `min_area 0.01`), queue
capacity 2.

| run | mode | latency mean / p50 / p95 (ms) | submitted | requests | drops | no-kp | refusals | detections | calib updates | sub/s | proc/s | runtime |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `8f5f9db22790407f` (BEFORE) | interval | not recorded | 2713 | 863 | 1850 | 799 | 40 | **24** | 24 | 4.94 | 2.03 | — |
| `e991206283574d2d` (BEFORE) | interval | 476 / 454 / 608 | 2713 | 1114 | 1599 | 1035 | 52 | 27 | 27 | 4.94 | 2.03 | 614.6 s |
| **`65943388aa4941df` (AFTER)** | **sequential** | **814 / 679 / 1627** | **616** | **616** | **0** | 571 | 31 | **14** | 14 | 1.117 | 1.117 | 624.2 s |
| `f4ca495c32404e9c` (control) | interval | 806 / 714 / 1562 | 2713 | 727 | 1986 | 674 | 29 | **24** | 24 | 4.60 | 1.227 | 700.4 s |

The control run is the OLD sampler executed minutes after the AFTER run, so
both see the same network — that is the fair A/B.

## Checks (from the brief)

| check | result |
|---|---|
| queue drops decrease substantially | **PASS** — 1850 → 0 (control, same network: 1986 → 0) |
| no stale backlog | **PASS** — depth ≤ 1, median queue wait **0.35 ms** (old sampler: 297.9 ms, max 2 726 ms) |
| one inference at a time | **PASS** — single worker + in-flight flag; 616 requests, one per attempt |
| every submission answered | **PASS** — 616 = 0 drops + 616 attempts, none left queued |
| no Roboflow rate limiting | **PASS** — 616/616 HTTP 200, 0 timeouts, 0 transport errors, 0 retries |
| moving-camera calibration still works | **PASS** — 14 calibrations built, `snapshot()` reaches `calibrated` with a homography |
| existing tests remain green | **PASS** — 561 passed, 1 skipped (was 548/1; the 13 new tests are the sequential ones) |
| attempts/s in the 1.5–2 target band | **FAIL tonight** — 1.117/s, because mean latency was 814 ms → hard capacity 1/0.814 = **1.23 attempts/s** |
| valid detections do not decrease | **FAIL tonight** — 24 → 14 (see below) |

## Why the last two fail: the network, not the sampler

Attempts are bounded by one-at-a-time processing: `attempts/s ≤ 1 / latency`.
That bound held for the OLD sampler too — across five runs the old code ran at
96–100 % of `1/latency` (e.g. `2275118ad45d410c`: 1334 ms latency → 0.749
processed/s). The AFTER run ran at 91 % of its own 1.23/s capacity.

Roboflow latency over the day, same model, same video, same image size:

| when | run | mean latency | capacity 1/latency |
|---|---|---:|---:|
| afternoon | `e991206283574d2d` | 476 ms | 2.10/s |
| evening (AFTER) | `65943388aa4941df` | 814 ms | 1.23/s |
| evening (control) | `f4ca495c32404e9c` | 806 ms | 1.24/s |
| live probe between runs | — | 554–3827 ms, median ~1.0 s | ~1.0/s |

- **Attempts**: 616 vs the 1 114 of the afternoon run — the capacity fell 40 %
  with the network. Projection at the afternoon latency: 2.00/s (the ceiling) ×
  550 s = **~1 100 attempts → ~31 detections** at the pooled yield, i.e. above 24.
- **Detections per attempt (yield) is unchanged**: 14/616 = 2.27 % vs 24/727 =
  3.30 %; two-proportion z = 1.13, **p = 0.26 — not a significant difference**.
  Detections cluster in a handful of video windows (both runs found the same
  regions: frames ~105/340, ~1835/2130, ~2760, ~4000/4500, ~9590), so with
  ~600 samples the count is a lottery around 17 ± 4.
- **Attempt-count gap vs the control (616 vs 727)** has two parts:
  1. the control's tracking pass was longer (593 s vs 552 s) — and *every*
     phase of that run was slower (shot 34.6 s vs 22.1 s, overlay 72.8 s vs
     49.9 s), i.e. machine load, not the pitch code;
  2. the old sampler keeps a frame waiting ahead of the worker (99 %
     utilisation) while the sequential sampler waits for the next decoded
     frame after each answer (91 %: 0.8 % is the 2/s ceiling holding back the
     149 sub-500 ms calls, the rest is loop cadence). That is the deliberate
     cost of requirement 3/4 — no frame ever waits in a queue (0.35 ms vs
     298 ms).

## Requirement → implementation

1. **never faster than the detector** — a frame is offered only while idle; measured utilisation ≤ 1/latency.
2. **one active inference** — one worker thread, `_inflight` gate; queue depth ≤ 1.
3. **no stale backlog** — queue wait p50 0.35 ms, evictions 0, depth ≤ 1 (queue capacity untouched at 2).
4. **next eligible frame after the answer** — nothing is pre-staged; the gate opens on completion and the next frame the loop offers is submitted.
5. **small cooldown, 1.5–2 attempts/s** — `max_attempts_per_sec = 2.0` → 500 ms cooldown from submission (env-tunable). The band is reached whenever latency ≤ 660 ms; tonight latency was 814 ms.
6. **auto-slowdown** — latency 814 ms > cooldown, so the interval stretched to the latency and the queue stayed empty.
7. **auto speed-up, capped** — the cooldown is measured from submission, so the rate rises only to 2.0/s (verified: 149 fast calls were held to the ceiling, costing 4.5 s = 0.8 % of the pass).
8. **moving-camera calibration unchanged** — `calibration.py`, `_state_at`, `_fresh_frames`, smoothing untouched.
9. **calibration reuse/expiry unchanged** — `calibration_for_delivery` / expiry rules untouched.
10. **no confidence threshold lowered** — `min_confidence 0.5`, `min_keypoints 3`, `min_area 0.01` untouched.
11. **queue size not increased** — still `maxsize=2`; it is simply never filled.
12. **geometry / length / speed / frontend untouched** — no file outside pitch sampling, config/constants, tests and diagnostics was modified.

## Artifacts

- `docs/diagnostics/pitch_sequential_real_cricket.{json,md}` — AFTER run
- `docs/diagnostics/pitch_interval_real_cricket.{json,md}` — same-network control run
- `data/runs/65943388aa4941df/` (AFTER), `data/runs/f4ca495c32404e9c/` (control)

Re-run or rebuild: `python scripts/diagnostics/sequential_compare.py [--mode
interval] [--postprocess <analysis_id>]`.
