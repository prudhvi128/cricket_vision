# Pitch sampling on `real_cricket.mp4` — run `sequential`

Run: `65943388aa4941df` · mode `sequential` · ceiling 2.0 attempts/s

| measure | BEFORE (`8f5f9db22790407f`) | AFTER |
|---|---:|---:|
| total pitch submissions | 2713 | 616 |
| actual Roboflow requests | 863 (attempts) | 616 |
| queue drops | 1850 | 0 |
| no-keypoint responses | 799 | 571 |
| local refusals | 40 | 31 |
| valid detections | 24 | 14 |
| calibration updates | — | 14 |
| submissions/sec | 4.941 | 1.117 |
| processed/sec | 2.026 | 1.117 |
| runtime (s) | — | 624.2 |

## Checks

| check | result |
|---|---|
| queue drops decrease substantially | PASS — 1850 → 0 |
| valid detections do not decrease | FAIL — 24 → 14 |
| calibration still happens | PASS — 14 updates |
| no stale backlog (queue never held more than one frame) | PASS — max depth 1, median wait 0.35 ms |
| every submission was answered (no lost frames) | PASS — 616 = 0 + 616 |
| attempts/s inside the 1.5–2 target band (detector-limited) | FAIL — 1.117/s at 814.32 ms mean latency (capacity 1.228/s) |
| no Roboflow rate limiting | PASS — non-200={} errors=0 |

## Diagnostics

- gated: `{"cooldown": 8013, "busy": 4936}`
- queue: `{"capacity": 2, "max_depth_observed": 1, "evictions": 0}`
- latency: `{"n": 616, "mean": 814.32, "p50": 679.46, "p95": 1626.8, "max": 4146.27}`
- queue wait: `{"n": 616, "mean": 0.37, "p50": 0.35, "p95": 0.53, "max": 0.83}`
- refusal reasons: `{"keypoint_hull_too_small": 16, "only_1_usable_keypoints": 6, "only_2_usable_keypoints": 5, "only_0_usable_keypoints": 4}`
- HTTP: 616 POSTs, non-200 `{}`, transport errors 0
