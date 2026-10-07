# Pitch sampling on `real_cricket.mp4` — run `interval`

Run: `f4ca495c32404e9c` · mode `interval` · ceiling 2.0 attempts/s

| measure | BEFORE (`8f5f9db22790407f`) | AFTER |
|---|---:|---:|
| total pitch submissions | 2713 | 2713 |
| actual Roboflow requests | 863 (attempts) | 727 |
| queue drops | 1850 | 1986 |
| no-keypoint responses | 799 | 674 |
| local refusals | 40 | 29 |
| valid detections | 24 | 24 |
| calibration updates | — | 24 |
| submissions/sec | 4.941 | 4.6 |
| processed/sec | 2.026 | 1.227 |
| runtime (s) | — | 700.4 |

## Checks

| check | result |
|---|---|

## Diagnostics

- gated: `{"spacing": 10852}`
- queue: `{"capacity": 2, "max_depth_observed": 2, "evictions": 1986}`
- latency: `{"n": 727, "mean": 806.01, "p50": 714.14, "p95": 1561.93, "max": 4414.36}`
- queue wait: `{"n": 727, "mean": 349.76, "p50": 297.94, "p95": 788.37, "max": 2726.77}`
- refusal reasons: `{"keypoint_hull_too_small": 15, "only_2_usable_keypoints": 9, "only_0_usable_keypoints": 4, "only_1_usable_keypoints": 1}`
- HTTP: 727 POSTs, non-200 `{}`, transport errors 0
