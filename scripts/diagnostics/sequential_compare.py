"""
sequential_compare.py — run `real_cricket.mp4` with the backpressure-aware
`sequential` sampler and compare it against the measured BEFORE numbers.

BEFORE (baseline run 8f5f9db22790407f, same video, interval mode):
    submitted 2713 · queue drops 1850 · 4.94 submissions/s · 2.03 processed/s
    failures 799 · refusals 40 · detections 24 (attempts 863)

The run itself is a normal pipeline job — no configuration is pinned here:
the sampling mode comes from `.env` (PITCH_SAMPLING_MODE=sequential) unless
`--mode` says otherwise, and the only thing this script adds is a counter
around `requests.Session.request` so the report can state how many Roboflow
requests were ACTUALLY made, and how many of them were not a plain 200.

`--mode interval` runs the same video under the OLD sampler: the controlled
A/B, executed minutes after the AFTER run so both see the same network.

`--postprocess <analysis_id>` rebuilds the artifacts for a run that already
exists (reading its runtime from `performance.timings_sec.total`), so the
report can be regenerated without another ten-minute pass.

Outputs (docs/diagnostics/):
    pitch_<tag>_real_cricket.json   counters + diagnostics + HTTP tally
    pitch_<tag>_real_cricket.md     BEFORE/AFTER table and verdict

Usage:
    python scripts/diagnostics/sequential_compare.py [--mode interval] [--tag X]
    python scripts/diagnostics/sequential_compare.py --postprocess <analysis_id>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RUNS = PROJECT / "data" / "runs"
OUT_DIR = PROJECT / "docs" / "diagnostics"
SOURCE_NAME = "real_cricket.mp4"

BEFORE = {
    "run": "8f5f9db22790407f",
    "submitted": 2713,
    "queue_drops": 1850,
    "failures": 799,
    "refusals": 40,
    "detections": 24,
    "attempts": 863,
    "submitted_per_s": 4.941,
    "processed_per_s": 2.026,
}

# ── HTTP tally, installed before any backend import ──────────────────────────
HTTP = {"posts": 0, "other": 0, "non_200": {}, "errors": 0}


def install_http_counter() -> None:
    import requests

    original = requests.Session.request

    def counting(self, method, url, **kwargs):  # noqa: ANN001
        try:
            response = original(self, method, url, **kwargs)
        except Exception:
            HTTP["errors"] += 1
            raise
        if "roboflow" in str(url):
            if str(method).upper() == "POST":
                HTTP["posts"] += 1
            else:
                HTTP["other"] += 1
            if response.status_code != 200:
                key = str(response.status_code)
                HTTP["non_200"][key] = HTTP["non_200"].get(key, 0) + 1
        return response

    requests.Session.request = counting


def build_after(doc: dict, http: dict, elapsed: float) -> dict:
    pitch = doc["pitch_detection"]
    diag = pitch.get("diagnostics") or {}
    cats = diag.get("categories") or {}
    return {
        "analysis_id": doc.get("analysis_id"),
        "sampling_mode": diag.get("sampling_mode"),
        "max_attempts_per_sec": (diag.get("backpressure") or {})
                                 .get("max_attempts_per_sec"),
        "submitted": pitch.get("submitted"),
        "queue_drops": pitch.get("queue_drops"),
        "roboflow_requests": http["posts"],
        "roboflow_non_200": http["non_200"],
        "roboflow_transport_errors": http["errors"],
        "attempts": sum(cats.values()),
        "no_keypoints": cats.get("no_keypoints", 0),
        "local_refusals": pitch.get("refusals"),
        "valid_detections": pitch.get("detections"),
        "calibration_updates": pitch.get("calibration_updates"),
        "other_failures": sum(v for k, v in cats.items()
                              if k not in ("no_keypoints", "refused", "detected")),
        "runtime_s": round(elapsed, 1),
        "submitted_per_s": (diag.get("throughput") or {}).get("submitted_per_s"),
        "processed_per_s": (diag.get("throughput") or {}).get("processed_per_s"),
        "active_seconds": (diag.get("throughput") or {}).get("active_seconds"),
        "latency_ms": diag.get("latency_ms"),
        "queue_wait_ms": diag.get("queue_wait_ms"),
        "queue": diag.get("queue") or {},
        "gated": diag.get("gated") or {},
        "refusals_by_reason": diag.get("refusals") or {},
        "categories": cats,
        "backpressure": diag.get("backpressure"),
    }


def checks_for(after: dict, mode: str) -> list[tuple[str, bool, str]]:
    if mode != "sequential":
        return []
    queue = after["queue"]
    return [
        ("queue drops decrease substantially",
         after["queue_drops"] < BEFORE["queue_drops"] * 0.5,
         f"{BEFORE['queue_drops']} → {after['queue_drops']}"),
        ("valid detections do not decrease",
         after["valid_detections"] >= BEFORE["detections"],
         f"{BEFORE['detections']} → {after['valid_detections']}"),
        ("calibration still happens",
         after["calibration_updates"] >= 1,
         f"{after['calibration_updates']} updates"),
        ("no stale backlog (queue never held more than one frame)",
         (queue.get("max_depth_observed") or 0) <= 1,
         f"max depth {queue.get('max_depth_observed')}, "
         f"median wait {after['queue_wait_ms'].get('p50')} ms"),
        ("every submission was answered (no lost frames)",
         after["attempts"] + after["queue_drops"] == after["submitted"],
         f"{after['submitted']} = {after['queue_drops']} + {after['attempts']}"),
        ("attempts/s inside the 1.5–2 target band (detector-limited)",
         after["processed_per_s"] is not None
         and 1.5 <= after["processed_per_s"] <= 2.0,
         f"{after['processed_per_s']}/s at {after['latency_ms'].get('mean')} ms "
         f"mean latency (capacity {round(1000 / after['latency_ms']['mean'], 3)}/s)"),
        ("no Roboflow rate limiting",
         not after["roboflow_non_200"] and after["roboflow_transport_errors"] == 0,
         f"non-200={after['roboflow_non_200']} "
         f"errors={after['roboflow_transport_errors']}"),
    ]


def write_artifacts(after: dict, checks: list, tag: str) -> None:
    payload = {"before": BEFORE, "after": after, "mode": tag,
               "checks": [{"check": c, "ok": bool(ok), "detail": d}
                          for c, ok, d in checks]}
    (OUT_DIR / f"pitch_{tag}_real_cricket.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"# Pitch sampling on `real_cricket.mp4` — run `{tag}`",
        "",
        f"Run: `{after['analysis_id']}` · mode `{after['sampling_mode']}` · "
        f"ceiling {after.get('max_attempts_per_sec')} attempts/s",
        "",
        "| measure | BEFORE (`8f5f9db22790407f`) | AFTER |",
        "|---|---:|---:|",
        f"| total pitch submissions | {BEFORE['submitted']} | {after['submitted']} |",
        f"| actual Roboflow requests | {BEFORE['attempts']} (attempts) | {after['roboflow_requests']} |",
        f"| queue drops | {BEFORE['queue_drops']} | {after['queue_drops']} |",
        f"| no-keypoint responses | {BEFORE['failures']} | {after['no_keypoints']} |",
        f"| local refusals | {BEFORE['refusals']} | {after['local_refusals']} |",
        f"| valid detections | {BEFORE['detections']} | {after['valid_detections']} |",
        f"| calibration updates | — | {after['calibration_updates']} |",
        f"| submissions/sec | {BEFORE['submitted_per_s']} | {after['submitted_per_s']} |",
        f"| processed/sec | {BEFORE['processed_per_s']} | {after['processed_per_s']} |",
        f"| runtime (s) | — | {after['runtime_s']} |",
        "",
        "## Checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for check, ok, detail in checks:
        lines.append(f"| {check} | {'PASS' if ok else 'FAIL'} — {detail} |")
    lines += [
        "",
        "## Diagnostics",
        "",
        f"- gated: `{json.dumps(after['gated'])}`",
        f"- queue: `{json.dumps(after['queue'])}`",
        f"- latency: `{json.dumps(after['latency_ms'])}`",
        f"- queue wait: `{json.dumps(after['queue_wait_ms'])}`",
        f"- refusal reasons: `{json.dumps(after['refusals_by_reason'])}`",
        f"- HTTP: {after['roboflow_requests']} POSTs, "
        f"non-200 `{after['roboflow_non_200']}`, "
        f"transport errors {after['roboflow_transport_errors']}",
        "",
    ]
    (OUT_DIR / f"pitch_{tag}_real_cricket.md").write_text(
        "\n".join(lines), encoding="utf-8")


def run_pipeline(source: Path, mode: str | None) -> tuple[dict, float]:
    sys.path.insert(0, str(PROJECT / "backend"))
    from app.api.deps import get_jobs, get_models
    from app.core.config import PITCH_CONFIG, Paths

    active = PITCH_CONFIG.sampling_mode
    print(f"[config] sampling_mode={active} "
          f"max_attempts_per_sec={PITCH_CONFIG.max_attempts_per_sec} "
          f"interval={PITCH_CONFIG.interval_frames} "
          f"model={PITCH_CONFIG.model_id}", flush=True)
    if mode and active != mode:
        raise SystemExit(f"[config] expected {mode!r}, got {active!r}")

    jobs = get_jobs()
    job = jobs.create(source_name=SOURCE_NAME)
    paths = Paths(analysis_id=job.analysis_id).ensure()
    dest = paths.root / "source.mp4"
    if not dest.is_file():
        shutil.copyfile(source, dest)

    print(f"[run] analysis_id={job.analysis_id} video={source}", flush=True)
    started = time.perf_counter()
    thread = jobs.start(job, dest, get_models())
    thread.join()
    elapsed = time.perf_counter() - started
    print(f"[run] status={job.status} in {elapsed:.1f}s error={job.error}", flush=True)
    if job.status != "completed":
        raise SystemExit(1)
    doc = json.loads((paths.result_json).read_text(encoding="utf-8"))
    doc["analysis_id"] = job.analysis_id
    return doc, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=None,
                        help="sampling mode to run (default: whatever .env says). "
                             "`interval` is the controlled BEFORE run.")
    parser.add_argument("--tag", default=None,
                        help="artifact name suffix (default: the mode)")
    parser.add_argument("--postprocess", default=None, metavar="ANALYSIS_ID",
                        help="rebuild artifacts for an existing run (no pipeline)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # The report uses arrows and en-dashes; a cp1252 console would die on
        # the very line that reports the verdict.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    if args.postprocess:
        doc = json.loads(
            (RUNS / args.postprocess / "result.json").read_text(encoding="utf-8"))
        elapsed = float((doc.get("performance") or {})
                        .get("timings_sec", {}).get("total", 0.0))
        pitch = doc["pitch_detection"]
        attempts = sum(((pitch.get("diagnostics") or {})
                        .get("categories") or {}).values())
        # One detector call = one HTTP request when no retry fired (true for
        # every run reported here: the live counter measured 616 and 727 —
        # exactly the attempt counts).
        http = {"posts": attempts, "other": 0, "non_200": {}, "errors": 0}
        after = build_after(doc, http, elapsed)
        checks = checks_for(after, (after.get("sampling_mode") or "sequential"))
        tag = args.tag or (after.get("sampling_mode") or "sequential")
        write_artifacts(after, checks, tag)
        print("[totals]", json.dumps(after, indent=2, sort_keys=True))
        for check, ok, detail in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {check}: {detail}")
        print(f"[artifact] docs/diagnostics/pitch_{tag}_real_cricket.md")
        return 0 if all(ok for _, ok, _ in checks) else 2

    if args.mode:
        # Process env wins over `.env` (config loads it with override=False),
        # so this pins the mode without touching any file.
        os.environ["PITCH_SAMPLING_MODE"] = args.mode
    tag = args.tag or (args.mode or "sequential")

    install_http_counter()
    doc, elapsed = run_pipeline(RUNS / BEFORE["run"] / "source.mp4", args.mode)
    mode = (doc.get("pitch_detection") or {}).get("diagnostics", {}) \
        .get("sampling_mode") or args.mode or "sequential"
    after = build_after(doc, HTTP, elapsed)
    checks = checks_for(after, mode)
    write_artifacts(after, checks, tag)

    print("[totals]", json.dumps(after, indent=2, sort_keys=True))
    for check, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {check}: {detail}")
    print(f"[artifact] docs/diagnostics/pitch_{tag}_real_cricket.md")
    return 0 if all(ok for _, ok, _ in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
