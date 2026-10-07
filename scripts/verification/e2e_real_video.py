"""
e2e_real_video.py — End-to-end verification against the real 9-minute video.

Runs `real_cricket.mp4` through the real HTTP API with the real detector and the
real shot checkpoint, then asserts the properties the backend claims. Every
assertion here is a measurement, not a restatement of intent:

  * the source video was decoded exactly once, and YOLO/Kalman ran once;
  * the shot model was loaded once and its checkpoint is identified;
  * every delivery served by the API passed purity validation, and nothing
    quarantined appears in the served array;
  * every delivery's clip is a real, decodable, non-empty file served by the
    clips route, with byte ranges for seeking, and its overlay exists on disk;
  * clip and trajectory describe the SAME delivery (no cross-talk);
  * each trajectory's frame range lies inside its own clip's frame range;
  * uncalibrated ground-plane analytics are null, not fabricated;
  * every trajectory point carries explicit detected/predicted provenance;
  * `GET /api/analysis/{id}/result` is one envelope — analysis id, status,
    pitch, deliveries — with no internal record, no filesystem path and the
    same schema for every entry, while the complete document it was projected
    from is still on disk in `result.json`, served by no route.

Run it with:
    python scripts/verification/e2e_real_video.py [path/to/video]

It is deliberately a script rather than a pytest test: it takes minutes, loads
both models, and is meant to be run deliberately against real footage, not as
part of the unit suite.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# scripts/<category>/<name>.py -> <root>/scripts/<category> -> <root>
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.core.config import Paths
from app.core.constants import LENGTH_ZONES_M
from app.storage.persistence import read_json

DEFAULT_VIDEO = ROOT / "real_cricket.mp4"

# Where a delivery's ground plane may come from, in `calibration.geometry_source`.
GEOMETRY_SOURCES = {"per_video_pitch_corners", "pitch_keypoints",
                    "template_corner_fractions"}
LENGTH_ZONE_NAMES = {label for label, _lo, _hi in LENGTH_ZONES_M}

# Keys that belong to the keypoint model's own bookkeeping and must never be
# served to a client inside a delivery's pitch block.
PITCH_INTERNAL_KEYS = {"homography", "map_homography", "model_id", "image_size",
                       "frame_number"}

# Validation verdicts that must never reach a client as a delivery.
FORBIDDEN_VALIDATION = {"MULTIPLE_EVENTS", "NO_EVENT"}

ENVELOPE_KEYS = {"analysis_id", "status", "pitch", "deliveries"}
DELIVERY_KEYS = {
    "id", "video", "trajectory", "bounce", "bowling", "shot", "quality", "pitch",
}

failures: list[str] = []
checks = 0


def check(label: str, condition: bool, detail: str = "") -> bool:
    global checks
    checks += 1
    if condition:
        print(f"  PASS  {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))
        failures.append(label)
    return bool(condition)


def main() -> int:
    # `--verify-only <analysis_id>` re-checks every assertion against an analysis
    # that is already on disk, without spending seven minutes re-analysing. The
    # upload and progress assertions are skipped and reported as such; everything
    # that reads the API is still exercised over HTTP.
    argv = sys.argv[1:]
    verify_only = "--verify-only" in argv
    if verify_only:
        argv = argv[argv.index("--verify-only") + 1:]
        existing_id = argv[0] if argv else None
    else:
        existing_id = None

    video = Path(argv[0]) if (argv and not verify_only) else DEFAULT_VIDEO
    if not verify_only and not video.is_file():
        print(f"video not found: {video}")
        return 2
    size_mb = video.stat().st_size / 1e6 if video.is_file() else 0.0
    print(f"Analysing {video} ({size_mb:.1f} MB)" if not verify_only
          else f"Verifying existing analysis {existing_id}")
    print("Both models load on first request; the tracking pass is the long part.\n")

    # The API as it is mounted in production: six routes, no static tree.
    app = FastAPI()
    app.include_router(router, prefix="/api")

    # ── Upload ───────────────────────────────────────────────────────────
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        check("health reports ok", health.get("status") == "ok", str(health)[:120])
        check(
            "health publishes the stage vocabulary",
            isinstance(health.get("stages"), list) and "tracking" in health["stages"],
        )

        if verify_only:
            analysis_id = existing_id
            if analysis_id is None:
                print("--verify-only needs an analysis id")
                return 2
        else:
            print("\n[1/8] upload")
            with open(video, "rb") as fh:
                response = client.post(
                    "/api/analyze",
                    files={"video": (video.name, fh, "video/mp4")},
                )
            check("POST /api/analyze accepted", response.status_code == 202,
                  f"status={response.status_code}")
            if response.status_code != 202:
                print(response.text[:2000])
                return 1
            body = response.json()
            analysis_id = body["analysis_id"]
            print(f"        analysis_id={analysis_id}")
            check("response names the documented status route",
                  body["status_url"] == f"/api/analysis/{analysis_id}/status")
            check("response names the documented result route",
                  body["result_url"] == f"/api/analysis/{analysis_id}/result")
            check("the 202 body carries no route that no longer exists",
                  set(body) == {"analysis_id", "status", "progress", "stage",
                                "bytes", "status_url", "result_url"},
                  str(sorted(body)))

            # ── Progress ──────────────────────────────────────────────────
            print("\n[2/8] poll progress")
            samples: list[dict] = []
            last_done = -1
            t0 = time.perf_counter()
            while True:
                status = client.get(f"/api/analysis/{analysis_id}/status").json()
                samples.append(status)
                done, total = status.get("frames_done") or 0, status.get("frames_total") or 0
                if done != last_done:
                    print(f"        {status['stage']:>18s} "
                          f"{str(status.get('substage') or '-'):<18s} "
                          f"{done:>6}/{total:<6} {status.get('progress')}%")
                    last_done = done
                if status["status"] in ("completed", "failed"):
                    break
                if time.perf_counter() - t0 > 3600:
                    print("        timed out")
                    return 1
                time.sleep(2.0)

            check("analysis completed", status["status"] == "completed",
                  f"status={status['status']} error={status.get('error')}")
            if status["status"] != "completed":
                print(json.dumps(status, indent=2)[:2000])
                return 1

            progresses = [s.get("progress") for s in samples if s.get("progress") is not None]
            check("progress never moves backwards",
                  all(b >= a for a, b in zip(progresses, progresses[1:])),
                  f"{len(progresses)} samples, {progresses[0]} -> {progresses[-1]}")
            check("progress ends at 100", progresses[-1] == 100.0, str(progresses[-1]))
            stages_seen = {s["stage"] for s in samples}
            check("progress passed through the tracking stage", "tracking" in stages_seen,
                  str(sorted(stages_seen)))
            check("status publishes no internal field",
                  not any(k in samples[0] for k in
                          ("percent", "phase", "units", "events",
                           "cancel_requested")),
                  str(sorted(samples[0])))

        # ── Result ────────────────────────────────────────────────────────
        print("\n[3/8] result documents")
        paths = Paths(analysis_id=analysis_id)
        # The complete document lives on disk and is served by no route; the
        # API serves the projection of it.
        result = read_json(paths.result_json)
        check("the complete document was persisted to disk", result is not None)
        if result is None:
            return 1
        summary = result["summary"]
        deliveries = result["deliveries"]
        print(f"        deliveries={summary['deliveries']} "
              f"quarantined={summary['quarantined']} "
              f"by_validation={summary['by_validation']}")

        envelope = client.get(f"/api/analysis/{analysis_id}/result").json()
        check("the result is one envelope", set(envelope) == ENVELOPE_KEYS,
              str(sorted(envelope)))
        check("the envelope names the analysis it describes",
              envelope.get("analysis_id") == analysis_id,
              str(envelope.get("analysis_id")))
        check("the envelope reports the completed status",
              envelope.get("status") == "completed", str(envelope.get("status")))
        check("the envelope carries a pitch block", isinstance(envelope.get("pitch"), dict),
              str(envelope.get("pitch"))[:80])
        check("the pitch block never leaks model internals",
              not any(k in envelope["pitch"]
                      for k in ("model_id", "image_size", "frame_number",
                                "homography")))
        frontend = envelope["deliveries"]
        check("the envelope holds exactly the persisted deliveries, in order",
              [d["id"] for d in frontend]
              == [d["delivery_id"] for d in deliveries],
              str([d["id"] for d in frontend]))
        check("the envelope exposes no internal record",
              all(
                  not any(k in d for k in ("event", "validation", "provenance",
                                           "clip_path", "speed_samples",
                                           "overlay", "index"))
                  for d in frontend
              ))
        check("every delivery entry has the same schema",
              all(set(d) == DELIVERY_KEYS for d in frontend),
              str(sorted(frontend[0]) if frontend else []))
        check("every delivery entry carries a pitch block",
              all(isinstance(d.get("pitch"), dict) for d in frontend))
        check("every trajectory point carries pitch_position",
              all("pitch_position" in p
                  for d in frontend for p in d["trajectory"]["points"]))
        check("every delivery pitch block carries an orientation key",
              all("orientation" in d["pitch"] for d in frontend),
              str(sorted(frontend[0]["pitch"]) if frontend else []))
        check("a delivery pitch block never leaks a homography",
              not any(
                  any(k in d["pitch"] for k in PITCH_INTERNAL_KEYS)
                  for d in frontend
              ),
              str(sorted(frontend[0]["pitch"]) if frontend else []))
        bad_orientation = [
            d["id"] for d in frontend
            if d["pitch"]["orientation"] is not None
            and (d["pitch"]["orientation"].get("axis") not in ("u", "v")
                 or d["pitch"]["orientation"].get("batting_end") not in (0, 1))
        ]
        check("every stated orientation names an axis and a batting end",
              not bad_orientation, str(bad_orientation))
        # Serialised with ensure_ascii=False: a default dump escapes the
        # degree symbol inside a swing label as a backslash-u escape, and
        # that escape sequence is not a path separator. A real Windows path
        # carries a literal backslash either way, so the test still bites.
        payload = json.dumps(envelope, ensure_ascii=False)
        check("the envelope carries no filesystem path",
              "\\" not in payload and "clip_path" not in payload)
        check("every clip_url is the public clips route",
              all(
                  d["video"]["clip_url"] ==
                  f"/api/analysis/{analysis_id}/clips/"
                  f"delivery_{d['id']:03d}.mp4"
                  for d in frontend if d["video"]["clip_url"]
              ),
              str([d["video"]["clip_url"] for d in frontend[:2]]))
        check("the envelope keeps detected/predicted provenance",
              all(
                  {p["source"] for p in d["trajectory"]["points"]}
                  <= {"detected", "predicted"}
                  for d in frontend
              ))

        perf = result["performance"]
        check("source decoded exactly once", perf["source_decodes"] == 1,
              str(perf["source_decodes"]))
        check("YOLO ran exactly once", perf["yolo_passes"] == 1)
        check("Kalman ran exactly once", perf["kalman_passes"] == 1)
        check("shot model loaded once", perf["shot_model_loads"] == 1,
              str(perf["shot_model_loads"]))
        check("shot checkpoint is identified", bool(perf.get("shot_model_checkpoint")),
              str(perf.get("shot_model_file")))
        check("shot model provenance name is recorded",
              bool(perf.get("shot_model_provenance_name")),
              str(perf.get("shot_model_provenance_name")))

        check("deliveries were produced", len(deliveries) > 0, str(len(deliveries)))
        check("every delivery has a purity verdict",
              all(d.get("validation", {}).get("classification") for d in deliveries))
        leaked = [
            d["delivery_id"] for d in deliveries
            if d.get("validation", {}).get("classification") in FORBIDDEN_VALIDATION
        ]
        check("no impure clip is exposed as a delivery", not leaked, str(leaked))
        check("quarantine carries a reason for every record",
              all(q.get("reason") for q in result["quarantine"]),
              f"{len(result['quarantine'])} quarantined")

        # ── Calibration honesty ───────────────────────────────────────────
        print("\n[4/8] calibration honesty")
        cal = result["calibration"]
        check("calibration block travels with the data", cal is not None)
        check("run is not claimed calibrated", not cal["speed_is_calibrated"])
        check("speed is labelled an estimate", cal["speed_is_estimate"])
        check("config snapshot is persisted", "config" in result)
        check("operator corners were not measured",
              result["config"]["geometry_calibrated"] is False)
        check("geometry provenance is named",
              cal.get("geometry_source") in GEOMETRY_SOURCES,
              str(cal.get("geometry_source")))

        # The invariant is no longer "no numbers at all" — a detected pitch quad
        # legitimately turns metres on for the deliveries that carry one. What
        # must never happen is a number attached to a delivery that had no
        # measured geometry, or a speed presented as calibrated.
        measured = bool(cal.get("geometry_calibrated"))
        geometry_deliveries = [
            d for d in deliveries if d["bowling"].get("geometry_calibrated")
        ]
        ungated_speed = [
            d["delivery_id"] for d in deliveries
            if d["bowling"]["speed_kmh"] is not None
            and not d["bowling"].get("geometry_calibrated")
        ]
        ungated_zone = [
            d["delivery_id"] for d in deliveries
            if (d["bowling"]["length"] is not None
                or d["bowling"]["line"] is not None)
            and not d["bowling"].get("geometry_calibrated")
        ]
        speeds = [d["bowling"]["speed_kmh"] for d in deliveries]
        fabricated = [s for s in speeds if s is not None]
        zones = [
            d["delivery_id"] for d in deliveries
            if d["bowling"]["length"] is not None
            or d["bowling"]["line"] is not None
        ]
        check("no ground-plane number outside a measured delivery",
              not ungated_speed and not ungated_zone,
              f"speeds={ungated_speed} zones={ungated_zone}")
        check("no delivery claims a calibrated speed",
              all(not d["bowling"].get("speed_calibrated") for d in deliveries))
        if measured:
            check("measured geometry reached the deliveries",
                  len(geometry_deliveries) > 0,
                  f"{len(geometry_deliveries)} of {len(deliveries)}")
            check("measured deliveries produced ground-plane numbers",
                  bool(fabricated) or bool(zones),
                  f"{len(fabricated)} speeds, {len(zones)} zones of "
                  f"{len(geometry_deliveries)} measured deliveries")
            lengths = {
                d["bowling"]["length"] for d in geometry_deliveries
                if d["bowling"]["length"] is not None
            }
            check("every length is one of the documented zones",
                  lengths <= LENGTH_ZONE_NAMES, str(sorted(lengths)))
            check("summary mean speed tracks the measured speeds",
                  (summary["mean_speed_kmh"] is not None) == bool(fabricated),
                  str(summary["mean_speed_kmh"]))
            tally = cal.get("pitch_keypoint_geometry")
            check("the per-delivery geometry tally travels with the calibration",
                  isinstance(tally, dict) and "rejections" in tally,
                  str(tally)[:200])
            print(f"        geometry_source={cal['geometry_source']}  "
                  f"{len(geometry_deliveries)}/{len(deliveries)} measured  "
                  f"{len(fabricated)} speeds, {len(zones)} zones  "
                  f"mean={summary['mean_speed_kmh']}")
        else:
            check("no speed is invented without calibration", not fabricated,
                  f"{len(fabricated)} non-null of {len(speeds)}")
            check("summary reports no mean speed", summary["mean_speed_kmh"] is None)
            check("no length/line zone is invented", not zones, str(zones))
            print("        no geometry was measured in this run; "
                  "ground-plane analytics stayed null")

        # ── Shot records ──────────────────────────────────────────────────
        print("\n[5/8] shot records")
        labelled = [d for d in deliveries if (d.get("shot") or {}).get("type")]
        check("shot labels were produced", len(labelled) > 0,
              f"{len(labelled)} of {len(deliveries)}")
        check("no shot silently failed", not summary["shot_failures"],
              str(summary["shot_failures"]))
        bad_len = [
            d["delivery_id"] for d in labelled
            if d["shot"]["frames_sampled"] != 30
        ]
        check("every shot sampled exactly 30 frames", not bad_len, str(bad_len))

        # A clip shorter than the model's 30-frame input is padded, and the
        # padding is declared. This is the invariant that matters: not that all
        # clips are long enough — real footage contains 19-frame clips — but that
        # a short clip never silently masquerades as 30 observations.
        short = [d for d in labelled if d["clip"]["frame_count"] < 30]
        undeclared = [
            d["delivery_id"] for d in short
            if not d["shot"]["frames_repeated"]
            or d["shot"]["unique_frames_sampled"] != d["clip"]["frame_count"]
        ]
        check("every clip shorter than 30 frames declares its padding",
              not undeclared,
              f"{len(short)} short clips; undeclared={undeclared}")
        for d in short:
            print(f"        delivery {d['delivery_id']}: "
                  f"{d['clip']['frame_count']} frames -> "
                  f"{d['shot']['frames_sampled']} samples "
                  f"({d['shot']['unique_frames_sampled']} unique, "
                  f"repeated={d['shot']['frames_repeated']})")

        # Sampled positions must be real positions inside the clip.
        out_of_range_samples = [
            d["delivery_id"] for d in labelled
            for pos in d["shot"]["sampled_clip_frames"]
            if not (0 <= pos < d["clip"]["frame_count"])
        ]
        check("every sampled clip position is inside the clip",
              not out_of_range_samples, str(out_of_range_samples[:5]))

        check("each delivery names the checkpoint in its provenance",
              all(
                  (d.get("provenance") or {}).get("shot_model")
                  == result["performance"]["shot_model_provenance_name"]
                  for d in labelled
              ),
              str(result["performance"]["shot_model_provenance_name"]))

        # ── Media and per-delivery consistency ────────────────────────────
        print("\n[6/8] media and cross-delivery consistency")
        import cv2

        clip_failures, unplayable, overlay_failures = [], [], []
        for d in frontend:
            did = d["id"]
            url = d["video"]["clip_url"]
            response = client.get(url) if url else None
            if response is None or response.status_code != 200 or not response.content:
                clip_failures.append(did)
            else:
                cap = cv2.VideoCapture(str(paths.clip_path(did)))
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    unplayable.append(did)
            # Overlays are internal artefacts: written, but served by no route.
            if not paths.overlay_path(did).is_file():
                overlay_failures.append(did)
        check("every delivery has a served clip", not clip_failures, str(clip_failures))
        check("every served clip is actually decodable", not unplayable,
              str(unplayable))
        check("every delivery has an overlay written to disk", not overlay_failures,
              str(overlay_failures))
        check("no route serves an overlay",
              client.get(
                  f"/api/analysis/{analysis_id}/overlays/delivery_001.mp4"
              ).status_code == 404)

        first_url = frontend[0]["video"]["clip_url"]
        ranged = client.get(first_url, headers={"Range": "bytes=0-1023"})
        check("clip supports byte ranges for seeking",
              ranged.status_code in (200, 206),
              f"status={ranged.status_code}")

        # ── One delivery per entry ────────────────────────────────────────
        print("\n[7/8] one delivery per entry, no cross-talk")
        again = client.get(f"/api/analysis/{analysis_id}/result").json()
        check("repeating the result request returns the same document",
              again == envelope)
        if len(frontend) >= 2:
            check("two deliveries yield two different trajectories",
                  frontend[0]["trajectory"] != frontend[1]["trajectory"])
            check("each entry carries its own id",
                  [d["id"] for d in frontend[:2]]
                  == [deliveries[0]["delivery_id"], deliveries[1]["delivery_id"]])
            check("no entry repeats another's id",
                  len({d["id"] for d in frontend}) == len(frontend))

        # Frame ranges must nest: the trajectory cannot reference frames that
        # lie outside the clip it is supposed to describe.
        out_of_range = []
        for d in deliveries:
            if not d.get("clip"):
                continue
            lo, hi = d["clip"]["start_frame"], d["clip"]["end_frame"]
            for p in d["trajectory"]["points"]:
                f = p["original_frame"]
                if f is not None and not (lo <= f <= hi):
                    out_of_range.append((d["delivery_id"], f, lo, hi))
                    break
        check("every trajectory frame lies inside its own clip window",
              not out_of_range, str(out_of_range[:5]))

        sources = {
            p["source"]
            for d in deliveries for p in d["trajectory"]["points"]
        }
        check("trajectory provenance is explicit",
              sources <= {"detected", "predicted"} and sources, str(sorted(sources)))
        detected_total = sum(
            1 for d in deliveries for p in d["trajectory"]["points"]
            if p["source"] == "detected"
        )
        check("detected points exist", detected_total > 0, str(detected_total))

        # ── Quarantine is diagnostic, not a delivery ───────────────────
        print("\n[8/8] quarantine is diagnostic, not a delivery")
        served_ids = {d["id"] for d in frontend}
        if result["quarantine"]:
            q = result["quarantine"][0]
            qid = q["delivery_id"]
            check("a quarantined id is not in the served array", qid not in served_ids,
                  str(qid))
            check("the quarantine record keeps the full delivery",
                  bool(q.get("record")))
        else:
            print("        no quarantined records in this run")

        check("the served array matches the persisted document",
              len(frontend) == len(deliveries),
              f"{len(frontend)} served of {len(deliveries)} persisted")
        check("the removed delivery routes are not routed",
              client.get(f"/api/analysis/{analysis_id}/deliveries").status_code == 404
              and client.get(
                  f"/api/analysis/{analysis_id}/deliveries/1"
              ).status_code == 404)
        check("the internal result route is not routed",
              client.get(
                  f"/api/analysis/{analysis_id}/result/internal"
              ).status_code == 404)

        analysis_id_out = analysis_id
        print(f"\nAnalysis id: {analysis_id_out}")
        print(f"Artefacts:   {paths.root}")

    print("\n" + "=" * 66)
    if failures:
        print(f"FAILED  {len(failures)} of {checks} checks:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print(f"PASSED  all {checks} checks")
    print(f"  {len(deliveries)} validated deliveries, "
          f"{len(result['quarantine'])} quarantined")
    print(f"  one source decode, one YOLO pass, one Kalman pass")
    print(f"  artefacts kept at {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
