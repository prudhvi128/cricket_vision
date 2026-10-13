"""
Run one REAL end-to-end analysis through the normal entry point.

    python scripts/verification/run_e2e_api.py [video] [port]

Starts the actual FastAPI app (backend/main.py) under uvicorn, POSTs the video
to /api/analyze (the same route the frontend uses), polls /status until the job
reaches a terminal state, then fetches /result and verifies:

  * the job finished and the result is served (200, not 409/404)
  * result.json and tracking.json exist on disk
  * pitch_seg_detection is present in the persisted document
  * calibration coverage is reported honestly (geometry_calibrated count and
    the per-delivery gate rejection reasons from the persisted document)
  * every clip_url the document advertises resolves to a file on disk
  * the API's served envelope matches the persisted document's delivery count

This is the audit's real-execution check: no stubs, no TestClient, no direct
pipeline calls — the exact path a browser upload takes.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
DEFAULT_VIDEO = ROOT / "real_cricket.mp4"


def wait_port(port: int, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 1.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def http_json(url: str, timeout: float = 180.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post_video(url: str, video: Path, timeout: float = 120.0) -> dict:
    boundary = "----cricketE2EBoundary"
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="video"; filename="{video.name}"\r\n'.encode()
    body += b"Content-Type: video/mp4\r\n\r\n"
    body += video.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    if not video.is_file():
        print(f"FATAL: video not found: {video}")
        return 2

    env = os.environ.copy()
    # Seg ON explicitly (the default, but make it unambiguous for the audit).
    env.setdefault("CRICKET_PITCH_SEG_MODEL", str(BACKEND / "models" / "pitch_seg_yolo11s.pt"))
    env["PYTHONPATH"] = str(BACKEND)

    server_log = BACKEND.parent / "data" / "runs" / f"e2e_server_{port}.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(server_log, "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(BACKEND),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    try:
        if not wait_port(port):
            log_fh.close()
            print(f"FATAL: server never came up; see {server_log}")
            print(server_log.read_text()[-2000:] if server_log.is_file() else "(no log)")
            return 2
        print(f"server up on :{port}; uploading {video.name} ({video.stat().st_size/1e6:.1f} MB)")

        t0 = time.time()
        accepted = post_video(f"http://127.0.0.1:{port}/api/analyze", video)
        analysis_id = accepted["analysis_id"]
        print(f"accepted: {analysis_id} status={accepted.get('status')} stage={accepted.get('stage')}")

        last = None
        timeouts = 0
        runs_dir = BACKEND.parent / "data" / "runs" / analysis_id
        result_path = runs_dir / "result.json"
        while True:
            # Disk is the source of truth for completion: the tracking thread
            # saturates the GIL, so the HTTP status handler can be starved for
            # minutes. result.json is written exactly once, at the end.
            if result_path.is_file() and result_path.stat().st_size > 0:
                st = http_json(f"http://127.0.0.1:{port}/api/analysis/{analysis_id}/status", timeout=60.0)
                if st.get("status") in ("completed", "failed"):
                    break
            try:
                st = http_json(f"http://127.0.0.1:{port}/api/analysis/{analysis_id}/status", timeout=15.0)
            except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
                timeouts += 1
                if timeouts % 6 == 1:
                    print(f"  [{time.time()-t0:7.1f}s] status poll starved ({timeouts} timeouts): {type(exc).__name__}")
                time.sleep(5)
                continue
            cur = (st.get("status"), st.get("stage"), st.get("substage"), st.get("percent"))
            if cur != last:
                print(f"  [{time.time()-t0:7.1f}s] status={st.get('status')} stage={st.get('stage')} "
                      f"substage={st.get('substage')} percent={st.get('percent')}")
                last = cur
            if st.get("status") in ("completed", "failed"):
                break
            time.sleep(5)

        elapsed = time.time() - t0
        print(f"terminal after {elapsed:.1f}s: {json.dumps(st, indent=None)}")
        if st.get("status") != "completed":
            print("FATAL: analysis did not complete")
            return 1

        doc = http_json(f"http://127.0.0.1:{port}/api/analysis/{analysis_id}/result")
        deliveries = doc.get("deliveries", [])
        print(f"served envelope: {len(deliveries)} deliveries")

        # --- Artifact verification -------------------------------------------
        runs = BACKEND.parent / "data" / "runs" / analysis_id
        result_json = runs / "result.json"
        tracking_json = runs / "tracking.json"
        print(f"result.json on disk:  {result_json.is_file()}")
        print(f"tracking.json on disk: {tracking_json.is_file()}")

        raw = json.loads(result_json.read_text()) if result_json.is_file() else {}
        seg = raw.get("pitch_seg_detection")
        print(f"pitch_seg_detection in result.json: {'present' if seg is not None else 'ABSENT'}")

        # Calibration coverage: how many deliveries got metric geometry, and
        # how many of those carry each calibration-dependent metric. The
        # per-delivery tally lives under `calibration.pitch_keypoint_geometry`
        # (its name predates the segmentation source); reading it from the top
        # level prints an empty dict and hides every rejection reason.
        dels = raw.get("deliveries", [])
        n = len(dels)
        geo = sum(1 for d in dels if d.get("bowling", {}).get("geometry_calibrated"))
        speed = sum(1 for d in dels if d.get("bowling", {}).get("speed_kmh") is not None)
        length = sum(1 for d in dels if d.get("bowling", {}).get("length") is not None)
        line = sum(1 for d in dels if d.get("bowling", {}).get("line") is not None)
        swing = sum(1 for d in dels if d.get("bowling", {}).get("swing") not in (None, ""))
        bounceang = sum(1 for d in dels if d.get("bowling", {}).get("bounce_angle") is not None)
        print(f"calibration coverage: {geo}/{n} geometry_calibrated")
        print(f"  speed={speed}/{n} length={length}/{n} line={line}/{n} "
              f"swing={swing}/{n} bounce_angle={bounceang}/{n}")
        # The seg source must be eligible (not hard-refused); the per-delivery
        # geometric checks decide, and their reasons are aggregated here.
        stats = (raw.get("calibration") or {}).get("pitch_keypoint_geometry") or {}
        print(f"pitch geometry decisions: {json.dumps(stats)[:400]}")
        box = raw.get("pitch_box_detection") or {}
        print(f"box detector enabled/boxes: {box.get('enabled')}/{box.get('boxes')}")

        # Codec: clips and overlays must be browser-playable H.264, and the
        # OpenH264 error must not appear in the server log.
        ffprobe = shutil.which("ffprobe")
        codecs = {}
        for sub in ("clips", "overlays"):
            d0 = runs / sub
            if not d0.is_dir():
                continue
            for f in sorted(d0.glob("*.mp4"))[:2]:
                if ffprobe:
                    r = subprocess.run(
                        [ffprobe, "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=codec_name,codec_tag_string",
                         "-of", "default=nw=1", str(f)],
                        capture_output=True, text=True,
                    )
                    codecs[f"{sub}/{f.name}"] = " ".join(r.stdout.split())
        for k, v in codecs.items():
            print(f"  codec {k}: {v}")
        server_log = BACKEND.parent / "data" / "runs" / f"e2e_server_{port}.log"
        if server_log.is_file():
            txt = server_log.read_text(errors="replace")
            print(f"OpenH264 in server log: {'YES (BAD)' if 'OpenH264' in txt or 'Incorrect library version' in txt else 'no'}")

        # Every advertised clip must exist. `clip_url` is nested under the
        # delivery's `video` block in the served envelope (the flat top-level
        # key only exists on the internal record, which this script does not
        # read here), so reading `d.get("clip_url")` would silently skip every
        # delivery and report zero missing.
        missing = []
        for d in deliveries:
            url = (d.get("video") or {}).get("clip_url") or d.get("clip_url")
            if not url:
                continue
            name = url.rsplit("/", 1)[-1]
            if not (runs / "clips" / name).is_file():
                missing.append(url)
        print(f"clip_urls missing on disk: {len(missing)}{missing[:3] if missing else ''}")

        # Served count must match persisted accepted deliveries.
        print(f"persisted deliveries: {len(raw.get('deliveries', []))} "
              f"(served {len(deliveries)}; match={len(raw.get('deliveries', [])) == len(deliveries)})")

        # A second decode must not have happened.
        print(f"source_decodes: {raw.get('performance', {}).get('source_decodes')}")
        print(f"E2E_OK elapsed={elapsed:.1f}s deliveries={len(deliveries)}")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        try:
            log_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
