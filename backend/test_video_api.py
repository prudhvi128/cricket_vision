import argparse
import json
import sys
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(
        description="Run a cricket video through the unified backend API"
    )
    parser.add_argument("video", help="Path to cricket video")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Backend URL",
    )
    parser.add_argument(
        "--output",
        default="analysis_result.json",
        help="Output JSON filename",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2,
        help="Seconds between status checks",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="Maximum analysis time in seconds",
    )

    args = parser.parse_args()

    video = Path(args.video).resolve()
    base_url = args.base_url.rstrip("/")

    if not video.exists():
        print(f"ERROR: Video not found: {video}")
        sys.exit(1)

    print("=" * 60)
    print("CRICKET VIDEO API TEST")
    print("=" * 60)
    print(f"Video:   {video}")
    print(f"Backend: {base_url}")

    # ---------------------------------------------------------
    # 1. Check backend
    # ---------------------------------------------------------
    print("\n[1/4] Checking backend...")

    try:
        response = requests.get(
            f"{base_url}/api/health",
            timeout=10,
        )
        response.raise_for_status()
        print("Backend is reachable.")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"ERROR: Cannot reach backend: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Upload video
    # ---------------------------------------------------------
    print("\n[2/4] Uploading video...")
    print("Please wait...")

    try:
        with video.open("rb") as f:
            response = requests.post(
                f"{base_url}/api/analyze",
                files={
                    "video": (
                        video.name,
                        f,
                        "video/mp4",
                    )
                },
                timeout=120,
            )
    except Exception as e:
        print(f"ERROR: Upload failed: {e}")
        sys.exit(1)

    if not response.ok:
        print(f"ERROR: HTTP {response.status_code}")
        print(response.text)
        sys.exit(1)

    upload = response.json()

    print("\nUpload response:")
    print(json.dumps(upload, indent=2))

    analysis_id = upload.get("analysis_id")

    if not analysis_id:
        print("\nERROR: Backend did not return analysis_id.")
        sys.exit(1)

    print(f"\nAnalysis ID: {analysis_id}")

    # ---------------------------------------------------------
    # 3. Poll analysis status
    # ---------------------------------------------------------
    print("\n[3/4] Waiting for analysis...")

    start_time = time.time()
    last_status = None
    last_progress = None

    while True:

        elapsed = time.time() - start_time

        if elapsed > args.timeout:
            print("\nERROR: Analysis timed out.")
            sys.exit(1)

        try:
            response = requests.get(
                f"{base_url}/api/analysis/{analysis_id}/status",
                timeout=30,
            )
        except Exception as e:
            print(f"ERROR: Status request failed: {e}")
            sys.exit(1)

        if not response.ok:
            print(f"ERROR: HTTP {response.status_code}")
            print(response.text)
            sys.exit(1)

        status_data = response.json()

        status = status_data.get("status", "unknown")
        progress = status_data.get("progress")

        if status != last_status or progress != last_progress:
            print(
                f"Status: {status}"
                + (
                    f" | Progress: {progress}"
                    if progress is not None
                    else ""
                )
            )

            last_status = status
            last_progress = progress

        # Successful states
        if status in {
            "completed",
            "complete",
            "success",
        }:
            print("\nAnalysis completed.")
            break

        # Failure states
        if status in {
            "failed",
            "error",
            "cancelled",
            "canceled",
        }:
            print("\nANALYSIS FAILED:")
            print(json.dumps(status_data, indent=2))
            sys.exit(1)

        time.sleep(args.interval)

    # ---------------------------------------------------------
    # 4. Get final result
    # ---------------------------------------------------------
    print("\n[4/4] Fetching final result...")

    try:
        response = requests.get(
            f"{base_url}/api/analysis/{analysis_id}/result",
            timeout=60,
        )
    except Exception as e:
        print(f"ERROR: Result request failed: {e}")
        sys.exit(1)

    if not response.ok:
        print(f"ERROR: HTTP {response.status_code}")
        print(response.text)
        sys.exit(1)

    result = response.json()

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------
    output = Path(args.output).resolve()

    output.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    print(f"Analysis ID: {analysis_id}")
    print(f"JSON saved:  {output}")

    print("\nFINAL JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()