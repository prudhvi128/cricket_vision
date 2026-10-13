"""
Inspect trajectory vs pitch quad geometry for one run.

    python scripts/verification/inspect_geometry.py [run_dir]

Prints, per delivery: the persisted quad with its axis-aligned spans and
aspect, the ball's image-space travel, and the bounce. With no argument the
newest run in data/runs is used.

For the question "why is there no metric geometry" use
`compare_geometry_gate.py`, which replays the gate itself and reports the
refusal reason for the stored quad and for a freshly detected one.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_run(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    runs = sorted(
        (ROOT / "data" / "runs").glob("*/tracking.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise SystemExit("FATAL: no run directory with tracking.json found")
    return runs[-1].parent


def main() -> int:
    run = resolve_run(sys.argv)
    tracking_path = run / "tracking.json"
    if not tracking_path.is_file():
        print(f"FATAL: {tracking_path} not found")
        return 2
    tracking = json.loads(tracking_path.read_text())
    print(f"run: {run}\n")

    for i, d in enumerate(tracking.get("deliveries", []), 1):
        pitch = d.get("pitch") or {}
        corners = pitch.get("keypoints") or pitch.get("corners") or {}
        pts = (d.get("trajectory") or {}).get("points") or []
        bounce = d.get("bounce") or {}

        print(f"--- delivery {i} ---")
        print(
            f"  frame={pitch.get('frame_number')} conf={pitch.get('confidence')} "
            f"model={pitch.get('model_id')}"
        )
        if corners:
            tl = corners.get("top_left")
            trr = corners.get("top_right")
            br = corners.get("bottom_right")
            bl = corners.get("bottom_left")
            if tl and bl and trr:
                len_px = math.dist(tl, bl)
                wid_px = math.dist(tl, trr)
                print(f"  quad: TL={tl} TR={trr} BR={br} BL={bl}")
                print(
                    f"  quad: vertical={len_px:.0f}px horizontal={wid_px:.0f}px "
                    f"aspect(v/h)={len_px / wid_px:.2f}"
                )
        if pts:
            f0 = pts[0]
            l0 = pts[-1]
            dx = l0["x"] - f0["x"]
            dy = l0["y"] - f0["y"]
            ang = math.degrees(math.atan2(dy, dx))
            print(
                f"  traj: {len(pts)} pts first=({f0.get('x'):.1f},{f0.get('y'):.1f}) "
                f"last=({l0.get('x'):.1f},{l0.get('y'):.1f})"
            )
            print(f"  travel: dx={dx:.1f} dy={dy:.1f} angle={ang:.1f}deg")
        print(
            f"  bounce: present={bounce.get('present')} frame={bounce.get('frame')} "
            f"x={bounce.get('x')} y={bounce.get('y')}"
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
