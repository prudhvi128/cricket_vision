"""
pitch_map.py — The trajectory plot returned by the API.

This is presentation, not measurement: it draws the points the tracker actually
recorded, in the pixels it actually recorded them, and nothing else. It is not a
to-scale pitch. That distinction is the reason it lives here rather than inside
the route handler that returns it — a coordinate plot of real measurements is a
view concern, and a reader should be able to check what it draws.
"""

from __future__ import annotations

from typing import Optional


def pitch_map_svg(delivery: dict) -> Optional[str]:
    """
    Render the stored trajectory as an SVG, or return None.

    Uses only pixel and normalised coordinates the tracker actually measured. A
    ground-plane pitch map needs a homography calibrated on this specific video,
    which this deployment does not have, so no metres are drawn and no axis is
    labelled with a distance. The SVG is a coordinate plot of real measurements,
    not a to-scale pitch.
    """
    traj = (delivery.get("trajectory") or {}).get("points") or []
    clip = delivery.get("clip") or {}
    w = clip.get("width")
    h = clip.get("height")
    if not w or not h:
        return None
    pts = [p for p in traj if p.get("x") is not None and p.get("y") is not None]
    if not pts:
        return None

    def esc(v) -> str:
        return str(v).replace("&", "&amp;").replace("<", "&lt;")

    segments = []
    run: list[str] = []
    prev_frame = None
    for p in pts:
        x = p.get("clip_frame")
        y = p.get("y")
        if p.get("x") is None:
            continue
        # A gap in clip-frame numbering means no point exists for that frame;
        # joining across it would draw a line the tracker never measured.
        if prev_frame is not None and x is not None and x != prev_frame + 1:
            segments.append(" ".join(run))
            run = []
        run.append(f"{p['x']},{p['y']}")
        prev_frame = x
    if run:
        segments.append(" ".join(run))

    strokes = "".join(
        f'<polyline points="{esc(s)}" fill="none" stroke="#ff2d55" '
        f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        for s in segments if len(s.split()) > 1
    )
    dots = "".join(
        f'<circle cx="{p["x"]}" cy="{p["y"]}" r="4" fill="{"#22d3ee" if p.get("source") == "detected" else "#f59e0b"}">'
        f'<title>frame {p.get("original_frame")} ({p.get("source")})</title></circle>'
        for p in pts
    )
    bounce = delivery.get("bounce") or {}
    marker = ""
    if bounce.get("x_px") is not None:
        marker = (
            f'<circle cx="{bounce["x_px"]}" cy="{bounce["y_px"]}" r="14" '
            f'fill="none" stroke="#ffffff" stroke-width="3"/>'
            f'<text x="{bounce["x_px"]}" y="{int(bounce["y_px"]) - 20}" '
            f'fill="#ffffff" font-size="24" text-anchor="middle">'
            f'bounce ({"scored" if bounce.get("detected") else "lowest tracked"})'
            f'</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}">'
        f'<rect width="{w}" height="{h}" fill="#000000"/>'
        f'{strokes}{dots}{marker}</svg>'
    )


__all__ = ["pitch_map_svg"]