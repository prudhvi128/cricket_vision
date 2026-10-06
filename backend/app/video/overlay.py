"""
overlay_renderer.py — Renders the stored trajectory onto a delivery clip.

THE OVERLAY IS A VIEW OF THE TRACKING RESULT, NOT A SECOND ANALYSIS
------------------------------------------------------------------
Upstream CricketTracker drew its overlay by calling `draw_trajectory_trail` on
the live frame inside the tracking loop, using `tracker.traj_points` — a
120-point `deque` that is cleared on every delivery boundary. That deque is
therefore far too short to draw a whole delivery, and it is in-memory state, not
a record: after the pass there is nothing to re-render from.

Here the overlay is drawn from `Delivery.trajectory` — the same persisted points
that feed the bounce marker, the pitch map, the bowling analytics and the API
response. There is no second detector and no second tracker involved. Decision
#6 and C3: the stored trajectory is the single source of truth.

WHY IT RENDERS FROM THE CLIP FILE
---------------------------------
Rendering from the written clip costs one sequential decode of a ~5-second file.
Rendering inside the tracking loop would require holding every clip frame in RAM
for the whole post-roll window — about 750 MB per delivery at 1080p, which the
ring buffer exists precisely to avoid. Decision #5 explicitly permits decoding
the clip later; it forbids re-opening the SOURCE video, which this never does.

FRAME MAPPING IS CHECKED, NOT ASSUMED
-------------------------------------
The overlay is only drawn when the clip's frame map is valid. If frames were lost
or the clip was truncated at end of video, `clip_frame` no longer corresponds to
the stored `x/y`, and a trajectory drawn on a mis-numbered clip looks entirely
plausible while being wrong. In that case the delivery gets
`overlay.rendered=false` and an explicit reason instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..core.config import Paths
from ..schemas.tracking import ClipMapError, Delivery, OverlayInfo
from .renderer import (
    draw_bounce_marker,
    draw_release_marker,
    draw_trajectory_trail,
)

log = logging.getLogger(__name__)


class OverlayRenderer:
    """
    Renders one overlay video per delivery.

    Stateless and reusable; call `render` once per delivery.
    """

    def __init__(self, codec_preference: tuple = ("avc1", "mp4v")) -> None:
        self.codec_preference = codec_preference

    def render(self, delivery: Delivery, paths: Paths) -> OverlayInfo:
        """Render `delivery`'s overlay. Never raises; failures are reported."""
        out_path = paths.overlay_path(delivery.delivery_id)

        # ── Preconditions ───────────────────────────────────────────────────
        if delivery.clip is None:
            return OverlayInfo(
                rendered=False,
                trajectory_source="stored",
                error="no clip was written for this delivery",
            )
        if not delivery.clip.frame_map_valid:
            return OverlayInfo(
                rendered=False,
                trajectory_source="stored",
                error=(
                    "clip frame map is invalid "
                    f"({delivery.clip.frame_map_error}); overlay suppressed "
                    "because stored coordinates would land on the wrong frames"
                ),
            )
        if not delivery.clip_path or not Path(delivery.clip_path).is_file():
            return OverlayInfo(
                rendered=False,
                trajectory_source="stored",
                error=f"clip file missing: {delivery.clip_path}",
            )

        frame_map = delivery.clip
        points = [p for p in delivery.trajectory.points if p.clip_frame is not None]

        if not points:
            # No trajectory to draw. Returning a copy of the clip would imply a
            # render happened; say plainly that there was nothing to draw.
            log.info(
                "Delivery %d has no trajectory points inside the clip; "
                "skipping overlay", delivery.delivery_id,
            )
            return OverlayInfo(
                rendered=False,
                trajectory_source="stored",
                error="no trajectory points fall inside the clip window",
            )

        # ── Read the clip sequentially ──────────────────────────────────────
        cap = cv2.VideoCapture(str(delivery.clip_path))
        if not cap.isOpened():
            return OverlayInfo(
                rendered=False,
                trajectory_source="stored",
                error=f"could not open clip for reading: {delivery.clip_path}",
            )

        try:
            ok, first = cap.read()
            if not ok or first is None:
                return OverlayInfo(
                    rendered=False, trajectory_source="stored",
                    error="clip decoded zero frames",
                )

            h, w = first.shape[:2]
            if (w, h) != (frame_map.width, frame_map.height):
                return OverlayInfo(
                    rendered=False,
                    trajectory_source="stored",
                    error=(
                        f"clip is {w}x{h} but the frame map was built for "
                        f"{frame_map.width}x{frame_map.height}; coordinates "
                        "would be misaligned"
                    ),
                )

            writer = self._open_writer(out_path, frame_map.fps, (w, h))
            if writer is None:
                return OverlayInfo(
                    rendered=False, trajectory_source="stored",
                    error="no usable codec to write the overlay",
                )

            try:
                rendered_frames = self._draw_pass(
                    cap, writer, first, delivery, frame_map, points
                )
            finally:
                writer.release()
        finally:
            cap.release()

        expected = frame_map.frame_count
        if rendered_frames != expected:
            return OverlayInfo(
                rendered=False,
                trajectory_source="stored",
                error=(
                    f"overlay wrote {rendered_frames} of {expected} frames; "
                    "clip and overlay lengths disagree"
                ),
            )

        return OverlayInfo(
            path=str(out_path),
            url=f"{paths.public_prefix}/overlays/delivery_{delivery.delivery_id:03d}.mp4",
            rendered=True,
            trajectory_source="stored",
        )

    # ── Internals ────────────────────────────────────────────────────────────
    def _open_writer(
        self, out_path: Path, fps: float, size: tuple[int, int]
    ) -> Optional[cv2.VideoWriter]:
        """Same isOpened()-checked fallback chain as ClipWriter."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        safe_fps = float(fps) if fps and fps > 0 else 25.0
        errors = []
        for codec in self.codec_preference:
            try:
                candidate = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*codec), safe_fps, size
                )
            except Exception as exc:
                errors.append(f"{codec}: {exc}")
                continue
            if candidate is not None and candidate.isOpened():
                return candidate
            if candidate is not None:
                candidate.release()
            errors.append(f"{codec}: isOpened() False")
        log.error("Overlay writer could not open %s (%s)", out_path.name, "; ".join(errors))
        return None

    def _draw_pass(
        self,
        cap: cv2.VideoCapture,
        writer: cv2.VideoWriter,
        first_frame: np.ndarray,
        delivery: Delivery,
        frame_map,
        points: list,
    ) -> int:
        """
        Stream the clip, growing the trail as each tracked frame comes due.

        Trajectory points are consumed in `clip_frame` order and revealed as the
        playhead passes them, so the overlay animates along the real tracked path
        instead of showing the finished line from frame one.
        """
        by_clip_frame: dict[int, list] = {}
        for p in points:
            by_clip_frame.setdefault(p.clip_frame, []).append(p)

        bounce_frame = None
        if delivery.bounce.original_frame is not None:
            bounce_frame = frame_map.to_clip_frame(delivery.bounce.original_frame)
            if not (1 <= bounce_frame <= frame_map.frame_count):
                bounce_frame = None

        release_frame = None
        release_point = None
        if delivery.release_frame is not None:
            release_frame = frame_map.to_clip_frame(delivery.release_frame)
            if not (1 <= release_frame <= frame_map.frame_count):
                release_frame = None
            else:
                # The release marker belongs on the tracked point at that frame,
                # not on the first point in the clip.
                release_point = next(
                    (p for p in points if p.clip_frame == release_frame), None
                )
                if release_point is None:
                    release_frame = None

        ordered = sorted(
            points, key=lambda p: (p.clip_frame is None, p.clip_frame)
        )
        revealed: list[tuple] = []
        cursor = 0

        rendered = 0
        clip_frame = 1
        frame = first_frame
        while frame is not None:
            while cursor < len(ordered) and ordered[cursor].clip_frame <= clip_frame:
                revealed.append((ordered[cursor].x, ordered[cursor].y))
                cursor += 1

            out = frame.copy()

            # Trailing line first, so markers sit on top of it.
            if len(revealed) >= 2:
                draw_trajectory_trail(out, revealed)

            if release_frame is not None and clip_frame == release_frame:
                draw_release_marker(out, (release_point.x, release_point.y))

            if bounce_frame is not None and clip_frame == bounce_frame:
                draw_bounce_marker(
                    out, (delivery.bounce.x_px, delivery.bounce.y_px)
                )

            writer.write(out)
            rendered += 1
            clip_frame += 1

            ok, frame = cap.read()
            if not ok:
                break

        return rendered