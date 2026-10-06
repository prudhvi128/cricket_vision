"""
media.py — Serving clip and overlay bytes.

Two families of route, both serving the same files from different directions:

  * by delivery id — `/api/analysis/{id}/deliveries/{did}/clip`
  * by filename     — `/api/analyses/{id}/clips/{name}` (legacy)

A missing overlay is reported as missing rather than served as a copy of the
clip. A client can then show the clip without a trajectory, instead of drawing a
trajectory that is not there.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...core.config import Paths
from ..helpers import parse_delivery_id, serve_file

router = APIRouter()


@router.get("/analysis/{analysis_id}/deliveries/{delivery_id}/clip")
def analysis_delivery_clip(analysis_id: str, delivery_id: str) -> FileResponse:
    """The delivery clip as a playable video, with byte-range support."""
    did = parse_delivery_id(delivery_id)
    path = Paths(analysis_id=analysis_id).clip_path(did)
    if not path.is_file():
        raise HTTPException(
            404,
            f"Clip for delivery {did} was not generated. The clip may have "
            "failed to write during analysis.",
        )
    return FileResponse(path, media_type="video/mp4")


@router.get("/analysis/{analysis_id}/deliveries/{delivery_id}/overlay")
def analysis_delivery_overlay(analysis_id: str, delivery_id: str) -> FileResponse:
    """
    The overlay video, or 404 when none was rendered.

    A missing overlay is reported as missing rather than served as a copy of the
    clip: a client can then show the clip without a trajectory instead of drawing
    a trajectory that is not there.
    """
    did = parse_delivery_id(delivery_id)
    path = Paths(analysis_id=analysis_id).overlay_path(did)
    if not path.is_file():
        raise HTTPException(
            404,
            f"No overlay for delivery {did}. See the delivery record's "
            "`overlay.error` for why.",
        )
    return FileResponse(path, media_type="video/mp4")


# ── Media (name-based, legacy) ───────────────────────────────────────────────
@router.get("/analyses/{analysis_id}/clips/{name}")
def serve_clip(analysis_id: str, name: str) -> FileResponse:
    paths = Paths(analysis_id=analysis_id)
    if not name.startswith("delivery_") or not name.endswith(".mp4"):
        raise HTTPException(400, "Invalid clip name")
    return serve_file(paths.clips_dir / name, "video/mp4", "Clip")


@router.get("/analyses/{analysis_id}/overlays/{name}")
def serve_overlay(analysis_id: str, name: str) -> FileResponse:
    paths = Paths(analysis_id=analysis_id)
    if not name.startswith("delivery_") or not name.endswith(".mp4"):
        raise HTTPException(400, "Invalid overlay name")
    return serve_file(paths.overlays_dir / name, "video/mp4", "Overlay")